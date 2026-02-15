"""
Multi-Agent Deliberation Orchestrator for the GitHub Issue Routing System.

This module implements the core deliberation loop that coordinates
multi-agent participation in round-based discussions about GitHub issues.

The orchestrator follows this flow:
  Phase 1: Agent Selection - Moderator selects relevant specialist agents
  Phase 2: Round Loop - Agents evaluate and comment in parallel rounds
  Phase 3: Convergence Check - After each round, check if discussion should continue
  Phase 4: Synthesis - Moderator synthesizes final recommendations

Key Design Decisions:
  - Agents decide independently whether to comment each round
  - All agent evaluations within a round happen concurrently (asyncio.gather)
  - State is persisted to database after each round for crash recovery
  - Graceful error handling: individual agent failures don't stop the deliberation
  - Configurable via WorkflowConfig (max_rounds, convergence_threshold, etc.)
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Optional

from src.agents.base import BaseAgent
from src.agents.registry import AgentRegistry
from src.database import crud
from src.database.engine import get_async_session
from src.utils.logging import get_logger
from src.models.agent import AgentDecision
from src.models.workflow import (
    Comment,
    ContinueDecision,
    WorkflowConfig,
    WorkflowInstance,
    WorkflowMetrics,
    WorkflowStatus,
)
from src.orchestration.convergence import (
    ConvergenceDetector,
    measure_agreement_level,
    measure_value_added,
)
from src.orchestration.moderator import AgentSelectionResult, ModeratorAgent
from src.orchestration.recovery import (
    WorkflowRecoveryError,
    get_last_convergence_state,
    load_workflow_for_recovery,
    reconstruct_workflow,
)
from src.orchestration.synthesis import RecommendationSynthesizer, SynthesisResult

logger = get_logger(__name__)


class DeliberationResult:
    """
    Result of a completed deliberation.

    Contains the final workflow state, summary, and metrics
    from the completed multi-agent discussion.
    """

    def __init__(
        self,
        workflow: WorkflowInstance,
        summary: Optional[str] = None,
        total_rounds: int = 0,
        total_comments: int = 0,
        final_convergence_score: float = 0.0,
        termination_reason: str = "",
        agent_participation: Optional[dict[str, int]] = None,
        round_metrics: Optional[list[dict[str, Any]]] = None,
        duration_seconds: float = 0.0,
    ) -> None:
        self.workflow = workflow
        self.summary = summary
        self.total_rounds = total_rounds
        self.total_comments = total_comments
        self.final_convergence_score = final_convergence_score
        self.termination_reason = termination_reason
        self.agent_participation = agent_participation or {}
        self.round_metrics = round_metrics or []
        self.duration_seconds = duration_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a dictionary."""
        return {
            "instance_id": self.workflow.instance_id,
            "status": self.workflow.status.value,
            "summary": self.summary,
            "total_rounds": self.total_rounds,
            "total_comments": self.total_comments,
            "final_convergence_score": round(self.final_convergence_score, 3),
            "termination_reason": self.termination_reason,
            "agent_participation": self.agent_participation,
            "round_metrics": self.round_metrics,
            "duration_seconds": round(self.duration_seconds, 2),
            "conversation_history": [
                {
                    "round": c.round,
                    "agent": c.agent,
                    "comment": c.comment,
                    "references": c.references,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in self.workflow.conversation_history
            ],
        }


class MultiAgentDeliberationOrchestrator:
    """
    Orchestrates multi-agent deliberation on GitHub issues.

    Coordinates the full deliberation lifecycle:
    1. Agent selection via ModeratorAgent
    2. Round-based parallel agent evaluation
    3. Convergence detection after each round
    4. Final synthesis of recommendations
    5. Database persistence throughout

    Usage:
        orchestrator = MultiAgentDeliberationOrchestrator(
            registry=agent_registry,
            moderator=moderator_agent,
        )
        result = await orchestrator.deliberate_on_issue(issue_data)
    """

    def __init__(
        self,
        registry: AgentRegistry,
        moderator: Optional[ModeratorAgent] = None,
        convergence_detector: Optional[ConvergenceDetector] = None,
        config: Optional[WorkflowConfig] = None,
        persist_to_db: bool = True,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            registry: Agent registry for loading specialist agents
            moderator: Moderator agent for selection and synthesis
            convergence_detector: Detector for convergence checking
            config: Default workflow configuration
            persist_to_db: Whether to persist state to database
        """
        self.registry = registry
        self.moderator = moderator or ModeratorAgent(registry=registry)
        self.convergence_detector = convergence_detector or ConvergenceDetector()
        self.default_config = config or WorkflowConfig()
        self.persist_to_db = persist_to_db
        self._logger = get_logger(__name__, component="orchestrator")

    async def deliberate_on_issue(
        self,
        issue: dict[str, Any],
        config: Optional[WorkflowConfig] = None,
        workflow_id: str = "github-issue-deliberation",
    ) -> DeliberationResult:
        """
        Run a full multi-agent deliberation on a GitHub issue.

        This is the main entry point that implements the complete
        deliberation lifecycle from agent selection through synthesis.

        Args:
            issue: GitHub issue data with 'title', 'body', 'labels', 'number'
            config: Optional workflow configuration override
            workflow_id: Workflow definition identifier

        Returns:
            DeliberationResult with complete deliberation outcome

        Raises:
            ValueError: If issue data is missing required fields
        """
        start_time = time.time()
        wf_config = config or self.default_config

        # Validate issue data
        issue_title = issue.get("title", "")
        issue_number = issue.get("number", 0)
        if not issue_title:
            raise ValueError("Issue must have a 'title' field")

        # Create workflow instance
        instance_id = self._generate_instance_id(issue_number)
        workflow = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            github_issue=issue,
            config=wf_config,
            metrics=WorkflowMetrics(
                started_at=datetime.utcnow(),
            ),
        )

        self._logger.info(
            "Starting deliberation for issue #%s: %s (instance=%s)",
            issue_number,
            issue_title[:80],
            instance_id,
        )

        # Persist initial workflow state
        db_workflow_pk = None
        if self.persist_to_db:
            db_workflow_pk = await self._persist_workflow_created(workflow, issue)

        round_metrics_list: list[dict[str, Any]] = []

        try:
            # ========================
            # Phase 1: Agent Selection
            # ========================
            workflow.status = WorkflowStatus.RUNNING
            if self.persist_to_db and db_workflow_pk:
                await self._update_db_status(db_workflow_pk, "running")

            selection_result = await self._select_agents(issue, workflow)
            workflow.selected_agents = selection_result.agent_names

            self._logger.info(
                "Phase 1 complete: %d agents selected - %s",
                selection_result.count,
                selection_result.agent_names,
            )

            # Persist agent selections to DB
            if self.persist_to_db and db_workflow_pk:
                await self._persist_agent_selections(
                    db_workflow_pk, selection_result
                )

            # ========================
            # Phase 2: Round Loop
            # ========================
            termination_reason = ""
            final_convergence_score = 0.0

            for round_num in range(1, wf_config.max_rounds + 1):
                workflow.current_round = round_num

                self._logger.info(
                    "=== Round %d/%d starting ===",
                    round_num,
                    wf_config.max_rounds,
                )

                # Execute round: all agents evaluate and comment in parallel
                round_comments = await self._execute_round(
                    round_num=round_num,
                    agents=selection_result.agents,
                    workflow=workflow,
                )

                # Add round comments to conversation history
                workflow.conversation_history.extend(round_comments)

                self._logger.info(
                    "Round %d: %d agents commented (total history: %d comments)",
                    round_num,
                    len(round_comments),
                    len(workflow.conversation_history),
                )

                # Persist round state to DB
                if self.persist_to_db and db_workflow_pk:
                    await self._persist_round(
                        db_workflow_pk,
                        round_num,
                        round_comments,
                        workflow,
                    )

                # ========================
                # Phase 3: Convergence Check
                # ========================
                decision = await self.convergence_detector.should_continue(
                    round_num=round_num,
                    round_comments=round_comments,
                    full_history=workflow.conversation_history,
                    config=wf_config,
                )

                # Collect round metrics
                metrics = await self.convergence_detector.get_round_metrics(
                    round_num=round_num,
                    round_comments=round_comments,
                    full_history=workflow.conversation_history,
                )
                round_metrics_list.append(metrics)

                final_convergence_score = decision.convergence_score

                # Persist convergence metrics to DB
                if self.persist_to_db and db_workflow_pk:
                    await self._persist_convergence_metric(
                        db_workflow_pk, round_num, decision, round_comments
                    )

                self._logger.info(
                    "Round %d convergence: score=%.2f, continue=%s, reason=%s",
                    round_num,
                    decision.convergence_score,
                    decision.should_continue,
                    decision.reason,
                )

                if not decision.should_continue:
                    termination_reason = decision.reason
                    break
            else:
                # Loop completed without breaking (max rounds reached)
                termination_reason = (
                    f"Maximum rounds ({wf_config.max_rounds}) reached"
                )

            # ========================
            # Phase 4: Synthesis
            # ========================
            summary = await self._synthesize_recommendations(workflow)
            workflow.summary = summary
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()

            # Update metrics
            agent_participation = self._calculate_agent_participation(workflow)
            workflow.metrics.current_round = workflow.current_round
            workflow.metrics.total_comments = len(workflow.conversation_history)
            workflow.metrics.convergence_score = final_convergence_score
            workflow.metrics.participating_agents = len(agent_participation)

            # Persist final state
            if self.persist_to_db and db_workflow_pk:
                await self._persist_workflow_completed(
                    db_workflow_pk,
                    workflow,
                    final_convergence_score,
                    summary,
                )

            duration = time.time() - start_time
            self._logger.info(
                "Deliberation complete: %d rounds, %d comments, "
                "convergence=%.2f, duration=%.1fs",
                workflow.current_round,
                len(workflow.conversation_history),
                final_convergence_score,
                duration,
            )

            return DeliberationResult(
                workflow=workflow,
                summary=summary,
                total_rounds=workflow.current_round,
                total_comments=len(workflow.conversation_history),
                final_convergence_score=final_convergence_score,
                termination_reason=termination_reason,
                agent_participation=agent_participation,
                round_metrics=round_metrics_list,
                duration_seconds=duration,
            )

        except Exception as e:
            # Handle deliberation failure
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()

            self._logger.error(
                "Deliberation failed for issue #%s: %s",
                issue_number,
                str(e),
                exc_info=True,
            )

            if self.persist_to_db and db_workflow_pk:
                await self._update_db_status(
                    db_workflow_pk, "failed",
                    summary=f"Error: {str(e)}",
                )

            duration = time.time() - start_time
            return DeliberationResult(
                workflow=workflow,
                summary=f"Deliberation failed: {str(e)}",
                total_rounds=workflow.current_round,
                total_comments=len(workflow.conversation_history),
                final_convergence_score=final_convergence_score,
                termination_reason=f"Error: {str(e)}",
                agent_participation=self._calculate_agent_participation(workflow),
                round_metrics=round_metrics_list,
                duration_seconds=duration,
            )

    # ========================
    # Workflow Recovery & Resume
    # ========================

    async def resume_deliberation(
        self,
        instance_id: str,
    ) -> DeliberationResult:
        """
        Resume an interrupted deliberation from its last saved state.

        Loads the workflow from the database, reconstructs the in-memory
        state, and continues the deliberation loop from the next round.

        Args:
            instance_id: The workflow instance ID to resume.

        Returns:
            DeliberationResult with the completed deliberation outcome.

        Raises:
            WorkflowRecoveryError: If the workflow cannot be found or reconstructed.
            ValueError: If the workflow is not in a resumable state.
        """
        start_time = time.time()

        self._logger.info(
            "Attempting to resume deliberation: %s", instance_id
        )

        # Load and reconstruct workflow from database
        workflow = await load_workflow_for_recovery(instance_id)
        if workflow is None:
            raise WorkflowRecoveryError(
                f"Workflow '{instance_id}' not found in database"
            )

        # Validate the workflow is in a resumable state
        if workflow.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PENDING):
            raise ValueError(
                f"Workflow '{instance_id}' has status '{workflow.status.value}' "
                f"and cannot be resumed. Only 'running' or 'pending' workflows "
                f"can be resumed."
            )

        wf_config = workflow.config
        resume_round = workflow.current_round + 1

        self._logger.info(
            "Resuming workflow %s from round %d (max=%d, "
            "existing comments=%d, agents=%s)",
            instance_id,
            resume_round,
            wf_config.max_rounds,
            len(workflow.conversation_history),
            workflow.selected_agents,
        )

        # Get the DB primary key for persistence
        db_workflow_pk = await self._get_db_workflow_pk(instance_id)

        # Re-resolve agents from the registry
        agents = []
        for agent_name in workflow.selected_agents:
            agent = self.registry.get_agent(agent_name)
            if agent is not None:
                agents.append(agent)
            else:
                self._logger.warning(
                    "Agent '%s' not found in registry during recovery, skipping",
                    agent_name,
                )

        if not agents:
            raise WorkflowRecoveryError(
                f"No agents could be resolved for workflow '{instance_id}'"
            )

        round_metrics_list: list[dict[str, Any]] = []
        termination_reason = ""
        final_convergence_score = 0.0

        # Check if deliberation was already converged
        if db_workflow_pk:
            session = await get_async_session()
            async with session:
                db_wf = await crud.get_workflow_by_id(
                    session, db_workflow_pk, load_relations=True
                )
                if db_wf:
                    conv_state = get_last_convergence_state(db_wf)
                    if not conv_state["should_continue"]:
                        self._logger.info(
                            "Workflow already converged at round %d, "
                            "proceeding to synthesis",
                            conv_state["last_round"],
                        )
                        resume_round = wf_config.max_rounds + 1  # Skip loop

        try:
            workflow.status = WorkflowStatus.RUNNING

            # Continue the deliberation loop from the resume round
            for round_num in range(resume_round, wf_config.max_rounds + 1):
                workflow.current_round = round_num

                self._logger.info(
                    "=== Resumed Round %d/%d ===",
                    round_num,
                    wf_config.max_rounds,
                )

                # Execute round: all agents evaluate in parallel
                round_comments = await self._execute_round(
                    round_num=round_num,
                    agents=agents,
                    workflow=workflow,
                )

                workflow.conversation_history.extend(round_comments)

                # Persist round state
                if self.persist_to_db and db_workflow_pk:
                    await self._persist_round(
                        db_workflow_pk, round_num, round_comments, workflow
                    )

                # Convergence check
                decision = await self.convergence_detector.should_continue(
                    round_num=round_num,
                    round_comments=round_comments,
                    full_history=workflow.conversation_history,
                    config=wf_config,
                )

                metrics = await self.convergence_detector.get_round_metrics(
                    round_num=round_num,
                    round_comments=round_comments,
                    full_history=workflow.conversation_history,
                )
                round_metrics_list.append(metrics)

                final_convergence_score = decision.convergence_score

                if self.persist_to_db and db_workflow_pk:
                    await self._persist_convergence_metric(
                        db_workflow_pk, round_num, decision, round_comments
                    )

                self._logger.info(
                    "Resumed round %d: score=%.2f, continue=%s",
                    round_num,
                    decision.convergence_score,
                    decision.should_continue,
                )

                if not decision.should_continue:
                    termination_reason = decision.reason
                    break
            else:
                termination_reason = (
                    f"Maximum rounds ({wf_config.max_rounds}) reached"
                )

            # Synthesis
            summary = await self._synthesize_recommendations(workflow)
            workflow.summary = summary
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()

            agent_participation = self._calculate_agent_participation(workflow)

            if self.persist_to_db and db_workflow_pk:
                await self._persist_workflow_completed(
                    db_workflow_pk, workflow, final_convergence_score, summary
                )

            duration = time.time() - start_time
            self._logger.info(
                "Resumed deliberation complete: %d rounds, %d comments, "
                "convergence=%.2f, duration=%.1fs",
                workflow.current_round,
                len(workflow.conversation_history),
                final_convergence_score,
                duration,
            )

            return DeliberationResult(
                workflow=workflow,
                summary=summary,
                total_rounds=workflow.current_round,
                total_comments=len(workflow.conversation_history),
                final_convergence_score=final_convergence_score,
                termination_reason=f"Resumed: {termination_reason}",
                agent_participation=agent_participation,
                round_metrics=round_metrics_list,
                duration_seconds=duration,
            )

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()

            self._logger.error(
                "Resumed deliberation failed for %s: %s",
                instance_id,
                str(e),
                exc_info=True,
            )

            if self.persist_to_db and db_workflow_pk:
                await self._update_db_status(
                    db_workflow_pk, "failed", summary=f"Error during resume: {str(e)}"
                )

            duration = time.time() - start_time
            return DeliberationResult(
                workflow=workflow,
                summary=f"Resumed deliberation failed: {str(e)}",
                total_rounds=workflow.current_round,
                total_comments=len(workflow.conversation_history),
                final_convergence_score=final_convergence_score,
                termination_reason=f"Error: {str(e)}",
                agent_participation=self._calculate_agent_participation(workflow),
                round_metrics=round_metrics_list,
                duration_seconds=duration,
            )

    async def _get_db_workflow_pk(self, instance_id: str) -> Optional[int]:
        """Get the database primary key for a workflow by instance ID."""
        try:
            session = await get_async_session()
            async with session:
                db_wf = await crud.get_workflow_by_instance_id(
                    session=session, instance_id=instance_id
                )
                return db_wf.id if db_wf else None
        except Exception as e:
            self._logger.warning(
                "Failed to get DB PK for workflow %s: %s", instance_id, str(e)
            )
            return None

    # ========================
    # Phase 1: Agent Selection
    # ========================

    async def _select_agents(
        self,
        issue: dict[str, Any],
        workflow: WorkflowInstance,
    ) -> AgentSelectionResult:
        """
        Select relevant agents for the issue using the moderator.

        Args:
            issue: GitHub issue data
            workflow: Current workflow instance

        Returns:
            AgentSelectionResult with selected agents and reasoning
        """
        try:
            result = await self.moderator.select_relevant_agents(
                issue=issue,
                registry=self.registry,
            )
            return result
        except Exception as e:
            self._logger.error(
                "Agent selection failed: %s. Using fallback.", str(e)
            )
            # Fallback: use keyword-based selection directly
            from src.orchestration.moderator import select_agents_by_keywords

            issue_title = issue.get("title", "")
            issue_body = issue.get("body", "")
            issue_labels = issue.get("labels", [])

            return select_agents_by_keywords(
                issue_title=issue_title,
                issue_body=issue_body,
                issue_labels=issue_labels,
                registry=self.registry,
            )

    # ========================
    # Phase 2: Round Execution
    # ========================

    async def _execute_round(
        self,
        round_num: int,
        agents: list[BaseAgent],
        workflow: WorkflowInstance,
    ) -> list[Comment]:
        """
        Execute a single deliberation round with parallel agent evaluation.

        For each agent:
        1. Call should_comment() to get participation decision
        2. If yes, call generate_comment() to get the actual comment
        3. Collect all comments for the round

        All agents are evaluated concurrently using asyncio.gather.

        Args:
            round_num: Current round number
            agents: List of agents participating in this deliberation
            workflow: Current workflow state

        Returns:
            List of comments generated in this round
        """
        self._logger.debug(
            "Executing round %d with %d agents", round_num, len(agents)
        )

        # Create tasks for all agents to evaluate concurrently
        tasks = [
            self._evaluate_agent(agent, workflow, round_num)
            for agent in agents
        ]

        # Run all agent evaluations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful comments, log errors
        round_comments: list[Comment] = []
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                self._logger.warning(
                    "Agent %s failed in round %d: %s",
                    agent.name,
                    round_num,
                    str(result),
                )
            elif isinstance(result, Comment):
                round_comments.append(result)
            # None means agent decided not to comment (expected)

        return round_comments

    async def _evaluate_agent(
        self,
        agent: BaseAgent,
        workflow: WorkflowInstance,
        round_num: int,
    ) -> Optional[Comment]:
        """
        Evaluate a single agent's participation in a round.

        First asks the agent if it should comment, then generates
        the comment if the answer is yes.

        Args:
            agent: The agent to evaluate
            workflow: Current workflow state
            round_num: Current round number

        Returns:
            Comment if agent chose to participate, None otherwise
        """
        try:
            # Step 1: Should this agent comment?
            decision = await agent.should_comment(
                workflow=workflow,
                current_round=round_num,
            )

            if not decision.should_comment:
                self._logger.debug(
                    "Agent %s staying silent in round %d: %s",
                    agent.name,
                    round_num,
                    decision.reason,
                )
                return None

            self._logger.debug(
                "Agent %s will comment in round %d (confidence=%.2f): %s",
                agent.name,
                round_num,
                decision.confidence,
                decision.reason,
            )

            # Step 2: Generate the comment
            comment = await agent.generate_comment(
                workflow=workflow,
                current_round=round_num,
                decision=decision,
            )

            return comment

        except Exception as e:
            self._logger.warning(
                "Agent %s evaluation failed in round %d: %s",
                agent.name,
                round_num,
                str(e),
            )
            raise

    # ========================
    # Phase 4: Synthesis
    # ========================

    async def _synthesize_recommendations(
        self,
        workflow: WorkflowInstance,
    ) -> str:
        """
        Synthesize final recommendations from the conversation history.

        Uses the RecommendationSynthesizer to create a structured summary
        with actionable recommendations, consensus points, unresolved
        conflicts, and agent participation statistics.

        Args:
            workflow: Completed workflow with full conversation history

        Returns:
            Summary string with synthesized recommendations (Markdown formatted)
        """
        synthesizer = RecommendationSynthesizer()
        synthesis_result = synthesizer.synthesize(workflow)

        # Store the full synthesis result on the workflow for downstream access
        self._last_synthesis_result = synthesis_result

        return synthesis_result.summary_text

    # ========================
    # Helper Methods
    # ========================

    def _generate_instance_id(self, issue_number: int) -> str:
        """Generate a unique instance ID for the workflow."""
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return f"issue-{issue_number}-{timestamp}"

    def _calculate_agent_participation(
        self,
        workflow: WorkflowInstance,
    ) -> dict[str, int]:
        """Calculate per-agent comment counts."""
        participation: dict[str, int] = {}
        for comment in workflow.conversation_history:
            participation[comment.agent] = participation.get(comment.agent, 0) + 1
        return participation

    # ========================
    # Database Persistence
    # ========================

    async def _persist_workflow_created(
        self,
        workflow: WorkflowInstance,
        issue: dict[str, Any],
    ) -> Optional[int]:
        """Persist initial workflow state to database. Returns the DB primary key."""
        try:
            session = await get_async_session()
            async with session:
                db_workflow = await crud.create_workflow(
                    session=session,
                    instance_id=workflow.instance_id,
                    issue_number=issue.get("number", 0),
                    issue_title=issue.get("title", ""),
                    issue_body=issue.get("body"),
                    issue_url=issue.get("html_url") or issue.get("url"),
                    repository=issue.get("repository"),
                    issue_labels=issue.get("labels"),
                    max_rounds=workflow.config.max_rounds,
                    workflow_id=workflow.workflow_id,
                )
                await session.commit()
                return db_workflow.id
        except Exception as e:
            self._logger.warning(
                "Failed to persist workflow creation: %s", str(e)
            )
            return None

    async def _update_db_status(
        self,
        workflow_pk: int,
        status: str,
        **kwargs,
    ) -> None:
        """Update workflow status in database."""
        try:
            session = await get_async_session()
            async with session:
                await crud.update_workflow_status(
                    session=session,
                    workflow_pk=workflow_pk,
                    status=status,
                    **kwargs,
                )
                await session.commit()
        except Exception as e:
            self._logger.warning(
                "Failed to update workflow status: %s", str(e)
            )

    async def _persist_agent_selections(
        self,
        workflow_pk: int,
        selection_result: AgentSelectionResult,
    ) -> None:
        """Persist agent selection results to database."""
        try:
            session = await get_async_session()
            async with session:
                for agent in selection_result.agents:
                    await crud.add_agent_to_workflow(
                        session=session,
                        workflow_pk=workflow_pk,
                        agent_name=agent.name,
                        agent_type=agent.agent_type.value,
                        expertise=agent.expertise,
                        selection_reason=selection_result.reasoning.get(
                            agent.name, ""
                        ),
                    )
                await session.commit()
        except Exception as e:
            self._logger.warning(
                "Failed to persist agent selections: %s", str(e)
            )

    async def _persist_round(
        self,
        workflow_pk: int,
        round_num: int,
        round_comments: list[Comment],
        workflow: WorkflowInstance,
    ) -> None:
        """Persist round results to database."""
        try:
            session = await get_async_session()
            async with session:
                # Update workflow round progress
                await crud.update_workflow_round(
                    session=session,
                    workflow_pk=workflow_pk,
                    current_round=round_num,
                    total_comments=len(workflow.conversation_history),
                )

                # Add conversation entries
                for comment in round_comments:
                    await crud.add_conversation_entry(
                        session=session,
                        workflow_pk=workflow_pk,
                        round_number=comment.round,
                        agent_name=comment.agent,
                        comment=comment.comment,
                        references=comment.references,
                    )

                    # Update agent participation
                    await crud.update_agent_participation(
                        session=session,
                        workflow_pk=workflow_pk,
                        agent_name=comment.agent,
                        round_number=round_num,
                    )

                await session.commit()
        except Exception as e:
            self._logger.warning(
                "Failed to persist round %d: %s", round_num, str(e)
            )

    async def _persist_convergence_metric(
        self,
        workflow_pk: int,
        round_num: int,
        decision: ContinueDecision,
        round_comments: list[Comment],
    ) -> None:
        """Persist convergence metrics to database."""
        try:
            value_added = measure_value_added(
                round_comments,
                [],  # We pass empty history here since the decision already computed it
            )
            agreement = measure_agreement_level(round_comments)

            session = await get_async_session()
            async with session:
                await crud.add_convergence_metric(
                    session=session,
                    workflow_pk=workflow_pk,
                    round_number=round_num,
                    convergence_score=decision.convergence_score,
                    should_continue=decision.should_continue,
                    reason=decision.reason,
                    agreement_level=agreement,
                    value_added=value_added,
                    rambling_detected=decision.rambling_detected,
                    value_trend=decision.value_trend,
                    comments_this_round=len(round_comments),
                    agents_participated=len(set(c.agent for c in round_comments)),
                )
                await session.commit()
        except Exception as e:
            self._logger.warning(
                "Failed to persist convergence metrics: %s", str(e)
            )

    async def _persist_workflow_completed(
        self,
        workflow_pk: int,
        workflow: WorkflowInstance,
        final_convergence_score: float,
        summary: Optional[str],
    ) -> None:
        """Persist final workflow state to database."""
        try:
            session = await get_async_session()
            async with session:
                await crud.update_workflow_status(
                    session=session,
                    workflow_pk=workflow_pk,
                    status="completed",
                    summary=summary,
                    final_convergence_score=final_convergence_score,
                    total_comments=len(workflow.conversation_history),
                    current_round=workflow.current_round,
                )
                await session.commit()
        except Exception as e:
            self._logger.warning(
                "Failed to persist workflow completion: %s", str(e)
            )
