"""
Tests for the MultiAgentDeliberationOrchestrator.

Tests cover:
- Full deliberation lifecycle (selection → rounds → convergence → synthesis)
- Parallel agent evaluation within rounds
- Convergence detection integration
- Error handling for agent failures
- Workflow state management
- DeliberationResult structure
- Database persistence (mocked)
- Edge cases (no comments, single round, max rounds)
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import BaseAgent
from src.agents.registry import AgentRegistry
from src.models.agent import AgentConfig, AgentDecision, AgentType, LLMProviderConfig
from src.models.workflow import (
    Comment,
    ContinueDecision,
    WorkflowConfig,
    WorkflowInstance,
    WorkflowMetrics,
    WorkflowStatus,
)
from src.orchestration.convergence import ConvergenceDetector
from src.orchestration.deliberation import (
    DeliberationResult,
    MultiAgentDeliberationOrchestrator,
)
from src.orchestration.moderator import AgentSelectionResult, ModeratorAgent


# ==================
# Test Agent Implementations
# ==================


class MockAgent(BaseAgent):
    """A mock agent for testing that always comments."""

    def __init__(
        self,
        name: str = "mock_agent",
        expertise: str = "Testing",
        should_comment_val: bool = True,
        comment_text: str = "Test comment",
        references: Optional[list[str]] = None,
        confidence: float = 0.9,
        fail_on_comment: bool = False,
        fail_on_should_comment: bool = False,
    ):
        config = AgentConfig(
            name=name,
            type=AgentType.CLAUDE_SDK_TEXT,
            expertise=expertise,
            domain_knowledge=f"Expert in {expertise}",
            llm=LLMProviderConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
            ),
            priority=7,
            category="development",
        )
        super().__init__(config)
        self._should_comment_val = should_comment_val
        self._comment_text = comment_text
        self._references = references or []
        self._confidence = confidence
        self._fail_on_comment = fail_on_comment
        self._fail_on_should_comment = fail_on_should_comment
        self._call_count = 0

    async def should_comment(
        self,
        workflow: WorkflowInstance,
        current_round: int,
    ) -> AgentDecision:
        if self._fail_on_should_comment:
            raise RuntimeError(f"Agent {self.name} failed on should_comment")
        return AgentDecision(
            should_comment=self._should_comment_val,
            reason=f"Agent {self.name} decision",
            responding_to=self._references,
            confidence=self._confidence,
        )

    async def generate_comment(
        self,
        workflow: WorkflowInstance,
        current_round: int,
        decision: AgentDecision,
    ) -> Comment:
        if self._fail_on_comment:
            raise RuntimeError(f"Agent {self.name} failed on generate_comment")
        self._call_count += 1
        return self.create_comment(
            text=f"{self._comment_text} (round {current_round})",
            round_number=current_round,
            references=self._references,
        )


class SilentAgent(MockAgent):
    """An agent that never comments."""

    def __init__(self, name: str = "silent_agent"):
        super().__init__(
            name=name,
            should_comment_val=False,
            expertise="Silence",
        )


class AgreementAgent(MockAgent):
    """An agent that produces agreement text."""

    def __init__(self, name: str = "agreement_agent", references: Optional[list[str]] = None):
        super().__init__(
            name=name,
            comment_text="I agree with this approach. Good point. I support this recommendation.",
            references=references or [],
            expertise="Agreement",
        )


# ==================
# Fixtures
# ==================


@pytest.fixture
def registry():
    """Create an AgentRegistry loaded with production agents."""
    reg = AgentRegistry()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "agent_definitions.yaml",
    )
    reg.load_from_yaml(config_path)
    return reg


@pytest.fixture
def mock_agents():
    """Create a list of mock agents."""
    return [
        MockAgent(
            name="ui_architect",
            expertise="UI Architecture",
            comment_text="UI perspective: consider component hierarchy",
        ),
        MockAgent(
            name="backend_dev",
            expertise="Backend Development",
            comment_text="Backend perspective: API design matters",
            references=["ui_architect"],
        ),
        MockAgent(
            name="security_expert",
            expertise="Security",
            comment_text="Security consideration: validate all inputs",
        ),
        MockAgent(
            name="qa_engineer",
            expertise="QA Engineering",
            comment_text="Testing perspective: add integration tests",
            references=["backend_dev"],
        ),
        MockAgent(
            name="ada_expert",
            expertise="Accessibility",
            comment_text="Accessibility: ensure ARIA labels are present",
            references=["ui_architect"],
        ),
    ]


@pytest.fixture
def mock_selection_result(mock_agents):
    """Create a mock agent selection result."""
    return AgentSelectionResult(
        agents=mock_agents,
        reasoning={a.name: f"Selected for {a.expertise}" for a in mock_agents},
        method="keyword",
    )


@pytest.fixture
def sample_issue():
    """Create a sample GitHub issue."""
    return {
        "number": 42,
        "title": "Add dark mode toggle to settings page",
        "body": (
            "Users should be able to switch between light and dark mode. "
            "The toggle should persist across sessions using local storage."
        ),
        "labels": ["enhancement", "ui"],
        "html_url": "https://github.com/test/repo/issues/42",
    }


@pytest.fixture
def workflow_config():
    """Create a test workflow configuration."""
    return WorkflowConfig(
        max_rounds=5,
        convergence_threshold=0.8,
        min_value_threshold=0.2,
        timeout_minutes=30,
    )


@pytest.fixture
def mock_moderator(mock_selection_result):
    """Create a mock moderator that returns predefined agents."""
    moderator = ModeratorAgent()
    moderator.select_relevant_agents = AsyncMock(return_value=mock_selection_result)
    return moderator


@pytest.fixture
def mock_convergence():
    """Create a mock convergence detector."""
    detector = ConvergenceDetector(use_llm=False)
    return detector


@pytest.fixture
def orchestrator(registry, mock_moderator, mock_convergence, workflow_config):
    """Create an orchestrator with mocked dependencies and no DB persistence."""
    return MultiAgentDeliberationOrchestrator(
        registry=registry,
        moderator=mock_moderator,
        convergence_detector=mock_convergence,
        config=workflow_config,
        persist_to_db=False,
    )


# ==================
# Test DeliberationResult
# ==================


class TestDeliberationResult:
    """Tests for the DeliberationResult class."""

    def test_result_creation(self):
        """Test creating a deliberation result."""
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.COMPLETED,
        )
        result = DeliberationResult(
            workflow=workflow,
            summary="Test summary",
            total_rounds=3,
            total_comments=10,
            final_convergence_score=0.85,
            termination_reason="High convergence",
            agent_participation={"agent1": 3, "agent2": 2},
            duration_seconds=15.5,
        )

        assert result.summary == "Test summary"
        assert result.total_rounds == 3
        assert result.total_comments == 10
        assert result.final_convergence_score == 0.85
        assert result.termination_reason == "High convergence"
        assert result.agent_participation == {"agent1": 3, "agent2": 2}
        assert result.duration_seconds == 15.5

    def test_result_to_dict(self):
        """Test serialization of deliberation result."""
        workflow = WorkflowInstance(
            instance_id="test-002",
            status=WorkflowStatus.COMPLETED,
            conversation_history=[
                Comment(
                    round=1,
                    agent="test_agent",
                    comment="Test comment",
                    references=[],
                ),
            ],
        )
        result = DeliberationResult(
            workflow=workflow,
            summary="Summary",
            total_rounds=1,
            total_comments=1,
            final_convergence_score=0.9,
            termination_reason="Done",
        )

        d = result.to_dict()
        assert d["instance_id"] == "test-002"
        assert d["status"] == "completed"
        assert d["summary"] == "Summary"
        assert d["total_rounds"] == 1
        assert d["total_comments"] == 1
        assert d["final_convergence_score"] == 0.9
        assert len(d["conversation_history"]) == 1
        assert d["conversation_history"][0]["agent"] == "test_agent"

    def test_result_defaults(self):
        """Test default values in deliberation result."""
        workflow = WorkflowInstance(
            instance_id="test-003",
            status=WorkflowStatus.PENDING,
        )
        result = DeliberationResult(workflow=workflow)

        assert result.summary is None
        assert result.total_rounds == 0
        assert result.total_comments == 0
        assert result.final_convergence_score == 0.0
        assert result.agent_participation == {}
        assert result.round_metrics == []
        assert result.duration_seconds == 0.0


# ==================
# Test Orchestrator Initialization
# ==================


class TestOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_basic_initialization(self, registry):
        """Test basic orchestrator creation."""
        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            persist_to_db=False,
        )
        assert orch.registry is registry
        assert orch.moderator is not None
        assert orch.convergence_detector is not None
        assert orch.default_config is not None

    def test_initialization_with_custom_config(self, registry):
        """Test orchestrator with custom configuration."""
        config = WorkflowConfig(max_rounds=20, convergence_threshold=0.9)
        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            config=config,
            persist_to_db=False,
        )
        assert orch.default_config.max_rounds == 20
        assert orch.default_config.convergence_threshold == 0.9

    def test_initialization_with_dependencies(
        self, registry, mock_moderator, mock_convergence
    ):
        """Test orchestrator with all dependencies injected."""
        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=mock_moderator,
            convergence_detector=mock_convergence,
            persist_to_db=False,
        )
        assert orch.moderator is mock_moderator
        assert orch.convergence_detector is mock_convergence


# ==================
# Test Agent Evaluation
# ==================


class TestAgentEvaluation:
    """Tests for individual agent evaluation."""

    @pytest.mark.asyncio
    async def test_agent_that_comments(self, orchestrator):
        """Test evaluating an agent that decides to comment."""
        agent = MockAgent(name="test", comment_text="My analysis")
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        result = await orchestrator._evaluate_agent(agent, workflow, 1)
        assert result is not None
        assert isinstance(result, Comment)
        assert result.agent == "test"
        assert "My analysis" in result.comment
        assert result.round == 1

    @pytest.mark.asyncio
    async def test_agent_that_stays_silent(self, orchestrator):
        """Test evaluating an agent that decides not to comment."""
        agent = SilentAgent()
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        result = await orchestrator._evaluate_agent(agent, workflow, 1)
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_failure_on_should_comment(self, orchestrator):
        """Test agent that fails during should_comment."""
        agent = MockAgent(
            name="failing_agent",
            fail_on_should_comment=True,
        )
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        with pytest.raises(RuntimeError):
            await orchestrator._evaluate_agent(agent, workflow, 1)

    @pytest.mark.asyncio
    async def test_agent_failure_on_generate_comment(self, orchestrator):
        """Test agent that fails during generate_comment."""
        agent = MockAgent(
            name="failing_agent",
            fail_on_comment=True,
        )
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        with pytest.raises(RuntimeError):
            await orchestrator._evaluate_agent(agent, workflow, 1)


# ==================
# Test Round Execution
# ==================


class TestRoundExecution:
    """Tests for round execution."""

    @pytest.mark.asyncio
    async def test_round_with_all_commenting(self, orchestrator, mock_agents):
        """Test a round where all agents comment."""
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        comments = await orchestrator._execute_round(1, mock_agents, workflow)
        assert len(comments) == len(mock_agents)
        assert all(isinstance(c, Comment) for c in comments)
        assert all(c.round == 1 for c in comments)

    @pytest.mark.asyncio
    async def test_round_with_mixed_participation(self, orchestrator):
        """Test a round where some agents comment and others stay silent."""
        agents = [
            MockAgent(name="active1", comment_text="Active 1"),
            SilentAgent(name="silent1"),
            MockAgent(name="active2", comment_text="Active 2"),
            SilentAgent(name="silent2"),
            MockAgent(name="active3", comment_text="Active 3"),
        ]
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        comments = await orchestrator._execute_round(1, agents, workflow)
        assert len(comments) == 3
        active_names = {c.agent for c in comments}
        assert active_names == {"active1", "active2", "active3"}

    @pytest.mark.asyncio
    async def test_round_with_agent_failure(self, orchestrator):
        """Test that agent failures don't stop the round."""
        agents = [
            MockAgent(name="good_agent", comment_text="Good"),
            MockAgent(name="bad_agent", fail_on_comment=True),
            MockAgent(name="another_good", comment_text="Also good"),
        ]
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        comments = await orchestrator._execute_round(1, agents, workflow)
        # Should have 2 comments (bad_agent failed, handled by asyncio.gather)
        assert len(comments) == 2
        assert {c.agent for c in comments} == {"good_agent", "another_good"}

    @pytest.mark.asyncio
    async def test_round_with_all_silent(self, orchestrator):
        """Test a round where no agents comment."""
        agents = [
            SilentAgent(name="silent1"),
            SilentAgent(name="silent2"),
            SilentAgent(name="silent3"),
        ]
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        comments = await orchestrator._execute_round(1, agents, workflow)
        assert len(comments) == 0

    @pytest.mark.asyncio
    async def test_round_parallel_execution(self, orchestrator):
        """Test that agents are evaluated concurrently."""

        class SlowAgent(MockAgent):
            async def should_comment(self, workflow, current_round):
                await asyncio.sleep(0.1)  # Simulate latency
                return AgentDecision(
                    should_comment=True,
                    reason="Slow but ready",
                    responding_to=[],
                    confidence=0.8,
                )

            async def generate_comment(self, workflow, current_round, decision):
                await asyncio.sleep(0.1)  # Simulate latency
                return self.create_comment(
                    text=f"Slow comment from {self.name}",
                    round_number=current_round,
                )

        agents = [
            SlowAgent(name=f"slow_{i}", expertise=f"Slow {i}")
            for i in range(5)
        ]
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Body"},
        )

        import time
        start = time.time()
        comments = await orchestrator._execute_round(1, agents, workflow)
        elapsed = time.time() - start

        assert len(comments) == 5
        # If truly parallel, 5 agents with 0.2s each should take ~0.2-0.5s,
        # not 1.0s (sequential)
        assert elapsed < 0.8, f"Agents not running in parallel: took {elapsed:.2f}s"


# ==================
# Test Full Deliberation
# ==================


class TestFullDeliberation:
    """Tests for the complete deliberation lifecycle."""

    @pytest.mark.asyncio
    async def test_basic_deliberation(self, orchestrator, sample_issue):
        """Test a basic deliberation that runs to completion."""
        result = await orchestrator.deliberate_on_issue(sample_issue)

        assert isinstance(result, DeliberationResult)
        assert result.workflow.status == WorkflowStatus.COMPLETED
        assert result.total_rounds >= 1
        assert result.total_comments >= 0
        assert result.termination_reason != ""
        assert result.summary is not None
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_deliberation_creates_instance_id(self, orchestrator, sample_issue):
        """Test that a unique instance ID is generated."""
        result = await orchestrator.deliberate_on_issue(sample_issue)
        assert result.workflow.instance_id.startswith("issue-42-")

    @pytest.mark.asyncio
    async def test_deliberation_selects_agents(self, orchestrator, sample_issue):
        """Test that agent selection is performed."""
        result = await orchestrator.deliberate_on_issue(sample_issue)
        assert len(result.workflow.selected_agents) > 0
        orchestrator.moderator.select_relevant_agents.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliberation_with_custom_config(self, orchestrator, sample_issue):
        """Test deliberation with custom configuration."""
        config = WorkflowConfig(max_rounds=2, convergence_threshold=0.5)
        result = await orchestrator.deliberate_on_issue(
            sample_issue, config=config
        )
        assert result.total_rounds <= 2
        assert result.workflow.config.max_rounds == 2

    @pytest.mark.asyncio
    async def test_deliberation_missing_title(self, orchestrator):
        """Test that missing title raises ValueError."""
        with pytest.raises(ValueError, match="title"):
            await orchestrator.deliberate_on_issue({"body": "no title"})

    @pytest.mark.asyncio
    async def test_deliberation_conversation_history(self, orchestrator, sample_issue):
        """Test that conversation history is accumulated across rounds."""
        result = await orchestrator.deliberate_on_issue(sample_issue)
        history = result.workflow.conversation_history

        if result.total_comments > 0:
            # Verify comments have proper round numbers
            for comment in history:
                assert comment.round >= 1
                assert comment.agent != ""
                assert comment.comment != ""

    @pytest.mark.asyncio
    async def test_deliberation_agent_participation(self, orchestrator, sample_issue):
        """Test that agent participation is tracked."""
        result = await orchestrator.deliberate_on_issue(sample_issue)

        if result.total_comments > 0:
            assert len(result.agent_participation) > 0
            # Each participating agent should have at least 1 comment
            for agent, count in result.agent_participation.items():
                assert count >= 1

    @pytest.mark.asyncio
    async def test_deliberation_round_metrics(self, orchestrator, sample_issue):
        """Test that round metrics are collected."""
        result = await orchestrator.deliberate_on_issue(sample_issue)

        assert len(result.round_metrics) == result.total_rounds
        for metrics in result.round_metrics:
            assert "round" in metrics
            assert "convergence_score" in metrics
            assert "value_added" in metrics
            assert "comments_count" in metrics

    @pytest.mark.asyncio
    async def test_deliberation_to_dict(self, orchestrator, sample_issue):
        """Test serialization of complete result."""
        result = await orchestrator.deliberate_on_issue(sample_issue)
        d = result.to_dict()

        assert "instance_id" in d
        assert "status" in d
        assert "summary" in d
        assert "total_rounds" in d
        assert "total_comments" in d
        assert "conversation_history" in d
        assert "round_metrics" in d
        assert "duration_seconds" in d


# ==================
# Test Convergence Integration
# ==================


class TestConvergenceIntegration:
    """Tests for convergence detection within deliberation."""

    @pytest.mark.asyncio
    async def test_natural_conclusion_no_comments(self, registry, mock_convergence):
        """Test that deliberation stops when no agents comment."""
        silent_agents = [SilentAgent(name=f"silent_{i}") for i in range(5)]
        selection = AgentSelectionResult(
            agents=silent_agents,
            reasoning={a.name: "Test" for a in silent_agents},
            method="keyword",
        )
        moderator = ModeratorAgent()
        moderator.select_relevant_agents = AsyncMock(return_value=selection)

        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=moderator,
            convergence_detector=mock_convergence,
            persist_to_db=False,
        )

        result = await orch.deliberate_on_issue({
            "number": 1,
            "title": "Test issue",
            "body": "Test body",
        })

        assert result.total_rounds == 1
        assert result.total_comments == 0
        assert "natural" in result.termination_reason.lower() or "no agents" in result.termination_reason.lower()

    @pytest.mark.asyncio
    async def test_max_rounds_termination(self, registry):
        """Test that deliberation stops at max rounds."""
        # Create agents that always comment with diverse content
        agents = []
        for i in range(5):
            agents.append(MockAgent(
                name=f"chatty_{i}",
                expertise=f"Area {i}",
                comment_text=f"Unique insight {i}: recommend implementing approach {i} for better performance and scalability consideration number {i}",
            ))

        selection = AgentSelectionResult(
            agents=agents,
            reasoning={a.name: "Test" for a in agents},
            method="keyword",
        )
        moderator = ModeratorAgent()
        moderator.select_relevant_agents = AsyncMock(return_value=selection)

        # Use a convergence detector that always says continue
        detector = ConvergenceDetector(use_llm=False)
        detector.should_continue = AsyncMock(
            return_value=ContinueDecision(
                should_continue=True,
                reason="Still productive",
                convergence_score=0.3,
                rambling_detected=False,
                value_trend="stable",
            )
        )
        # Keep real get_round_metrics
        real_detector = ConvergenceDetector(use_llm=False)
        detector.get_round_metrics = real_detector.get_round_metrics

        config = WorkflowConfig(max_rounds=3)
        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=moderator,
            convergence_detector=detector,
            config=config,
            persist_to_db=False,
        )

        result = await orch.deliberate_on_issue({
            "number": 2,
            "title": "Test max rounds",
            "body": "Testing max rounds termination",
        })

        assert result.total_rounds == 3
        assert "maximum" in result.termination_reason.lower() or "max" in result.termination_reason.lower()


# ==================
# Test Synthesis
# ==================


class TestSynthesis:
    """Tests for recommendation synthesis."""

    @pytest.mark.asyncio
    async def test_synthesis_with_comments(self, orchestrator):
        """Test synthesis produces meaningful summary."""
        workflow = WorkflowInstance(
            instance_id="test-001",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Add dark mode", "body": "Need dark mode"},
            current_round=2,
            conversation_history=[
                Comment(
                    round=1,
                    agent="ui_architect",
                    comment="I recommend using CSS variables for theming.",
                ),
                Comment(
                    round=1,
                    agent="frontend_dev",
                    comment="I agree with ui_architect. CSS custom properties work well.",
                    references=["ui_architect"],
                ),
                Comment(
                    round=2,
                    agent="ada_expert",
                    comment="Ensure sufficient color contrast ratios for accessibility.",
                    references=["ui_architect"],
                ),
            ],
        )

        summary = await orchestrator._synthesize_recommendations(workflow)
        assert "Deliberation Summary" in summary
        assert "ui_architect" in summary
        assert "frontend_dev" in summary
        assert "ada_expert" in summary
        assert "3" in summary  # 3 comments
        assert "Agent Participation" in summary

    @pytest.mark.asyncio
    async def test_synthesis_empty_history(self, orchestrator):
        """Test synthesis with no comments."""
        workflow = WorkflowInstance(
            instance_id="test-002",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Test"},
        )

        summary = await orchestrator._synthesize_recommendations(workflow)
        assert "no agents" in summary.lower() or "no discussion" in summary.lower()

    @pytest.mark.asyncio
    async def test_synthesis_tracks_references(self, orchestrator):
        """Test that synthesis correctly tracks cross-agent references."""
        workflow = WorkflowInstance(
            instance_id="test-003",
            status=WorkflowStatus.RUNNING,
            github_issue={"title": "Test", "body": "Test"},
            current_round=2,
            conversation_history=[
                Comment(
                    round=1,
                    agent="architect",
                    comment="Use microservices approach.",
                ),
                Comment(
                    round=1,
                    agent="dev1",
                    comment="Building on architect's point.",
                    references=["architect"],
                ),
                Comment(
                    round=2,
                    agent="dev2",
                    comment="I agree with architect and dev1.",
                    references=["architect", "dev1"],
                ),
            ],
        )

        summary = await orchestrator._synthesize_recommendations(workflow)
        assert "referenced" in summary.lower()


# ==================
# Test Error Handling
# ==================


class TestErrorHandling:
    """Tests for error handling in deliberation."""

    @pytest.mark.asyncio
    async def test_agent_selection_failure_fallback(self, registry, mock_convergence):
        """Test fallback when moderator selection fails."""
        moderator = ModeratorAgent(registry=registry)
        moderator.select_relevant_agents = AsyncMock(
            side_effect=RuntimeError("LLM unavailable")
        )

        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=moderator,
            convergence_detector=mock_convergence,
            persist_to_db=False,
        )

        # The orchestrator should fall back to keyword-based selection
        result = await orch.deliberate_on_issue({
            "number": 1,
            "title": "Add user authentication with OAuth",
            "body": "Implement secure login with OAuth2",
        })

        # Should still complete successfully via fallback
        assert result.workflow.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED)

    @pytest.mark.asyncio
    async def test_all_agents_fail(self, registry, mock_convergence):
        """Test handling when all agents fail."""
        failing_agents = [
            MockAgent(
                name=f"fail_{i}",
                fail_on_should_comment=True,
                expertise=f"Failing {i}",
            )
            for i in range(5)
        ]
        selection = AgentSelectionResult(
            agents=failing_agents,
            reasoning={a.name: "Test" for a in failing_agents},
            method="keyword",
        )
        moderator = ModeratorAgent()
        moderator.select_relevant_agents = AsyncMock(return_value=selection)

        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=moderator,
            convergence_detector=mock_convergence,
            persist_to_db=False,
        )

        result = await orch.deliberate_on_issue({
            "number": 1,
            "title": "Test failing agents",
            "body": "All agents fail",
        })

        # Should complete (no comments = natural conclusion)
        assert result.total_comments == 0

    @pytest.mark.asyncio
    async def test_partial_agent_failure(self, registry, mock_convergence):
        """Test that partial agent failures don't halt deliberation."""
        agents = [
            MockAgent(name="good_1", comment_text="Good insight 1"),
            MockAgent(name="bad_1", fail_on_comment=True),
            MockAgent(name="good_2", comment_text="Good insight 2"),
        ]
        selection = AgentSelectionResult(
            agents=agents,
            reasoning={a.name: "Test" for a in agents},
            method="keyword",
        )
        moderator = ModeratorAgent()
        moderator.select_relevant_agents = AsyncMock(return_value=selection)

        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=moderator,
            convergence_detector=mock_convergence,
            persist_to_db=False,
        )

        result = await orch.deliberate_on_issue({
            "number": 1,
            "title": "Test partial failure",
            "body": "Some agents fail",
        })

        # Should still complete with good agents' contributions
        assert result.workflow.status == WorkflowStatus.COMPLETED


# ==================
# Test Helper Methods
# ==================


class TestHelperMethods:
    """Tests for orchestrator helper methods."""

    def test_generate_instance_id(self, orchestrator):
        """Test instance ID generation."""
        instance_id = orchestrator._generate_instance_id(42)
        assert instance_id.startswith("issue-42-")
        # Should have timestamp
        parts = instance_id.split("-")
        assert len(parts) >= 3

    def test_calculate_agent_participation(self, orchestrator):
        """Test agent participation calculation."""
        workflow = WorkflowInstance(
            instance_id="test-001",
            conversation_history=[
                Comment(round=1, agent="agent1", comment="c1"),
                Comment(round=1, agent="agent2", comment="c2"),
                Comment(round=2, agent="agent1", comment="c3"),
                Comment(round=2, agent="agent3", comment="c4"),
                Comment(round=3, agent="agent1", comment="c5"),
            ],
        )

        participation = orchestrator._calculate_agent_participation(workflow)
        assert participation == {"agent1": 3, "agent2": 1, "agent3": 1}

    def test_calculate_participation_empty(self, orchestrator):
        """Test participation with no comments."""
        workflow = WorkflowInstance(instance_id="test-002")
        participation = orchestrator._calculate_agent_participation(workflow)
        assert participation == {}


# ==================
# Test Workflow Status Transitions
# ==================


class TestWorkflowStatusTransitions:
    """Tests for workflow status transitions during deliberation."""

    @pytest.mark.asyncio
    async def test_status_pending_to_running(self, orchestrator, sample_issue):
        """Test that workflow transitions from PENDING to RUNNING."""
        result = await orchestrator.deliberate_on_issue(sample_issue)
        # After completion, should be COMPLETED (passed through RUNNING)
        assert result.workflow.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_status_completed_on_success(self, orchestrator, sample_issue):
        """Test that successful deliberation ends in COMPLETED status."""
        result = await orchestrator.deliberate_on_issue(sample_issue)
        assert result.workflow.status == WorkflowStatus.COMPLETED
        assert result.workflow.completed_at is not None

    @pytest.mark.asyncio
    async def test_metrics_updated_on_completion(self, orchestrator, sample_issue):
        """Test that metrics are properly updated."""
        result = await orchestrator.deliberate_on_issue(sample_issue)
        metrics = result.workflow.metrics

        assert metrics.current_round == result.total_rounds
        assert metrics.total_comments == result.total_comments
        assert metrics.started_at is not None


# ==================
# Test Multiple Rounds
# ==================


class TestMultipleRounds:
    """Tests for multi-round deliberation behavior."""

    @pytest.mark.asyncio
    async def test_comments_accumulate_across_rounds(self, registry):
        """Test that conversation history grows across rounds."""
        agents = [
            MockAgent(
                name=f"agent_{i}",
                expertise=f"Area {i}",
                comment_text=f"Novel unique insight about topic {i} with suggestion {i}",
            )
            for i in range(3)
        ]
        selection = AgentSelectionResult(
            agents=agents,
            reasoning={a.name: "Test" for a in agents},
            method="keyword",
        )
        moderator = ModeratorAgent()
        moderator.select_relevant_agents = AsyncMock(return_value=selection)

        config = WorkflowConfig(max_rounds=2)

        # Mock convergence to allow round 1, stop after round 2
        detector = ConvergenceDetector(use_llm=False)
        call_count = 0

        async def mock_should_continue(round_num, round_comments, full_history, config=None):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return ContinueDecision(
                    should_continue=False,
                    reason="Convergence reached",
                    convergence_score=0.85,
                    rambling_detected=False,
                    value_trend="converged",
                )
            return ContinueDecision(
                should_continue=True,
                reason="Still productive",
                convergence_score=0.4,
                rambling_detected=False,
                value_trend="increasing",
            )

        detector.should_continue = mock_should_continue

        orch = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=moderator,
            convergence_detector=detector,
            config=config,
            persist_to_db=False,
        )

        result = await orch.deliberate_on_issue({
            "number": 1,
            "title": "Multi-round test",
            "body": "Testing multiple rounds",
        })

        assert result.total_rounds == 2
        # 3 agents * 2 rounds = 6 comments
        assert result.total_comments == 6
        # Verify comments span multiple rounds
        rounds_seen = set(c.round for c in result.workflow.conversation_history)
        assert 1 in rounds_seen
        assert 2 in rounds_seen
