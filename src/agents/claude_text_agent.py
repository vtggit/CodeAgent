"""
Claude SDK Text Agent for the Multi-Agent GitHub Issue Routing System.

This module implements ClaudeTextAgent, a text-only agent that uses
LLM APIs (via litellm) for pure reasoning tasks. This is the primary
agent type used by most specialist agents who don't need code editing
or tool-use capabilities.

The agent makes two types of LLM calls:
1. should_comment() - Evaluates whether the agent has valuable input
2. generate_comment() - Produces the actual analysis/recommendation
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.agents.base import BaseAgent
from src.models.agent import AgentConfig, AgentDecision, AgentType
from src.models.workflow import Comment, WorkflowInstance


logger = logging.getLogger(__name__)

# Suppress litellm verbose logging
litellm.set_verbose = False


class ClaudeTextAgent(BaseAgent):
    """
    Text-only agent using LLM for reasoning and analysis.

    This agent uses litellm to call LLM APIs (Anthropic Claude, OpenAI, etc.)
    with no tools enabled. It's designed for pure reasoning tasks where agents
    analyze issues and provide expert recommendations.

    Features:
    - Configurable LLM provider and model via AgentConfig
    - Automatic retry with exponential backoff for API failures
    - Rate limiting to prevent API abuse
    - Structured decision-making for should_comment/generate_comment
    - Cost tracking support (when enabled)
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize a Claude text agent.

        Args:
            config: Agent configuration with LLM settings
        """
        super().__init__(config)

        # Track API usage for rate limiting
        self._last_call_time: float = 0.0
        self._min_call_interval: float = 0.5  # seconds between API calls

        # Build the LLM model identifier for litellm
        self._llm_model = self._build_llm_model_string()

        self._logger.info(
            "ClaudeTextAgent initialized with model: %s", self._llm_model
        )

    def _build_llm_model_string(self) -> str:
        """
        Build the litellm model string from configuration.

        litellm uses provider/model format for routing, e.g.:
        - "anthropic/claude-sonnet-4-5-20250929"
        - "openai/gpt-4"
        - "openai/local-model" (with custom base_url for LM Studio)

        Returns:
            litellm-compatible model string
        """
        provider = self._config.llm.provider.lower()
        model = self._config.llm.model

        if provider == "anthropic":
            # Anthropic models work directly in litellm
            return f"anthropic/{model}"
        elif provider == "openai":
            return f"openai/{model}"
        elif provider in ("lm_studio", "ollama"):
            # Local models use openai-compatible format
            return f"openai/{model}"
        else:
            # Default: try the model name directly
            return model

    async def _rate_limit(self) -> None:
        """
        Simple rate limiter to avoid API abuse.

        Ensures minimum interval between API calls.
        """
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_call_interval:
            await asyncio.sleep(self._min_call_interval - elapsed)
        self._last_call_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call the LLM API with retry logic.

        Uses litellm for multi-provider support. Retries up to 3 times
        with exponential backoff on failures.

        Args:
            messages: Chat messages in OpenAI format
            temperature: Override for temperature (uses config default if None)
            max_tokens: Override for max tokens (uses config default if None)

        Returns:
            The LLM response text

        Raises:
            Exception: If all retry attempts fail
        """
        await self._rate_limit()

        # Build kwargs for litellm
        kwargs: dict[str, Any] = {
            "model": self._llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._config.llm.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.llm.max_tokens,
        }

        # Add base_url for local models
        if self._config.llm.base_url:
            kwargs["api_base"] = self._config.llm.base_url

        self._logger.debug(
            "Calling LLM: model=%s, messages=%d, temp=%.2f",
            self._llm_model,
            len(messages),
            kwargs["temperature"],
        )

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content

            self._logger.debug(
                "LLM response received: %d chars, model=%s",
                len(content) if content else 0,
                response.model if hasattr(response, 'model') else 'unknown',
            )

            return content or ""

        except Exception as e:
            self._logger.error(
                "LLM call failed for agent %s: %s", self.name, str(e)
            )
            raise

    def _build_should_comment_prompt(
        self,
        workflow: WorkflowInstance,
        current_round: int,
    ) -> list[dict[str, str]]:
        """
        Build the prompt for the should_comment decision.

        Creates a structured prompt that asks the LLM to evaluate whether
        this agent should participate in the current round.

        Args:
            workflow: Current workflow state
            current_round: Current round number

        Returns:
            List of chat messages for the LLM
        """
        issue_ctx = self.get_issue_context(workflow)

        # Build conversation summary
        conversation_summary = ""
        if workflow.conversation_history:
            conversation_summary = "\n## Previous Comments\n"
            for comment in workflow.conversation_history:
                conversation_summary += (
                    f"\n### Round {comment.round} - {comment.agent}\n"
                    f"{comment.comment}\n"
                )

        # Check if anyone mentioned this agent
        mentions = self.get_comments_mentioning(workflow)
        mention_note = ""
        if mentions:
            mention_note = (
                f"\n**Note:** {len(mentions)} comment(s) reference you directly. "
                f"Consider responding if you have relevant insights.\n"
            )

        user_message = f"""## Issue to Evaluate
**Title:** {issue_ctx['title']}
**Labels:** {', '.join(issue_ctx['labels']) if issue_ctx['labels'] else 'none'}

**Description:**
{issue_ctx['body'] or 'No description provided'}

## Current Round: {current_round}
## Your Expertise: {self.expertise}
## Your Domain Knowledge: {self.domain_knowledge}
{mention_note}
{conversation_summary}

## Your Task
Evaluate whether you should comment on this issue in round {current_round}.

Consider:
1. Is this issue relevant to your expertise ({self.expertise})?
2. Do you have unique insights not yet covered by other agents?
3. Are there comments from other agents you should respond to?
4. Would your input add genuine value to the discussion?

Respond with a JSON object:
```json
{{
    "should_comment": true/false,
    "reason": "Brief explanation of your decision",
    "responding_to": ["agent_name1", "agent_name2"],
    "confidence": 0.0 to 1.0
}}
```

ONLY output the JSON object, nothing else."""

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

    def _build_generate_comment_prompt(
        self,
        workflow: WorkflowInstance,
        current_round: int,
        decision: AgentDecision,
    ) -> list[dict[str, str]]:
        """
        Build the prompt for generating a comment.

        Creates a structured prompt that asks the LLM to provide
        domain-specific analysis and recommendations.

        Args:
            workflow: Current workflow state
            current_round: Current round number
            decision: The participation decision with context

        Returns:
            List of chat messages for the LLM
        """
        issue_ctx = self.get_issue_context(workflow)

        # Build conversation summary
        conversation_summary = ""
        if workflow.conversation_history:
            conversation_summary = "\n## Previous Comments\n"
            for comment in workflow.conversation_history:
                conversation_summary += (
                    f"\n### Round {comment.round} - {comment.agent}\n"
                    f"{comment.comment}\n"
                )

        responding_note = ""
        if decision.responding_to:
            responding_note = (
                f"\n**You are responding to:** {', '.join(decision.responding_to)}\n"
                f"Make sure to reference their specific points.\n"
            )

        user_message = f"""## Issue Under Discussion
**Title:** {issue_ctx['title']}
**Labels:** {', '.join(issue_ctx['labels']) if issue_ctx['labels'] else 'none'}

**Description:**
{issue_ctx['body'] or 'No description provided'}

## Current Round: {current_round}
{responding_note}
{conversation_summary}

## Your Task
Provide your expert analysis as {self.name} (expertise: {self.expertise}).

Decision context: {decision.reason}

Guidelines:
- Focus on your area of expertise: {self.expertise}
- Reference other agents by name when responding to their points
- Be concise but thorough (aim for 100-300 words)
- Provide actionable recommendations where possible
- Acknowledge when something is outside your expertise
- Don't repeat points already well-covered by other agents
- If this is round 1, provide your initial analysis
- If later rounds, focus on new insights, responses, or refinements

Provide your comment directly (no JSON wrapping, no metadata, just your analysis)."""

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

    def _parse_should_comment_response(self, response: str) -> AgentDecision:
        """
        Parse the LLM response for should_comment into an AgentDecision.

        Handles both clean JSON and JSON embedded in markdown code blocks.

        Args:
            response: Raw LLM response text

        Returns:
            AgentDecision parsed from the response
        """
        # Try to extract JSON from the response
        text = response.strip()

        # Remove markdown code block wrapping if present
        if text.startswith("```"):
            # Remove opening ```json or ``` line
            lines = text.split("\n")
            # Find start of JSON
            start = 1 if lines[0].startswith("```") else 0
            # Find end of JSON
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()

        try:
            data = json.loads(text)
            return AgentDecision(
                should_comment=bool(data.get("should_comment", False)),
                reason=str(data.get("reason", "No reason provided")),
                responding_to=data.get("responding_to", []),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._logger.warning(
                "Failed to parse should_comment response as JSON: %s. "
                "Defaulting to should_comment=False. Response: %s",
                str(e),
                text[:200],
            )
            # Default to not commenting if we can't parse
            return AgentDecision(
                should_comment=False,
                reason=f"Failed to parse LLM response: {str(e)}",
                responding_to=[],
                confidence=0.0,
            )

    async def should_comment(
        self,
        workflow: WorkflowInstance,
        current_round: int,
    ) -> AgentDecision:
        """
        Use LLM to decide whether to comment in the current round.

        Calls the configured LLM with context about the issue and
        conversation history to make an informed participation decision.

        Args:
            workflow: Current workflow state
            current_round: Current round number (1-based)

        Returns:
            AgentDecision with the participation decision
        """
        self._logger.info(
            "Evaluating should_comment for round %d (issue: %s)",
            current_round,
            workflow.github_issue.get("title", "unknown") if workflow.github_issue else "unknown",
        )

        try:
            messages = self._build_should_comment_prompt(workflow, current_round)

            # Use lower temperature for decision-making
            response = await self._call_llm(
                messages,
                temperature=0.3,
                max_tokens=500,
            )

            decision = self._parse_should_comment_response(response)

            self._logger.info(
                "Decision for round %d: should_comment=%s, confidence=%.2f, reason=%s",
                current_round,
                decision.should_comment,
                decision.confidence,
                decision.reason[:80],
            )

            return decision

        except Exception as e:
            self._logger.error(
                "Error in should_comment for agent %s: %s",
                self.name,
                str(e),
            )
            # On error, default to not commenting
            return AgentDecision(
                should_comment=False,
                reason=f"Error evaluating participation: {str(e)}",
                responding_to=[],
                confidence=0.0,
            )

    async def generate_comment(
        self,
        workflow: WorkflowInstance,
        current_round: int,
        decision: AgentDecision,
    ) -> Comment:
        """
        Use LLM to generate a comment for the current round.

        Called when should_comment() returns True. Generates domain-specific
        analysis and recommendations based on the issue and conversation.

        Args:
            workflow: Current workflow state
            current_round: Current round number (1-based)
            decision: The participation decision with context

        Returns:
            Comment with the generated analysis

        Raises:
            Exception: If LLM call fails after all retries
        """
        self._logger.info(
            "Generating comment for round %d (responding to: %s)",
            current_round,
            ", ".join(decision.responding_to) if decision.responding_to else "none",
        )

        try:
            messages = self._build_generate_comment_prompt(
                workflow, current_round, decision
            )

            response = await self._call_llm(messages)

            # Clean up the response
            comment_text = response.strip()

            if not comment_text:
                comment_text = (
                    f"[{self.name}] Unable to generate substantive comment "
                    f"for this round."
                )

            comment = self.create_comment(
                text=comment_text,
                round_number=current_round,
                references=decision.responding_to,
            )

            self._logger.info(
                "Generated comment: %d chars, references: %s",
                len(comment_text),
                ", ".join(decision.responding_to) if decision.responding_to else "none",
            )

            return comment

        except Exception as e:
            self._logger.error(
                "Error generating comment for agent %s: %s",
                self.name,
                str(e),
            )
            # Return an error comment rather than crashing the workflow
            return self.create_comment(
                text=(
                    f"[{self.name}] Error generating analysis: {str(e)}. "
                    f"Please disregard this comment."
                ),
                round_number=current_round,
                references=decision.responding_to,
            )
