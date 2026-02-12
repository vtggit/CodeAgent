"""
Moderator Agent for the Multi-Agent GitHub Issue Routing System.

The ModeratorAgent is a meta-agent that orchestrates the deliberation process.
Its responsibilities include:
- Selecting relevant specialist agents for a given GitHub issue
- Detecting convergence in multi-round deliberation (future: Issue #14)
- Preventing rambling and repetitive discussion (future: Issue #14)
- Synthesizing final recommendations (future: Issue #15)

This module implements the agent selection logic using LLM-powered
analysis with a keyword-based fallback when LLM is unavailable.
"""

import json
import logging
import os
import re
from typing import Any, Optional

import litellm

from src.agents.base import BaseAgent
from src.agents.registry import AgentRegistry
from src.models.agent import AgentConfig

logger = logging.getLogger(__name__)

# Selection constraints
MIN_AGENTS = 5
MAX_AGENTS = 15
DEFAULT_AGENT_COUNT = 8


class AgentSelectionResult:
    """
    Result of the agent selection process.

    Contains the selected agents along with metadata about the
    selection process, including reasoning for each agent's selection.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        reasoning: dict[str, str],
        method: str = "llm",
    ) -> None:
        """
        Initialize an AgentSelectionResult.

        Args:
            agents: Ordered list of selected agents (most relevant first)
            reasoning: Map of agent_name -> selection reason
            method: Selection method used ("llm" or "keyword")
        """
        self.agents = agents
        self.reasoning = reasoning
        self.method = method

    @property
    def agent_names(self) -> list[str]:
        """List of selected agent names."""
        return [a.name for a in self.agents]

    @property
    def count(self) -> int:
        """Number of selected agents."""
        return len(self.agents)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "agents": self.agent_names,
            "reasoning": self.reasoning,
            "method": self.method,
            "count": self.count,
        }


# ==================
# Keyword Matching
# ==================

# Maps keywords/patterns to relevant agent names for fallback selection
KEYWORD_AGENT_MAP: dict[str, list[str]] = {
    # UI/Frontend keywords
    r"(?:ui|interface|button|toggle|modal|dialog|form|input|dropdown|menu|navigation|navbar|sidebar|tab|layout|page|view|component|widget|dark.?mode|light.?mode|theme|color|font|style|css|animation|responsive|mobile)":
        ["ui_architect", "ux_designer", "frontend_dev", "ada_expert", "ui_designer"],

    # Backend/API keywords
    r"(?:api|endpoint|route|middleware|server|backend|rest|graphql|request|response|authentication|authorization|jwt|oauth|session|cookie|cors|rate.?limit)":
        ["api_architect", "backend_dev", "security_expert", "auth_expert"],

    # Database keywords
    r"(?:database|db|sql|query|migration|schema|table|index|orm|model|data|storage|persist|cache|redis|postgresql|mysql|sqlite|mongodb)":
        ["data_architect", "database_expert", "backend_dev"],

    # Security keywords
    r"(?:security|vulnerab|xss|csrf|injection|auth|password|encrypt|token|secret|permission|access.?control|firewall|audit|compliance|owasp)":
        ["security_expert", "auth_expert", "privacy_expert", "legal_compliance"],

    # Performance keywords
    r"(?:performance|speed|slow|fast|latency|optimize|cache|memory|cpu|load|scalab|benchmark|profil|bottleneck|concurren)":
        ["performance_expert", "system_architect", "database_expert", "cloud_architect"],

    # Testing keywords
    r"(?:test|qa|quality|bug|regression|coverage|unit.?test|integration|e2e|playwright|cypress|jest|pytest)":
        ["qa_engineer", "backend_dev", "frontend_dev"],

    # DevOps/Infrastructure keywords
    r"(?:deploy|ci|cd|pipeline|docker|kubernetes|container|cloud|aws|gcp|azure|terraform|monitor|log|alert|uptime|sre)":
        ["devops_engineer", "cloud_architect", "sre"],

    # Accessibility keywords
    r"(?:accessib|wcag|aria|screen.?reader|keyboard.?nav|contrast|a11y|disability|impair)":
        ["ada_expert", "ui_architect", "ux_designer", "qa_engineer"],

    # Documentation keywords
    r"(?:document|readme|guide|tutorial|api.?doc|swagger|openapi|comment|changelog)":
        ["tech_writer", "api_architect", "product_manager"],

    # Architecture keywords
    r"(?:architect|design.?pattern|microservice|monolith|event.?driven|message.?queue|pub.?sub|service.?mesh|domain.?driven)":
        ["system_architect", "api_architect", "data_architect"],

    # Mobile keywords
    r"(?:ios|android|swift|kotlin|mobile|app.?store|react.?native|flutter|pwa)":
        ["ios_developer", "android_developer", "mobile_web_dev", "mobile_performance_expert"],

    # Data/ML keywords
    r"(?:machine.?learn|ml|ai|model|train|predict|neural|nlp|data.?science|analytics|a/b.?test|experiment)":
        ["ml_engineer", "data_scientist", "analytics_expert", "data_engineer"],

    # Search keywords
    r"(?:search|elasticsearch|index|relevance|autocomplete|full.?text)":
        ["search_expert", "backend_dev"],

    # Payment keywords
    r"(?:payment|billing|subscription|stripe|invoice|checkout|cart|price|refund|transaction)":
        ["payment_systems_expert", "security_expert", "backend_dev"],

    # Email keywords
    r"(?:email|smtp|notification|alert|newsletter|template|sendgrid|mailgun)":
        ["email_systems_expert", "notification_expert"],

    # Internationalization keywords
    r"(?:i18n|l10n|internation|localiz|translat|language|locale|rtl|multi.?language)":
        ["i18n_expert", "frontend_dev", "ux_researcher"],

    # Real-time keywords
    r"(?:realtime|real.?time|websocket|socket|sse|server.?sent|live|stream|push|notification)":
        ["realtime_systems_expert", "backend_dev", "frontend_dev"],

    # Error handling keywords
    r"(?:error|exception|crash|fail|retry|circuit.?breaker|fallback|graceful|resilien)":
        ["error_handling_expert", "sre", "backend_dev"],

    # SEO keywords
    r"(?:seo|search.?engine|meta.?tag|sitemap|robot|canonical|structured.?data|ranking)":
        ["seo_expert", "frontend_dev", "content_strategist"],

    # Product/UX keywords
    r"(?:user.?story|requirement|feature|roadmap|stakeholder|feedback|persona|journey|usability|ux.?research)":
        ["product_manager", "ux_researcher", "ux_designer"],

    # Privacy keywords
    r"(?:privacy|gdpr|ccpa|consent|cookie|data.?protection|anonymi|retention|pii)":
        ["privacy_expert", "legal_compliance", "security_expert"],
}

# Agents that should always be considered for compliance/oversight
COMPLIANCE_AGENTS = ["security_expert", "qa_engineer"]


def select_agents_by_keywords(
    issue_title: str,
    issue_body: str,
    issue_labels: list[str],
    registry: AgentRegistry,
) -> AgentSelectionResult:
    """
    Select relevant agents using keyword matching.

    This is the fallback method when LLM-based selection is unavailable.
    It analyzes the issue text against known keyword patterns to identify
    relevant agents.

    Args:
        issue_title: The GitHub issue title
        issue_body: The GitHub issue body/description
        issue_labels: List of issue label strings
        registry: The agent registry to select from

    Returns:
        AgentSelectionResult with selected agents and reasoning
    """
    # Combine all text for matching
    text = f"{issue_title} {issue_body} {' '.join(issue_labels)}".lower()

    # Score each agent based on keyword matches
    agent_scores: dict[str, float] = {}
    agent_reasons: dict[str, str] = {}

    for pattern, agent_names in KEYWORD_AGENT_MAP.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            match_str = ", ".join(set(matches[:3]))
            for i, agent_name in enumerate(agent_names):
                if registry.has_agent(agent_name):
                    # Score decreases for agents later in the list (less directly relevant)
                    score = len(matches) * (1.0 - i * 0.15)
                    agent_scores[agent_name] = agent_scores.get(agent_name, 0) + score
                    if agent_name not in agent_reasons:
                        agent_reasons[agent_name] = f"Keyword match: {match_str}"

    # Add compliance agents if not already included
    for agent_name in COMPLIANCE_AGENTS:
        if agent_name not in agent_scores and registry.has_agent(agent_name):
            agent_scores[agent_name] = 0.5
            agent_reasons[agent_name] = "Compliance/oversight perspective"

    # Also boost agents based on their configured priority
    for agent_name in agent_scores:
        agent = registry.get_agent(agent_name)
        agent_scores[agent_name] += agent.priority * 0.1

    # Sort by score (highest first) and apply limits
    sorted_agents = sorted(
        agent_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Select top agents within bounds
    selected_names = [name for name, _ in sorted_agents[:MAX_AGENTS]]

    # Ensure minimum count by adding high-priority agents
    if len(selected_names) < MIN_AGENTS:
        high_priority = registry.get_agents_by_priority(min_priority=7)
        for agent in high_priority:
            if agent.name not in selected_names:
                selected_names.append(agent.name)
                agent_reasons[agent.name] = "High priority agent (general coverage)"
            if len(selected_names) >= MIN_AGENTS:
                break

    # Build result
    selected_agents = []
    for name in selected_names:
        try:
            selected_agents.append(registry.get_agent(name))
        except Exception:
            pass

    logger.info(
        "Keyword-based selection: %d agents selected from %d candidates",
        len(selected_agents),
        len(agent_scores),
    )

    return AgentSelectionResult(
        agents=selected_agents,
        reasoning=agent_reasons,
        method="keyword",
    )


# ==================
# ModeratorAgent
# ==================


class ModeratorAgent:
    """
    Meta-agent that manages the deliberation process.

    The ModeratorAgent is responsible for:
    - Selecting relevant specialist agents for a given issue
    - Analyzing issue content to determine which expertise is needed
    - Ensuring proper team composition (direct + secondary + compliance)

    Uses LLM-powered analysis when available, with keyword-based
    fallback for environments without API keys.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        registry: Optional[AgentRegistry] = None,
    ) -> None:
        """
        Initialize the ModeratorAgent.

        Args:
            model: LLM model identifier (defaults to ANTHROPIC_MODEL env var)
            registry: Optional pre-loaded AgentRegistry
        """
        self.model = model or os.getenv(
            "ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"
        )
        self._registry = registry
        self._logger = logging.getLogger(f"{__name__}.ModeratorAgent")
        self._logger.info("ModeratorAgent initialized with model: %s", self.model)

    @property
    def registry(self) -> Optional[AgentRegistry]:
        """The agent registry used for selection."""
        return self._registry

    @registry.setter
    def registry(self, value: AgentRegistry) -> None:
        """Set the agent registry."""
        self._registry = value

    # ==================
    # Agent Selection
    # ==================

    async def select_relevant_agents(
        self,
        issue: dict[str, Any],
        registry: Optional[AgentRegistry] = None,
        workflow_config: Optional[dict[str, Any]] = None,
    ) -> AgentSelectionResult:
        """
        Analyze a GitHub issue and select the most relevant specialist agents.

        Uses LLM-powered analysis to intelligently match issue content to
        agent expertise. Falls back to keyword matching if LLM is unavailable.

        Args:
            issue: GitHub issue data (title, body, labels, etc.)
            registry: Agent registry (overrides instance registry)
            workflow_config: Optional workflow configuration overrides

        Returns:
            AgentSelectionResult with ordered list of selected agents

        Raises:
            ValueError: If no registry is available
        """
        reg = registry or self._registry
        if reg is None:
            raise ValueError(
                "No agent registry available. "
                "Pass a registry or set it on the ModeratorAgent instance."
            )

        issue_title = issue.get("title", "")
        issue_body = issue.get("body", "")
        issue_labels = issue.get("labels", [])

        self._logger.info(
            "Selecting agents for issue: %s (labels: %s)",
            issue_title[:80],
            issue_labels,
        )

        # Try LLM-based selection first
        try:
            result = await self._select_with_llm(
                issue_title=issue_title,
                issue_body=issue_body,
                issue_labels=issue_labels,
                registry=reg,
            )
            self._logger.info(
                "LLM selection: %d agents selected - %s",
                result.count,
                result.agent_names,
            )
            return result
        except Exception as e:
            self._logger.warning(
                "LLM-based selection failed, falling back to keyword matching: %s",
                str(e),
            )

        # Fallback to keyword matching
        result = select_agents_by_keywords(
            issue_title=issue_title,
            issue_body=issue_body,
            issue_labels=issue_labels,
            registry=reg,
        )
        self._logger.info(
            "Keyword selection: %d agents selected - %s",
            result.count,
            result.agent_names,
        )
        return result

    async def _select_with_llm(
        self,
        issue_title: str,
        issue_body: str,
        issue_labels: list[str],
        registry: AgentRegistry,
    ) -> AgentSelectionResult:
        """
        Use LLM to intelligently select agents for the issue.

        Args:
            issue_title: The GitHub issue title
            issue_body: The issue description
            issue_labels: Issue labels
            registry: Agent registry to select from

        Returns:
            AgentSelectionResult from LLM analysis
        """
        # Build agent catalog for the prompt
        agent_catalog = []
        for agent in registry.get_all_agents():
            agent_catalog.append(
                f"- {agent.name} ({agent.category or 'general'}): "
                f"{agent.expertise}"
            )
        agent_descriptions = "\n".join(agent_catalog)

        prompt = f"""Analyze this GitHub issue and determine which specialist agents should participate in the deliberation.

Issue Title: {issue_title}
Issue Body: {issue_body}
Labels: {', '.join(issue_labels) if issue_labels else 'none'}

Available specialist agents:
{agent_descriptions}

Consider:
1. Direct relevance (agents obviously needed for this issue)
2. Secondary impacts (agents whose domain might be affected)
3. Compliance requirements (security, accessibility, privacy perspectives)
4. Typical team composition for this type of work

Return a JSON object with:
- "agents": array of agent names ordered by relevance (most relevant first)
- "reasoning": object mapping agent name to a brief reason for selection

Select 5-15 agents. Include all critical perspectives."""

        # Determine the model string for litellm
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set - cannot use LLM selection")

        model_string = f"anthropic/{self.model}"

        response = await litellm.acompletion(
            model=model_string,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful issue router for a multi-agent system. "
                        "Output only valid JSON matching the requested format. "
                        "Do not include any text outside the JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            api_key=api_key,
        )

        # Parse the LLM response
        raw_text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        parsed = json.loads(raw_text)

        # Extract agent names and reasoning
        if isinstance(parsed, list):
            # Simple array of names (spec format)
            selected_names = parsed
            reasoning = {name: "Selected by LLM analysis" for name in selected_names}
        elif isinstance(parsed, dict):
            selected_names = parsed.get("agents", [])
            reasoning = parsed.get("reasoning", {})
        else:
            raise ValueError(f"Unexpected LLM response format: {type(parsed)}")

        # Validate and filter to existing agents
        valid_agents = []
        valid_reasoning = {}
        for name in selected_names:
            if registry.has_agent(name):
                valid_agents.append(registry.get_agent(name))
                valid_reasoning[name] = reasoning.get(
                    name, "Selected by LLM analysis"
                )

        # Enforce bounds
        if len(valid_agents) > MAX_AGENTS:
            valid_agents = valid_agents[:MAX_AGENTS]
        elif len(valid_agents) < MIN_AGENTS:
            # Supplement with keyword-based selection
            keyword_result = select_agents_by_keywords(
                issue_title, issue_body, issue_labels, registry
            )
            for agent in keyword_result.agents:
                if agent.name not in [a.name for a in valid_agents]:
                    valid_agents.append(agent)
                    valid_reasoning[agent.name] = keyword_result.reasoning.get(
                        agent.name, "Added to meet minimum agent count"
                    )
                if len(valid_agents) >= MIN_AGENTS:
                    break

        return AgentSelectionResult(
            agents=valid_agents,
            reasoning=valid_reasoning,
            method="llm",
        )
