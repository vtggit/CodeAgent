"""
Recommendation Synthesis for the Multi-Agent GitHub Issue Routing System.

This module implements intelligent synthesis of multi-agent deliberation
conversations into structured, actionable recommendations. It replaces
the simple heuristic-based summary with deep analysis of:

- Consensus points (where agents agree)
- Conflicts and disagreements (with both sides presented)
- Actionable recommendations (prioritized by support level)
- Agent participation statistics
- Key insights referenced by multiple agents

The synthesizer can operate in two modes:
1. Heuristic mode: Fast, pattern-based analysis (no LLM needed)
2. LLM-enhanced mode: Uses AI to extract nuanced insights (future)

Design Decisions:
- Recommendations are extracted using keyword patterns and cross-references
- Consensus is measured by agreement patterns and agent alignment
- Conflicts are detected by disagreement patterns between agents
- All output is Markdown-formatted for GitHub comment compatibility
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from src.models.workflow import Comment, WorkflowInstance
from src.orchestration.convergence import (
    AGREEMENT_PATTERNS,
    DISAGREEMENT_PATTERNS,
    VALUE_PATTERNS,
    count_pattern_matches,
    extract_unique_words,
    measure_agreement_level,
    measure_cross_reference_density,
    text_similarity,
)

logger = logging.getLogger(__name__)


# ==================
# Data Models
# ==================


@dataclass
class Recommendation:
    """A single actionable recommendation extracted from the deliberation."""

    text: str
    """The recommendation text (cleaned and concise)."""

    source_agent: str
    """The agent who first proposed this recommendation."""

    supporting_agents: list[str] = field(default_factory=list)
    """Agents who agreed with or built on this recommendation."""

    dissenting_agents: list[str] = field(default_factory=list)
    """Agents who disagreed or raised concerns."""

    priority: str = "medium"
    """Priority level: high, medium, or low."""

    category: str = "general"
    """Category: architecture, implementation, security, testing, etc."""

    round_introduced: int = 1
    """Round in which this recommendation was first made."""

    support_score: float = 0.0
    """Normalized support score (0.0 to 1.0)."""


@dataclass
class ConsensusPoint:
    """A point of agreement among agents."""

    topic: str
    """Brief description of the consensus topic."""

    agents: list[str] = field(default_factory=list)
    """Agents who contributed to this consensus."""

    strength: str = "moderate"
    """Consensus strength: strong, moderate, or weak."""

    evidence: str = ""
    """Key quote or paraphrase supporting this consensus."""


@dataclass
class Conflict:
    """An unresolved disagreement between agents."""

    topic: str
    """Brief description of the disagreement."""

    position_a: str = ""
    """First position/perspective."""

    agents_a: list[str] = field(default_factory=list)
    """Agents holding position A."""

    position_b: str = ""
    """Second position/perspective."""

    agents_b: list[str] = field(default_factory=list)
    """Agents holding position B."""

    resolution_suggestion: str = ""
    """Suggested resolution or compromise, if any."""


@dataclass
class SynthesisResult:
    """Complete synthesis of a deliberation conversation."""

    recommendations: list[Recommendation] = field(default_factory=list)
    """Prioritized list of actionable recommendations."""

    consensus_points: list[ConsensusPoint] = field(default_factory=list)
    """Points of agreement among agents."""

    conflicts: list[Conflict] = field(default_factory=list)
    """Unresolved disagreements."""

    agent_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-agent participation statistics."""

    key_themes: list[str] = field(default_factory=list)
    """Major themes identified in the discussion."""

    overall_agreement: float = 0.0
    """Overall agreement score (0.0 to 1.0)."""

    total_recommendations: int = 0
    """Total number of recommendations extracted."""

    summary_text: str = ""
    """Final markdown-formatted summary text."""


# ==================
# Pattern Extractors
# ==================

# Recommendation extraction patterns
RECOMMENDATION_PATTERNS = [
    r"(?:I\s+)?(?:recommend|suggest|propose)\s+(?:that\s+)?(.{20,200}?)(?:\.|$)",
    r"(?:we\s+should|you\s+should|they\s+should)\s+(.{20,200}?)(?:\.|$)",
    r"(?:the\s+best\s+approach\s+(?:is|would\s+be))\s+(.{20,200}?)(?:\.|$)",
    r"(?:consider\s+(?:using|implementing|adding))\s+(.{20,200}?)(?:\.|$)",
    r"(?:my\s+recommendation\s+is)\s+(.{20,200}?)(?:\.|$)",
    r"(?:the\s+solution\s+(?:is|should\s+be))\s+(.{20,200}?)(?:\.|$)",
    r"(?:I\s+would\s+(?:suggest|advise|recommend))\s+(.{20,200}?)(?:\.|$)",
    r"(?:it(?:'s|\s+is)\s+(?:important|critical|essential)\s+to)\s+(.{20,200}?)(?:\.|$)",
]

# Category detection patterns
CATEGORY_PATTERNS = {
    "architecture": [
        r"\b(?:architect|design\s+pattern|microservice|monolith|component|module|layer|structure)\b",
    ],
    "implementation": [
        r"\b(?:implement|code|develop|build|create|function|method|class|library|framework)\b",
    ],
    "security": [
        r"\b(?:security|vulnerab|auth|encrypt|token|xss|csrf|inject|sanitiz|validat)\b",
    ],
    "testing": [
        r"\b(?:test|qa|coverage|regression|unit|integration|e2e|assert)\b",
    ],
    "performance": [
        r"\b(?:performance|speed|latency|cache|optimize|scalab|memory|cpu)\b",
    ],
    "accessibility": [
        r"\b(?:accessib|wcag|aria|screen.?reader|a11y|contrast|keyboard)\b",
    ],
    "ux": [
        r"\b(?:user\s+experience|ux|usability|intuitive|ui|interface|design)\b",
    ],
    "infrastructure": [
        r"\b(?:deploy|ci|cd|docker|kubernetes|cloud|monitor|log)\b",
    ],
    "data": [
        r"\b(?:database|data|schema|model|migration|query|storage|persist)\b",
    ],
}

# Agreement patterns for detecting which agents support a recommendation
SUPPORT_PATTERNS = [
    r"\bagree\s+with\s+(\w+)",
    r"\bsecond\s+(\w+)(?:'s|\s+)",
    r"\bsupport\s+(\w+)(?:'s|\s+)",
    r"\bbuilding\s+on\s+(\w+)(?:'s|\s+)",
    r"\bas\s+(\w+)\s+(?:mentioned|suggested|noted|said|pointed\s+out)",
    r"\b(\w+)\s+(?:makes?\s+a\s+good|raises?\s+an?\s+(?:excellent|important))\s+point",
]

# Disagreement patterns for detecting which agents disagree
DISSENT_PATTERNS = [
    r"\bdisagree\s+with\s+(\w+)",
    r"\bcontrary\s+to\s+(\w+)(?:'s|\s+)",
    r"\bunlike\s+(\w+)(?:'s|\s+)",
    r"\bwhile\s+(\w+)\s+(?:suggest|recommend|propos)",
    r"\b(\w+)(?:'s|\s+)(?:approach|suggestion|recommendation)\s+(?:may|might|could)\s+(?:not|miss)",
]


# ==================
# Recommendation Synthesizer
# ==================


class RecommendationSynthesizer:
    """
    Synthesizes multi-agent deliberation into structured recommendations.

    Analyzes conversation history to extract:
    - Actionable recommendations with priority and support level
    - Consensus points where agents agree
    - Conflicts where agents disagree
    - Agent participation statistics
    - Key discussion themes

    Usage:
        synthesizer = RecommendationSynthesizer()
        result = synthesizer.synthesize(workflow)
        markdown = result.summary_text
    """

    def __init__(self) -> None:
        """Initialize the synthesizer."""
        self._logger = logging.getLogger(f"{__name__}.RecommendationSynthesizer")

    def synthesize(self, workflow: WorkflowInstance) -> SynthesisResult:
        """
        Perform full synthesis of a deliberation conversation.

        Analyzes the workflow's conversation history and produces
        a structured SynthesisResult with recommendations, consensus
        points, conflicts, and statistics.

        Args:
            workflow: WorkflowInstance with conversation_history populated

        Returns:
            SynthesisResult with all synthesis outputs
        """
        result = SynthesisResult()

        if not workflow.conversation_history:
            result.summary_text = "No discussion occurred - no agents commented."
            return result

        history = workflow.conversation_history

        self._logger.info(
            "Synthesizing %d comments from %d agents across %d round(s)",
            len(history),
            len(set(c.agent for c in history)),
            workflow.current_round,
        )

        # Step 1: Calculate agent statistics
        result.agent_stats = self._calculate_agent_stats(history)

        # Step 2: Extract recommendations
        result.recommendations = self._extract_recommendations(history)
        result.total_recommendations = len(result.recommendations)

        # Step 3: Identify consensus points
        result.consensus_points = self._identify_consensus(history)

        # Step 4: Identify conflicts
        result.conflicts = self._identify_conflicts(history)

        # Step 5: Extract key themes
        result.key_themes = self._extract_themes(history)

        # Step 6: Calculate overall agreement
        result.overall_agreement = measure_agreement_level(history)

        # Step 7: Format as Markdown
        result.summary_text = self._format_markdown(workflow, result)

        self._logger.info(
            "Synthesis complete: %d recommendations, %d consensus points, "
            "%d conflicts, agreement=%.2f",
            result.total_recommendations,
            len(result.consensus_points),
            len(result.conflicts),
            result.overall_agreement,
        )

        return result

    # ==================
    # Extraction Methods
    # ==================

    def _calculate_agent_stats(
        self, history: list[Comment]
    ) -> dict[str, dict[str, Any]]:
        """
        Calculate per-agent participation statistics.

        Args:
            history: Full conversation history

        Returns:
            Dict mapping agent name to stats dict
        """
        stats: dict[str, dict[str, Any]] = {}

        # Group comments by agent
        agent_comments: dict[str, list[Comment]] = defaultdict(list)
        for comment in history:
            agent_comments[comment.agent].append(comment)

        # Reference counts (how often each agent is referenced by others)
        reference_counts: dict[str, int] = Counter()
        for comment in history:
            for ref in comment.references:
                reference_counts[ref] += 1

        for agent_name, comments in agent_comments.items():
            rounds_active = sorted(set(c.round for c in comments))

            # Count recommendation-like statements
            rec_count = 0
            for c in comments:
                rec_count += count_pattern_matches(c.comment, VALUE_PATTERNS)

            # Count agreement/disagreement indicators
            agree_count = sum(
                count_pattern_matches(c.comment, AGREEMENT_PATTERNS)
                for c in comments
            )
            disagree_count = sum(
                count_pattern_matches(c.comment, DISAGREEMENT_PATTERNS)
                for c in comments
            )

            # Total word count
            total_words = sum(len(c.comment.split()) for c in comments)

            stats[agent_name] = {
                "comment_count": len(comments),
                "rounds_active": rounds_active,
                "referenced_by_count": reference_counts.get(agent_name, 0),
                "references_made": sum(len(c.references) for c in comments),
                "recommendation_count": rec_count,
                "agreement_indicators": agree_count,
                "disagreement_indicators": disagree_count,
                "total_words": total_words,
                "avg_words_per_comment": total_words // max(len(comments), 1),
            }

        return stats

    def _extract_recommendations(
        self, history: list[Comment]
    ) -> list[Recommendation]:
        """
        Extract actionable recommendations from agent comments.

        Uses regex patterns to identify recommendation-like statements,
        then analyzes cross-references to determine support/dissent.

        Args:
            history: Full conversation history

        Returns:
            List of Recommendation objects, sorted by support score
        """
        raw_recommendations: list[Recommendation] = []

        # Group comments by agent for reference lookup
        agent_comments: dict[str, list[Comment]] = defaultdict(list)
        for c in history:
            agent_comments[c.agent].append(c)

        # Extract recommendation candidates from each comment
        for comment in history:
            for pattern in RECOMMENDATION_PATTERNS:
                matches = re.findall(pattern, comment.comment, re.IGNORECASE)
                for match in matches:
                    # Clean up the extracted text
                    rec_text = match.strip()
                    if len(rec_text) < 20:
                        continue  # Skip very short matches

                    # Check for duplicates (similar recommendations)
                    is_duplicate = False
                    for existing in raw_recommendations:
                        if text_similarity(rec_text, existing.text) > 0.6:
                            # Merge: add this agent as supporting
                            if comment.agent not in [existing.source_agent] + existing.supporting_agents:
                                existing.supporting_agents.append(comment.agent)
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        category = self._categorize_text(rec_text)
                        raw_recommendations.append(
                            Recommendation(
                                text=rec_text,
                                source_agent=comment.agent,
                                round_introduced=comment.round,
                                category=category,
                            )
                        )

        # Analyze support and dissent for each recommendation
        for rec in raw_recommendations:
            rec.supporting_agents, rec.dissenting_agents = (
                self._find_support_and_dissent(rec, history)
            )
            # Calculate support score
            total_agents = len(set(c.agent for c in history))
            supporters = len(set([rec.source_agent] + rec.supporting_agents))
            dissenters = len(set(rec.dissenting_agents))
            if total_agents > 0:
                rec.support_score = (supporters - dissenters * 0.5) / total_agents
            rec.support_score = max(0.0, min(rec.support_score, 1.0))

            # Assign priority based on support
            if rec.support_score >= 0.6:
                rec.priority = "high"
            elif rec.support_score >= 0.3:
                rec.priority = "medium"
            else:
                rec.priority = "low"

        # Sort by support score (highest first), then by round introduced
        raw_recommendations.sort(
            key=lambda r: (-r.support_score, r.round_introduced)
        )

        return raw_recommendations

    def _find_support_and_dissent(
        self,
        recommendation: Recommendation,
        history: list[Comment],
    ) -> tuple[list[str], list[str]]:
        """
        Find agents who support or dissent from a recommendation.

        Analyzes subsequent comments for agreement/disagreement
        with the source agent and similar content.

        Args:
            recommendation: The recommendation to analyze
            history: Full conversation history

        Returns:
            Tuple of (supporting_agents, dissenting_agents)
        """
        supporting: set[str] = set()
        dissenting: set[str] = set()
        source = recommendation.source_agent
        rec_words = extract_unique_words(recommendation.text)

        for comment in history:
            if comment.agent == source:
                continue

            comment_lower = comment.comment.lower()

            # Check if this comment references the source agent
            references_source = source in comment.references
            # Check if comment discusses similar topics
            comment_words = extract_unique_words(comment.comment)
            topic_overlap = len(rec_words & comment_words) / max(len(rec_words), 1)

            if references_source or topic_overlap > 0.3:
                # Check for agreement
                agree = count_pattern_matches(comment.comment, AGREEMENT_PATTERNS)
                disagree = count_pattern_matches(comment.comment, DISAGREEMENT_PATTERNS)

                if agree > disagree:
                    supporting.add(comment.agent)
                elif disagree > agree:
                    dissenting.add(comment.agent)

        return list(supporting), list(dissenting)

    def _identify_consensus(self, history: list[Comment]) -> list[ConsensusPoint]:
        """
        Identify points of consensus among agents.

        Looks for topics where multiple agents agree, building
        consensus points from clusters of agreement.

        Args:
            history: Full conversation history

        Returns:
            List of ConsensusPoint objects
        """
        consensus_points: list[ConsensusPoint] = []

        # Group comments by round
        rounds: dict[int, list[Comment]] = defaultdict(list)
        for c in history:
            rounds[c.round].append(c)

        # Track topic clusters: sets of agents discussing similar things
        # Use a simplified approach: group comments with high word overlap
        topic_clusters: list[dict[str, Any]] = []

        for comment in history:
            comment_words = extract_unique_words(comment.comment)
            if not comment_words:
                continue

            # Try to add to an existing cluster
            added = False
            for cluster in topic_clusters:
                overlap = len(comment_words & cluster["words"]) / max(
                    len(comment_words), 1
                )
                if overlap > 0.25:
                    cluster["agents"].add(comment.agent)
                    cluster["words"] |= comment_words
                    cluster["comments"].append(comment)
                    added = True
                    break

            if not added:
                topic_clusters.append({
                    "agents": {comment.agent},
                    "words": comment_words,
                    "comments": [comment],
                })

        # Convert clusters with multiple agents showing agreement to consensus points
        for cluster in topic_clusters:
            if len(cluster["agents"]) < 2:
                continue

            # Check if the cluster shows agreement
            cluster_agreement = measure_agreement_level(cluster["comments"])
            if cluster_agreement < 0.4:
                continue

            # Extract the topic from the most common significant words
            all_text = " ".join(c.comment for c in cluster["comments"])
            topic_words = extract_unique_words(all_text)
            # Pick top keywords (most common in the cluster)
            word_counts = Counter()
            for c in cluster["comments"]:
                for w in extract_unique_words(c.comment):
                    word_counts[w] += 1

            top_words = [w for w, _ in word_counts.most_common(5)]
            topic = ", ".join(top_words[:3]) if top_words else "general discussion"

            # Get the best evidence quote
            evidence = ""
            for c in cluster["comments"]:
                if count_pattern_matches(c.comment, AGREEMENT_PATTERNS) > 0:
                    evidence = c.comment[:150]
                    break
            if not evidence and cluster["comments"]:
                evidence = cluster["comments"][0].comment[:150]

            # Determine strength
            if cluster_agreement >= 0.7 and len(cluster["agents"]) >= 3:
                strength = "strong"
            elif cluster_agreement >= 0.5:
                strength = "moderate"
            else:
                strength = "weak"

            consensus_points.append(
                ConsensusPoint(
                    topic=topic,
                    agents=sorted(cluster["agents"]),
                    strength=strength,
                    evidence=evidence + ("..." if len(evidence) >= 150 else ""),
                )
            )

        # Sort by strength (strong first) and number of agents
        strength_order = {"strong": 0, "moderate": 1, "weak": 2}
        consensus_points.sort(
            key=lambda cp: (strength_order.get(cp.strength, 3), -len(cp.agents))
        )

        return consensus_points

    def _identify_conflicts(self, history: list[Comment]) -> list[Conflict]:
        """
        Identify unresolved conflicts in the deliberation.

        Looks for agent pairs with disagreement patterns, and tries
        to extract both positions.

        Args:
            history: Full conversation history

        Returns:
            List of Conflict objects
        """
        conflicts: list[Conflict] = []

        # Build agent pairs with disagreement
        disagreements: dict[tuple[str, str], list[Comment]] = defaultdict(list)

        for comment in history:
            if not comment.references:
                continue

            disagree_count = count_pattern_matches(
                comment.comment, DISAGREEMENT_PATTERNS
            )
            if disagree_count < 1:
                continue

            for ref in comment.references:
                pair = tuple(sorted([comment.agent, ref]))
                disagreements[pair].append(comment)

        # Convert to conflicts
        agent_comments: dict[str, list[Comment]] = defaultdict(list)
        for c in history:
            agent_comments[c.agent].append(c)

        for (agent_a, agent_b), dispute_comments in disagreements.items():
            # Get representative positions
            position_a = ""
            position_b = ""

            # Find the first substantial comment from each agent
            for c in agent_comments.get(agent_a, []):
                if len(c.comment) > 30:
                    position_a = c.comment[:200]
                    break

            for c in agent_comments.get(agent_b, []):
                if len(c.comment) > 30:
                    position_b = c.comment[:200]
                    break

            # Extract topic from the dispute
            all_dispute_text = " ".join(c.comment for c in dispute_comments)
            dispute_words = extract_unique_words(all_dispute_text)
            topic_words = list(dispute_words)[:3]
            topic = ", ".join(topic_words) if topic_words else "approach disagreement"

            # Check if there's a resolution suggestion
            resolution = ""
            for c in history:
                if c.round > dispute_comments[0].round:
                    c_lower = c.comment.lower()
                    if any(
                        p in c_lower
                        for p in ["compromise", "middle ground", "both", "combine", "hybrid"]
                    ):
                        resolution = c.comment[:200]
                        break

            conflicts.append(
                Conflict(
                    topic=topic,
                    position_a=position_a + ("..." if len(position_a) >= 200 else ""),
                    agents_a=[agent_a],
                    position_b=position_b + ("..." if len(position_b) >= 200 else ""),
                    agents_b=[agent_b],
                    resolution_suggestion=resolution,
                )
            )

        return conflicts

    def _extract_themes(self, history: list[Comment]) -> list[str]:
        """
        Extract major discussion themes from conversation history.

        Uses word frequency and category detection to identify
        the main themes discussed by agents.

        Args:
            history: Full conversation history

        Returns:
            List of theme strings (most prominent first)
        """
        # Count words across all comments
        word_counts: Counter = Counter()
        for comment in history:
            words = extract_unique_words(comment.comment)
            word_counts.update(words)

        # Detect categories discussed
        all_text = " ".join(c.comment for c in history)
        category_scores: dict[str, int] = {}

        for category, patterns in CATEGORY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, all_text, re.IGNORECASE))
            if score > 0:
                category_scores[category] = score

        # Sort categories by score
        sorted_categories = sorted(
            category_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Return top themes as formatted strings
        themes = []
        for category, score in sorted_categories[:5]:
            themes.append(category.replace("_", " ").title())

        return themes

    def _categorize_text(self, text: str) -> str:
        """
        Categorize a text snippet into a domain category.

        Args:
            text: Text to categorize

        Returns:
            Category string
        """
        best_category = "general"
        best_score = 0

        for category, patterns in CATEGORY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, text, re.IGNORECASE))
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    # ==================
    # Markdown Formatting
    # ==================

    def _format_markdown(
        self,
        workflow: WorkflowInstance,
        result: SynthesisResult,
    ) -> str:
        """
        Format the synthesis result as a Markdown summary.

        Produces a well-structured GitHub-compatible Markdown document
        with all synthesis sections.

        Args:
            workflow: The workflow instance
            result: The synthesis result to format

        Returns:
            Markdown-formatted string
        """
        sections: list[str] = []

        # Header
        issue_title = (
            workflow.github_issue.get("title", "N/A")
            if workflow.github_issue
            else "N/A"
        )
        total_agents = len(result.agent_stats)
        total_comments = len(workflow.conversation_history)
        total_rounds = workflow.current_round

        sections.append("## Deliberation Summary\n")
        sections.append(
            f"**Issue:** {issue_title}\n"
            f"**Rounds:** {total_rounds} | "
            f"**Agents:** {total_agents} | "
            f"**Comments:** {total_comments} | "
            f"**Agreement:** {result.overall_agreement:.0%}\n"
        )

        # Recommendations section
        rec_section = self._format_recommendations(result.recommendations)
        if rec_section:
            sections.append(rec_section)

        # Consensus section
        consensus_section = self._format_consensus(result.consensus_points)
        if consensus_section:
            sections.append(consensus_section)

        # Conflicts section
        conflict_section = self._format_conflicts(result.conflicts)
        if conflict_section:
            sections.append(conflict_section)

        # Key themes section
        if result.key_themes:
            themes_str = ", ".join(f"**{t}**" for t in result.key_themes)
            sections.append(f"### Key Themes\n\n{themes_str}\n")

        # Agent participation section
        agent_section = self._format_agent_participation(result.agent_stats)
        if agent_section:
            sections.append(agent_section)

        # Consensus status
        sections.append(self._format_consensus_status(result.overall_agreement))

        return "\n".join(sections)

    def _format_recommendations(
        self, recommendations: list[Recommendation]
    ) -> Optional[str]:
        """Format recommendations as Markdown."""
        if not recommendations:
            return None

        lines = ["### Actionable Recommendations\n"]

        # Group by priority
        high = [r for r in recommendations if r.priority == "high"]
        medium = [r for r in recommendations if r.priority == "medium"]
        low = [r for r in recommendations if r.priority == "low"]

        if high:
            lines.append("#### High Priority")
            for i, rec in enumerate(high, 1):
                supporters_str = ""
                if rec.supporting_agents:
                    supporters_str = f" (supported by: {', '.join(rec.supporting_agents)})"
                lines.append(
                    f"{i}. **[{rec.category.title()}]** {rec.text}"
                    f"\n   - *Proposed by:* {rec.source_agent}{supporters_str}"
                )
            lines.append("")

        if medium:
            lines.append("#### Medium Priority")
            for i, rec in enumerate(medium, 1):
                lines.append(
                    f"{i}. **[{rec.category.title()}]** {rec.text}"
                    f"\n   - *Proposed by:* {rec.source_agent}"
                )
            lines.append("")

        if low:
            lines.append("#### Additional Suggestions")
            for i, rec in enumerate(low, 1):
                lines.append(
                    f"{i}. {rec.text} *(from {rec.source_agent})*"
                )
            lines.append("")

        return "\n".join(lines)

    def _format_consensus(
        self, consensus_points: list[ConsensusPoint]
    ) -> Optional[str]:
        """Format consensus points as Markdown."""
        if not consensus_points:
            return None

        lines = ["### Consensus Points\n"]

        for cp in consensus_points:
            strength_icon = {
                "strong": "+++",
                "moderate": "++",
                "weak": "+",
            }.get(cp.strength, "+")

            agents_str = ", ".join(cp.agents)
            lines.append(
                f"- [{strength_icon}] **{cp.topic}**"
                f"\n  - Agents: {agents_str}"
            )
            if cp.evidence:
                lines.append(f'  - > "{cp.evidence}"')

        lines.append("")
        return "\n".join(lines)

    def _format_conflicts(self, conflicts: list[Conflict]) -> Optional[str]:
        """Format conflicts as Markdown."""
        if not conflicts:
            return None

        lines = ["### Unresolved Disagreements\n"]

        for i, conflict in enumerate(conflicts, 1):
            lines.append(f"**{i}. {conflict.topic}**")
            if conflict.position_a:
                lines.append(
                    f"  - {', '.join(conflict.agents_a)}: "
                    f'"{conflict.position_a}"'
                )
            if conflict.position_b:
                lines.append(
                    f"  - {', '.join(conflict.agents_b)}: "
                    f'"{conflict.position_b}"'
                )
            if conflict.resolution_suggestion:
                lines.append(
                    f"  - *Potential resolution:* {conflict.resolution_suggestion[:200]}"
                )
            lines.append("")

        return "\n".join(lines)

    def _format_agent_participation(
        self, agent_stats: dict[str, dict[str, Any]]
    ) -> Optional[str]:
        """Format agent participation as Markdown table."""
        if not agent_stats:
            return None

        lines = ["### Agent Participation\n"]
        lines.append(
            "| Agent | Comments | Rounds | Referenced By | Recommendations |"
        )
        lines.append("|-------|----------|--------|---------------|-----------------|")

        # Sort by comment count
        sorted_agents = sorted(
            agent_stats.items(),
            key=lambda x: x[1]["comment_count"],
            reverse=True,
        )

        for agent_name, stats in sorted_agents:
            rounds = ", ".join(str(r) for r in stats["rounds_active"])
            lines.append(
                f"| {agent_name} | {stats['comment_count']} "
                f"| {rounds} "
                f"| {stats['referenced_by_count']} "
                f"| {stats['recommendation_count']} |"
            )

        lines.append("")
        return "\n".join(lines)

    def _format_consensus_status(self, agreement: float) -> str:
        """Format the overall consensus status."""
        lines = ["### Consensus Status\n"]

        if agreement >= 0.8:
            lines.append(
                "**Strong consensus** reached among participating agents. "
                "The recommendations above reflect a high degree of alignment."
            )
        elif agreement >= 0.6:
            lines.append(
                "**Moderate agreement** with some differing perspectives. "
                "Most agents converged on key points, though some nuances remain."
            )
        elif agreement >= 0.4:
            lines.append(
                "**Mixed opinions** across the panel. Multiple valid approaches "
                "were discussed. Review the conflicts section for areas needing resolution."
            )
        else:
            lines.append(
                "**Significant disagreement** among agents. Manual review is recommended "
                "to weigh the different perspectives and make a final decision."
            )

        return "\n".join(lines)
