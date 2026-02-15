"""
Result Poster for posting deliberation results to GitHub issues.

Provides formatting of DeliberationResult into clean, readable Markdown
comments and posting them via the GitHubClient.

Usage:
    from src.integrations.result_poster import ResultPoster

    poster = ResultPoster()
    poster.post_results("owner/repo", 42, deliberation_result)
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

from src.config.settings import get_settings
from src.integrations.github_client import (
    CommentData,
    GitHubClient,
    get_github_client,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Maximum comment length for GitHub (65536 chars)
GITHUB_MAX_COMMENT_LENGTH = 65536

# Leave some buffer for truncation message
TRUNCATION_BUFFER = 500


class ResultFormatter:
    """
    Formats deliberation results into Markdown for GitHub issue comments.

    Produces a well-structured comment with sections for:
    - Status overview (rounds, convergence, duration)
    - Summary / Recommendations
    - Consensus points
    - Conflicts / Disagreements
    - Agent participation statistics
    - Convergence metrics
    - Round-by-round breakdown
    """

    def format(self, result: Any) -> str:
        """
        Format a DeliberationResult as a Markdown comment.

        Args:
            result: DeliberationResult object from the orchestrator.

        Returns:
            Formatted Markdown string.
        """
        sections = []

        # Header
        sections.append(self._format_header(result))

        # Status overview
        sections.append(self._format_status_overview(result))

        # Summary / Recommendations
        summary_section = self._format_summary(result)
        if summary_section:
            sections.append(summary_section)

        # Consensus and conflicts (extracted from conversation)
        consensus_section = self._format_consensus(result)
        if consensus_section:
            sections.append(consensus_section)

        conflicts_section = self._format_conflicts(result)
        if conflicts_section:
            sections.append(conflicts_section)

        # Agent participation
        participation_section = self._format_agent_participation(result)
        if participation_section:
            sections.append(participation_section)

        # Convergence metrics
        metrics_section = self._format_convergence_metrics(result)
        if metrics_section:
            sections.append(metrics_section)

        # Round-by-round breakdown (if available)
        rounds_section = self._format_round_breakdown(result)
        if rounds_section:
            sections.append(rounds_section)

        # Footer
        sections.append(self._format_footer())

        full_comment = "\n\n".join(sections)

        # Truncate if too long for GitHub
        if len(full_comment) > GITHUB_MAX_COMMENT_LENGTH - TRUNCATION_BUFFER:
            full_comment = self._truncate_comment(full_comment)

        return full_comment

    # ========================
    # Section Formatters
    # ========================

    def _format_header(self, result: Any) -> str:
        """Format the main header."""
        status_emoji = self._get_status_emoji(result)
        return (
            f"## {status_emoji} Multi-Agent Deliberation Results\n\n"
            f"*Automated analysis by the Multi-Agent GitHub Issue Routing System*"
        )

    def _format_status_overview(self, result: Any) -> str:
        """Format the status overview section."""
        lines = ["### Overview"]

        status = getattr(result.workflow, "status", None)
        status_value = status.value if status else "unknown"

        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| **Status** | {self._format_status_badge(status_value)} |")
        lines.append(f"| **Rounds** | {result.total_rounds} |")
        lines.append(f"| **Total Comments** | {result.total_comments} |")
        lines.append(
            f"| **Convergence** | {result.final_convergence_score:.0%} |"
        )
        lines.append(f"| **Duration** | {self._format_duration(result.duration_seconds)} |")
        lines.append(f"| **Termination** | {result.termination_reason} |")

        # Add agent count
        if result.agent_participation:
            lines.append(
                f"| **Agents** | {len(result.agent_participation)} participated |"
            )

        return "\n".join(lines)

    def _format_summary(self, result: Any) -> Optional[str]:
        """Format the summary/recommendations section."""
        if not result.summary:
            return None

        lines = [
            "### Recommendations",
            "",
            result.summary,
        ]
        return "\n".join(lines)

    def _format_consensus(self, result: Any) -> Optional[str]:
        """
        Extract and format consensus points from conversation.

        Analyzes conversation history to identify points of agreement.
        """
        if not hasattr(result, "workflow") or not result.workflow:
            return None

        conversation = getattr(result.workflow, "conversation_history", [])
        if not conversation:
            return None

        # Look for agreement patterns in comments
        agreement_indicators = [
            "agree", "consensus", "align", "concur", "support",
            "recommended", "should", "best practice",
        ]

        agreement_comments = []
        for comment in conversation:
            comment_text = comment.comment.lower() if hasattr(comment, "comment") else ""
            if any(indicator in comment_text for indicator in agreement_indicators):
                agent = comment.agent if hasattr(comment, "agent") else "unknown"
                # Extract a brief snippet (first sentence)
                text = comment.comment if hasattr(comment, "comment") else ""
                snippet = text.split(".")[0][:200] if text else ""
                if snippet:
                    agreement_comments.append((agent, snippet))

        if not agreement_comments:
            return None

        lines = [
            "### Consensus Points",
            "",
        ]

        # Show up to 5 key consensus items
        seen_snippets = set()
        for agent, snippet in agreement_comments[:10]:
            # Deduplicate similar snippets
            normalized = snippet.lower().strip()
            if normalized not in seen_snippets:
                seen_snippets.add(normalized)
                lines.append(f"- **{agent}**: {snippet}")
                if len(seen_snippets) >= 5:
                    break

        return "\n".join(lines)

    def _format_conflicts(self, result: Any) -> Optional[str]:
        """
        Extract and format conflicts/disagreements from conversation.

        Analyzes conversation history to identify unresolved conflicts.
        """
        if not hasattr(result, "workflow") or not result.workflow:
            return None

        conversation = getattr(result.workflow, "conversation_history", [])
        if not conversation:
            return None

        # Look for disagreement patterns
        conflict_indicators = [
            "disagree", "concern", "risk", "however", "but",
            "alternative", "instead", "caution", "warning",
        ]

        conflict_comments = []
        for comment in conversation:
            comment_text = comment.comment.lower() if hasattr(comment, "comment") else ""
            if any(indicator in comment_text for indicator in conflict_indicators):
                agent = comment.agent if hasattr(comment, "agent") else "unknown"
                text = comment.comment if hasattr(comment, "comment") else ""
                snippet = text.split(".")[0][:200] if text else ""
                if snippet:
                    conflict_comments.append((agent, snippet))

        if not conflict_comments:
            return None

        lines = [
            "### Points of Discussion",
            "",
        ]

        seen_snippets = set()
        for agent, snippet in conflict_comments[:10]:
            normalized = snippet.lower().strip()
            if normalized not in seen_snippets:
                seen_snippets.add(normalized)
                lines.append(f"- **{agent}**: {snippet}")
                if len(seen_snippets) >= 5:
                    break

        return "\n".join(lines)

    def _format_agent_participation(self, result: Any) -> Optional[str]:
        """Format the agent participation section."""
        if not result.agent_participation:
            return None

        lines = [
            "### Agent Participation",
            "",
            "| Agent | Comments | Role |",
            "|-------|----------|------|",
        ]

        # Sort by comment count (descending)
        sorted_agents = sorted(
            result.agent_participation.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for agent_name, count in sorted_agents:
            # Try to infer role from agent name
            role = self._infer_agent_role(agent_name)
            bar = self._make_bar(count, max(result.agent_participation.values()))
            lines.append(f"| **{agent_name}** | {count} {bar} | {role} |")

        return "\n".join(lines)

    def _format_convergence_metrics(self, result: Any) -> Optional[str]:
        """Format convergence metrics as a summary."""
        if not result.round_metrics:
            # Still show basic convergence info if available
            if result.final_convergence_score > 0:
                lines = [
                    "### Convergence",
                    "",
                    f"Final convergence score: **{result.final_convergence_score:.0%}**",
                ]
                if result.final_convergence_score >= 0.8:
                    lines.append("")
                    lines.append("Strong consensus was reached among the participating agents.")
                elif result.final_convergence_score >= 0.5:
                    lines.append("")
                    lines.append("Moderate agreement was reached. Some areas may need further discussion.")
                else:
                    lines.append("")
                    lines.append("Limited consensus was achieved. Multiple perspectives remain.")
                return "\n".join(lines)
            return None

        lines = [
            "### Convergence Trend",
            "",
        ]

        # Show convergence progression
        for i, metrics in enumerate(result.round_metrics):
            round_num = i + 1
            score = metrics.get("convergence_score", 0.0)
            comments = metrics.get("comments", 0)
            bar = self._make_convergence_bar(score)
            lines.append(
                f"- **Round {round_num}**: {bar} {score:.0%} "
                f"({comments} comment{'s' if comments != 1 else ''})"
            )

        lines.append("")
        lines.append(
            f"**Final Score**: {result.final_convergence_score:.0%} - "
            f"{result.termination_reason}"
        )

        return "\n".join(lines)

    def _format_round_breakdown(self, result: Any) -> Optional[str]:
        """Format a brief round-by-round breakdown."""
        if not hasattr(result, "workflow") or not result.workflow:
            return None

        conversation = getattr(result.workflow, "conversation_history", [])
        if not conversation or len(conversation) <= 3:
            return None

        # Group comments by round
        rounds: dict[int, list] = {}
        for comment in conversation:
            round_num = comment.round if hasattr(comment, "round") else 0
            if round_num not in rounds:
                rounds[round_num] = []
            rounds[round_num].append(comment)

        if len(rounds) <= 1:
            return None

        lines = [
            "<details>",
            "<summary><strong>Round-by-Round Breakdown</strong></summary>",
            "",
        ]

        for round_num in sorted(rounds.keys()):
            round_comments = rounds[round_num]
            agents = [c.agent for c in round_comments if hasattr(c, "agent")]
            lines.append(f"**Round {round_num}** ({len(round_comments)} comments)")
            lines.append(f"- Participants: {', '.join(agents)}")
            lines.append("")

        lines.append("</details>")

        return "\n".join(lines)

    def _format_footer(self) -> str:
        """Format the footer."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            "---\n\n"
            f"*Generated at {timestamp} by "
            f"[Multi-Agent GitHub Issue Routing System]"
            f"(https://github.com/vtggit/CodeAgent)*"
        )

    # ========================
    # Helper Methods
    # ========================

    def _get_status_emoji(self, result: Any) -> str:
        """Get status emoji for the header."""
        status = getattr(result.workflow, "status", None)
        if status:
            status_value = status.value
            return {
                "completed": "\u2705",  # check mark
                "failed": "\u274c",      # cross mark
                "timeout": "\u23f0",     # alarm clock
                "running": "\u23f3",     # hourglass
                "pending": "\u23f3",     # hourglass
            }.get(status_value, "\u2139\ufe0f")   # info
        return "\u2139\ufe0f"

    def _format_status_badge(self, status: str) -> str:
        """Format status as a badge-like text."""
        badges = {
            "completed": "`Completed`",
            "failed": "`Failed`",
            "timeout": "`Timeout`",
            "running": "`Running`",
            "pending": "`Pending`",
        }
        return badges.get(status, f"`{status}`")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _infer_agent_role(self, agent_name: str) -> str:
        """Infer a human-readable role from agent name."""
        role_map = {
            "system_architect": "Architecture",
            "ui_architect": "UI Architecture",
            "data_architect": "Data Architecture",
            "api_architect": "API Design",
            "frontend_dev": "Frontend",
            "backend_dev": "Backend",
            "ios_developer": "iOS",
            "android_developer": "Android",
            "qa_engineer": "QA",
            "security_expert": "Security",
            "ada_expert": "Accessibility",
            "performance_expert": "Performance",
            "devops_engineer": "DevOps",
            "cloud_architect": "Cloud",
            "sre": "Reliability",
            "product_manager": "Product",
            "ux_designer": "UX",
            "ui_designer": "UI Design",
            "tech_writer": "Documentation",
            "ml_engineer": "ML/AI",
            "database_expert": "Database",
            "privacy_expert": "Privacy",
            "legal_compliance": "Compliance",
        }
        return role_map.get(agent_name, agent_name.replace("_", " ").title())

    def _make_bar(self, value: int, max_value: int) -> str:
        """Create a simple text bar chart."""
        if max_value <= 0:
            return ""
        filled = int((value / max_value) * 5)
        return "\u2588" * filled + "\u2591" * (5 - filled)

    def _make_convergence_bar(self, score: float) -> str:
        """Create a convergence progress bar."""
        filled = int(score * 10)
        return "\u2588" * filled + "\u2591" * (10 - filled)

    def _truncate_comment(self, comment: str) -> str:
        """Truncate comment to fit GitHub's limit."""
        max_length = GITHUB_MAX_COMMENT_LENGTH - TRUNCATION_BUFFER
        truncated = comment[:max_length]

        # Try to truncate at a section boundary
        last_section = truncated.rfind("\n### ")
        if last_section > max_length * 0.5:
            truncated = truncated[:last_section]

        truncated += (
            "\n\n---\n\n"
            "*Note: This comment was truncated due to length. "
            "See the full results in the workflow database.*"
        )
        return truncated


class ResultPoster:
    """
    Posts deliberation results to GitHub issues.

    Combines ResultFormatter for Markdown generation with
    GitHubClient for API interaction.

    Args:
        client: Optional GitHubClient instance. If None, uses global client.
        formatter: Optional ResultFormatter. If None, creates default.
        repo_full_name: Optional repository name. If None, reads from env.
    """

    def __init__(
        self,
        client: Optional[GitHubClient] = None,
        formatter: Optional[ResultFormatter] = None,
        repo_full_name: Optional[str] = None,
    ):
        self._client = client
        self._formatter = formatter or ResultFormatter()
        self._repo_full_name = repo_full_name

    @property
    def client(self) -> Optional[GitHubClient]:
        """Get the GitHub client (lazy initialization)."""
        if self._client is None:
            self._client = get_github_client()
        return self._client

    @property
    def repo_full_name(self) -> Optional[str]:
        """Get the repository full name."""
        if self._repo_full_name is None:
            self._repo_full_name = os.getenv("GITHUB_REPOSITORY")
        return self._repo_full_name

    def post_results(
        self,
        repo_full_name: Optional[str],
        issue_number: int,
        result: Any,
    ) -> Optional[CommentData]:
        """
        Post deliberation results to a GitHub issue.

        Args:
            repo_full_name: Repository in "owner/repo" format. If None, uses default.
            issue_number: GitHub issue number.
            result: DeliberationResult from the orchestrator.

        Returns:
            CommentData if posted successfully, None if posting is skipped.

        Raises:
            ValueError: If no repository is configured.
        """
        repo = repo_full_name or self.repo_full_name
        if not repo:
            logger.warning(
                "No repository configured - cannot post results to issue #%d",
                issue_number,
            )
            return None

        if self.client is None:
            logger.debug(
                "GitHub client not available - skipping result posting for issue #%d",
                issue_number,
            )
            return None

        try:
            # Format the comment
            comment_body = self._formatter.format(result)

            logger.info(
                "Posting deliberation results to %s#%d (%d chars)",
                repo,
                issue_number,
                len(comment_body),
            )

            # Post the comment
            comment = self.client.post_comment(repo, issue_number, comment_body)

            logger.info(
                "Successfully posted results to %s#%d (comment_id=%d)",
                repo,
                issue_number,
                comment.id,
            )

            return comment

        except Exception as e:
            logger.error(
                "Failed to post results to %s#%d: %s",
                repo,
                issue_number,
                str(e),
            )
            return None

    def format_only(self, result: Any) -> str:
        """
        Format results without posting (useful for testing/preview).

        Args:
            result: DeliberationResult from the orchestrator.

        Returns:
            Formatted Markdown string.
        """
        return self._formatter.format(result)
