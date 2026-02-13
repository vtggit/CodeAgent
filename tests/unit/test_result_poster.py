"""
Unit tests for the ResultFormatter and ResultPoster.

Tests cover:
- Markdown formatting of deliberation results
- All sections: header, status, summary, consensus, conflicts, agents, metrics
- Comment truncation for very long results
- ResultPoster integration with GitHubClient
- Error handling for posting failures
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.integrations.result_poster import (
    ResultFormatter,
    ResultPoster,
    GITHUB_MAX_COMMENT_LENGTH,
)
from src.models.workflow import (
    Comment,
    WorkflowInstance,
    WorkflowStatus,
    WorkflowMetrics,
)


# ========================
# Fixtures
# ========================


@pytest.fixture
def formatter():
    """Create a ResultFormatter instance."""
    return ResultFormatter()


@pytest.fixture
def sample_workflow():
    """Create a sample workflow instance."""
    return WorkflowInstance(
        instance_id="issue-42-20260208-143022",
        workflow_id="github-issue-deliberation",
        status=WorkflowStatus.COMPLETED,
        github_issue={
            "number": 42,
            "title": "Add dark mode toggle",
            "body": "Need dark mode support",
            "labels": ["enhancement", "ui"],
        },
        selected_agents=["ui_architect", "ada_expert", "frontend_dev", "security_expert"],
        conversation_history=[
            Comment(
                round=1,
                agent="ui_architect",
                comment="I recommend using a CSS variables approach for dark mode. This allows for easy theme switching and should support all major browsers.",
                references=[],
            ),
            Comment(
                round=1,
                agent="ada_expert",
                comment="I agree with the CSS approach. However, we need to ensure contrast ratios meet WCAG 2.1 AA standards. The minimum contrast ratio should be 4.5:1 for normal text.",
                references=["ui_architect"],
            ),
            Comment(
                round=1,
                agent="frontend_dev",
                comment="I support the CSS variables approach. We should also consider using prefers-color-scheme media query for automatic detection. The best practice would be to store user preference in localStorage.",
                references=["ui_architect"],
            ),
            Comment(
                round=2,
                agent="security_expert",
                comment="There is a concern about localStorage usage - we should sanitize any stored values. However, I agree the overall approach is sound.",
                references=["frontend_dev"],
            ),
            Comment(
                round=2,
                agent="ui_architect",
                comment="I concur with the security concern. We should validate the theme value against an allowed list. I recommend a simple toggle instead of free-text input.",
                references=["security_expert"],
            ),
        ],
        current_round=2,
        summary="Use CSS variables for theming with localStorage persistence and WCAG-compliant contrast ratios.",
    )


@pytest.fixture
def sample_result(sample_workflow):
    """Create a sample DeliberationResult."""
    result = MagicMock()
    result.workflow = sample_workflow
    result.summary = "Use CSS variables for theming with localStorage persistence and WCAG-compliant contrast ratios."
    result.total_rounds = 2
    result.total_comments = 5
    result.final_convergence_score = 0.85
    result.termination_reason = "High consensus reached"
    result.duration_seconds = 45.3
    result.agent_participation = {
        "ui_architect": 2,
        "ada_expert": 1,
        "frontend_dev": 1,
        "security_expert": 1,
    }
    result.round_metrics = [
        {"convergence_score": 0.5, "comments": 3},
        {"convergence_score": 0.85, "comments": 2},
    ]
    return result


@pytest.fixture
def minimal_result():
    """Create a minimal DeliberationResult with few fields."""
    workflow = WorkflowInstance(
        instance_id="issue-1-20260208-100000",
        status=WorkflowStatus.COMPLETED,
        conversation_history=[],
    )
    result = MagicMock()
    result.workflow = workflow
    result.summary = None
    result.total_rounds = 1
    result.total_comments = 0
    result.final_convergence_score = 0.0
    result.termination_reason = "No comments generated"
    result.duration_seconds = 1.2
    result.agent_participation = {}
    result.round_metrics = []
    return result


@pytest.fixture
def failed_result():
    """Create a failed DeliberationResult."""
    workflow = WorkflowInstance(
        instance_id="issue-99-20260208-120000",
        status=WorkflowStatus.FAILED,
        conversation_history=[],
    )
    result = MagicMock()
    result.workflow = workflow
    result.summary = None
    result.total_rounds = 0
    result.total_comments = 0
    result.final_convergence_score = 0.0
    result.termination_reason = "Agent selection failed"
    result.duration_seconds = 0.5
    result.agent_participation = {}
    result.round_metrics = []
    return result


# ========================
# ResultFormatter Tests
# ========================


class TestResultFormatterHeader:
    """Tests for header formatting."""

    def test_completed_header(self, formatter, sample_result):
        """Completed result uses check mark emoji."""
        output = formatter.format(sample_result)
        assert "\u2705" in output  # check mark
        assert "Multi-Agent Deliberation Results" in output

    def test_failed_header(self, formatter, failed_result):
        """Failed result uses cross mark emoji."""
        output = formatter.format(failed_result)
        assert "\u274c" in output  # cross mark

    def test_header_contains_system_name(self, formatter, sample_result):
        """Header mentions the system name."""
        output = formatter.format(sample_result)
        assert "Multi-Agent GitHub Issue Routing System" in output


class TestResultFormatterStatusOverview:
    """Tests for status overview section."""

    def test_contains_all_metrics(self, formatter, sample_result):
        """Status overview contains all required metrics."""
        output = formatter.format(sample_result)
        assert "Rounds" in output
        assert "2" in output  # total_rounds
        assert "Total Comments" in output
        assert "5" in output  # total_comments
        assert "Convergence" in output
        assert "85%" in output  # convergence score
        assert "Duration" in output
        assert "45.3s" in output  # duration
        assert "High consensus reached" in output  # termination reason

    def test_status_badge_completed(self, formatter, sample_result):
        """Completed status shows correct badge."""
        output = formatter.format(sample_result)
        assert "`Completed`" in output

    def test_status_badge_failed(self, formatter, failed_result):
        """Failed status shows correct badge."""
        output = formatter.format(failed_result)
        assert "`Failed`" in output

    def test_agent_count_shown(self, formatter, sample_result):
        """Agent count is shown in overview."""
        output = formatter.format(sample_result)
        assert "4 participated" in output


class TestResultFormatterSummary:
    """Tests for summary/recommendations section."""

    def test_summary_present(self, formatter, sample_result):
        """Summary section is included when available."""
        output = formatter.format(sample_result)
        assert "### Recommendations" in output
        assert "CSS variables" in output

    def test_no_summary_section_when_empty(self, formatter, minimal_result):
        """Summary section is omitted when no summary available."""
        output = formatter.format(minimal_result)
        assert "### Recommendations" not in output


class TestResultFormatterConsensus:
    """Tests for consensus points section."""

    def test_consensus_extracted(self, formatter, sample_result):
        """Consensus points are extracted from conversation."""
        output = formatter.format(sample_result)
        assert "Consensus Points" in output or "Points of Discussion" in output

    def test_no_consensus_when_no_conversation(self, formatter, minimal_result):
        """Consensus section omitted with no conversation."""
        output = formatter.format(minimal_result)
        assert "Consensus Points" not in output


class TestResultFormatterConflicts:
    """Tests for conflicts/discussion section."""

    def test_conflicts_detected(self, formatter, sample_result):
        """Discussion points are detected from conversation."""
        output = formatter.format(sample_result)
        # The sample has "concern" and "however" which are conflict indicators
        assert "Points of Discussion" in output

    def test_no_conflicts_when_no_conversation(self, formatter, minimal_result):
        """Conflicts section omitted with no conversation."""
        output = formatter.format(minimal_result)
        assert "Points of Discussion" not in output


class TestResultFormatterAgentParticipation:
    """Tests for agent participation section."""

    def test_agent_participation_table(self, formatter, sample_result):
        """Agent participation table is formatted."""
        output = formatter.format(sample_result)
        assert "### Agent Participation" in output
        assert "ui_architect" in output
        assert "ada_expert" in output
        assert "frontend_dev" in output
        assert "security_expert" in output

    def test_agent_sorted_by_count(self, formatter, sample_result):
        """Agents are sorted by comment count (descending)."""
        output = formatter.format(sample_result)
        ui_pos = output.find("ui_architect")
        ada_pos = output.find("ada_expert")
        # ui_architect has 2 comments, should appear first
        assert ui_pos < ada_pos

    def test_agent_roles_inferred(self, formatter, sample_result):
        """Agent roles are inferred from names."""
        output = formatter.format(sample_result)
        assert "UI Architecture" in output or "Security" in output

    def test_no_participation_when_empty(self, formatter, minimal_result):
        """Participation section omitted when no agents."""
        output = formatter.format(minimal_result)
        assert "### Agent Participation" not in output


class TestResultFormatterConvergenceMetrics:
    """Tests for convergence metrics section."""

    def test_convergence_trend(self, formatter, sample_result):
        """Convergence trend is shown with round metrics."""
        output = formatter.format(sample_result)
        assert "Convergence Trend" in output
        assert "Round 1" in output
        assert "Round 2" in output
        assert "50%" in output
        assert "85%" in output

    def test_basic_convergence_without_metrics(self, formatter):
        """Basic convergence shown when no round_metrics but has score."""
        result = MagicMock()
        result.workflow = MagicMock()
        result.workflow.status = WorkflowStatus.COMPLETED
        result.workflow.conversation_history = []
        result.summary = None
        result.total_rounds = 1
        result.total_comments = 1
        result.final_convergence_score = 0.9
        result.termination_reason = "Consensus"
        result.duration_seconds = 10.0
        result.agent_participation = {}
        result.round_metrics = []

        output = formatter.format(result)
        assert "Convergence" in output
        assert "90%" in output
        assert "Strong consensus" in output


class TestResultFormatterRoundBreakdown:
    """Tests for round-by-round breakdown."""

    def test_round_breakdown_present(self, formatter, sample_result):
        """Round breakdown is shown for multi-round results."""
        output = formatter.format(sample_result)
        assert "Round-by-Round Breakdown" in output
        assert "<details>" in output  # Collapsible section

    def test_no_breakdown_for_minimal(self, formatter, minimal_result):
        """No breakdown for results with few comments."""
        output = formatter.format(minimal_result)
        assert "Round-by-Round Breakdown" not in output


class TestResultFormatterFooter:
    """Tests for footer formatting."""

    def test_footer_present(self, formatter, sample_result):
        """Footer is present with timestamp."""
        output = formatter.format(sample_result)
        assert "Generated at" in output
        assert "Multi-Agent GitHub Issue Routing System" in output


class TestResultFormatterDuration:
    """Tests for duration formatting."""

    def test_seconds_format(self, formatter):
        """Short durations formatted in seconds."""
        assert formatter._format_duration(45.3) == "45.3s"

    def test_minutes_format(self, formatter):
        """Medium durations formatted in minutes."""
        assert formatter._format_duration(125.0) == "2.1m"

    def test_hours_format(self, formatter):
        """Long durations formatted in hours."""
        assert formatter._format_duration(7200.0) == "2.0h"


class TestResultFormatterTruncation:
    """Tests for comment truncation."""

    def test_short_comments_not_truncated(self, formatter, sample_result):
        """Normal-length comments are not truncated."""
        output = formatter.format(sample_result)
        assert "truncated" not in output

    def test_long_comments_truncated(self, formatter):
        """Very long comments are truncated."""
        result = MagicMock()
        result.workflow = MagicMock()
        result.workflow.status = WorkflowStatus.COMPLETED
        result.workflow.conversation_history = []
        result.summary = "x" * 70000  # Very long summary
        result.total_rounds = 1
        result.total_comments = 1
        result.final_convergence_score = 0.5
        result.termination_reason = "Max rounds"
        result.duration_seconds = 10.0
        result.agent_participation = {}
        result.round_metrics = []

        output = formatter.format(result)
        assert len(output) < GITHUB_MAX_COMMENT_LENGTH
        assert "truncated" in output


class TestResultFormatterHelpers:
    """Tests for helper methods."""

    def test_make_bar(self, formatter):
        """Bar chart generation works."""
        bar = formatter._make_bar(5, 5)
        assert "\u2588" in bar  # Full block

    def test_make_convergence_bar(self, formatter):
        """Convergence bar generation works."""
        bar = formatter._make_convergence_bar(0.5)
        assert len(bar) == 10  # Always 10 chars

    def test_infer_agent_role(self, formatter):
        """Agent role inference works."""
        assert formatter._infer_agent_role("security_expert") == "Security"
        assert formatter._infer_agent_role("qa_engineer") == "QA"
        assert formatter._infer_agent_role("unknown_agent") == "Unknown Agent"


# ========================
# ResultPoster Tests
# ========================


class TestResultPoster:
    """Tests for ResultPoster."""

    def test_post_results_success(self, sample_result):
        """Successfully posts results to GitHub."""
        mock_client = MagicMock()
        mock_comment = MagicMock()
        mock_comment.id = 456
        mock_client.post_comment.return_value = mock_comment

        poster = ResultPoster(client=mock_client, repo_full_name="test/repo")
        result = poster.post_results("test/repo", 42, sample_result)

        assert result is not None
        assert result.id == 456
        mock_client.post_comment.assert_called_once()
        # Verify the comment body contains expected content
        call_args = mock_client.post_comment.call_args
        body = call_args[0][2]  # Third positional arg
        assert "Multi-Agent Deliberation Results" in body

    def test_post_results_no_client(self, sample_result):
        """Returns None when no client available."""
        poster = ResultPoster(client=None, repo_full_name="test/repo")
        # Mock get_github_client to return None
        with patch(
            "src.integrations.result_poster.get_github_client",
            return_value=None,
        ):
            result = poster.post_results("test/repo", 42, sample_result)
        assert result is None

    def test_post_results_no_repo(self, sample_result):
        """Returns None when no repository configured."""
        mock_client = MagicMock()
        poster = ResultPoster(client=mock_client, repo_full_name=None)

        with patch.dict("os.environ", {}, clear=True):
            result = poster.post_results(None, 42, sample_result)
        assert result is None

    def test_post_results_handles_error(self, sample_result):
        """Handles posting errors gracefully."""
        mock_client = MagicMock()
        mock_client.post_comment.side_effect = Exception("API error")

        poster = ResultPoster(client=mock_client, repo_full_name="test/repo")
        result = poster.post_results("test/repo", 42, sample_result)

        assert result is None

    def test_format_only(self, sample_result):
        """format_only returns formatted string without posting."""
        poster = ResultPoster(client=None)
        output = poster.format_only(sample_result)

        assert isinstance(output, str)
        assert "Multi-Agent Deliberation Results" in output
        assert len(output) > 100

    def test_custom_formatter(self, sample_result):
        """Accepts custom formatter."""
        mock_formatter = MagicMock()
        mock_formatter.format.return_value = "Custom formatted output"

        mock_client = MagicMock()
        mock_comment = MagicMock()
        mock_comment.id = 789
        mock_client.post_comment.return_value = mock_comment

        poster = ResultPoster(
            client=mock_client,
            formatter=mock_formatter,
            repo_full_name="test/repo",
        )
        poster.post_results("test/repo", 42, sample_result)

        mock_formatter.format.assert_called_once_with(sample_result)
        mock_client.post_comment.assert_called_once_with(
            "test/repo", 42, "Custom formatted output"
        )
