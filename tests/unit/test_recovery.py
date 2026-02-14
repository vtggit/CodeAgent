"""
Tests for the Workflow Recovery module.

Tests cover:
- Reconstructing WorkflowInstance from database records
- Recovery of conversation history from ConversationRecords
- Agent selection recovery
- Convergence state extraction
- Finding interrupted workflows
- Resume deliberation flow
- Edge cases (empty history, missing agents, completed workflows)
- Save/recovery round-trip integrity
"""

import json
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.database import crud
from src.database.models import (
    AgentRecord,
    Base,
    ConversationRecord,
    ConvergenceMetricRecord,
    WorkflowRecord,
)
from src.models.workflow import (
    Comment,
    WorkflowConfig,
    WorkflowInstance,
    WorkflowStatus,
)
from src.orchestration.recovery import (
    WorkflowRecoveryError,
    _reconstruct_conversation,
    _reconstruct_issue,
    get_last_convergence_state,
    reconstruct_workflow,
)


# ==================
# Test Fixtures
# ==================


@pytest_asyncio.fixture
async def async_engine():
    """Create an in-memory async SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(async_engine):
    """Create a test database session."""
    factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def populated_workflow(session):
    """Create a workflow with conversation history and metrics in the DB."""
    # Create workflow
    wf = await crud.create_workflow(
        session=session,
        instance_id="issue-42-20260214-120000",
        issue_number=42,
        issue_title="Add dark mode toggle",
        issue_body="We need dark mode support for the app",
        issue_url="https://github.com/test/repo/issues/42",
        repository="test/repo",
        issue_labels=["enhancement", "ui"],
        max_rounds=10,
    )

    # Update to running
    await crud.update_workflow_status(session, wf.id, "running")
    await crud.update_workflow_round(session, wf.id, current_round=2, total_comments=5)

    # Add agents
    await crud.add_agent_to_workflow(
        session, wf.id, "ui_architect", "claude_sdk_text", "UI design", "UI expertise"
    )
    await crud.add_agent_to_workflow(
        session, wf.id, "frontend_dev", "claude_sdk_text", "Frontend development", "React expert"
    )
    await crud.add_agent_to_workflow(
        session, wf.id, "ada_expert", "claude_sdk_text", "Accessibility", "WCAG compliance"
    )

    # Add conversation entries
    await crud.add_conversation_entry(
        session, wf.id, 1, "ui_architect",
        "I recommend using CSS custom properties for theming.",
        references=None,
    )
    await crud.add_conversation_entry(
        session, wf.id, 1, "frontend_dev",
        "I agree with ui_architect. CSS variables work well with React.",
        references=["ui_architect"],
    )
    await crud.add_conversation_entry(
        session, wf.id, 1, "ada_expert",
        "Ensure color contrast meets WCAG AA standards.",
        references=["ui_architect"],
    )
    await crud.add_conversation_entry(
        session, wf.id, 2, "ui_architect",
        "Good point about accessibility. We should also support prefers-color-scheme.",
        references=["ada_expert"],
    )
    await crud.add_conversation_entry(
        session, wf.id, 2, "frontend_dev",
        "I concur. Let me summarize our approach: CSS vars + React context.",
        references=["ui_architect", "ada_expert"],
    )

    # Add convergence metrics
    await crud.add_convergence_metric(
        session, wf.id, 1,
        convergence_score=0.3,
        should_continue=True,
        reason="Discussion still productive",
        agreement_level=0.5,
        value_added=0.8,
        comments_this_round=3,
        agents_participated=3,
    )
    await crud.add_convergence_metric(
        session, wf.id, 2,
        convergence_score=0.7,
        should_continue=True,
        reason="Agreement increasing but more discussion needed",
        agreement_level=0.75,
        value_added=0.5,
        comments_this_round=2,
        agents_participated=2,
    )

    await session.commit()

    # Re-fetch with relations
    result = await crud.get_workflow_by_instance_id(
        session, "issue-42-20260214-120000", load_relations=True
    )
    return result


@pytest.fixture
def mock_workflow_record():
    """Create a minimal mock WorkflowRecord."""
    wf = MagicMock(spec=WorkflowRecord)
    wf.id = 1
    wf.instance_id = "issue-99-20260214-150000"
    wf.workflow_id = "github-issue-deliberation"
    wf.status = "running"
    wf.issue_number = 99
    wf.issue_title = "Test Issue"
    wf.issue_body = "Test body"
    wf.issue_url = "https://github.com/test/repo/issues/99"
    wf.repository = "test/repo"
    wf.issue_labels = json.dumps(["bug"])
    wf.current_round = 1
    wf.max_rounds = 10
    wf.total_comments = 2
    wf.summary = None
    wf.final_convergence_score = None
    wf.created_at = datetime(2026, 2, 14, 12, 0, 0)
    wf.started_at = datetime(2026, 2, 14, 12, 0, 5)
    wf.completed_at = None
    wf.updated_at = datetime(2026, 2, 14, 12, 1, 0)

    # Conversation records
    conv1 = MagicMock(spec=ConversationRecord)
    conv1.round_number = 1
    conv1.agent_name = "agent_a"
    conv1.comment = "I recommend approach X."
    conv1.references = None
    conv1.created_at = datetime(2026, 2, 14, 12, 0, 10)
    conv1.github_comment_id = None

    conv2 = MagicMock(spec=ConversationRecord)
    conv2.round_number = 1
    conv2.agent_name = "agent_b"
    conv2.comment = "I agree with agent_a."
    conv2.references = json.dumps(["agent_a"])
    conv2.created_at = datetime(2026, 2, 14, 12, 0, 15)
    conv2.github_comment_id = 12345

    wf.conversations = [conv1, conv2]

    # Agent records
    ag1 = MagicMock(spec=AgentRecord)
    ag1.agent_name = "agent_a"
    ag2 = MagicMock(spec=AgentRecord)
    ag2.agent_name = "agent_b"
    wf.agents = [ag1, ag2]

    # Convergence metrics
    cm = MagicMock(spec=ConvergenceMetricRecord)
    cm.round_number = 1
    cm.convergence_score = 0.5
    cm.should_continue = True
    cm.reason = "Still productive"
    cm.rambling_detected = False
    cm.value_trend = "increasing"
    wf.convergence_metrics = [cm]

    return wf


# ==================
# Reconstruction Tests
# ==================


class TestReconstructWorkflow:
    """Tests for reconstructing WorkflowInstance from DB records."""

    def test_reconstruct_from_mock(self, mock_workflow_record):
        """Reconstruct a workflow from mock DB record."""
        workflow = reconstruct_workflow(mock_workflow_record)

        assert workflow.instance_id == "issue-99-20260214-150000"
        assert workflow.status == WorkflowStatus.RUNNING
        assert workflow.current_round == 1
        assert len(workflow.conversation_history) == 2
        assert len(workflow.selected_agents) == 2
        assert "agent_a" in workflow.selected_agents
        assert "agent_b" in workflow.selected_agents

    def test_reconstruct_conversation_history(self, mock_workflow_record):
        """Conversation history is correctly reconstructed."""
        workflow = reconstruct_workflow(mock_workflow_record)

        assert workflow.conversation_history[0].agent == "agent_a"
        assert workflow.conversation_history[0].round == 1
        assert "approach X" in workflow.conversation_history[0].comment

        assert workflow.conversation_history[1].agent == "agent_b"
        assert workflow.conversation_history[1].references == ["agent_a"]
        assert workflow.conversation_history[1].github_comment_id == 12345

    def test_reconstruct_github_issue(self, mock_workflow_record):
        """GitHub issue data is correctly reconstructed."""
        workflow = reconstruct_workflow(mock_workflow_record)

        assert workflow.github_issue is not None
        assert workflow.github_issue["number"] == 99
        assert workflow.github_issue["title"] == "Test Issue"
        assert workflow.github_issue["labels"] == ["bug"]

    def test_reconstruct_config(self, mock_workflow_record):
        """Workflow config is correctly reconstructed."""
        workflow = reconstruct_workflow(mock_workflow_record)

        assert workflow.config.max_rounds == 10

    def test_reconstruct_metrics(self, mock_workflow_record):
        """Workflow metrics are correctly reconstructed."""
        workflow = reconstruct_workflow(mock_workflow_record)

        assert workflow.metrics.current_round == 1
        assert workflow.metrics.total_comments == 2
        assert workflow.metrics.participating_agents == 2

    def test_reconstruct_none_raises(self):
        """Reconstructing from None raises an error."""
        with pytest.raises(WorkflowRecoveryError, match="Cannot reconstruct from None"):
            reconstruct_workflow(None)

    def test_reconstruct_unknown_status(self, mock_workflow_record):
        """Unknown status defaults to RUNNING."""
        mock_workflow_record.status = "bogus_status"
        workflow = reconstruct_workflow(mock_workflow_record)
        assert workflow.status == WorkflowStatus.RUNNING

    def test_reconstruct_completed_status(self, mock_workflow_record):
        """Completed status is preserved."""
        mock_workflow_record.status = "completed"
        workflow = reconstruct_workflow(mock_workflow_record)
        assert workflow.status == WorkflowStatus.COMPLETED

    def test_reconstruct_with_summary(self, mock_workflow_record):
        """Summary is preserved in reconstruction."""
        mock_workflow_record.summary = "Final recommendations here"
        workflow = reconstruct_workflow(mock_workflow_record)
        assert workflow.summary == "Final recommendations here"


# ==================
# Issue Reconstruction Tests
# ==================


class TestReconstructIssue:
    """Tests for reconstructing GitHub issue data."""

    def test_basic_issue_reconstruction(self, mock_workflow_record):
        """Issue data is reconstructed from DB fields."""
        issue = _reconstruct_issue(mock_workflow_record)

        assert issue["number"] == 99
        assert issue["title"] == "Test Issue"
        assert issue["body"] == "Test body"
        assert issue["labels"] == ["bug"]
        assert issue["repository"] == "test/repo"

    def test_issue_with_no_labels(self, mock_workflow_record):
        """Issue with None labels returns empty list."""
        mock_workflow_record.issue_labels = None
        issue = _reconstruct_issue(mock_workflow_record)
        assert issue["labels"] == []

    def test_issue_with_invalid_labels_json(self, mock_workflow_record):
        """Invalid JSON labels returns empty list."""
        mock_workflow_record.issue_labels = "not-valid-json"
        issue = _reconstruct_issue(mock_workflow_record)
        assert issue["labels"] == []

    def test_issue_with_no_body(self, mock_workflow_record):
        """Issue with None body returns empty string."""
        mock_workflow_record.issue_body = None
        issue = _reconstruct_issue(mock_workflow_record)
        assert issue["body"] == ""


# ==================
# Conversation Reconstruction Tests
# ==================


class TestReconstructConversation:
    """Tests for reconstructing conversation from DB records."""

    def test_empty_conversation(self):
        """Empty records returns empty conversation."""
        comments = _reconstruct_conversation([])
        assert comments == []

    def test_ordering_by_round_and_time(self):
        """Comments are sorted by round then creation time."""
        rec1 = MagicMock(spec=ConversationRecord)
        rec1.round_number = 2
        rec1.agent_name = "agent_b"
        rec1.comment = "Later comment"
        rec1.references = None
        rec1.created_at = datetime(2026, 2, 14, 12, 1, 0)
        rec1.github_comment_id = None

        rec2 = MagicMock(spec=ConversationRecord)
        rec2.round_number = 1
        rec2.agent_name = "agent_a"
        rec2.comment = "First comment"
        rec2.references = None
        rec2.created_at = datetime(2026, 2, 14, 12, 0, 0)
        rec2.github_comment_id = None

        comments = _reconstruct_conversation([rec1, rec2])
        assert comments[0].round == 1
        assert comments[0].agent == "agent_a"
        assert comments[1].round == 2
        assert comments[1].agent == "agent_b"

    def test_references_parsing(self):
        """JSON references are correctly parsed."""
        rec = MagicMock(spec=ConversationRecord)
        rec.round_number = 1
        rec.agent_name = "agent_a"
        rec.comment = "I agree with agent_b"
        rec.references = json.dumps(["agent_b", "agent_c"])
        rec.created_at = datetime(2026, 2, 14, 12, 0, 0)
        rec.github_comment_id = None

        comments = _reconstruct_conversation([rec])
        assert comments[0].references == ["agent_b", "agent_c"]

    def test_invalid_references_json(self):
        """Invalid JSON references defaults to empty list."""
        rec = MagicMock(spec=ConversationRecord)
        rec.round_number = 1
        rec.agent_name = "agent_a"
        rec.comment = "Comment"
        rec.references = "not-valid-json"
        rec.created_at = datetime(2026, 2, 14, 12, 0, 0)
        rec.github_comment_id = None

        comments = _reconstruct_conversation([rec])
        assert comments[0].references == []


# ==================
# Convergence State Tests
# ==================


class TestConvergenceState:
    """Tests for extracting convergence state."""

    def test_convergence_from_metrics(self, mock_workflow_record):
        """Extracts convergence state from the latest metric."""
        state = get_last_convergence_state(mock_workflow_record)

        assert state["last_round"] == 1
        assert state["convergence_score"] == 0.5
        assert state["should_continue"] is True
        assert state["reason"] == "Still productive"
        assert state["rambling_detected"] is False
        assert state["value_trend"] == "increasing"

    def test_convergence_no_metrics(self, mock_workflow_record):
        """Returns defaults when no metrics exist."""
        mock_workflow_record.convergence_metrics = []
        state = get_last_convergence_state(mock_workflow_record)

        assert state["last_round"] == 0
        assert state["convergence_score"] == 0.0
        assert state["should_continue"] is True

    def test_convergence_latest_round(self, mock_workflow_record):
        """Returns the highest round number metric."""
        cm2 = MagicMock(spec=ConvergenceMetricRecord)
        cm2.round_number = 3
        cm2.convergence_score = 0.9
        cm2.should_continue = False
        cm2.reason = "Converged"
        cm2.rambling_detected = False
        cm2.value_trend = "stable"

        mock_workflow_record.convergence_metrics.append(cm2)

        state = get_last_convergence_state(mock_workflow_record)
        assert state["last_round"] == 3
        assert state["convergence_score"] == 0.9
        assert state["should_continue"] is False


# ==================
# Database Round-Trip Tests
# ==================


class TestDatabaseRoundTrip:
    """Tests for save/recovery round-trip integrity."""

    @pytest.mark.asyncio
    async def test_full_round_trip(self, populated_workflow):
        """Full round-trip: save to DB, reconstruct, verify data."""
        workflow = reconstruct_workflow(populated_workflow)

        # Verify basic fields
        assert workflow.instance_id == "issue-42-20260214-120000"
        assert workflow.status == WorkflowStatus.RUNNING
        assert workflow.current_round == 2

        # Verify issue data
        assert workflow.github_issue["number"] == 42
        assert workflow.github_issue["title"] == "Add dark mode toggle"
        assert workflow.github_issue["labels"] == ["enhancement", "ui"]

        # Verify conversation history
        assert len(workflow.conversation_history) == 5
        assert workflow.conversation_history[0].agent == "ui_architect"
        assert workflow.conversation_history[0].round == 1
        assert workflow.conversation_history[2].agent == "ada_expert"

        # Verify agent list
        assert len(workflow.selected_agents) == 3
        assert "ada_expert" in workflow.selected_agents
        assert "frontend_dev" in workflow.selected_agents
        assert "ui_architect" in workflow.selected_agents

        # Verify config
        assert workflow.config.max_rounds == 10

    @pytest.mark.asyncio
    async def test_conversation_references_preserved(self, populated_workflow):
        """Agent references are preserved in round-trip."""
        workflow = reconstruct_workflow(populated_workflow)

        # Comment at index 1 (frontend_dev round 1) referenced ui_architect
        frontend_round1 = [
            c for c in workflow.conversation_history
            if c.agent == "frontend_dev" and c.round == 1
        ]
        assert len(frontend_round1) == 1
        assert "ui_architect" in frontend_round1[0].references

    @pytest.mark.asyncio
    async def test_convergence_metrics_recovery(self, populated_workflow):
        """Convergence metrics are available after recovery."""
        state = get_last_convergence_state(populated_workflow)

        assert state["last_round"] == 2
        assert state["convergence_score"] == 0.7
        assert state["should_continue"] is True

    @pytest.mark.asyncio
    async def test_round_ordering(self, populated_workflow):
        """Comments are in correct round and temporal order."""
        workflow = reconstruct_workflow(populated_workflow)

        rounds = [c.round for c in workflow.conversation_history]
        # First 3 comments should be round 1, last 2 should be round 2
        assert rounds == [1, 1, 1, 2, 2]


# ==================
# CRUD Recovery Tests
# ==================


class TestCrudRecovery:
    """Tests for CRUD recovery functions."""

    @pytest.mark.asyncio
    async def test_get_interrupted_workflows(self, session, populated_workflow):
        """Finds running workflows as interrupted."""
        workflows = await crud.get_interrupted_workflows(session)
        assert len(workflows) >= 1
        assert any(w.instance_id == "issue-42-20260214-120000" for w in workflows)

    @pytest.mark.asyncio
    async def test_get_interrupted_excludes_completed(self, session):
        """Completed workflows are not returned as interrupted."""
        wf = await crud.create_workflow(
            session=session,
            instance_id="completed-wf",
            issue_number=1,
            issue_title="Completed issue",
        )
        await crud.update_workflow_status(session, wf.id, "completed")
        await session.commit()

        workflows = await crud.get_interrupted_workflows(session)
        assert not any(w.instance_id == "completed-wf" for w in workflows)

    @pytest.mark.asyncio
    async def test_get_workflow_for_recovery(self, session, populated_workflow):
        """Loads workflow with all relations for recovery."""
        wf = await crud.get_workflow_for_recovery(
            session, "issue-42-20260214-120000"
        )
        assert wf is not None
        assert len(wf.conversations) == 5
        assert len(wf.agents) == 3
        assert len(wf.convergence_metrics) == 2

    @pytest.mark.asyncio
    async def test_get_workflow_for_recovery_not_found(self, session):
        """Returns None for non-existent workflow."""
        wf = await crud.get_workflow_for_recovery(session, "nonexistent")
        assert wf is None

    @pytest.mark.asyncio
    async def test_update_workflow_summary(self, session, populated_workflow):
        """Updates workflow summary and convergence score."""
        await crud.update_workflow_summary(
            session, populated_workflow.id, "Final summary", 0.95
        )
        await session.commit()

        wf = await crud.get_workflow_by_instance_id(
            session, "issue-42-20260214-120000"
        )
        assert wf.summary == "Final summary"
        assert wf.final_convergence_score == 0.95


# ==================
# Edge Cases
# ==================


class TestEdgeCases:
    """Tests for edge cases in recovery."""

    def test_workflow_with_no_conversations(self, mock_workflow_record):
        """Handles workflow with empty conversation history."""
        mock_workflow_record.conversations = []
        workflow = reconstruct_workflow(mock_workflow_record)
        assert workflow.conversation_history == []

    def test_workflow_with_no_agents(self, mock_workflow_record):
        """Handles workflow with no agents."""
        mock_workflow_record.agents = []
        workflow = reconstruct_workflow(mock_workflow_record)
        assert workflow.selected_agents == []

    def test_workflow_with_null_convergence_score(self, mock_workflow_record):
        """Handles null convergence score."""
        mock_workflow_record.final_convergence_score = None
        workflow = reconstruct_workflow(mock_workflow_record)
        assert workflow.metrics.convergence_score == 0.0

    def test_workflow_with_no_url(self, mock_workflow_record):
        """Handles workflow with no issue URL."""
        mock_workflow_record.issue_url = None
        workflow = reconstruct_workflow(mock_workflow_record)
        assert workflow.github_issue["url"] == ""

    def test_conversation_with_none_references(self):
        """Handles conversation record with None references."""
        rec = MagicMock(spec=ConversationRecord)
        rec.round_number = 1
        rec.agent_name = "agent_a"
        rec.comment = "Comment"
        rec.references = None
        rec.created_at = datetime(2026, 2, 14, 12, 0, 0)
        rec.github_comment_id = None

        comments = _reconstruct_conversation([rec])
        assert comments[0].references == []

    @pytest.mark.asyncio
    async def test_round_trip_with_special_chars(self, session):
        """Round-trip preserves special characters in issue data."""
        wf = await crud.create_workflow(
            session=session,
            instance_id="special-chars-test",
            issue_number=100,
            issue_title='Issue with "quotes" & <tags>',
            issue_body="Body with\nnewlines\tand\ttabs",
            issue_labels=["label:colon", "label/slash"],
        )
        await crud.update_workflow_status(session, wf.id, "running")
        await session.commit()

        recovered = await crud.get_workflow_by_instance_id(
            session, "special-chars-test", load_relations=True
        )
        workflow = reconstruct_workflow(recovered)

        assert workflow.github_issue["title"] == 'Issue with "quotes" & <tags>'
        assert "\n" in workflow.github_issue["body"]
        assert "label:colon" in workflow.github_issue["labels"]
