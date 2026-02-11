"""
Unit tests for database models and CRUD operations.

Tests use an in-memory SQLite database for isolation and speed.
"""

import asyncio
import json
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.database.models import (
    Base,
    WorkflowRecord,
    ConversationRecord,
    AgentRecord,
    ConvergenceMetricRecord,
)
from src.database import crud


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


# ==========================
# Model Tests
# ==========================


class TestWorkflowModel:
    """Tests for WorkflowRecord model."""

    @pytest.mark.asyncio
    async def test_create_workflow(self, session):
        """Test creating a basic workflow record."""
        workflow = WorkflowRecord(
            instance_id="test-issue-1-20260210",
            workflow_id="github-issue-deliberation",
            status="pending",
            issue_number=1,
            issue_title="Test Issue",
            issue_body="Test body content",
            repository="test-org/test-repo",
            max_rounds=10,
        )
        session.add(workflow)
        await session.flush()

        assert workflow.id is not None
        assert workflow.instance_id == "test-issue-1-20260210"
        assert workflow.status == "pending"
        assert workflow.current_round == 0
        assert workflow.total_comments == 0

    @pytest.mark.asyncio
    async def test_workflow_repr(self, session):
        """Test workflow string representation."""
        workflow = WorkflowRecord(
            instance_id="repr-test",
            status="running",
            issue_number=42,
            issue_title="Repr Test",
            current_round=3,
        )
        session.add(workflow)
        await session.flush()

        repr_str = repr(workflow)
        assert "repr-test" in repr_str
        assert "running" in repr_str

    @pytest.mark.asyncio
    async def test_workflow_defaults(self, session):
        """Test that default values are set correctly."""
        workflow = WorkflowRecord(
            instance_id="defaults-test",
            issue_number=1,
            issue_title="Defaults",
        )
        session.add(workflow)
        await session.flush()

        assert workflow.status == "pending"
        assert workflow.current_round == 0
        assert workflow.max_rounds == 10
        assert workflow.total_comments == 0
        assert workflow.created_at is not None

    @pytest.mark.asyncio
    async def test_workflow_labels_json(self, session):
        """Test storing labels as JSON."""
        labels = ["bug", "enhancement", "ui"]
        workflow = WorkflowRecord(
            instance_id="labels-test",
            issue_number=1,
            issue_title="Labels",
            issue_labels=json.dumps(labels),
        )
        session.add(workflow)
        await session.flush()

        loaded_labels = json.loads(workflow.issue_labels)
        assert loaded_labels == labels


class TestConversationModel:
    """Tests for ConversationRecord model."""

    @pytest.mark.asyncio
    async def test_create_conversation_entry(self, session):
        """Test creating a conversation record."""
        workflow = WorkflowRecord(
            instance_id="conv-test",
            issue_number=1,
            issue_title="Conv Test",
        )
        session.add(workflow)
        await session.flush()

        conv = ConversationRecord(
            workflow_id=workflow.id,
            round_number=1,
            agent_name="ui_architect",
            comment="I recommend using a toggle component...",
            references=json.dumps(["frontend_dev"]),
        )
        session.add(conv)
        await session.flush()

        assert conv.id is not None
        assert conv.agent_name == "ui_architect"
        assert conv.round_number == 1
        refs = json.loads(conv.references)
        assert "frontend_dev" in refs

    @pytest.mark.asyncio
    async def test_conversation_relationship(self, session):
        """Test workflow-conversation relationship."""
        workflow = WorkflowRecord(
            instance_id="rel-test",
            issue_number=2,
            issue_title="Rel Test",
        )
        session.add(workflow)
        await session.flush()

        conv1 = ConversationRecord(
            workflow_id=workflow.id,
            round_number=1,
            agent_name="agent_a",
            comment="Comment A",
        )
        conv2 = ConversationRecord(
            workflow_id=workflow.id,
            round_number=1,
            agent_name="agent_b",
            comment="Comment B",
        )
        session.add_all([conv1, conv2])
        await session.flush()

        # Refresh to load relationships
        await session.refresh(workflow, ["conversations"])
        assert len(workflow.conversations) == 2


class TestAgentModel:
    """Tests for AgentRecord model."""

    @pytest.mark.asyncio
    async def test_create_agent_record(self, session):
        """Test creating an agent record."""
        workflow = WorkflowRecord(
            instance_id="agent-test",
            issue_number=3,
            issue_title="Agent Test",
        )
        session.add(workflow)
        await session.flush()

        agent = AgentRecord(
            workflow_id=workflow.id,
            agent_name="security_expert",
            agent_type="claude_sdk_text",
            expertise="Application security",
            selection_reason="Issue involves authentication",
        )
        session.add(agent)
        await session.flush()

        assert agent.id is not None
        assert agent.total_comments == 0
        assert agent.rounds_participated == 0


class TestConvergenceMetricModel:
    """Tests for ConvergenceMetricRecord model."""

    @pytest.mark.asyncio
    async def test_create_convergence_metric(self, session):
        """Test creating a convergence metric record."""
        workflow = WorkflowRecord(
            instance_id="conv-metric-test",
            issue_number=4,
            issue_title="Convergence Test",
        )
        session.add(workflow)
        await session.flush()

        metric = ConvergenceMetricRecord(
            workflow_id=workflow.id,
            round_number=1,
            convergence_score=0.45,
            agreement_level=0.6,
            value_added=0.8,
            rambling_detected=False,
            value_trend="increasing",
            comments_this_round=5,
            agents_participated=3,
            should_continue=True,
            reason="Discussion still productive",
        )
        session.add(metric)
        await session.flush()

        assert metric.id is not None
        assert metric.convergence_score == 0.45
        assert metric.should_continue is True


# ==========================
# CRUD Tests
# ==========================


class TestWorkflowCRUD:
    """Tests for workflow CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_workflow(self, session):
        """Test creating a workflow via CRUD."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="crud-create-test",
            issue_number=100,
            issue_title="CRUD Create Test",
            issue_body="Testing CRUD creation",
            repository="test/repo",
            issue_labels=["bug", "ui"],
        )

        assert workflow.id is not None
        assert workflow.instance_id == "crud-create-test"
        assert workflow.issue_number == 100
        assert json.loads(workflow.issue_labels) == ["bug", "ui"]

    @pytest.mark.asyncio
    async def test_get_workflow_by_instance_id(self, session):
        """Test retrieving a workflow by instance ID."""
        await crud.create_workflow(
            session=session,
            instance_id="get-by-id-test",
            issue_number=101,
            issue_title="Get By ID Test",
        )
        await session.flush()

        found = await crud.get_workflow_by_instance_id(session, "get-by-id-test")
        assert found is not None
        assert found.issue_number == 101

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self, session):
        """Test retrieving a non-existent workflow."""
        found = await crud.get_workflow_by_instance_id(session, "nonexistent")
        assert found is None

    @pytest.mark.asyncio
    async def test_list_workflows(self, session):
        """Test listing workflows with filtering."""
        await crud.create_workflow(
            session=session,
            instance_id="list-test-1",
            issue_number=1,
            issue_title="List Test 1",
        )
        w2 = await crud.create_workflow(
            session=session,
            instance_id="list-test-2",
            issue_number=2,
            issue_title="List Test 2",
        )
        await crud.update_workflow_status(session, w2.id, "running")
        await session.flush()

        all_workflows = await crud.list_workflows(session)
        assert len(all_workflows) == 2

        running = await crud.list_workflows(session, status="running")
        assert len(running) == 1
        assert running[0].instance_id == "list-test-2"

    @pytest.mark.asyncio
    async def test_update_workflow_status(self, session):
        """Test updating workflow status."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="status-update-test",
            issue_number=102,
            issue_title="Status Update",
        )
        await session.flush()

        await crud.update_workflow_status(session, workflow.id, "running")
        await session.flush()

        updated = await crud.get_workflow_by_id(session, workflow.id)
        assert updated.status == "running"
        assert updated.started_at is not None

    @pytest.mark.asyncio
    async def test_update_workflow_completed(self, session):
        """Test marking workflow as completed."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="completed-test",
            issue_number=103,
            issue_title="Completed Test",
        )
        await session.flush()

        await crud.update_workflow_status(
            session, workflow.id, "completed",
            summary="All agents agree on approach.",
            final_convergence_score=0.92,
        )
        await session.flush()

        updated = await crud.get_workflow_by_id(session, workflow.id)
        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.final_convergence_score == 0.92

    @pytest.mark.asyncio
    async def test_update_workflow_round(self, session):
        """Test updating workflow round progress."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="round-update-test",
            issue_number=104,
            issue_title="Round Update",
        )
        await session.flush()

        await crud.update_workflow_round(session, workflow.id, 3, 15)
        await session.flush()

        updated = await crud.get_workflow_by_id(session, workflow.id)
        assert updated.current_round == 3
        assert updated.total_comments == 15


class TestConversationCRUD:
    """Tests for conversation CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_conversation_entry(self, session):
        """Test adding a conversation entry."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="conv-crud-test",
            issue_number=200,
            issue_title="Conv CRUD",
        )
        await session.flush()

        entry = await crud.add_conversation_entry(
            session=session,
            workflow_pk=workflow.id,
            round_number=1,
            agent_name="qa_engineer",
            comment="We should add integration tests for this.",
            references=["frontend_dev", "backend_dev"],
        )
        await session.flush()

        assert entry.id is not None
        assert entry.agent_name == "qa_engineer"
        refs = json.loads(entry.references)
        assert len(refs) == 2

    @pytest.mark.asyncio
    async def test_get_conversation_history(self, session):
        """Test retrieving conversation history."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="history-test",
            issue_number=201,
            issue_title="History",
        )
        await session.flush()

        # Add entries across rounds
        for round_num in range(1, 4):
            await crud.add_conversation_entry(
                session=session,
                workflow_pk=workflow.id,
                round_number=round_num,
                agent_name=f"agent_{round_num}",
                comment=f"Comment in round {round_num}",
            )
        await session.flush()

        # Get all history
        history = await crud.get_conversation_history(session, workflow.id)
        assert len(history) == 3

        # Get specific round
        round2 = await crud.get_conversation_history(
            session, workflow.id, round_number=2
        )
        assert len(round2) == 1
        assert round2[0].agent_name == "agent_2"


class TestAgentCRUD:
    """Tests for agent CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_agent_to_workflow(self, session):
        """Test adding an agent to a workflow."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="agent-crud-test",
            issue_number=300,
            issue_title="Agent CRUD",
        )
        await session.flush()

        agent = await crud.add_agent_to_workflow(
            session=session,
            workflow_pk=workflow.id,
            agent_name="ui_architect",
            agent_type="claude_sdk_text",
            expertise="UI architecture",
            selection_reason="Issue is UI-related",
        )
        await session.flush()

        assert agent.id is not None
        assert agent.total_comments == 0

    @pytest.mark.asyncio
    async def test_update_agent_participation(self, session):
        """Test updating agent participation metrics."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="participation-test",
            issue_number=301,
            issue_title="Participation",
        )
        await session.flush()

        await crud.add_agent_to_workflow(
            session=session,
            workflow_pk=workflow.id,
            agent_name="test_agent",
        )
        await session.flush()

        await crud.update_agent_participation(
            session, workflow.id, "test_agent", round_number=1
        )
        await crud.update_agent_participation(
            session, workflow.id, "test_agent", round_number=3
        )
        await session.flush()

        agents = await crud.get_workflow_agents(session, workflow.id)
        assert len(agents) == 1
        assert agents[0].total_comments == 2
        assert agents[0].first_comment_round == 1
        assert agents[0].last_comment_round == 3


class TestConvergenceCRUD:
    """Tests for convergence metrics CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_convergence_metric(self, session):
        """Test adding a convergence metric."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="convergence-crud-test",
            issue_number=400,
            issue_title="Convergence CRUD",
        )
        await session.flush()

        metric = await crud.add_convergence_metric(
            session=session,
            workflow_pk=workflow.id,
            round_number=1,
            convergence_score=0.3,
            should_continue=True,
            reason="Discussion just starting",
            comments_this_round=5,
            agents_participated=4,
        )
        await session.flush()

        assert metric.id is not None
        assert metric.convergence_score == 0.3

    @pytest.mark.asyncio
    async def test_get_convergence_metrics(self, session):
        """Test retrieving convergence metrics."""
        workflow = await crud.create_workflow(
            session=session,
            instance_id="metrics-list-test",
            issue_number=401,
            issue_title="Metrics List",
        )
        await session.flush()

        for i in range(1, 4):
            await crud.add_convergence_metric(
                session=session,
                workflow_pk=workflow.id,
                round_number=i,
                convergence_score=0.2 * i,
                should_continue=i < 3,
            )
        await session.flush()

        metrics = await crud.get_convergence_metrics(session, workflow.id)
        assert len(metrics) == 3
        assert metrics[0].round_number == 1
        assert metrics[2].convergence_score == pytest.approx(0.6)


class TestUtilityQueries:
    """Tests for utility query functions."""

    @pytest.mark.asyncio
    async def test_count_workflows_by_status(self, session):
        """Test counting workflows by status."""
        await crud.create_workflow(
            session=session,
            instance_id="count-1",
            issue_number=500,
            issue_title="Count 1",
        )
        w2 = await crud.create_workflow(
            session=session,
            instance_id="count-2",
            issue_number=501,
            issue_title="Count 2",
        )
        w3 = await crud.create_workflow(
            session=session,
            instance_id="count-3",
            issue_number=502,
            issue_title="Count 3",
        )
        await session.flush()

        await crud.update_workflow_status(session, w2.id, "running")
        await crud.update_workflow_status(session, w3.id, "running")
        await session.flush()

        counts = await crud.count_workflows_by_status(session)
        assert counts.get("pending", 0) == 1
        assert counts.get("running", 0) == 2
