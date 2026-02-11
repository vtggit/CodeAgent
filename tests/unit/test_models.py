"""
Unit tests for Pydantic models.

Tests cover model creation, validation, serialization, and schema generation.
"""

import json

import pytest
from pydantic import ValidationError

from src.models.workflow import (
    WorkflowInstance,
    WorkflowConfig,
    WorkflowMetrics,
    WorkflowStatus,
    Comment,
    ContinueDecision,
)
from src.models.agent import (
    AgentConfig,
    AgentType,
    AgentParticipation,
    AgentDecision,
    LLMProviderConfig,
)
from src.models.github import (
    GitHubIssue,
    GitHubWebhookPayload,
    GitHubLabel,
    GitHubUser,
    GitHubRepository,
)


# ==========================
# Comment Model Tests
# ==========================


class TestComment:
    """Tests for the Comment model."""

    def test_create_valid_comment(self):
        """Test creating a valid comment."""
        comment = Comment(
            round=1,
            agent="ui_architect",
            comment="I recommend using a toggle component in the header.",
        )
        assert comment.round == 1
        assert comment.agent == "ui_architect"
        assert comment.references == []
        assert comment.timestamp is not None

    def test_comment_with_references(self):
        """Test comment with references to other agents."""
        comment = Comment(
            round=2,
            agent="ada_expert",
            comment="Regarding ui_architect's point about the header...",
            references=["ui_architect", "frontend_dev"],
        )
        assert len(comment.references) == 2
        assert "ui_architect" in comment.references

    def test_comment_round_validation(self):
        """Test that round must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            Comment(round=0, agent="test", comment="Test")
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_comment_empty_text_rejected(self):
        """Test that empty comment text is rejected."""
        with pytest.raises(ValidationError):
            Comment(round=1, agent="test", comment="")

    def test_comment_json_serialization(self):
        """Test JSON round-trip serialization."""
        comment = Comment(
            round=1,
            agent="security_expert",
            comment="We should consider XSS prevention.",
            references=["frontend_dev"],
        )
        json_str = comment.model_dump_json()
        restored = Comment.model_validate_json(json_str)
        assert restored.agent == comment.agent
        assert restored.comment == comment.comment
        assert restored.references == comment.references


# ==========================
# ContinueDecision Tests
# ==========================


class TestContinueDecision:
    """Tests for the ContinueDecision model."""

    def test_create_continue_decision(self):
        """Test creating a valid continue decision."""
        decision = ContinueDecision(
            should_continue=True,
            reason="Discussion still productive",
            convergence_score=0.45,
        )
        assert decision.should_continue is True
        assert decision.rambling_detected is False
        assert decision.value_trend == "stable"

    def test_stop_decision(self):
        """Test creating a stop decision."""
        decision = ContinueDecision(
            should_continue=False,
            reason="High consensus reached",
            convergence_score=0.92,
            rambling_detected=False,
            value_trend="decreasing",
        )
        assert decision.should_continue is False
        assert decision.convergence_score == 0.92

    def test_convergence_score_bounds(self):
        """Test convergence score validation bounds."""
        with pytest.raises(ValidationError):
            ContinueDecision(
                should_continue=True,
                reason="test",
                convergence_score=1.5,
            )
        with pytest.raises(ValidationError):
            ContinueDecision(
                should_continue=True,
                reason="test",
                convergence_score=-0.1,
            )

    def test_invalid_value_trend(self):
        """Test that invalid value_trend is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ContinueDecision(
                should_continue=True,
                reason="test",
                convergence_score=0.5,
                value_trend="invalid_trend",
            )
        assert "value_trend" in str(exc_info.value)

    def test_valid_value_trends(self):
        """Test all valid value trend values."""
        for trend in ["increasing", "decreasing", "stable", "unknown"]:
            decision = ContinueDecision(
                should_continue=True,
                reason="test",
                convergence_score=0.5,
                value_trend=trend,
            )
            assert decision.value_trend == trend


# ==========================
# WorkflowConfig Tests
# ==========================


class TestWorkflowConfig:
    """Tests for the WorkflowConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkflowConfig()
        assert config.max_rounds == 10
        assert config.convergence_threshold == 0.8
        assert config.min_value_threshold == 0.2
        assert config.timeout_minutes == 60

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkflowConfig(
            max_rounds=20,
            convergence_threshold=0.9,
            timeout_minutes=120,
        )
        assert config.max_rounds == 20
        assert config.convergence_threshold == 0.9

    def test_config_bounds(self):
        """Test configuration value bounds."""
        with pytest.raises(ValidationError):
            WorkflowConfig(max_rounds=0)
        with pytest.raises(ValidationError):
            WorkflowConfig(convergence_threshold=1.5)


# ==========================
# WorkflowInstance Tests
# ==========================


class TestWorkflowInstance:
    """Tests for the WorkflowInstance model."""

    def test_create_workflow_instance(self):
        """Test creating a valid workflow instance."""
        instance = WorkflowInstance(
            instance_id="issue-42-20260208-143022",
            github_issue={
                "number": 42,
                "title": "Add dark mode toggle",
                "body": "Need dark mode",
                "labels": ["enhancement"],
            },
        )
        assert instance.instance_id == "issue-42-20260208-143022"
        assert instance.status == WorkflowStatus.PENDING
        assert instance.current_round == 0
        assert instance.conversation_history == []
        assert instance.selected_agents == []

    def test_workflow_with_conversation(self):
        """Test workflow with conversation history."""
        comments = [
            Comment(round=1, agent="agent_a", comment="First comment"),
            Comment(round=1, agent="agent_b", comment="Second comment"),
            Comment(
                round=2,
                agent="agent_a",
                comment="Follow-up",
                references=["agent_b"],
            ),
        ]
        instance = WorkflowInstance(
            instance_id="test-workflow",
            conversation_history=comments,
            current_round=2,
            status=WorkflowStatus.RUNNING,
        )
        assert len(instance.conversation_history) == 3
        assert instance.status == WorkflowStatus.RUNNING

    def test_workflow_json_roundtrip(self):
        """Test JSON serialization/deserialization of full workflow."""
        instance = WorkflowInstance(
            instance_id="json-test",
            workflow_id="github-issue-deliberation",
            status=WorkflowStatus.RUNNING,
            selected_agents=["ui_architect", "security_expert"],
            conversation_history=[
                Comment(round=1, agent="ui_architect", comment="My recommendation..."),
            ],
            current_round=1,
        )
        json_str = instance.model_dump_json()
        restored = WorkflowInstance.model_validate_json(json_str)
        assert restored.instance_id == "json-test"
        assert len(restored.selected_agents) == 2
        assert len(restored.conversation_history) == 1

    def test_workflow_schema_generation(self):
        """Test that JSON Schema can be generated."""
        schema = WorkflowInstance.model_json_schema()
        assert "properties" in schema
        assert "instance_id" in schema["properties"]
        assert "status" in schema["properties"]


# ==========================
# AgentConfig Tests
# ==========================


class TestAgentConfig:
    """Tests for the AgentConfig model."""

    def test_create_text_agent(self):
        """Test creating a text-based agent config."""
        config = AgentConfig(
            name="ui_architect",
            type=AgentType.CLAUDE_SDK_TEXT,
            expertise="User interface architecture and component design",
            domain_knowledge="React, Vue, design systems",
            category="architecture",
        )
        assert config.name == "ui_architect"
        assert config.type == AgentType.CLAUDE_SDK_TEXT
        assert config.can_edit_files is False
        assert config.tools == []

    def test_create_code_agent(self):
        """Test creating a code agent with tools."""
        config = AgentConfig(
            name="frontend_dev",
            type=AgentType.CLAUDE_SDK_CODE,
            expertise="Frontend implementation",
            domain_knowledge="JavaScript/TypeScript, React",
            tools=["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
            can_edit_files=True,
            category="development",
        )
        assert config.type == AgentType.CLAUDE_SDK_CODE
        assert config.can_edit_files is True
        assert len(config.tools) == 6

    def test_agent_with_custom_llm(self):
        """Test agent with custom LLM provider."""
        config = AgentConfig(
            name="cost_optimizer",
            expertise="Cost optimization",
            llm=LLMProviderConfig(
                provider="lm_studio",
                model="mistral-7b-instruct",
                base_url="http://10.1.1.58:1234/v1",
                temperature=0.3,
            ),
        )
        assert config.llm.provider == "lm_studio"
        assert config.llm.base_url == "http://10.1.1.58:1234/v1"

    def test_agent_default_llm(self):
        """Test agent uses default Anthropic LLM config."""
        config = AgentConfig(
            name="test_agent",
            expertise="Testing",
        )
        assert config.llm.provider == "anthropic"
        assert "claude" in config.llm.model

    def test_agent_json_serialization(self):
        """Test agent config JSON round-trip."""
        config = AgentConfig(
            name="security_expert",
            type=AgentType.CLAUDE_SDK_TEXT,
            expertise="Application security",
            domain_knowledge="OWASP, XSS prevention, CSRF",
            priority=8,
        )
        json_str = config.model_dump_json()
        restored = AgentConfig.model_validate_json(json_str)
        assert restored.name == "security_expert"
        assert restored.priority == 8

    def test_agent_schema_generation(self):
        """Test that JSON Schema can be generated."""
        schema = AgentConfig.model_json_schema()
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "type" in schema["properties"]


# ==========================
# AgentDecision Tests
# ==========================


class TestAgentDecision:
    """Tests for the AgentDecision model."""

    def test_comment_decision(self):
        """Test a decision to comment."""
        decision = AgentDecision(
            should_comment=True,
            reason="I have relevant expertise on this topic",
            responding_to=["ui_architect"],
            confidence=0.85,
        )
        assert decision.should_comment is True
        assert len(decision.responding_to) == 1

    def test_skip_decision(self):
        """Test a decision to skip commenting."""
        decision = AgentDecision(
            should_comment=False,
            reason="This topic is outside my expertise",
            confidence=0.2,
        )
        assert decision.should_comment is False
        assert decision.responding_to == []


# ==========================
# GitHubIssue Tests
# ==========================


class TestGitHubIssue:
    """Tests for the GitHubIssue model."""

    def test_create_issue(self):
        """Test creating a valid GitHub issue."""
        issue = GitHubIssue(
            number=42,
            title="Add dark mode toggle to settings",
            body="Users want ability to switch between light and dark themes.",
            state="open",
            labels=["enhancement", "ui"],
            url="https://github.com/myorg/myrepo/issues/42",
        )
        assert issue.number == 42
        assert len(issue.labels) == 2

    def test_issue_minimal(self):
        """Test creating an issue with minimal required fields."""
        issue = GitHubIssue(number=1, title="Bug report")
        assert issue.number == 1
        assert issue.body is None
        assert issue.labels == []

    def test_issue_json_roundtrip(self):
        """Test JSON serialization of issues."""
        issue = GitHubIssue(
            number=100,
            title="Test Issue",
            labels=["bug", "high-priority"],
        )
        json_str = issue.model_dump_json()
        restored = GitHubIssue.model_validate_json(json_str)
        assert restored.number == 100
        assert restored.labels == ["bug", "high-priority"]


# ==========================
# GitHubWebhookPayload Tests
# ==========================


class TestGitHubWebhookPayload:
    """Tests for the GitHubWebhookPayload model."""

    def test_create_payload(self):
        """Test creating a webhook payload."""
        payload = GitHubWebhookPayload(
            action="opened",
            issue=GitHubIssue(
                number=42,
                title="Test Issue",
                labels=["bug"],
            ),
            repository="myorg/myrepo",
            sender="octocat",
        )
        assert payload.action == "opened"
        assert payload.issue.number == 42

    def test_from_raw_webhook(self):
        """Test parsing a raw GitHub webhook payload."""
        raw = {
            "action": "opened",
            "issue": {
                "number": 42,
                "title": "Add dark mode toggle",
                "body": "Need dark mode support",
                "state": "open",
                "labels": [
                    {"id": 1, "name": "enhancement", "color": "a2eeef"},
                    {"id": 2, "name": "ui", "color": "d4c5f9"},
                ],
                "html_url": "https://github.com/test/repo/issues/42",
            },
            "repository": {
                "full_name": "test/repo",
            },
            "sender": {
                "login": "octocat",
            },
        }
        payload = GitHubWebhookPayload.from_raw_webhook(raw)
        assert payload.action == "opened"
        assert payload.issue.number == 42
        assert payload.issue.labels == ["enhancement", "ui"]
        assert payload.repository == "test/repo"
        assert payload.sender == "octocat"

    def test_from_raw_webhook_missing_issue(self):
        """Test that missing issue raises ValueError."""
        with pytest.raises(ValueError, match="missing 'issue' field"):
            GitHubWebhookPayload.from_raw_webhook({"action": "opened"})

    def test_from_raw_webhook_minimal(self):
        """Test parsing a minimal webhook payload."""
        raw = {
            "action": "opened",
            "issue": {
                "number": 1,
                "title": "Minimal Issue",
            },
        }
        payload = GitHubWebhookPayload.from_raw_webhook(raw)
        assert payload.issue.number == 1
        assert payload.issue.labels == []
        assert payload.repository is None

    def test_payload_schema_generation(self):
        """Test that JSON Schema can be generated."""
        schema = GitHubWebhookPayload.model_json_schema()
        assert "properties" in schema


# ==========================
# LLMProviderConfig Tests
# ==========================


class TestLLMProviderConfig:
    """Tests for the LLMProviderConfig model."""

    def test_default_config(self):
        """Test default LLM config."""
        config = LLMProviderConfig()
        assert config.provider == "anthropic"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_custom_config(self):
        """Test custom LLM config."""
        config = LLMProviderConfig(
            provider="lm_studio",
            model="qwen/qwen3-coder-next",
            base_url="http://localhost:1234/v1",
            temperature=0.3,
            max_tokens=8192,
        )
        assert config.provider == "lm_studio"
        assert config.base_url == "http://localhost:1234/v1"

    def test_temperature_bounds(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            LLMProviderConfig(temperature=3.0)
        with pytest.raises(ValidationError):
            LLMProviderConfig(temperature=-0.1)


# ==========================
# Model Import Tests
# ==========================


class TestModelImports:
    """Test that all models can be imported from the package."""

    def test_import_all_models(self):
        """Test importing all models from src.models."""
        from src.models import (
            WorkflowInstance,
            WorkflowConfig,
            WorkflowMetrics,
            WorkflowStatus,
            Comment,
            ContinueDecision,
            AgentConfig,
            AgentType,
            AgentParticipation,
            AgentDecision,
            LLMProviderConfig,
            GitHubIssue,
            GitHubWebhookPayload,
            GitHubLabel,
            GitHubUser,
            GitHubRepository,
        )
        # All imports succeed
        assert WorkflowInstance is not None
        assert AgentConfig is not None
        assert GitHubIssue is not None

    def test_nested_model_validation(self):
        """Test deeply nested model validation works."""
        from src.models import WorkflowInstance, Comment

        instance = WorkflowInstance(
            instance_id="nested-test",
            conversation_history=[
                Comment(
                    round=1,
                    agent="test_agent",
                    comment="Nested validation test",
                    references=["other_agent"],
                ),
            ],
            config={
                "max_rounds": 15,
                "convergence_threshold": 0.85,
            },
        )
        assert instance.config.max_rounds == 15
        assert len(instance.conversation_history) == 1


# ==========================
# Schema Export Tests
# ==========================


class TestSchemaExport:
    """Test JSON Schema generation for API documentation."""

    def test_all_models_generate_schema(self):
        """Test that all models can generate valid JSON schemas."""
        models = [
            WorkflowInstance,
            WorkflowConfig,
            Comment,
            ContinueDecision,
            AgentConfig,
            GitHubIssue,
            GitHubWebhookPayload,
        ]
        for model in models:
            schema = model.model_json_schema()
            assert isinstance(schema, dict), f"{model.__name__} schema is not a dict"
            assert "properties" in schema, f"{model.__name__} schema missing properties"

    def test_schema_to_json(self):
        """Test that schemas can be serialized to JSON strings."""
        schema = WorkflowInstance.model_json_schema()
        json_str = json.dumps(schema, indent=2)
        parsed = json.loads(json_str)
        assert parsed["properties"]["instance_id"] is not None
