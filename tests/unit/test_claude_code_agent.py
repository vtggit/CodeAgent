"""
Tests for the ClaudeCodeAgent class.

Tests cover:
- Agent initialization and configuration
- Tool definitions and filtering
- Tool executor (sandbox mode, path validation, execution)
- Tool audit trail tracking
- should_comment() decision making (no tools used)
- generate_comment() with tool-use loop
- Error handling for tool failures
- Sandbox safety restrictions
- Integration with agent registry
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.claude_code_agent import (
    ClaudeCodeAgent,
    ToolAuditTrail,
    ToolCall,
    ToolDeniedError,
    ToolExecutionError,
    ToolExecutor,
    ToolStatus,
    DEFAULT_CODE_AGENT_TOOLS,
    TOOL_DEFINITIONS,
    get_tool_definitions,
)
from src.agents.registry import AgentRegistry, AGENT_TYPE_MAP
from src.models.agent import AgentConfig, AgentDecision, AgentType, LLMProviderConfig
from src.models.workflow import Comment, WorkflowInstance


# ==================
# Fixtures
# ==================


@pytest.fixture
def code_agent_config():
    """Create a code agent configuration."""
    return AgentConfig(
        name="test_code_agent",
        type=AgentType.CLAUDE_SDK_CODE,
        expertise="Backend development and code review",
        domain_knowledge="Python, FastAPI, databases, testing",
        llm=LLMProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_tokens=4096,
        ),
        tools=["Read", "Grep", "Glob", "Bash"],
        can_edit_files=True,
        priority=7,
        category="development",
    )


@pytest.fixture
def code_agent_config_no_tools():
    """Create a code agent config with no explicit tools (should use defaults)."""
    return AgentConfig(
        name="test_default_tools",
        type=AgentType.CLAUDE_SDK_CODE,
        expertise="General development",
        domain_knowledge="Various",
        llm=LLMProviderConfig(),
        tools=[],
        priority=5,
    )


@pytest.fixture
def code_agent(code_agent_config):
    """Create a ClaudeCodeAgent instance."""
    return ClaudeCodeAgent(code_agent_config)


@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    return WorkflowInstance(
        instance_id="test-workflow-001",
        github_issue={
            "title": "Refactor authentication module",
            "body": "The auth module needs better error handling and token validation.",
            "labels": ["enhancement", "backend"],
            "number": 42,
            "state": "open",
            "html_url": "https://github.com/test/repo/issues/42",
        },
        conversation_history=[],
        current_round=1,
        selected_agents=["test_code_agent", "security_expert"],
    )


@pytest.fixture
def workflow_with_history(sample_workflow):
    """Create a workflow with conversation history."""
    sample_workflow.conversation_history = [
        Comment(
            round=1,
            agent="security_expert",
            comment="The auth module has potential token leakage issues. We need to sanitize error responses.",
            references=[],
            timestamp=datetime(2026, 2, 15, 10, 0, 0),
        ),
    ]
    sample_workflow.current_round = 2
    return sample_workflow


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write("def main():\n    print('hello')\n")
        with open(os.path.join(tmpdir, "test.py"), "w") as f:
            f.write("def test_main():\n    assert True\n")
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        with open(os.path.join(tmpdir, "src", "auth.py"), "w") as f:
            f.write("class Auth:\n    def validate(self, token):\n        return True\n")
        yield tmpdir


# ==================
# Tool Definitions Tests
# ==================


class TestToolDefinitions:
    """Tests for tool definition management."""

    def test_default_tools_list(self):
        """Verify default tools include all expected tools."""
        assert "Read" in DEFAULT_CODE_AGENT_TOOLS
        assert "Write" in DEFAULT_CODE_AGENT_TOOLS
        assert "Edit" in DEFAULT_CODE_AGENT_TOOLS
        assert "Grep" in DEFAULT_CODE_AGENT_TOOLS
        assert "Glob" in DEFAULT_CODE_AGENT_TOOLS
        assert "Bash" in DEFAULT_CODE_AGENT_TOOLS
        assert len(DEFAULT_CODE_AGENT_TOOLS) == 6

    def test_tool_definitions_structure(self):
        """Verify tool definitions have correct structure."""
        for tool_def in TOOL_DEFINITIONS:
            assert "type" in tool_def
            assert tool_def["type"] == "function"
            assert "function" in tool_def
            func = tool_def["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_get_tool_definitions_all(self):
        """Get all tool definitions."""
        defs = get_tool_definitions(DEFAULT_CODE_AGENT_TOOLS)
        assert len(defs) == 6
        names = {d["function"]["name"] for d in defs}
        assert names == {"Read", "Write", "Edit", "Grep", "Glob", "Bash"}

    def test_get_tool_definitions_subset(self):
        """Get a subset of tool definitions."""
        defs = get_tool_definitions(["Read", "Grep"])
        assert len(defs) == 2
        names = {d["function"]["name"] for d in defs}
        assert names == {"Read", "Grep"}

    def test_get_tool_definitions_empty(self):
        """Get definitions with no allowed tools returns empty."""
        defs = get_tool_definitions([])
        assert len(defs) == 0

    def test_get_tool_definitions_unknown_tool(self):
        """Unknown tools are silently ignored."""
        defs = get_tool_definitions(["Read", "NonExistentTool"])
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "Read"


# ==================
# ToolCall Tests
# ==================


class TestToolCall:
    """Tests for ToolCall data class."""

    def test_create_tool_call(self):
        """Create a basic tool call record."""
        tc = ToolCall(
            tool_name="Read",
            arguments={"file_path": "/tmp/test.py"},
            result="file contents",
            status=ToolStatus.SUCCESS,
        )
        assert tc.tool_name == "Read"
        assert tc.status == ToolStatus.SUCCESS
        assert tc.result == "file contents"
        assert tc.error_message is None

    def test_tool_call_to_dict(self):
        """Serialize tool call to dictionary."""
        tc = ToolCall(
            tool_name="Grep",
            arguments={"pattern": "def main"},
            result="main.py:1:def main():",
            status=ToolStatus.SUCCESS,
            duration_ms=15.5,
        )
        d = tc.to_dict()
        assert d["tool_name"] == "Grep"
        assert d["status"] == "success"
        assert d["duration_ms"] == 15.5
        assert d["result_length"] == len("main.py:1:def main():")
        assert "call_id" in d
        assert "timestamp" in d

    def test_tool_call_error(self):
        """Create an error tool call."""
        tc = ToolCall(
            tool_name="Bash",
            arguments={"command": "invalid_cmd"},
            status=ToolStatus.ERROR,
            error_message="Command not found",
        )
        assert tc.status == ToolStatus.ERROR
        assert tc.error_message == "Command not found"

    def test_tool_call_denied(self):
        """Create a denied tool call."""
        tc = ToolCall(
            tool_name="Write",
            arguments={"file_path": "/etc/passwd", "content": "hack"},
            status=ToolStatus.DENIED,
            error_message="Sandbox restriction",
        )
        assert tc.status == ToolStatus.DENIED


# ==================
# ToolAuditTrail Tests
# ==================


class TestToolAuditTrail:
    """Tests for the tool audit trail."""

    def test_empty_trail(self):
        """Empty audit trail has zero counts."""
        trail = ToolAuditTrail()
        assert trail.total_calls == 0
        assert trail.successful_calls == 0
        assert trail.total_duration_ms == 0.0

    def test_add_calls(self):
        """Add multiple tool calls."""
        trail = ToolAuditTrail()
        trail.add_call(ToolCall(tool_name="Read", arguments={}, status=ToolStatus.SUCCESS, duration_ms=10))
        trail.add_call(ToolCall(tool_name="Grep", arguments={}, status=ToolStatus.SUCCESS, duration_ms=20))
        trail.add_call(ToolCall(tool_name="Read", arguments={}, status=ToolStatus.ERROR, duration_ms=5))

        assert trail.total_calls == 3
        assert trail.successful_calls == 2
        assert trail.total_duration_ms == 35.0

    def test_get_calls_by_tool(self):
        """Filter calls by tool name."""
        trail = ToolAuditTrail()
        trail.add_call(ToolCall(tool_name="Read", arguments={}))
        trail.add_call(ToolCall(tool_name="Grep", arguments={}))
        trail.add_call(ToolCall(tool_name="Read", arguments={}))

        read_calls = trail.get_calls_by_tool("Read")
        assert len(read_calls) == 2

        grep_calls = trail.get_calls_by_tool("Grep")
        assert len(grep_calls) == 1

    def test_get_failed_calls(self):
        """Get all failed calls."""
        trail = ToolAuditTrail()
        trail.add_call(ToolCall(tool_name="Read", arguments={}, status=ToolStatus.SUCCESS))
        trail.add_call(ToolCall(tool_name="Bash", arguments={}, status=ToolStatus.ERROR))
        trail.add_call(ToolCall(tool_name="Write", arguments={}, status=ToolStatus.DENIED))

        failed = trail.get_failed_calls()
        assert len(failed) == 2

    def test_get_summary(self):
        """Get aggregated statistics."""
        trail = ToolAuditTrail()
        trail.add_call(ToolCall(tool_name="Read", arguments={}, status=ToolStatus.SUCCESS, duration_ms=10))
        trail.add_call(ToolCall(tool_name="Read", arguments={}, status=ToolStatus.SUCCESS, duration_ms=15))
        trail.add_call(ToolCall(tool_name="Grep", arguments={}, status=ToolStatus.ERROR, duration_ms=5))

        summary = trail.get_summary()
        assert summary["total_calls"] == 3
        assert summary["successful_calls"] == 2
        assert summary["failed_calls"] == 1
        assert summary["total_duration_ms"] == 30.0
        assert summary["tools_used"] == {"Read": 2, "Grep": 1}

    def test_to_list(self):
        """Serialize all calls."""
        trail = ToolAuditTrail()
        trail.add_call(ToolCall(tool_name="Read", arguments={"file_path": "x.py"}))
        trail.add_call(ToolCall(tool_name="Grep", arguments={"pattern": "test"}))

        result = trail.to_list()
        assert len(result) == 2
        assert result[0]["tool_name"] == "Read"
        assert result[1]["tool_name"] == "Grep"

    def test_clear(self):
        """Clear the audit trail."""
        trail = ToolAuditTrail()
        trail.add_call(ToolCall(tool_name="Read", arguments={}))
        assert trail.total_calls == 1

        trail.clear()
        assert trail.total_calls == 0


# ==================
# ToolExecutor Tests
# ==================


class TestToolExecutor:
    """Tests for the tool executor."""

    def test_init_sandbox_mode(self):
        """Initialize with sandbox mode enabled."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir="/tmp/sandbox")
        assert executor.sandbox_mode is True
        assert executor.workspace_dir == "/tmp/sandbox"

    def test_init_no_sandbox(self):
        """Initialize with sandbox mode disabled."""
        executor = ToolExecutor(sandbox_mode=False)
        assert executor.sandbox_mode is False

    @pytest.mark.asyncio
    async def test_execute_read_file(self, temp_workspace):
        """Read a file successfully."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        result = await executor.execute("Read", {"file_path": os.path.join(temp_workspace, "main.py")})
        assert "def main()" in result
        assert "print('hello')" in result

    @pytest.mark.asyncio
    async def test_execute_read_nonexistent(self, temp_workspace):
        """Read a nonexistent file returns error."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        result = await executor.execute("Read", {"file_path": os.path.join(temp_workspace, "nonexistent.py")})
        assert "Error" in result or "not found" in result

    @pytest.mark.asyncio
    async def test_execute_read_no_path(self):
        """Read without file_path raises error."""
        executor = ToolExecutor(sandbox_mode=False)
        with pytest.raises(ToolExecutionError, match="file_path is required"):
            await executor.execute("Read", {})

    @pytest.mark.asyncio
    async def test_execute_write_sandbox_denied(self, temp_workspace):
        """Write in sandbox mode is denied."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        with pytest.raises(ToolDeniedError, match="Write operations are disabled"):
            await executor.execute("Write", {
                "file_path": os.path.join(temp_workspace, "new.py"),
                "content": "print('hack')",
            })

    @pytest.mark.asyncio
    async def test_execute_edit_sandbox_denied(self, temp_workspace):
        """Edit in sandbox mode is denied."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        with pytest.raises(ToolDeniedError, match="Edit operations are disabled"):
            await executor.execute("Edit", {
                "file_path": os.path.join(temp_workspace, "main.py"),
                "old_string": "hello",
                "new_string": "world",
            })

    @pytest.mark.asyncio
    async def test_execute_grep(self, temp_workspace):
        """Grep for a pattern."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        result = await executor.execute("Grep", {
            "pattern": "def main",
            "path": temp_workspace,
        })
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_execute_grep_no_match(self, temp_workspace):
        """Grep with no matches."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        result = await executor.execute("Grep", {
            "pattern": "nonexistent_pattern_xyz",
            "path": temp_workspace,
        })
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_execute_glob(self, temp_workspace):
        """Glob for Python files."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        result = await executor.execute("Glob", {
            "pattern": "**/*.py",
            "path": temp_workspace,
        })
        assert "main.py" in result
        assert "test.py" in result

    @pytest.mark.asyncio
    async def test_execute_glob_no_match(self, temp_workspace):
        """Glob with no matches."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        result = await executor.execute("Glob", {
            "pattern": "*.xyz",
            "path": temp_workspace,
        })
        assert "No files found" in result

    @pytest.mark.asyncio
    async def test_execute_bash(self, temp_workspace):
        """Execute a bash command."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        result = await executor.execute("Bash", {"command": "echo hello"})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_bash_blocked_command(self, temp_workspace):
        """Blocked commands are denied in sandbox mode."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        with pytest.raises(ToolDeniedError, match="blocked"):
            await executor.execute("Bash", {"command": "rm -rf /"})

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Unknown tool raises error."""
        executor = ToolExecutor(sandbox_mode=False)
        with pytest.raises(ToolExecutionError, match="Unknown tool"):
            await executor.execute("UnknownTool", {})

    def test_validate_path_in_sandbox(self, temp_workspace):
        """Path within sandbox is allowed."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        # Should not raise
        executor._validate_path(os.path.join(temp_workspace, "main.py"))

    def test_validate_path_outside_sandbox(self, temp_workspace):
        """Path outside sandbox is denied."""
        executor = ToolExecutor(sandbox_mode=True, workspace_dir=temp_workspace)
        with pytest.raises(ToolDeniedError, match="outside the sandbox"):
            executor._validate_path("/etc/passwd")

    def test_validate_path_no_sandbox(self):
        """Path validation skipped when sandbox is disabled."""
        executor = ToolExecutor(sandbox_mode=False)
        # Should not raise
        executor._validate_path("/etc/passwd")


# ==================
# ClaudeCodeAgent Initialization Tests
# ==================


class TestClaudeCodeAgentInit:
    """Tests for ClaudeCodeAgent initialization."""

    def test_basic_init(self, code_agent):
        """Basic initialization with default settings."""
        assert code_agent.name == "test_code_agent"
        assert code_agent.agent_type == AgentType.CLAUDE_SDK_CODE
        assert code_agent.expertise == "Backend development and code review"
        assert code_agent.can_edit_files is True

    def test_tools_from_config(self, code_agent):
        """Tools are loaded from config."""
        assert code_agent.allowed_tools == ["Read", "Grep", "Glob", "Bash"]
        assert "Read" in code_agent.allowed_tools
        assert "Grep" in code_agent.allowed_tools

    def test_default_tools_when_empty(self, code_agent_config_no_tools):
        """Default tools are used when config has empty tools list."""
        agent = ClaudeCodeAgent(code_agent_config_no_tools)
        assert agent.allowed_tools == DEFAULT_CODE_AGENT_TOOLS

    def test_sandbox_enabled_by_default(self, code_agent):
        """Sandbox mode is enabled by default."""
        assert code_agent.sandbox_enabled is True

    def test_set_sandbox_mode(self, code_agent):
        """Can toggle sandbox mode."""
        code_agent.set_sandbox_mode(False)
        assert code_agent.sandbox_enabled is False
        code_agent.set_sandbox_mode(True)
        assert code_agent.sandbox_enabled is True

    def test_set_workspace_dir(self, code_agent):
        """Can set workspace directory."""
        code_agent.set_workspace_dir("/tmp/workspace")
        assert code_agent._tool_executor.workspace_dir == "/tmp/workspace"

    def test_audit_trail_initially_empty(self, code_agent):
        """Audit trail starts empty."""
        assert code_agent.audit_trail.total_calls == 0

    def test_inherits_from_base_agent(self, code_agent):
        """ClaudeCodeAgent is a BaseAgent."""
        from src.agents.base import BaseAgent
        assert isinstance(code_agent, BaseAgent)

    def test_tool_definitions_filtered(self, code_agent):
        """Tool definitions match allowed tools."""
        tool_names = {d["function"]["name"] for d in code_agent._tool_defs}
        assert tool_names == {"Read", "Grep", "Glob", "Bash"}

    def test_llm_model_string_anthropic(self, code_agent):
        """LLM model string for Anthropic provider."""
        assert code_agent._llm_model == "anthropic/claude-sonnet-4-5-20250929"

    def test_llm_model_string_openai(self):
        """LLM model string for OpenAI provider."""
        config = AgentConfig(
            name="openai_agent",
            type=AgentType.CLAUDE_SDK_CODE,
            expertise="Test",
            llm=LLMProviderConfig(provider="openai", model="gpt-4"),
        )
        agent = ClaudeCodeAgent(config)
        assert agent._llm_model == "openai/gpt-4"

    def test_llm_model_string_local(self):
        """LLM model string for local LM Studio."""
        config = AgentConfig(
            name="local_agent",
            type=AgentType.CLAUDE_SDK_CODE,
            expertise="Test",
            llm=LLMProviderConfig(provider="lm_studio", model="local-model"),
        )
        agent = ClaudeCodeAgent(config)
        assert agent._llm_model == "openai/local-model"


# ==================
# Agent Registry Integration Tests
# ==================


class TestRegistryIntegration:
    """Tests for ClaudeCodeAgent integration with the registry."""

    def test_code_agent_in_type_map(self):
        """ClaudeCodeAgent is registered in the type map."""
        assert AgentType.CLAUDE_SDK_CODE in AGENT_TYPE_MAP
        assert AGENT_TYPE_MAP[AgentType.CLAUDE_SDK_CODE] is ClaudeCodeAgent

    def test_registry_creates_code_agent(self):
        """Registry creates ClaudeCodeAgent for claude_sdk_code type."""
        registry = AgentRegistry()
        agents_data = {
            "test_coder": {
                "type": "claude_sdk_code",
                "expertise": "Code development",
                "domain_knowledge": "Python, APIs",
                "tools": ["Read", "Grep", "Glob"],
                "can_edit_files": True,
                "priority": 7,
                "category": "development",
            }
        }
        count = registry.load_from_dict(agents_data)
        assert count == 1

        agent = registry.get_agent("test_coder")
        assert isinstance(agent, ClaudeCodeAgent)
        assert agent.agent_type == AgentType.CLAUDE_SDK_CODE
        assert agent.can_edit_files is True
        assert "Read" in agent.allowed_tools

    def test_registry_mixed_agents(self):
        """Registry handles both text and code agents."""
        registry = AgentRegistry()
        agents_data = {
            "text_agent": {
                "type": "claude_sdk_text",
                "expertise": "Analysis",
                "priority": 5,
            },
            "code_agent": {
                "type": "claude_sdk_code",
                "expertise": "Development",
                "tools": ["Read", "Grep"],
                "priority": 7,
            },
        }
        count = registry.load_from_dict(agents_data)
        assert count == 2

        from src.agents.claude_text_agent import ClaudeTextAgent

        text = registry.get_agent("text_agent")
        code = registry.get_agent("code_agent")

        assert isinstance(text, ClaudeTextAgent)
        assert isinstance(code, ClaudeCodeAgent)

    def test_registry_code_agents_by_type(self):
        """Can filter code agents by type."""
        registry = AgentRegistry()
        agents_data = {
            "text1": {"type": "claude_sdk_text", "expertise": "A"},
            "code1": {"type": "claude_sdk_code", "expertise": "B"},
            "code2": {"type": "claude_sdk_code", "expertise": "C"},
        }
        registry.load_from_dict(agents_data)

        code_agents = registry.get_agents_by_type(AgentType.CLAUDE_SDK_CODE)
        assert len(code_agents) == 2
        for agent in code_agents:
            assert isinstance(agent, ClaudeCodeAgent)


# ==================
# should_comment Tests
# ==================


class TestShouldComment:
    """Tests for the should_comment method."""

    @pytest.mark.asyncio
    async def test_should_comment_returns_true(self, code_agent, sample_workflow):
        """should_comment returns True when LLM decides to participate."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "should_comment": True,
            "reason": "I can analyze the authentication code",
            "responding_to": [],
            "confidence": 0.85,
        })

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            decision = await code_agent.should_comment(sample_workflow, 1)

        assert decision.should_comment is True
        assert decision.confidence == 0.85
        assert "authentication" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_should_comment_returns_false(self, code_agent, sample_workflow):
        """should_comment returns False when not relevant."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "should_comment": False,
            "reason": "Not relevant to my expertise",
            "responding_to": [],
            "confidence": 0.2,
        })

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            decision = await code_agent.should_comment(sample_workflow, 1)

        assert decision.should_comment is False

    @pytest.mark.asyncio
    async def test_should_comment_no_tools_used(self, code_agent, sample_workflow):
        """should_comment does not use tools (decision only)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "should_comment": True,
            "reason": "Relevant",
            "responding_to": [],
            "confidence": 0.8,
        })

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await code_agent.should_comment(sample_workflow, 1)

            # Verify tools were NOT passed to LLM
            call_args = mock_llm.call_args
            assert "tools" not in call_args.kwargs or call_args.kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_should_comment_error_handling(self, code_agent, sample_workflow):
        """should_comment returns False on error."""
        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, side_effect=Exception("API error")):
            decision = await code_agent.should_comment(sample_workflow, 1)

        assert decision.should_comment is False
        assert "Error" in decision.reason

    @pytest.mark.asyncio
    async def test_should_comment_malformed_json(self, code_agent, sample_workflow):
        """should_comment handles malformed JSON gracefully."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            decision = await code_agent.should_comment(sample_workflow, 1)

        assert decision.should_comment is False
        assert decision.confidence == 0.0


# ==================
# generate_comment Tests
# ==================


class TestGenerateComment:
    """Tests for the generate_comment method."""

    @pytest.mark.asyncio
    async def test_generate_comment_no_tools(self, code_agent, sample_workflow):
        """Generate a comment without tool calls."""
        decision = AgentDecision(
            should_comment=True,
            reason="Relevant to code review",
            responding_to=[],
            confidence=0.8,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The auth module needs better input validation."
        mock_response.choices[0].message.tool_calls = None

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            comment = await code_agent.generate_comment(sample_workflow, 1, decision)

        assert isinstance(comment, Comment)
        assert "auth module" in comment.comment
        assert comment.agent == "test_code_agent"
        assert comment.round == 1

    @pytest.mark.asyncio
    async def test_generate_comment_with_tool_calls(self, code_agent, sample_workflow):
        """Generate a comment using tools."""
        decision = AgentDecision(
            should_comment=True,
            reason="Need to inspect code",
            responding_to=[],
            confidence=0.9,
        )

        # First response: LLM makes a tool call
        tool_call_mock = MagicMock()
        tool_call_mock.id = "call_001"
        tool_call_mock.function.name = "Read"
        tool_call_mock.function.arguments = json.dumps({"file_path": "/tmp/auth.py"})

        response1 = MagicMock()
        response1.choices = [MagicMock()]
        response1.choices[0].message.content = ""
        response1.choices[0].message.tool_calls = [tool_call_mock]

        # Second response: LLM provides final answer
        response2 = MagicMock()
        response2.choices = [MagicMock()]
        response2.choices[0].message.content = "After reading the auth file, I recommend..."
        response2.choices[0].message.tool_calls = None

        call_count = 0

        async def mock_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return response1
            return response2

        with patch.object(code_agent, "_call_llm", side_effect=mock_call_llm):
            with patch.object(
                code_agent._tool_executor, "execute",
                new_callable=AsyncMock,
                return_value="class Auth:\n    def validate(self):\n        return True"
            ):
                comment = await code_agent.generate_comment(sample_workflow, 1, decision)

        assert isinstance(comment, Comment)
        assert "After reading the auth file" in comment.comment
        # Tool usage summary should be appended
        assert "Tools used:" in comment.comment
        assert code_agent.audit_trail.total_calls == 1

    @pytest.mark.asyncio
    async def test_generate_comment_tool_error(self, code_agent, sample_workflow):
        """Handle tool execution errors gracefully."""
        decision = AgentDecision(
            should_comment=True,
            reason="Need to inspect code",
            responding_to=[],
            confidence=0.8,
        )

        tool_call_mock = MagicMock()
        tool_call_mock.id = "call_002"
        tool_call_mock.function.name = "Read"
        tool_call_mock.function.arguments = json.dumps({"file_path": "/nonexistent"})

        response1 = MagicMock()
        response1.choices = [MagicMock()]
        response1.choices[0].message.content = ""
        response1.choices[0].message.tool_calls = [tool_call_mock]

        response2 = MagicMock()
        response2.choices = [MagicMock()]
        response2.choices[0].message.content = "Based on the error, the file doesn't exist."
        response2.choices[0].message.tool_calls = None

        call_count = 0

        async def mock_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return response1
            return response2

        with patch.object(code_agent, "_call_llm", side_effect=mock_call_llm):
            with patch.object(
                code_agent._tool_executor, "execute",
                new_callable=AsyncMock,
                side_effect=ToolExecutionError("File not found")
            ):
                comment = await code_agent.generate_comment(sample_workflow, 1, decision)

        assert isinstance(comment, Comment)
        # Should still generate a comment even with tool error
        assert "file doesn't exist" in comment.comment

    @pytest.mark.asyncio
    async def test_generate_comment_with_references(self, code_agent, workflow_with_history):
        """Generate comment responding to another agent."""
        decision = AgentDecision(
            should_comment=True,
            reason="Responding to security concerns",
            responding_to=["security_expert"],
            confidence=0.85,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Responding to security_expert: I agree about token sanitization."
        )
        mock_response.choices[0].message.tool_calls = None

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            comment = await code_agent.generate_comment(workflow_with_history, 2, decision)

        assert comment.references == ["security_expert"]
        assert comment.round == 2

    @pytest.mark.asyncio
    async def test_generate_comment_api_error(self, code_agent, sample_workflow):
        """Handle API errors gracefully."""
        decision = AgentDecision(
            should_comment=True,
            reason="Relevant",
            responding_to=[],
            confidence=0.8,
        )

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, side_effect=Exception("API down")):
            comment = await code_agent.generate_comment(sample_workflow, 1, decision)

        assert isinstance(comment, Comment)
        assert "Error" in comment.comment
        assert "API down" in comment.comment

    @pytest.mark.asyncio
    async def test_generate_comment_clears_audit_trail(self, code_agent, sample_workflow):
        """Audit trail is cleared at start of each generate_comment."""
        # Pre-populate audit trail
        code_agent._audit_trail.add_call(ToolCall(tool_name="Read", arguments={}))
        assert code_agent.audit_trail.total_calls == 1

        decision = AgentDecision(
            should_comment=True,
            reason="Test",
            responding_to=[],
            confidence=0.8,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Analysis here."
        mock_response.choices[0].message.tool_calls = None

        with patch.object(code_agent, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            await code_agent.generate_comment(sample_workflow, 1, decision)

        # Trail should have been cleared (no new tool calls)
        assert code_agent.audit_trail.total_calls == 0


# ==================
# Tool Execution Integration Tests
# ==================


class TestToolExecution:
    """Tests for tool execution through the agent."""

    @pytest.mark.asyncio
    async def test_execute_denied_tool(self, code_agent):
        """Denied tool is recorded in audit trail."""
        tool_call = await code_agent._execute_tool_call(
            "Write", {"file_path": "/tmp/test", "content": "hello"}
        )
        assert tool_call.status == ToolStatus.DENIED
        assert code_agent.audit_trail.total_calls == 1

    @pytest.mark.asyncio
    async def test_execute_disallowed_tool(self, code_agent):
        """Tool not in allowed list is denied."""
        # code_agent only allows Read, Grep, Glob, Bash - not Write
        # But Write might be in ToolExecutor... the _execute_tool_call checks allowed_tools first
        tool_call = await code_agent._execute_tool_call(
            "UnknownTool", {"foo": "bar"}
        )
        assert tool_call.status == ToolStatus.DENIED

    @pytest.mark.asyncio
    async def test_execute_tool_audit_tracking(self, code_agent, temp_workspace):
        """Tool calls are tracked in audit trail."""
        code_agent.set_workspace_dir(temp_workspace)

        await code_agent._execute_tool_call(
            "Read", {"file_path": os.path.join(temp_workspace, "main.py")}
        )
        await code_agent._execute_tool_call(
            "Glob", {"pattern": "*.py", "path": temp_workspace}
        )

        assert code_agent.audit_trail.total_calls == 2
        assert code_agent.audit_trail.successful_calls == 2

        summary = code_agent.get_tool_audit_summary()
        assert summary["agent"] == "test_code_agent"
        assert summary["total_calls"] == 2
        assert summary["sandbox_mode"] is True


# ==================
# System Prompt Tests
# ==================


class TestSystemPrompt:
    """Tests for the code agent's system prompt."""

    def test_code_system_prompt_includes_tools(self, code_agent):
        """Code system prompt mentions available tools."""
        prompt = code_agent._build_code_system_prompt()
        assert "Code Agent Capabilities" in prompt
        assert "Read" in prompt
        assert "Grep" in prompt

    def test_code_system_prompt_extends_base(self, code_agent):
        """Code system prompt includes base prompt content."""
        prompt = code_agent._build_code_system_prompt()
        assert code_agent.name in prompt
        assert code_agent.expertise in prompt

    def test_build_should_comment_prompt(self, code_agent, sample_workflow):
        """should_comment prompt includes tool capabilities."""
        messages = code_agent._build_should_comment_prompt(sample_workflow, 1)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "tools" in messages[1]["content"].lower()

    def test_build_generate_comment_prompt(self, code_agent, sample_workflow):
        """generate_comment prompt includes tool descriptions."""
        decision = AgentDecision(
            should_comment=True,
            reason="Relevant",
            responding_to=[],
            confidence=0.8,
        )
        messages = code_agent._build_generate_comment_prompt(sample_workflow, 1, decision)
        assert len(messages) == 2
        user_msg = messages[1]["content"]
        assert "Read" in user_msg
        assert "Grep" in user_msg


# ==================
# Serialization Tests
# ==================


class TestSerialization:
    """Tests for agent serialization."""

    def test_to_dict(self, code_agent):
        """Serialize code agent to dictionary."""
        d = code_agent.to_dict()
        assert d["name"] == "test_code_agent"
        assert d["type"] == "claude_sdk_code"
        assert d["can_edit_files"] is True
        assert "Read" in d["tools"]
        assert d["category"] == "development"

    def test_repr(self, code_agent):
        """String representation."""
        r = repr(code_agent)
        assert "ClaudeCodeAgent" in r
        assert "test_code_agent" in r

    def test_str(self, code_agent):
        """String format."""
        s = str(code_agent)
        assert "test_code_agent" in s
        assert "Backend development" in s
