"""
Claude SDK Code Agent for the Multi-Agent GitHub Issue Routing System.

This module implements ClaudeCodeAgent, an agent with tool-use capabilities
that can examine and modify code. It extends the text-only ClaudeTextAgent
with support for tools like Read, Write, Edit, Grep, Glob, and Bash.

Key differences from ClaudeTextAgent:
- Tools are enabled in LLM configuration
- Tool calls are tracked in an audit trail
- Sandbox mode can be enabled for safety
- Supports iterative tool-use loops (agent can call tools, see results, call more)

The agent is used for specialist roles like frontend_dev and backend_dev
that need to inspect or modify codebases as part of their analysis.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.agents.base import BaseAgent
from src.integrations.llm_provider import (
    LLMProviderManager,
    build_model_string,
    get_provider_manager,
)
from src.models.agent import AgentConfig, AgentDecision, AgentType
from src.models.workflow import Comment, WorkflowInstance
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Suppress litellm verbose logging
litellm.set_verbose = False


# ==================
# Tool Definitions
# ==================

class ToolStatus(str, Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class ToolCall:
    """
    Record of a single tool invocation.

    Used for audit trail and tracking tool usage patterns.
    """
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[str] = None
    status: ToolStatus = ToolStatus.SUCCESS
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging and storage."""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status.value,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "result_length": len(self.result) if self.result else 0,
        }


@dataclass
class ToolAuditTrail:
    """
    Tracks all tool usage for a session/request.

    Provides aggregated statistics and full history for auditing.
    """
    calls: list[ToolCall] = field(default_factory=list)

    def add_call(self, call: ToolCall) -> None:
        """Add a tool call record."""
        self.calls.append(call)

    def get_calls_by_tool(self, tool_name: str) -> list[ToolCall]:
        """Get all calls for a specific tool."""
        return [c for c in self.calls if c.tool_name == tool_name]

    def get_failed_calls(self) -> list[ToolCall]:
        """Get all failed tool calls."""
        return [c for c in self.calls if c.status != ToolStatus.SUCCESS]

    @property
    def total_calls(self) -> int:
        """Total number of tool calls."""
        return len(self.calls)

    @property
    def successful_calls(self) -> int:
        """Number of successful tool calls."""
        return sum(1 for c in self.calls if c.status == ToolStatus.SUCCESS)

    @property
    def total_duration_ms(self) -> float:
        """Total duration of all tool calls in milliseconds."""
        return sum(c.duration_ms for c in self.calls)

    def get_summary(self) -> dict[str, Any]:
        """Get aggregated statistics."""
        tool_counts: dict[str, int] = {}
        for call in self.calls:
            tool_counts[call.tool_name] = tool_counts.get(call.tool_name, 0) + 1

        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.total_calls - self.successful_calls,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "tools_used": tool_counts,
        }

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all calls to list of dicts."""
        return [c.to_dict() for c in self.calls]

    def clear(self) -> None:
        """Clear the audit trail."""
        self.calls.clear()


# ==================
# Allowed Tools Configuration
# ==================

# Default tools available to code agents
DEFAULT_CODE_AGENT_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "Bash",
]

# Tool definitions for the LLM function calling interface
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the contents of a file. Returns the file content as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file, creating it if it doesn't exist or overwriting if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Edit a file by replacing a specific string with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory or file to search in (default: current directory)",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.py')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern. Returns matching file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match (e.g., '**/*.py')",
                    },
                    "path": {
                        "type": "string",
                        "description": "The base directory to search in",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a bash command. Returns the command output (stdout and stderr).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def get_tool_definitions(allowed_tools: list[str]) -> list[dict[str, Any]]:
    """
    Get tool definitions filtered by the allowed tools list.

    Args:
        allowed_tools: List of tool names the agent is allowed to use

    Returns:
        List of tool definition dicts for the LLM
    """
    return [
        tool for tool in TOOL_DEFINITIONS
        if tool["function"]["name"] in allowed_tools
    ]


# ==================
# Tool Executor (Sandboxed)
# ==================

class ToolExecutor:
    """
    Executes tool calls in a sandboxed environment.

    This executor simulates tool execution for the deliberation context.
    In a real deployment, tools would interact with actual filesystems
    and processes. For the deliberation use case, tools operate on
    the issue's associated repository or a sandboxed workspace.

    Safety features:
    - Sandbox mode restricts file operations to a workspace directory
    - Command execution has timeouts
    - Dangerous commands are blocked
    - All operations are logged for audit
    """

    # Commands that are blocked in sandbox mode
    BLOCKED_COMMANDS = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",
        "chmod -R 777 /",
        "wget",
        "curl",  # block in sandbox for safety
    ]

    def __init__(
        self,
        sandbox_mode: bool = True,
        workspace_dir: Optional[str] = None,
        command_timeout: int = 30,
    ) -> None:
        """
        Initialize the tool executor.

        Args:
            sandbox_mode: If True, restrict operations to workspace_dir
            workspace_dir: Directory for sandboxed file operations
            command_timeout: Timeout in seconds for Bash commands
        """
        self.sandbox_mode = sandbox_mode
        self.workspace_dir = workspace_dir
        self.command_timeout = command_timeout
        self._logger = get_logger(__name__)

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result as a string

        Raises:
            ToolExecutionError: If the tool execution fails
            ToolDeniedError: If the tool call is blocked by sandbox
        """
        handler = getattr(self, f"_execute_{tool_name.lower()}", None)
        if handler is None:
            raise ToolExecutionError(f"Unknown tool: {tool_name}")

        return await handler(arguments)

    async def _execute_read(self, args: dict[str, Any]) -> str:
        """Execute Read tool."""
        file_path = args.get("file_path", "")
        if not file_path:
            raise ToolExecutionError("file_path is required")

        self._validate_path(file_path)

        try:
            import aiofiles
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
            return content
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    async def _execute_write(self, args: dict[str, Any]) -> str:
        """Execute Write tool."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            raise ToolExecutionError("file_path is required")

        self._validate_path(file_path)

        if self.sandbox_mode:
            raise ToolDeniedError(
                "Write operations are disabled in sandbox mode for deliberation. "
                "The agent can describe proposed changes but cannot modify files directly."
            )

        try:
            import aiofiles
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    async def _execute_edit(self, args: dict[str, Any]) -> str:
        """Execute Edit tool."""
        if self.sandbox_mode:
            raise ToolDeniedError(
                "Edit operations are disabled in sandbox mode for deliberation. "
                "The agent can describe proposed edits but cannot modify files directly."
            )

        file_path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")

        if not file_path:
            raise ToolExecutionError("file_path is required")

        self._validate_path(file_path)

        try:
            import aiofiles
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()

            if old_string not in content:
                return f"Error: old_string not found in {file_path}"

            new_content = content.replace(old_string, new_string, 1)
            async with aiofiles.open(file_path, "w") as f:
                await f.write(new_content)

            return f"Successfully edited {file_path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    async def _execute_grep(self, args: dict[str, Any]) -> str:
        """Execute Grep tool."""
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        glob_pattern = args.get("glob", "")

        if not pattern:
            raise ToolExecutionError("pattern is required")

        self._validate_path(path)

        try:
            import subprocess
            cmd = ["grep", "-r", "-n", pattern, path]
            if glob_pattern:
                cmd = ["grep", "-r", "-n", f"--include={glob_pattern}", pattern, path]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.command_timeout
            )

            result = stdout.decode("utf-8", errors="replace")
            if not result:
                return f"No matches found for pattern '{pattern}'"
            # Limit output size
            if len(result) > 10000:
                result = result[:10000] + "\n... (output truncated)"
            return result
        except asyncio.TimeoutError:
            return f"Error: Grep timed out after {self.command_timeout}s"
        except Exception as e:
            return f"Error running grep: {str(e)}"

    async def _execute_glob(self, args: dict[str, Any]) -> str:
        """Execute Glob tool."""
        import glob as glob_module

        pattern = args.get("pattern", "")
        path = args.get("path", ".")

        if not pattern:
            raise ToolExecutionError("pattern is required")

        self._validate_path(path)

        try:
            import os
            full_pattern = os.path.join(path, pattern)
            matches = sorted(glob_module.glob(full_pattern, recursive=True))

            if not matches:
                return f"No files found matching pattern '{pattern}'"

            # Limit output
            if len(matches) > 100:
                result_lines = matches[:100]
                result_lines.append(f"... and {len(matches) - 100} more files")
                return "\n".join(result_lines)

            return "\n".join(matches)
        except Exception as e:
            return f"Error running glob: {str(e)}"

    async def _execute_bash(self, args: dict[str, Any]) -> str:
        """Execute Bash tool."""
        command = args.get("command", "")
        if not command:
            raise ToolExecutionError("command is required")

        if self.sandbox_mode:
            # Check for blocked commands
            for blocked in self.BLOCKED_COMMANDS:
                if blocked in command.lower():
                    raise ToolDeniedError(
                        f"Command blocked by sandbox: contains '{blocked}'"
                    )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.command_timeout
            )

            result = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")

            if err and proc.returncode != 0:
                result += f"\nSTDERR: {err}"

            if not result:
                result = f"(command completed with exit code {proc.returncode})"

            # Limit output size
            if len(result) > 10000:
                result = result[:10000] + "\n... (output truncated)"

            return result
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {self.command_timeout}s"
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _validate_path(self, path: str) -> None:
        """
        Validate that a file path is within the sandbox.

        Args:
            path: Path to validate

        Raises:
            ToolDeniedError: If the path is outside the sandbox
        """
        if not self.sandbox_mode or not self.workspace_dir:
            return

        import os
        # Resolve the path to an absolute path
        abs_path = os.path.abspath(path)
        abs_workspace = os.path.abspath(self.workspace_dir)

        if not abs_path.startswith(abs_workspace):
            raise ToolDeniedError(
                f"Path '{path}' is outside the sandbox workspace '{self.workspace_dir}'"
            )


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""
    pass


class ToolDeniedError(Exception):
    """Raised when a tool call is blocked by sandbox policy."""
    pass


# ==================
# Claude Code Agent
# ==================

class ClaudeCodeAgent(BaseAgent):
    """
    Code-capable agent using LLM with tool-use for code analysis.

    This agent extends the text-only agent with the ability to use tools
    (Read, Write, Edit, Grep, Glob, Bash) for examining and modifying code.
    It's designed for specialist roles that need to inspect codebases as part
    of their deliberation analysis.

    Features:
    - All ClaudeTextAgent features (LLM reasoning, rate limiting, retries)
    - Tool-use support via LLM function calling
    - Iterative tool-use loops (call tools, process results, call more)
    - Audit trail for all tool usage
    - Sandbox mode for safe operation
    - Configurable allowed tools per agent
    - Maximum tool call limits to prevent runaway execution

    In deliberation mode (default), Write/Edit operations are blocked by
    the sandbox - agents can read and search code but not modify it.
    They can describe proposed changes in their comments instead.
    """

    # Maximum number of tool call iterations per LLM interaction
    MAX_TOOL_ITERATIONS = 5

    # Maximum total tool calls per generate_comment invocation
    MAX_TOTAL_TOOL_CALLS = 15

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize a Claude code agent.

        Args:
            config: Agent configuration with tool settings
        """
        super().__init__(config)

        # Track API usage for rate limiting
        self._last_call_time: float = 0.0
        self._min_call_interval: float = 0.5

        # Build the LLM model identifier for litellm
        self._llm_model = self._build_llm_model_string()

        # Build fallback model strings
        self._fallback_models = self._build_fallback_model_strings()

        # Get the global provider manager
        self._provider_manager: LLMProviderManager = get_provider_manager()

        # Determine allowed tools from config or defaults
        self._allowed_tools = config.tools if config.tools else DEFAULT_CODE_AGENT_TOOLS

        # Initialize tool executor with sandbox mode
        self._tool_executor = ToolExecutor(
            sandbox_mode=True,
            workspace_dir=None,  # Will be set per-workflow
            command_timeout=30,
        )

        # Audit trail for tool usage
        self._audit_trail = ToolAuditTrail()

        # Get tool definitions for allowed tools
        self._tool_defs = get_tool_definitions(self._allowed_tools)

        self._logger.info(
            "ClaudeCodeAgent initialized: model=%s, tools=%s, sandbox=%s, fallbacks=%s",
            self._llm_model,
            self._allowed_tools,
            self._tool_executor.sandbox_mode,
            ", ".join(self._fallback_models) if self._fallback_models else "none",
        )

    @property
    def allowed_tools(self) -> list[str]:
        """List of tools this agent is allowed to use."""
        return list(self._allowed_tools)

    @property
    def audit_trail(self) -> ToolAuditTrail:
        """Access the tool usage audit trail."""
        return self._audit_trail

    @property
    def sandbox_enabled(self) -> bool:
        """Whether sandbox mode is enabled."""
        return self._tool_executor.sandbox_mode

    def set_sandbox_mode(self, enabled: bool) -> None:
        """Enable or disable sandbox mode."""
        self._tool_executor.sandbox_mode = enabled
        self._logger.info("Sandbox mode set to: %s", enabled)

    def set_workspace_dir(self, workspace_dir: Optional[str]) -> None:
        """Set the workspace directory for sandboxed operations."""
        self._tool_executor.workspace_dir = workspace_dir
        self._logger.info("Workspace directory set to: %s", workspace_dir)

    def _build_llm_model_string(self) -> str:
        """
        Build the litellm model string from configuration.

        Uses the centralized build_model_string utility for consistency.

        Returns:
            litellm-compatible model string
        """
        return build_model_string(
            self._config.llm.provider,
            self._config.llm.model,
        )

    def _build_fallback_model_strings(self) -> list[str]:
        """
        Build litellm model strings for all configured fallback providers.

        Returns:
            List of fallback model strings
        """
        fallbacks = []
        for fb in self._config.llm.fallback_providers:
            fallbacks.append(build_model_string(fb.provider, fb.model))
        return fallbacks

    async def _rate_limit(self) -> None:
        """Simple rate limiter to avoid API abuse."""
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
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Any:
        """
        Call the LLM API via the LLMProviderManager with retry logic
        and optional tool support.

        Uses the centralized provider manager for multi-provider support,
        rate limiting, cost tracking, and fallback handling.

        Args:
            messages: Chat messages
            temperature: Override temperature
            max_tokens: Override max tokens
            tools: Tool definitions for function calling

        Returns:
            The full LLM response object (to inspect tool_calls)

        Raises:
            Exception: If all retry attempts fail
        """
        await self._rate_limit()

        temp = temperature if temperature is not None else self._config.llm.temperature
        tokens = max_tokens if max_tokens is not None else self._config.llm.max_tokens

        self._logger.debug(
            "Calling LLM: model=%s, messages=%d, tools=%s",
            self._llm_model,
            len(messages),
            bool(tools),
        )

        try:
            response = await self._provider_manager.completion(
                model=self._llm_model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                api_base=self._config.llm.base_url,
                tools=tools,
                fallback_models=self._fallback_models if self._fallback_models else None,
            )
            return response
        except Exception as e:
            self._logger.error("LLM call failed: %s", str(e))
            raise

    async def _execute_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCall:
        """
        Execute a single tool call and record it in the audit trail.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            ToolCall record with result
        """
        start_time = time.time()
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
        )

        self._logger.info(
            "Executing tool: %s (args: %s)",
            tool_name,
            {k: str(v)[:100] for k, v in arguments.items()},
        )

        try:
            if tool_name not in self._allowed_tools:
                raise ToolDeniedError(
                    f"Tool '{tool_name}' is not in the allowed tools list: {self._allowed_tools}"
                )

            result = await self._tool_executor.execute(tool_name, arguments)
            tool_call.result = result
            tool_call.status = ToolStatus.SUCCESS

        except ToolDeniedError as e:
            tool_call.status = ToolStatus.DENIED
            tool_call.error_message = str(e)
            tool_call.result = f"DENIED: {str(e)}"
            self._logger.warning("Tool call denied: %s - %s", tool_name, str(e))

        except ToolExecutionError as e:
            tool_call.status = ToolStatus.ERROR
            tool_call.error_message = str(e)
            tool_call.result = f"ERROR: {str(e)}"
            self._logger.error("Tool execution error: %s - %s", tool_name, str(e))

        except asyncio.TimeoutError:
            tool_call.status = ToolStatus.TIMEOUT
            tool_call.error_message = "Tool execution timed out"
            tool_call.result = "TIMEOUT: Tool execution timed out"
            self._logger.error("Tool timeout: %s", tool_name)

        except Exception as e:
            tool_call.status = ToolStatus.ERROR
            tool_call.error_message = str(e)
            tool_call.result = f"ERROR: {str(e)}"
            self._logger.error("Unexpected error in tool %s: %s", tool_name, str(e))

        finally:
            tool_call.duration_ms = (time.time() - start_time) * 1000
            self._audit_trail.add_call(tool_call)

        self._logger.info(
            "Tool %s completed: status=%s, duration=%.1fms",
            tool_name,
            tool_call.status.value,
            tool_call.duration_ms,
        )

        return tool_call

    async def _process_tool_calls(
        self,
        response: Any,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Process tool calls from the LLM response.

        Executes each tool call, adds results to messages,
        and returns the final text response.

        Args:
            response: LLM response potentially containing tool calls
            messages: Current message history

        Returns:
            Tuple of (final_text_response, updated_messages)
        """
        total_tool_calls = 0
        iterations = 0

        while iterations < self.MAX_TOOL_ITERATIONS:
            message = response.choices[0].message

            # Check if there are tool calls to process
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                # No more tool calls - return the text content
                return message.content or "", messages

            # Add the assistant's message (with tool_calls) to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in tool_calls:
                total_tool_calls += 1
                if total_tool_calls > self.MAX_TOTAL_TOOL_CALLS:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "ERROR: Maximum tool call limit reached. Please provide your analysis with the information gathered so far.",
                    })
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_result = await self._execute_tool_call(
                    tc.function.name, args
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result.result or "No output",
                })

            # Call LLM again with tool results
            iterations += 1
            response = await self._call_llm(
                messages,
                tools=self._tool_defs if total_tool_calls < self.MAX_TOTAL_TOOL_CALLS else None,
            )

        # If we exhausted iterations, return whatever we have
        final_content = response.choices[0].message.content or ""
        return final_content, messages

    # ==================
    # should_comment
    # ==================

    def _build_should_comment_prompt(
        self,
        workflow: WorkflowInstance,
        current_round: int,
    ) -> list[dict[str, str]]:
        """Build the prompt for the should_comment decision."""
        issue_ctx = self.get_issue_context(workflow)

        conversation_summary = ""
        if workflow.conversation_history:
            conversation_summary = "\n## Previous Comments\n"
            for comment in workflow.conversation_history:
                conversation_summary += (
                    f"\n### Round {comment.round} - {comment.agent}\n"
                    f"{comment.comment}\n"
                )

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
## Your Capabilities: Code agent with tools ({', '.join(self._allowed_tools)})
{mention_note}
{conversation_summary}

## Your Task
Evaluate whether you should comment on this issue in round {current_round}.

You have access to code analysis tools ({', '.join(self._allowed_tools)}).
Consider if using these tools would help provide better analysis.

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

    def _parse_should_comment_response(self, response: str) -> AgentDecision:
        """Parse the LLM response for should_comment into an AgentDecision."""
        text = response.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            start = 1 if lines[0].startswith("```") else 0
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
                "Failed to parse should_comment response: %s. Response: %s",
                str(e),
                text[:200],
            )
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

        Does NOT use tools for the decision step - only for comment generation.

        Args:
            workflow: Current workflow state
            current_round: Current round number

        Returns:
            AgentDecision with the participation decision
        """
        self._logger.info(
            "Evaluating should_comment for round %d (code agent: %s)",
            current_round,
            self.name,
        )

        try:
            messages = self._build_should_comment_prompt(workflow, current_round)

            # No tools for the decision step
            response = await self._call_llm(
                messages,
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content or ""
            decision = self._parse_should_comment_response(content)

            self._logger.info(
                "Decision: should_comment=%s, confidence=%.2f, reason=%s",
                decision.should_comment,
                decision.confidence,
                decision.reason[:80],
            )

            return decision

        except Exception as e:
            self._logger.error("Error in should_comment: %s", str(e))
            return AgentDecision(
                should_comment=False,
                reason=f"Error evaluating participation: {str(e)}",
                responding_to=[],
                confidence=0.0,
            )

    # ==================
    # generate_comment
    # ==================

    def _build_generate_comment_prompt(
        self,
        workflow: WorkflowInstance,
        current_round: int,
        decision: AgentDecision,
    ) -> list[dict[str, str]]:
        """Build the prompt for generating a comment with tool access."""
        issue_ctx = self.get_issue_context(workflow)

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

You have access to code analysis tools. Use them to examine relevant code if needed:
- **Read**: Read file contents
- **Grep**: Search for patterns in code
- **Glob**: Find files by pattern
- **Bash**: Run commands (read-only operations recommended)

Guidelines:
- Use tools to examine code when it would strengthen your analysis
- Focus on your area of expertise: {self.expertise}
- Reference other agents by name when responding to their points
- Be concise but thorough (aim for 100-300 words)
- Include specific code references when relevant
- Provide actionable recommendations
- Don't repeat points already well-covered

Provide your comment directly (no JSON wrapping, just your analysis)."""

        return [
            {"role": "system", "content": self._build_code_system_prompt()},
            {"role": "user", "content": user_message},
        ]

    def _build_code_system_prompt(self) -> str:
        """
        Build a system prompt enhanced for code agent capabilities.

        Returns:
            System prompt string with tool-use guidance
        """
        base = self.build_system_prompt()
        return (
            f"{base}\n\n"
            f"## Code Agent Capabilities\n"
            f"You are a code-capable agent with access to the following tools: "
            f"{', '.join(self._allowed_tools)}.\n\n"
            f"When analyzing issues, you can use these tools to:\n"
            f"- Read source files to understand the current implementation\n"
            f"- Search code with grep/glob to find relevant patterns\n"
            f"- Run commands to inspect the project structure\n\n"
            f"Use tools judiciously - only when examining code would provide "
            f"concrete, specific insights that strengthen your analysis.\n"
            f"Always include relevant code snippets or file references in your response."
        )

    async def generate_comment(
        self,
        workflow: WorkflowInstance,
        current_round: int,
        decision: AgentDecision,
    ) -> Comment:
        """
        Use LLM with tools to generate a code-informed comment.

        The agent can use tools to read code, search files, etc. before
        or during comment generation. Tool usage is tracked in the audit trail.

        Args:
            workflow: Current workflow state
            current_round: Current round number
            decision: The participation decision

        Returns:
            Comment with code-informed analysis
        """
        self._logger.info(
            "Generating code-agent comment for round %d (tools: %s)",
            current_round,
            self._allowed_tools,
        )

        # Clear audit trail for this invocation
        self._audit_trail.clear()

        try:
            messages = self._build_generate_comment_prompt(
                workflow, current_round, decision
            )

            # Call LLM with tools enabled
            response = await self._call_llm(
                messages,
                tools=self._tool_defs,
            )

            # Process any tool calls iteratively
            comment_text, _ = await self._process_tool_calls(response, messages)

            if not comment_text:
                comment_text = (
                    f"[{self.name}] Unable to generate substantive comment "
                    f"for this round."
                )

            # Append tool usage summary if tools were used
            if self._audit_trail.total_calls > 0:
                summary = self._audit_trail.get_summary()
                tools_note = (
                    f"\n\n---\n*Tools used: {summary['total_calls']} calls "
                    f"({summary['successful_calls']} successful) - "
                    f"{', '.join(f'{k}: {v}' for k, v in summary['tools_used'].items())}*"
                )
                comment_text += tools_note

            comment = self.create_comment(
                text=comment_text,
                round_number=current_round,
                references=decision.responding_to,
            )

            self._logger.info(
                "Generated code-agent comment: %d chars, tool_calls=%d, references=%s",
                len(comment_text),
                self._audit_trail.total_calls,
                ", ".join(decision.responding_to) if decision.responding_to else "none",
            )

            return comment

        except Exception as e:
            self._logger.error("Error generating comment: %s", str(e))
            return self.create_comment(
                text=(
                    f"[{self.name}] Error generating analysis: {str(e)}. "
                    f"Please disregard this comment."
                ),
                round_number=current_round,
                references=decision.responding_to,
            )

    def get_tool_audit_summary(self) -> dict[str, Any]:
        """
        Get a summary of tool usage for this agent.

        Returns:
            Dictionary with tool usage statistics
        """
        return {
            "agent": self.name,
            "sandbox_mode": self._tool_executor.sandbox_mode,
            "allowed_tools": self._allowed_tools,
            **self._audit_trail.get_summary(),
        }
