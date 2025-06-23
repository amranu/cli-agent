#!/usr/bin/env python3
"""Base MCP Agent implementation with shared functionality."""

import asyncio
import json
import logging
import os
import re
import select
import subprocess
import sys
import termios
import time
import tty
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastmcp.client import Client as FastMCPClient
from fastmcp.client import StdioTransport

from cli_agent.core.builtin_tool_executor import BuiltinToolExecutor
from cli_agent.core.formatting import ResponseFormatter
from cli_agent.core.formatting_utils import FormattingUtils
from cli_agent.core.slash_commands import SlashCommandManager
from cli_agent.core.subagent_coordinator import SubagentCoordinator
from cli_agent.core.system_prompt_builder import SystemPromptBuilder
from cli_agent.core.token_manager import TokenManager
from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt
from cli_agent.core.tool_schema import ToolSchemaManager
from cli_agent.tools.builtin_tools import get_all_builtin_tools
from config import HostConfig

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Suppress WARNING messages during interactive chat
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BaseMCPAgent(ABC):
    """Base class for MCP agents with shared functionality."""

    def __init__(self, config: HostConfig, is_subagent: bool = False):
        self.config = config
        self.is_subagent = is_subagent
        self.mcp_clients: Dict[str, FastMCPClient] = {}
        self.available_tools: Dict[str, Dict] = {}
        self.conversation_history: List[Dict[str, Any]] = []

        # Communication socket for subagent forwarding (set by parent process)
        self.comm_socket = None

        # Streaming callbacks for JSON output mode
        self.streaming_json_callback = None
        self.streaming_json_tool_use_callback = None
        self.streaming_json_tool_result_callback = None

        # Centralized subagent management system
        if not is_subagent:
            try:
                # Add project root directory to path for subagent import
                # subagent.py is in the root directory, not in cli_agent/core/
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from subagent import SubagentManager

                self.subagent_manager = SubagentManager(config)

                # Event-driven message handling
                self.subagent_message_queue = asyncio.Queue()
                self.subagent_manager.add_message_callback(self._on_subagent_message)

                # Track last message time for timeout management
                self.last_subagent_message_time = None

                logger.info("Initialized centralized subagent management system")
            except ImportError as e:
                logger.warning(f"Failed to import subagent manager: {e}")
                self.subagent_manager = None
                self.subagent_message_queue = None
                self.last_subagent_message_time = None
        else:
            self.subagent_manager = None
            self.subagent_message_queue = None
            self.last_subagent_message_time = None

        # Add built-in tools
        self._add_builtin_tools()

        # Initialize slash command manager
        self.slash_commands = SlashCommandManager(self)

        # Initialize tool permission manager
        try:
            from cli_agent.core.tool_permissions import (
                ToolPermissionConfig,
                ToolPermissionManager,
            )

            # Create default permission config (prompts for all tools by default)
            # Set empty session file to ensure no persistent approvals across sessions
            permission_config = ToolPermissionConfig()
            permission_config.session_permissions_file = (
                None  # Disable persistent storage
            )
            self.permission_manager = ToolPermissionManager(permission_config)
            logger.info(
                f"Initialized tool permission manager with session file: '{permission_config.session_permissions_file}'"
            )
        except ImportError as e:
            logger.warning(f"Failed to import tool permission manager: {e}")
            self.permission_manager = None

        # Initialize token manager
        self.token_manager = TokenManager(config)
        logger.debug("Initialized token manager")

        # Initialize tool schema manager
        self.tool_schema = ToolSchemaManager()
        logger.debug("Initialized tool schema manager")

        # Initialize response formatter
        self.formatter = ResponseFormatter(config)
        logger.debug("Initialized response formatter")

        # Initialize utility classes
        self.builtin_executor = BuiltinToolExecutor(self)
        self.subagent_coordinator = SubagentCoordinator(self)
        self.system_prompt_builder = SystemPromptBuilder(self)
        self.formatting_utils = FormattingUtils(self)
        logger.debug("Initialized utility classes")

        # Centralized LLM client initialization
        self._initialize_llm_client()

        logger.info(
            f"Initialized Base MCP Agent with {len(self.available_tools)} built-in tools"
        )

    def _add_builtin_tools(self):
        """Add built-in tools to the available tools."""
        builtin_tools = get_all_builtin_tools()

        # Configure tools based on agent type
        if self.is_subagent:
            # Remove subagent management tools for subagents to prevent recursion
            subagent_tools = [
                "builtin:task",
                "builtin:task_status",
                "builtin:task_results",
            ]
            for tool_key in subagent_tools:
                if tool_key in builtin_tools:
                    del builtin_tools[tool_key]
                    logger.info(f"Removed {tool_key} from subagent tools")
        else:
            # Remove emit_result tool for main agents (subagents only)
            if "builtin:emit_result" in builtin_tools:
                del builtin_tools["builtin:emit_result"]
                logger.info("Removed emit_result from main agent tools")

        self.available_tools.update(builtin_tools)

        # Debug: Log available tools for subagents
        if self.is_subagent:
            logger.info(
                f"Subagent initialized with {len(self.available_tools)} tools: {list(self.available_tools.keys())}"
            )

    async def _execute_builtin_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a built-in tool."""
        if tool_name == "bash_execute":
            return self.builtin_executor.bash_execute(args)
        elif tool_name == "read_file":
            return self.builtin_executor.read_file(args)
        elif tool_name == "write_file":
            return self.builtin_executor.write_file(args)
        elif tool_name == "list_directory":
            return self.builtin_executor.list_directory(args)
        elif tool_name == "get_current_directory":
            return self.builtin_executor.get_current_directory(args)
        elif tool_name == "todo_read":
            return self.builtin_executor.todo_read(args)
        elif tool_name == "todo_write":
            return self.builtin_executor.todo_write(args)
        elif tool_name == "replace_in_file":
            return self.builtin_executor.replace_in_file(args)
        elif tool_name == "webfetch":
            return self.builtin_executor.webfetch(args)
        elif tool_name == "task":
            return await self.builtin_executor.task(args)
        elif tool_name == "task_status":
            return self.builtin_executor.task_status(args)
        elif tool_name == "task_results":
            return self.builtin_executor.task_results(args)
        elif tool_name == "emit_result":
            return await self._emit_result(args)
        else:
            return f"Unknown built-in tool: {tool_name}"

    async def _emit_result(self, args: Dict[str, Any]) -> str:
        """Emit the final result of a subagent task and terminate (subagents only)."""
        if not self.is_subagent:
            return "Error: emit_result can only be used by subagents"

        result = args.get("result", "")
        summary = args.get("summary", "")

        try:
            # Import emit functions
            from subagent import emit_result

            # Emit the final result
            if summary:
                full_result = f"{result}\n\nSummary: {summary}"
            else:
                full_result = result

            emit_result(full_result)

            # Exit the subagent process to terminate cleanly
            sys.exit(0)

        except Exception as e:
            return f"Error emitting result: {str(e)}"

    async def start_mcp_server(self, server_name: str, server_config) -> bool:
        """Start and connect to an MCP server using FastMCP."""
        try:
            logger.info(f"Starting MCP server: {server_name}")

            # Construct command and args for FastMCP client
            command = server_config.command[0]
            args = server_config.command[1:] + server_config.args

            # Create FastMCP client with stdio transport
            transport = StdioTransport(
                command=command, args=args, env=server_config.env
            )
            client = FastMCPClient(transport=transport)

            # Enter the context manager and store it for cleanup
            context_manager = client.__aenter__()
            await context_manager

            # Store the client and context manager
            self.mcp_clients[server_name] = client
            self._mcp_contexts = getattr(self, "_mcp_contexts", {})
            self._mcp_contexts[server_name] = client

            # Get available tools from this server
            tools_result = await client.list_tools()
            if tools_result and hasattr(tools_result, "tools"):
                for tool in tools_result.tools:
                    tool_key = f"{server_name}:{tool.name}"
                    self.available_tools[tool_key] = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "schema": (
                            tool.inputSchema if hasattr(tool, "inputSchema") else {}
                        ),
                        "client": client,
                    }
                    logger.info(f"Registered tool: {tool_key}")
            elif hasattr(tools_result, "__len__"):
                # Handle list format
                for tool in tools_result:
                    tool_key = f"{server_name}:{tool.name}"
                    self.available_tools[tool_key] = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "schema": (
                            tool.inputSchema if hasattr(tool, "inputSchema") else {}
                        ),
                        "client": client,
                    }
                    logger.info(f"Registered tool: {tool_key}")

            logger.info(f"Successfully connected to MCP server: {server_name}")
            return True

        except Exception as e:
            import traceback

            logger.error(f"Failed to start MCP server {server_name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    async def shutdown(self):
        """Shutdown all MCP connections."""
        logger.info("Shutting down MCP connections...")

        # Close FastMCP client sessions
        for server_name, client in self.mcp_clients.items():
            try:
                # Exit the context manager properly
                await client.__aexit__(None, None, None)
                logger.info(f"Closed client session for {server_name}")
            except Exception as e:
                logger.error(f"Error closing client session for {server_name}: {e}")

        self.mcp_clients.clear()
        self.available_tools.clear()
        if hasattr(self, "_mcp_contexts"):
            self._mcp_contexts.clear()

        # Shutdown subagent manager if present
        if hasattr(self, "subagent_manager") and self.subagent_manager:
            await self.subagent_manager.terminate_all()

    # Centralized Subagent Management Methods

    def _on_subagent_message(self, message):
        """Callback for when a subagent message is received - display during yield period."""
        try:
            # Update timeout tracking - reset timer whenever we receive any message
            import time

            self.last_subagent_message_time = time.time()

            # Get task_id for identification (if available in message data)
            task_id = (
                message.data.get("task_id", "unknown")
                if hasattr(message, "data") and message.data
                else "unknown"
            )

            if message.type == "output":
                formatted = f"🤖 [SUBAGENT-{task_id}] {message.content}"
            elif message.type == "status":
                formatted = f"📋 [SUBAGENT-{task_id}] {message.content}"
            elif message.type == "error":
                formatted = f"❌ [SUBAGENT-{task_id}] {message.content}"
            elif message.type == "result":
                formatted = f"✅ [SUBAGENT-{task_id}] Result: {message.content}"
            elif message.type == "permission_request":
                # Handle permission request from subagent
                self.subagent_coordinator.handle_subagent_permission_request(
                    message, task_id
                )
                return  # Don't display permission requests, handle them directly
            else:
                formatted = f"[SUBAGENT-{task_id} {message.type}] {message.content}"

            # Only display immediately if we're in yielding mode (subagents active)
            # This ensures clean separation between main agent and subagent output
            if self.subagent_manager and self.subagent_manager.get_active_count() > 0:
                # Subagents are active - display immediately during yield period
                self.subagent_coordinator.display_subagent_message_immediately(
                    formatted, message.type
                )
            else:
                # No active subagents - just log for now (main agent controls display)
                logger.info(
                    f"Subagent message logged: {message.type} - {message.content[:50]}"
                )

        except Exception as e:
            logger.error(f"Error handling subagent message: {e}")

    async def execute_function_calls(
        self,
        function_calls: List,
        interactive: bool = True,
        input_handler=None,
        streaming_mode: bool = False,
    ) -> tuple:
        """Centralized function call execution for all host implementations."""
        function_results = []
        all_tool_output = []

        # Prepare tool info for parallel execution
        tool_info_list = []
        tool_coroutines = []

        # Check for interruption before starting any tool execution
        if input_handler and input_handler.interrupted:
            all_tool_output.append("🛑 Tool execution interrupted by user")
            return function_results, all_tool_output

        # Check if there's any buffered text that needs to be displayed before tool execution
        text_buffer = getattr(self, "_text_buffer", "")
        if text_buffer.strip() and interactive:
            # Format and display the buffered text before showing tool execution
            formatted_response = self.formatter.format_markdown(text_buffer)
            # Replace newlines with \r\n for proper terminal handling
            formatted_response = formatted_response.replace("\n", "\r\n")
            print(f"\r\x1b[K\r\nAssistant: {formatted_response}")
            # Clear the buffer
            self._text_buffer = ""

        for i, function_call in enumerate(function_calls, 1):
            tool_name = function_call.name.replace(
                "_", ":", 1
            )  # Convert back to MCP format

            # Parse arguments from function call
            arguments = {}
            if hasattr(function_call, "args") and function_call.args:
                try:
                    import json

                    # First try to access as dict directly
                    if hasattr(function_call.args, "items"):
                        arguments = dict(function_call.args)
                    elif hasattr(function_call.args, "__iter__"):
                        arguments = dict(function_call.args)
                    else:
                        # If args is a string, try to parse as JSON
                        if isinstance(function_call.args, str):
                            arguments = json.loads(function_call.args)
                        else:
                            arguments = {}
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in function call args: {e}")
                    logger.warning(f"Raw args: {function_call.args}")
                    arguments = {}
                except Exception as e:
                    logger.warning(f"Error parsing function call args: {e}")
                    logger.warning(f"Raw args: {function_call.args}")
                    arguments = {}

            # Store tool info for processing
            tool_info_list.append((i, tool_name, arguments))

            # Display tool execution step
            tool_execution_msg = self.formatter.display_tool_execution_step(
                i, tool_name, arguments, self.is_subagent, interactive=interactive
            )
            if interactive and not streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            elif interactive and streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            else:
                all_tool_output.append(tool_execution_msg)

            # Emit tool use if streaming JSON callback is set
            import uuid

            tool_use_id = f"toolu_{i}_{uuid.uuid4().hex[:16]}"
            if (
                hasattr(self, "streaming_json_tool_use_callback")
                and self.streaming_json_tool_use_callback
            ):
                self.streaming_json_tool_use_callback(tool_name, arguments, tool_use_id)

            # Create coroutine for parallel execution with tool_use_id tracking
            tool_coroutines.append(
                (tool_use_id, self._execute_mcp_tool(tool_name, arguments))
            )

        # Execute all tools in parallel
        if tool_coroutines:
            try:
                # Execute all tool calls concurrently
                # Extract just the coroutines for asyncio.gather
                coroutines = [coroutine for _, coroutine in tool_coroutines]
                tool_results = await asyncio.gather(*coroutines, return_exceptions=True)

                # Process results in order
                for (i, tool_name, arguments), (tool_use_id, _), tool_result in zip(
                    tool_info_list, tool_coroutines, tool_results
                ):
                    tool_success = True

                    # Handle exceptions
                    if isinstance(tool_result, Exception):
                        # Re-raise tool permission denials so they can be handled at the chat level
                        from cli_agent.core.tool_permissions import (
                            ToolDeniedReturnToPrompt,
                        )

                        if isinstance(tool_result, ToolDeniedReturnToPrompt):
                            raise tool_result  # Re-raise the exception to bubble up to interactive chat

                        tool_success = False
                        tool_result = f"Exception during execution: {str(tool_result)}"
                    elif isinstance(tool_result, str):
                        # Check if tool result indicates an error
                        if (
                            tool_result.startswith("Error:")
                            or "error" in tool_result.lower()[:100]
                        ):
                            tool_success = False
                    else:
                        # Convert non-string results to string
                        tool_result = str(tool_result)

                    # Format result with success/failure status
                    status = "SUCCESS" if tool_success else "FAILED"
                    result_content = f"Tool {tool_name} {status}: {tool_result}"
                    if not tool_success:
                        result_content += "\n⚠️  Command failed - take this into account for your next action."
                    function_results.append(result_content)

                    # Emit tool result if streaming JSON callback is set
                    if (
                        hasattr(self, "streaming_json_tool_result_callback")
                        and self.streaming_json_tool_result_callback
                    ):
                        self.streaming_json_tool_result_callback(
                            tool_use_id, str(tool_result), not tool_success
                        )

                    # Use unified tool result display
                    tool_result_msg = self.formatter.display_tool_execution_result(
                        tool_result,
                        not tool_success,
                        self.is_subagent,
                        interactive=interactive,
                    )
                    if interactive and not streaming_mode:
                        print(f"\r\x1b[K{tool_result_msg}\n", flush=True)
                    elif interactive and streaming_mode:
                        print(f"\r\x1b[K{tool_result_msg}\n", flush=True)
                    else:
                        all_tool_output.append(tool_result_msg)

            except Exception as e:
                # Handle any errors during parallel execution
                from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

                if isinstance(e, ToolDeniedReturnToPrompt):
                    raise  # Re-raise permission denials

                error_msg = f"Error during tool execution: {str(e)}"
                logger.error(error_msg)
                all_tool_output.append(error_msg)
                function_results.append(f"Tool execution FAILED: {error_msg}")

        return function_results, all_tool_output

    async def _execute_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool (built-in or external) and return the result."""
        try:
            if tool_key not in self.available_tools:
                # Debug: show available tools when tool not found
                available_list = list(self.available_tools.keys())[
                    :10
                ]  # First 10 tools
                return f"Error: Tool {tool_key} not found. Available tools: {available_list}"

            tool_info = self.available_tools[tool_key]
            tool_name = tool_info["name"]

            # Check tool permissions (both main agent and subagents)
            if hasattr(self, "permission_manager") and self.permission_manager:
                from cli_agent.core.tool_permissions import (
                    ToolDeniedReturnToPrompt,
                    ToolPermissionResult,
                )

                input_handler = getattr(self, "_input_handler", None)
                permission_result = await self.permission_manager.check_tool_permission(
                    tool_name, arguments, input_handler
                )

                if not permission_result.allowed:
                    if permission_result.return_to_prompt and not self.is_subagent:
                        # Only return to prompt for main agent, not subagents
                        raise ToolDeniedReturnToPrompt(permission_result.reason)
                    else:
                        # For subagents or config-based denials, return error message
                        return f"Tool execution denied: {permission_result.reason}"

            # Forward to parent if this is a subagent (except for subagent management tools)
            if self.is_subagent and self.comm_socket:
                excluded_tools = ["task", "task_status", "task_results"]
                if tool_name not in excluded_tools:
                    # Tool forwarding happens silently
                    return await self._forward_tool_to_parent(
                        tool_key, tool_name, arguments
                    )
            elif self.is_subagent:
                sys.stderr.write(
                    f"🤖 [SUBAGENT] WARNING: is_subagent=True but no comm_socket for tool {tool_name}\n"
                )
                sys.stderr.flush()

            # Check if it's a built-in tool
            if tool_info["server"] == "builtin":
                logger.info(f"Executing built-in tool: {tool_name}")
                return await self._execute_builtin_tool(tool_name, arguments)

            # Handle external MCP tools with FastMCP
            client = tool_info["client"]
            if client is None:
                return f"Error: No client session for tool {tool_key}"

            logger.info(f"Executing MCP tool: {tool_name}")
            result = await client.call_tool(tool_name, arguments)

            # Format the result for FastMCP
            if hasattr(result, "content") and result.content:
                content_parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        content_parts.append(content.text)
                    elif hasattr(content, "data"):
                        content_parts.append(str(content.data))
                    else:
                        content_parts.append(str(content))
                return "\n".join(content_parts)
            elif isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return f"Tool executed successfully. Result type: {type(result)}, Content: {result}"

        except Exception as e:
            # Re-raise tool permission denials so they can be handled at the chat level
            from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

            if isinstance(e, ToolDeniedReturnToPrompt):
                raise  # Re-raise the exception to bubble up to interactive chat

            logger.error(f"Error executing tool {tool_key}: {e}")
            return f"Error executing tool {tool_key}: {str(e)}"

    async def _forward_tool_to_parent(
        self, tool_key: str, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Forward tool execution to parent agent via communication socket."""
        try:
            import json
            import uuid

            # Create unique request ID for tracking
            request_id = str(uuid.uuid4())

            # Prepare tool execution message
            message = {
                "type": "tool_execution_request",
                "request_id": request_id,
                "tool_key": tool_key,
                "tool_name": tool_name,
                "tool_args": arguments,
                "timestamp": time.time(),
            }

            # Send request to parent (synchronous)
            message_json = json.dumps(message) + "\n"
            self.comm_socket.send(message_json.encode("utf-8"))

            # Wait for response with timeout
            response_timeout = 300.0  # 5 minutes timeout for tool execution
            self.comm_socket.settimeout(response_timeout)

            # Read response (synchronous)
            buffer = ""
            while True:
                try:
                    data = self.comm_socket.recv(4096).decode("utf-8")
                    if not data:
                        break

                    buffer += data

                    # Process complete messages (newline-delimited JSON)
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            try:
                                response = json.loads(line.strip())
                                if (
                                    response.get("type") == "tool_execution_response"
                                    and response.get("request_id") == request_id
                                ):

                                    # Return tool result
                                    if response.get("success", False):
                                        return response.get(
                                            "result", "Tool executed successfully"
                                        )
                                    else:
                                        error = response.get("error", "Unknown error")
                                        return f"Error from parent: {error}"

                            except json.JSONDecodeError:
                                continue

                except Exception as e:
                    logger.error(f"Error receiving response from parent: {e}")
                    break

            return f"Error: No response received from parent for tool {tool_name}"

        except Exception as e:
            logger.error(f"Error forwarding tool {tool_name} to parent: {e}")
            return f"Error forwarding tool to parent: {str(e)}"

    async def _execute_mcp_tool_with_keepalive(
        self,
        tool_key: str,
        arguments: Dict[str, Any],
        input_handler=None,
        keepalive_interval: float = 5.0,
    ) -> tuple:
        """Execute an MCP tool with keep-alive messages, returning (result, keepalive_messages)."""
        import asyncio

        # Create the tool execution task
        tool_task = asyncio.create_task(self._execute_mcp_tool(tool_key, arguments))

        # Keep-alive configuration
        keepalive_messages = []
        start_time = asyncio.get_event_loop().time()

        # Monitor the task and collect keep-alive messages
        while not tool_task.done():
            try:
                # Check for interruption before waiting
                if input_handler and input_handler.interrupted:
                    tool_task.cancel()
                    keepalive_messages.append("🛑 Tool execution cancelled by user")
                    try:
                        await tool_task
                    except asyncio.CancelledError:
                        pass
                    return "Tool execution cancelled", keepalive_messages

                # Wait for either task completion or timeout
                await asyncio.wait_for(
                    asyncio.shield(tool_task), timeout=keepalive_interval
                )
                break  # Task completed
            except asyncio.TimeoutError:
                # Task is still running, send keep-alive message
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time

                # Create a keep-alive message
                keepalive_msg = (
                    f"⏳ Tool {tool_key} still running... ({elapsed:.1f}s elapsed)"
                )
                if input_handler:
                    keepalive_msg += ", press ESC to cancel"
                keepalive_messages.append(keepalive_msg)
                logger.debug(f"Keep-alive: {keepalive_msg}")
                continue

        # Get the final result
        try:
            result = await tool_task
        except ToolDeniedReturnToPrompt:
            # Re-raise this exception immediately without wrapping in tuple
            raise
        except Exception as e:
            # Other exceptions become part of the result
            result = e
        return result, keepalive_messages

    def _create_system_prompt(self, for_first_message: bool = False) -> str:
        """Create a centralized system prompt with LLM-specific customization points."""
        # Build the system prompt using centralized template system
        base_prompt = self.system_prompt_builder.build_base_system_prompt()

        # Add LLM-specific customizations
        llm_customizations = self._get_llm_specific_instructions()

        if llm_customizations:
            final_prompt = base_prompt + "\n\n" + llm_customizations
        else:
            final_prompt = base_prompt

        return final_prompt

    def _get_llm_specific_instructions(self) -> str:
        """Override in subclasses to add LLM-specific instructions.

        This is a hook for LLM implementations to add:
        - Model-specific behavior instructions
        - API usage guidelines
        - Tool execution requirements
        - Context management specifics
        """
        return ""

    # Centralized Tool Result Integration
    # ===================================

    @abstractmethod
    def _normalize_tool_calls_to_standard_format(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert LLM-specific tool calls to standardized format.

        Each LLM implementation should convert their tool call format to:
        {
            "id": "call_123",
            "name": "tool_name",
            "arguments": {...}  # dict or JSON string
        }

        Args:
            tool_calls: LLM-specific tool call objects

        Returns:
            List of standardized tool call dicts
        """
        pass

    def _add_tool_results_to_conversation(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
        tool_results: List[str],
    ) -> List[Dict[str, Any]]:
        """Add tool results to conversation using standardized format.

        This method normalizes tool calls and integrates results consistently.
        """
        if not tool_calls or not tool_results:
            # Check if we should modify in place for session persistence
            if getattr(self, "_modify_messages_in_place", False):
                return messages  # Return original list, don't copy
            else:
                return messages.copy()

        # Validate input lengths match
        if len(tool_calls) != len(tool_results):
            logger.warning(
                f"Tool calls ({len(tool_calls)}) and results ({len(tool_results)}) count mismatch"
            )
            # Truncate to shorter length to avoid index errors
            min_length = min(len(tool_calls), len(tool_results))
            tool_calls = tool_calls[:min_length]
            tool_results = tool_results[:min_length]

        # Normalize tool calls to standard format
        normalized_calls = self._normalize_tool_calls_to_standard_format(tool_calls)

        # Build standardized tool result messages
        # Check if we should modify in place for session persistence
        if getattr(self, "_modify_messages_in_place", False):
            updated_messages = messages  # Use original list for in-place modification
        else:
            updated_messages = messages.copy()  # Create copy for backward compatibility

        # Add assistant message with normalized tool calls
        if normalized_calls:
            # Convert to OpenAI-style format for assistant message
            openai_tool_calls = []
            for call in normalized_calls:
                openai_tool_calls.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": (
                                call["arguments"]
                                if isinstance(call["arguments"], str)
                                else json.dumps(call["arguments"])
                            ),
                        },
                    }
                )

            updated_messages.append(
                {"role": "assistant", "content": "", "tool_calls": openai_tool_calls}
            )

        # Add tool result messages
        for call, result in zip(normalized_calls, tool_results):
            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": result,
                }
            )

        return updated_messages

    # Centralized Agent.md Integration
    # ===============================

    # Centralized Text Processing Utilities
    # ====================================

    @abstractmethod
    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract text that appears before tool calls in provider-specific format.

        Must be implemented by subclasses to handle their specific tool call formats.
        """
        pass

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: Optional[bool] = None,
        modify_messages_in_place: bool = False,
    ) -> Union[str, Any]:
        """Generate a response using the specific LLM. Centralized implementation with subagent yielding."""
        # For subagents, use interactive=False to avoid terminal formatting issues
        interactive = not self.is_subagent

        # Use provided stream parameter, or fall back to instance/default behavior
        # Subagents should not stream to avoid generator issues
        if stream is not None:
            # Use explicitly provided stream parameter
            use_stream = stream and not self.is_subagent
        else:
            # Fall back to instance attribute or default
            use_stream = getattr(self, "stream", True) and not self.is_subagent

        # Store the modify_messages_in_place flag for use by implementations
        self._modify_messages_in_place = modify_messages_in_place

        # Call the concrete implementation's _generate_completion method
        tools_list = (
            self.convert_tools_to_llm_format() if self.available_tools else None
        )
        return await self._generate_completion(
            messages, tools_list, use_stream, interactive
        )

    # Tool conversion and parsing helper methods
    def normalize_tool_name(self, tool_key: str) -> str:
        """Normalize tool name by replacing colons with underscores."""
        return self.tool_schema.normalize_tool_name(tool_key)

    def generate_default_description(self, tool_info: dict) -> str:
        """Generate a default description for a tool if none exists."""
        return self.tool_schema.generate_default_description(tool_info)

    def get_tool_schema(self, tool_info: dict) -> dict:
        """Get tool schema with fallback to basic object schema."""
        return self.tool_schema.get_tool_schema(tool_info)

    def validate_json_arguments(self, args_json: str) -> bool:
        """Validate that a string contains valid JSON."""
        return self.tool_schema.validate_json_arguments(args_json)

    def validate_tool_name(self, tool_name: str) -> bool:
        """Validate tool name format."""
        return self.tool_schema.validate_tool_name(tool_name)

    def create_tool_call_object(self, name: str, args: str, call_id: str = None):
        """Create a standardized tool call object."""
        return self.tool_schema.create_tool_call_object(name, args, call_id)

    # Centralized Client Initialization
    # =================================

    def _initialize_llm_client(self):
        """Centralized LLM client initialization with common patterns."""
        try:
            # Get provider-specific config
            provider_config = self._get_provider_config()

            # Set streaming preference
            self.stream = self._get_streaming_preference(provider_config)

            # Calculate timeout
            timeout_seconds = self._calculate_timeout(provider_config)

            # Initialize client with error handling
            self._client = self._create_llm_client(provider_config, timeout_seconds)

            # Log successful initialization
            self._log_successful_initialization(provider_config)

        except Exception as e:
            self._handle_client_initialization_error(e)

    def _log_successful_initialization(self, provider_config):
        """Common logging pattern for successful initialization."""
        model_name = getattr(provider_config, "model", "unknown")
        provider_name = self.__class__.__name__.replace("MCP", "").replace("Host", "")
        logger.info(f"Initialized MCP {provider_name} Host with model: {model_name}")

    def _handle_client_initialization_error(self, error: Exception):
        """Common error handling for client initialization failures."""
        provider_name = self.__class__.__name__.replace("MCP", "").replace("Host", "")
        logger.error(f"Failed to initialize {provider_name} client: {error}")
        raise

    @abstractmethod
    def _get_provider_config(self):
        """Get provider-specific configuration. Must implement in subclass."""
        pass

    @abstractmethod
    def _get_streaming_preference(self, provider_config) -> bool:
        """Get streaming preference for this provider. Must implement in subclass."""
        pass

    @abstractmethod
    def _calculate_timeout(self, provider_config) -> float:
        """Calculate timeout based on provider and model. Must implement in subclass."""
        pass

    @abstractmethod
    def _create_llm_client(self, provider_config, timeout_seconds):
        """Create the actual LLM client. Must implement in subclass."""
        pass

    @abstractmethod
    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to the specific LLM's format. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from the LLM response. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def _generate_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        interactive: bool = True,
    ) -> Any:
        """Generate completion using the specific LLM. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_current_runtime_model(self) -> str:
        """Get the actual model being used at runtime. Must be implemented by subclasses."""
        pass

    async def handle_tool_execution(
        self,
        tool_calls: List[Any],
        messages: List[Dict[str, Any]],
        interactive: bool = True,
        streaming_mode: bool = False,
    ) -> List[Dict[str, Any]]:
        """Centralized tool execution handler.

        This method handles:
        1. Displaying buffered text before tools
        2. Showing tool execution start message
        3. Executing tools with proper error handling
        4. Updating conversation with results

        Returns updated messages list with tool results added.
        Raises ToolDeniedReturnToPrompt if user denies permission.
        """
        try:
            # Display buffered text before tool execution if interactive
            if interactive and hasattr(self, "_text_buffer"):
                text_buffer = getattr(self, "_text_buffer", "")
                if text_buffer.strip():
                    formatted_response = self.formatter.format_markdown(text_buffer)
                    formatted_response = formatted_response.replace("\n", "\r\n")
                    print(f"\r\x1b[K\r\nAssistant: {formatted_response}")
                    self._text_buffer = ""

            # Display tool execution start
            if interactive and not self.is_subagent:
                print(
                    f"\r\n{self.formatter.display_tool_execution_start(len(tool_calls), self.is_subagent, interactive=True)}"
                )

            # Execute the tools (this will raise ToolDeniedReturnToPrompt if denied)
            function_results, _ = await self.execute_function_calls(
                tool_calls, interactive=interactive, streaming_mode=streaming_mode
            )

            # Add tool results to the conversation
            updated_messages = self._add_tool_results_to_conversation(
                messages, tool_calls, function_results
            )

            return updated_messages

        except ToolDeniedReturnToPrompt:
            # Clear any buffered content
            if hasattr(self, "_text_buffer"):
                self._text_buffer = ""
            # Clear the last line that might have tool execution start message
            if interactive:
                print("\r\x1b[K", end="", flush=True)
            # Re-raise to bubble up
            raise

    async def _handle_streaming_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]],
        interactive: bool,
    ) -> Any:
        """Handle streaming chat completion with tool execution."""
        # Generate streaming response
        response = await self._generate_completion(messages, tools, stream=True)

        # For streaming, we need to collect the full response first
        if hasattr(response, "__aiter__"):
            # It's an async generator, collect the full response
            full_content = ""
            collected_response = None

            async for chunk in response:
                if isinstance(chunk, str):
                    full_content += chunk
                else:
                    # Store the last non-string chunk as it may contain tool calls
                    collected_response = chunk

            # If we collected a response object, parse tool calls from it
            if collected_response:
                tool_calls = self.parse_tool_calls(collected_response)
            else:
                # Try to parse tool calls from the full content string
                tool_calls = self._extract_tool_calls_from_content(full_content)

            if tool_calls:
                # Execute tools and continue conversation
                return await self._execute_tools_and_continue(
                    messages, full_content, tool_calls, True, interactive
                )
            else:
                # No tool calls, return the full content
                return full_content
        else:
            # Not a generator, handle as non-streaming
            return await self._handle_non_streaming_chat_completion(
                messages, tools, interactive
            )

    async def _handle_non_streaming_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]],
        interactive: bool,
    ) -> Any:
        """Handle non-streaming chat completion with tool execution."""
        # Generate completion
        response = await self._generate_completion(messages, tools, stream=False)

        # Parse tool calls from response
        tool_calls = self.parse_tool_calls(response)

        if tool_calls:
            # Execute tools and continue conversation
            return await self._execute_tools_and_continue(
                messages, response, tool_calls, False, interactive
            )
        else:
            # No tool calls, return the response
            return self._extract_content_from_response(response)

    def _extract_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from content string. Override in subclasses if needed."""
        # Default implementation - try to find JSON-like tool calls
        import json

        tool_calls = []
        # Look for function call patterns in the content
        # This is a simple fallback - subclasses should override for better parsing
        function_pattern = r"function_call\s*:\s*({[^}]+})"
        matches = re.findall(function_pattern, content)

        for match in matches:
            try:
                call_data = json.loads(match)
                if "name" in call_data:
                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "function": {
                                "name": call_data["name"],
                                "arguments": call_data.get("arguments", {}),
                            },
                        }
                    )
            except:
                pass

        return tool_calls

    def _extract_content_from_response(self, response: Any) -> str:
        """Extract text content from response. Override in subclasses for model-specific parsing."""
        if isinstance(response, str):
            return response

        # Try common response formats
        if hasattr(response, "choices") and response.choices:
            if hasattr(response.choices[0], "message"):
                return response.choices[0].message.content or ""

        if hasattr(response, "candidates") and response.candidates:
            if hasattr(response.candidates[0], "content"):
                if hasattr(response.candidates[0].content, "parts"):
                    parts = response.candidates[0].content.parts
                    text_parts = [
                        part.text
                        for part in parts
                        if hasattr(part, "text") and part.text
                    ]
                    return "".join(text_parts)

        return str(response)

    async def _execute_tools_and_continue(
        self,
        messages: List[Dict[str, Any]],
        response: Any,
        tool_calls: List[Dict[str, Any]],
        stream: bool,
        interactive: bool,
    ) -> Any:
        """Execute tool calls and continue the conversation."""
        # Extract text content from response for assistant message
        response_content = self._extract_content_from_response(response)

        # Add assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": response_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Execute tools in parallel
        tool_results = await asyncio.gather(
            *[self._execute_single_tool(tool_call) for tool_call in tool_calls],
            return_exceptions=True,
        )

        # Convert results to strings
        string_results = []
        for result in tool_results:
            if isinstance(result, Exception):
                string_results.append(f"Error: {str(result)}")
            else:
                string_results.append(str(result))

        # Let the model-specific implementation handle how to add tool results to conversation
        messages = self._add_tool_results_to_conversation(
            messages, tool_calls, string_results
        )

        # Generate next completion with tool results
        tools = self.convert_tools_to_llm_format() if self.available_tools else None
        next_response = await self._generate_completion(messages, tools, stream)

        # Check for more tool calls
        next_tool_calls = self.parse_tool_calls(next_response)

        if next_tool_calls:
            # Continue recursively if there are more tool calls
            return await self._execute_tools_and_continue(
                messages, next_response, next_tool_calls, stream, interactive
            )
        else:
            # No more tool calls, return final response
            return self._extract_content_from_response(next_response)

    async def _execute_single_tool(self, tool_call: Any) -> str:
        """Execute a single tool call."""
        try:
            # Handle different tool call formats (dict vs SimpleNamespace)
            if hasattr(tool_call, "get"):
                # Dictionary format (DeepSeek)
                function_name = tool_call.get("function", {}).get("name", "")
                arguments = tool_call.get("function", {}).get("arguments", {})
            else:
                # SimpleNamespace format (Gemini)
                function_name = getattr(tool_call, "name", "")
                arguments = getattr(tool_call, "args", {})

            if isinstance(arguments, str):
                import json

                arguments = json.loads(arguments)

            # Create a simple namespace object that execute_function_calls expects
            from types import SimpleNamespace

            function_call = SimpleNamespace()
            function_call.name = function_name
            function_call.args = arguments

            # Use centralized tool execution
            results, outputs = await self.execute_function_calls([function_call])
            return results[0] if results else ""

        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {e}")
            return f"Error executing tool: {str(e)}"

    async def interactive_chat(
        self, input_handler, existing_messages: List[Dict[str, Any]] = None
    ):
        """Interactive chat session with shared functionality."""
        from cli_agent.core.input_handler import InterruptibleInput

        # Store input handler for tool permission prompts
        self._input_handler = input_handler

        messages = existing_messages or []
        current_task = None

        print(
            "Starting interactive chat. Type /quit or /exit to end, /tools to list available tools."
        )
        print(
            "Use /help for slash commands. Press ESC at any time to interrupt operations.\n"
        )

        while True:
            try:
                # Cancel any pending task if interrupted
                if (
                    input_handler.interrupted
                    and current_task
                    and not current_task.done()
                ):
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                    input_handler.interrupted = False
                    current_task = None
                    continue

                # Check for subagent interruption before getting input
                if input_handler.interrupted and self.subagent_manager:
                    active_count = self.subagent_manager.get_active_count()
                    if active_count > 0:
                        print(f"\n🛑 Interrupting {active_count} active subagent(s)...")
                        await self.subagent_manager.terminate_all()
                        print("✅ All subagents terminated. Returning to prompt.")
                        input_handler.interrupted = False
                        continue

                # Get user input with smart multiline detection, but check for subagents first
                prompt_text = "You: "
                if (
                    self.subagent_manager
                    and self.subagent_manager.get_active_count() > 0
                ):
                    active_tasks = self.subagent_manager.get_active_task_ids()
                    prompt_text = f"You (🤖 {len(active_tasks)} subagents active - ESC to cancel): "

                user_input = input_handler.get_multiline_input(prompt_text)

                if user_input is None:  # Interrupted
                    # Check if we should terminate subagents
                    if (
                        self.subagent_manager
                        and self.subagent_manager.get_active_count() > 0
                    ):
                        active_count = self.subagent_manager.get_active_count()
                        print(f"\n🛑 Interrupting {active_count} active subagent(s)...")
                        await self.subagent_manager.terminate_all()
                        print("✅ All subagents terminated. Returning to prompt.")
                    elif current_task and not current_task.done():
                        current_task.cancel()
                        print("🛑 Operation cancelled by user")

                    input_handler.interrupted = False
                    current_task = None
                    continue

                # Handle slash commands
                if user_input.strip().startswith("/"):
                    try:
                        slash_response = await self.slash_commands.handle_slash_command(
                            user_input.strip(), messages
                        )
                        if slash_response:
                            # Handle special command responses
                            if isinstance(slash_response, dict):
                                if "compacted_messages" in slash_response:
                                    print(f"\n{slash_response['status']}\n")
                                    messages[:] = slash_response[
                                        "compacted_messages"
                                    ]  # Update messages in place
                                elif "clear_messages" in slash_response:
                                    print(f"\n{slash_response['status']}\n")
                                    messages.clear()  # Clear the local messages list
                                elif "quit" in slash_response:
                                    print(f"\n{slash_response['status']}")
                                    break  # Exit the chat loop
                                elif "reload_host" in slash_response:
                                    print(f"\n{slash_response['status']}")
                                    return {
                                        "reload_host": slash_response["reload_host"],
                                        "messages": messages,
                                    }
                                elif "send_to_llm" in slash_response:
                                    # Special case: send the content to LLM for processing
                                    if "status" in slash_response:
                                        print(f"\n{slash_response['status']}\n")
                                    user_input = slash_response["send_to_llm"]
                                    # Don't continue - fall through to normal LLM processing
                                else:
                                    print(
                                        f"\n{slash_response.get('status', str(slash_response))}\n"
                                    )
                            else:
                                print(f"\n{slash_response}\n")
                            # Only continue if we're not sending to LLM
                            if not (
                                isinstance(slash_response, dict)
                                and "send_to_llm" in slash_response
                            ):
                                continue
                    except Exception as e:
                        print(f"\nError handling slash command: {e}\n")
                        continue

                if not user_input.strip():
                    # Empty input, just continue
                    continue

                # Add user message
                messages.append({"role": "user", "content": user_input})

                # Check if this is the first message and enhance with AGENT.md if so
                is_first_message = len(messages) == 1
                if is_first_message:
                    enhanced_messages = self.system_prompt_builder.enhance_first_message_with_agent_md(
                        messages.copy()  # Use a copy so original messages list can be modified in-place
                    )
                    # For first message, we need to use the enhanced version for LLM but ensure
                    # tool execution results are added to the original messages list
                    working_messages = (
                        messages  # The list that will receive tool execution results
                    )
                else:
                    enhanced_messages = messages
                    working_messages = messages

                # Reset tool execution state for new message
                self._tool_execution_started = False
                self._post_tool_buffer = ""
                self._text_buffer = ""

                # Show thinking message
                print("\nThinking...")

                # Create response task - enable in-place message modification for session persistence
                tools_list = self.convert_tools_to_llm_format()
                current_task = asyncio.create_task(
                    self.generate_response(
                        working_messages, tools_list, modify_messages_in_place=True
                    )
                )

                # Wait for response with simple interruption handling
                try:
                    await current_task
                except asyncio.CancelledError:
                    print("\n🛑 Request cancelled")
                    input_handler.interrupted = False
                    current_task = None
                    continue
                except Exception as e:
                    # Check if this is a tool permission denial that should return to prompt
                    from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

                    if isinstance(e, ToolDeniedReturnToPrompt):
                        # Tool denial message already printed by permission manager
                        # Clear any partial output that might have been displayed
                        print("\r\x1b[K", end="", flush=True)  # Clear current line
                        # Remove the last user message since we're not processing it
                        if messages and messages[-1]["role"] == "user":
                            messages.pop()
                        # Clear any buffered text
                        self._text_buffer = ""
                        current_task = None
                        # Show clean prompt on next line
                        print()  # New line for clean prompt
                        continue
                    else:
                        print(f"\nError generating response: {e}")
                        current_task = None
                        continue

                # Get the response
                response = current_task.result()
                current_task = None

                if hasattr(response, "__aiter__"):
                    # Streaming response
                    print("\nAssistant (press ESC to interrupt):")
                    sys.stdout.flush()
                    full_response = ""
                    # Use instance variables that are reset per message
                    tool_execution_started = getattr(
                        self, "_tool_execution_started", False
                    )
                    post_tool_buffer = getattr(self, "_post_tool_buffer", "")

                    # Set up non-blocking input monitoring
                    stdin_fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(stdin_fd)
                    tty.setraw(stdin_fd)

                    interrupted = False
                    try:
                        async for chunk in response:
                            # Check for escape key on each chunk
                            if select.select([sys.stdin], [], [], 0)[
                                0
                            ]:  # Non-blocking check
                                char = sys.stdin.read(1)
                                if char == "\x1b":  # Escape key
                                    interrupted = True
                                    break

                            # Check for interruption flag
                            if input_handler.interrupted:
                                interrupted = True
                                input_handler.interrupted = False
                                break

                            if isinstance(chunk, str):
                                # Check if this chunk indicates tool calls are starting
                                tool_start = (
                                    "🔧 Using" in chunk
                                    or "🔧 Executing" in chunk
                                    or "Tool 1" in chunk
                                    or "tool_calls" in chunk
                                    or 'function":' in chunk
                                    or "Executing " in chunk
                                )

                                # Check if this chunk indicates tool processing is complete
                                tool_processing_end = (
                                    "⚙️ Processing tool results..." in chunk
                                    or "✅ Result:" in chunk
                                    or "❌ Error:" in chunk
                                    or ("Tool " in chunk and "SUCCESS:" in chunk)
                                    or ("Tool " in chunk and "FAILED:" in chunk)
                                    or chunk.strip().endswith(
                                        "FAILED: Tool execution cancelled"
                                    )
                                )

                                if tool_start and not tool_execution_started:
                                    # Tool execution starting - format and display any buffered content first
                                    tool_execution_started = True
                                    self._tool_execution_started = True

                                    # Get the most current buffer content
                                    current_buffer = getattr(self, "_text_buffer", "")

                                    # Display buffered pre-tool text if any
                                    if current_buffer.strip():
                                        # Emit streaming JSON event before formatting
                                        if (
                                            hasattr(self, "streaming_json_callback")
                                            and self.streaming_json_callback
                                        ):
                                            self.streaming_json_callback(current_buffer)

                                        formatted_response = (
                                            self.formatter.format_markdown(
                                                current_buffer
                                            )
                                        )
                                        display_response = formatted_response.replace(
                                            "\n", "\r\n"
                                        )
                                        print(display_response, end="", flush=True)
                                        print("\r\n", end="", flush=True)  # Separator

                                    # Clear the text buffer since we displayed it
                                    self._text_buffer = ""

                                # Handle tool processing completion
                                if tool_processing_end and tool_execution_started:
                                    # Tool processing complete - stream this chunk then switch back to buffering mode
                                    display_chunk = chunk.replace("\n", "\r\n")
                                    print(display_chunk, end="", flush=True)
                                    full_response += chunk

                                    # Reset for post-tool text buffering
                                    tool_execution_started = False
                                    self._tool_execution_started = False
                                    self._text_buffer = ""
                                    continue  # Don't process this chunk again below

                                # Fallback: if we're in tool execution mode but see regular text patterns,
                                # assume tool execution is complete and reset to buffering mode
                                if (
                                    tool_execution_started
                                    and not tool_start
                                    and not tool_processing_end
                                ):
                                    # Check if this looks like regular response text (not tool-related)
                                    regular_text_patterns = [
                                        chunk.strip().startswith(
                                            (
                                                "I'll",
                                                "I will",
                                                "I can",
                                                "I need",
                                                "Let me",
                                                "Now",
                                                "The",
                                                "This",
                                                "Based on",
                                            )
                                        ),
                                        len(chunk.strip()) > 20
                                        and not any(
                                            pattern in chunk
                                            for pattern in [
                                                "Tool",
                                                "Executing",
                                                "🔧",
                                                "✅",
                                                "❌",
                                                "function",
                                                "arguments",
                                            ]
                                        ),
                                        chunk.count(".") > 0
                                        and chunk.count(" ")
                                        > 5,  # Likely sentence structure
                                    ]

                                    if any(regular_text_patterns):
                                        # We're seeing regular text, reset tool execution state
                                        tool_execution_started = False
                                        self._tool_execution_started = False
                                        self._text_buffer = ""

                                # Output behavior based on current mode
                                force_buffering = getattr(
                                    self, "_force_buffering_for_streaming_json", False
                                )

                                if tool_execution_started and not force_buffering:
                                    # In tool execution mode - stream directly (unless forcing buffer)
                                    display_chunk = chunk.replace("\n", "\r\n")
                                    print(display_chunk, end="", flush=True)
                                    full_response += chunk
                                else:
                                    # In text mode (pre-tool or post-tool) - buffer for markdown formatting
                                    # OR when streaming JSON (to hit buffer emission callbacks)
                                    self._text_buffer = (
                                        getattr(self, "_text_buffer", "") + chunk
                                    )
                                    print(
                                        f"DEBUG: Added to buffer, now: {repr(self._text_buffer[:50])}",
                                        file=sys.stderr,
                                    )
                                    full_response += chunk
                            else:
                                # Handle any non-string chunks if needed
                                chunk_str = str(chunk)
                                formatted_chunk = (
                                    self.formatting_utils.format_chunk_safely(chunk_str)
                                )
                                display_chunk = formatted_chunk.replace("\n", "\r\n")
                                print(display_chunk, end="", flush=True)
                                full_response += chunk_str
                    finally:
                        # Always restore terminal settings first
                        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)

                        # Handle any remaining buffered text and final formatting
                        if not interrupted:
                            # Check if there's buffered text that needs to be displayed
                            remaining_text_buffer = getattr(self, "_text_buffer", "")
                            if remaining_text_buffer.strip():
                                # Emit streaming JSON event before formatting
                                if (
                                    hasattr(self, "streaming_json_callback")
                                    and self.streaming_json_callback
                                ):
                                    print(
                                        f"DEBUG: Buffer callback with: {repr(remaining_text_buffer[:50])}",
                                        file=sys.stderr,
                                    )
                                    self.streaming_json_callback(remaining_text_buffer)

                                # Format and display the remaining buffered text
                                formatted_response = self.formatter.format_markdown(
                                    remaining_text_buffer
                                )
                                display_response = formatted_response.replace(
                                    "\n", "\r\n"
                                )
                                print(display_response, end="", flush=True)
                                print()  # Final newline
                            elif full_response and not tool_execution_started:
                                # No tools were executed and no separate buffer - format the entire response
                                # Emit streaming JSON event before formatting the complete response
                                if (
                                    hasattr(self, "streaming_json_callback")
                                    and self.streaming_json_callback
                                ):
                                    print(
                                        f"DEBUG: Full response callback with: {repr(full_response[:50])}",
                                        file=sys.stderr,
                                    )
                                    self.streaming_json_callback(full_response)

                                print()  # New line after streaming
                                print(
                                    f"\033[1A\033[KAssistant: ", end=""
                                )  # Move up one line, clear it, write "Assistant: "
                                formatted_response = self.formatter.format_markdown(
                                    full_response
                                )
                                print(formatted_response)
                                sys.stdout.flush()
                            else:
                                # Tools were executed - just add a newline (content already displayed)
                                print()
                        elif interrupted:
                            print("\n🛑 Streaming interrupted by user")
                            sys.stdout.flush()
                        else:
                            print()  # Just a newline if no response

                    # Add assistant response to messages
                    if full_response:  # Only add if not interrupted
                        messages.append({"role": "assistant", "content": full_response})
                else:
                    # Non-streaming response with markdown formatting
                    formatted_response = self.formatter.format_markdown(str(response))
                    print(f"\nAssistant: {formatted_response}")
                    messages.append({"role": "assistant", "content": str(response)})

            except KeyboardInterrupt:
                # Move to beginning of line and clear, then print exit message
                sys.stdout.write("\r\x1b[KExiting...\n")
                sys.stdout.flush()
                break
            except Exception as e:
                print(f"\nError: {e}")

        # Return the updated messages list for session saving
        return messages

    # Centralized Tool Call Processing Methods
    # ========================================

    def _extract_and_normalize_tool_calls(
        self, response: Any, accumulated_content: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extract tool calls from LLM response and normalize to common format.
        Delegates to LLM-specific parsing methods via parse_tool_calls().

        Returns list of normalized tool call dicts with keys:
        - id: tool call identifier
        - function: dict with 'name' and 'arguments' keys
        """
        # Use the existing parse_tool_calls method (implemented by subclasses)
        # This maintains compatibility with existing implementations
        tool_calls = self.parse_tool_calls(response)

        # Normalize to consistent format if needed
        normalized_calls = []
        for i, call in enumerate(tool_calls):
            if isinstance(call, dict):
                # Already in dict format (DeepSeek style)
                if "function" in call:
                    # Ensure type field is present for DeepSeek compatibility
                    if "type" not in call:
                        call = call.copy()
                        call["type"] = "function"
                    normalized_calls.append(call)
                elif "name" in call and "args" in call:
                    # Convert from simple format to standard format
                    normalized_calls.append(
                        {
                            "id": f"call_{i}_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call.get("args", {}),
                            },
                        }
                    )
            elif hasattr(call, "function"):
                # SimpleNamespace format (already standardized)
                normalized_calls.append(
                    {
                        "id": getattr(call, "id", f"call_{i}_{int(time.time())}"),
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )
            elif hasattr(call, "name"):
                # Simple object format
                normalized_calls.append(
                    {
                        "id": getattr(call, "id", f"call_{i}_{int(time.time())}"),
                        "function": {
                            "name": call.name,
                            "arguments": getattr(call, "args", {}),
                        },
                    }
                )

        return normalized_calls

    def _validate_and_convert_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        streaming_mode: bool = False,
        error_handler=None,
    ) -> tuple[List[Any], List[Dict[str, Any]]]:
        """
        Validate tool calls and convert to function call format.

        Args:
            tool_calls: List of normalized tool call dicts
            streaming_mode: Whether in streaming mode (affects error handling)
            error_handler: Optional callable for handling errors (streaming mode)

        Returns:
            - List of valid function calls ready for execution (SimpleNamespace objects)
            - List of error messages for invalid calls
        """
        from types import SimpleNamespace

        function_calls = []
        error_messages = []

        for tool_call in tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                call_id = tool_call.get("id", f"call_{len(function_calls)}")

                # Parse arguments if they're a string
                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                    except json.JSONDecodeError as e:
                        # Handle JSON parsing errors
                        error_content = (
                            f"Error parsing arguments for {function_name}: {e}\\n"
                            "⚠️  Command failed - take this into account for your next action."
                        )
                        error_msg = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": error_content,
                        }
                        error_messages.append(error_msg)

                        # Handle streaming mode error display
                        if streaming_mode and error_handler:
                            error_handler(
                                f"\\n🔧 Tool parsing error: {error_content}\\n"
                            )

                        continue  # Skip this tool call
                else:
                    parsed_args = arguments

                # Create function call object
                function_call = SimpleNamespace()
                function_call.name = function_name
                function_call.args = parsed_args
                function_calls.append(function_call)

            except (KeyError, TypeError) as e:
                # Handle malformed tool call structure
                error_content = f"Malformed tool call structure: {e}"
                error_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.get(
                        "id", f"call_error_{len(error_messages)}"
                    ),
                    "content": error_content,
                }
                error_messages.append(error_msg)

                if streaming_mode and error_handler:
                    error_handler(f"\\n🔧 Tool structure error: {error_content}\\n")

        return function_calls, error_messages

    def _build_tool_result_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[str],
        error_messages: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build standardized tool result messages for conversation."""
        messages = []

        # Add error messages first
        if error_messages:
            messages.extend(error_messages)

        # Add successful tool result messages
        for i, (tool_call, result) in enumerate(zip(tool_calls, tool_results)):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{i}"),
                    "content": result,
                }
            )

        return messages

    def _display_tool_execution_info(
        self,
        tool_calls: List[Dict[str, Any]],
        interactive: bool,
        streaming_mode: bool,
    ) -> None:
        """Display tool execution information to user."""
        if not interactive or not tool_calls:
            return

        if streaming_mode:
            print(f"\\n🔧 Executing {len(tool_calls)} tool(s)...", flush=True)
        else:
            print(f"\\n🔧 Executing {len(tool_calls)} tool(s):", flush=True)
            for i, tc in enumerate(tool_calls, 1):
                tool_name = tc["function"]["name"].replace("_", ":", 1)
                try:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    args_preview = (
                        str(args)[:100] + "..." if len(str(args)) > 100 else str(args)
                    )
                    print(f"   {i}. {tool_name} - {args_preview}", flush=True)
                except Exception:
                    print(f"   {i}. {tool_name}", flush=True)

    async def _process_tool_calls_centralized(
        self,
        response: Any,
        current_messages: List[Dict[str, Any]],
        original_messages: List[Dict[str, Any]],
        interactive: bool = True,
        streaming_mode: bool = False,
        accumulated_content: str = "",
    ) -> tuple[List[Dict[str, Any]], Optional[Dict], bool]:
        """
        Centralized tool call processing coordinator.

        Args:
            response: LLM response object
            current_messages: Current conversation messages
            original_messages: Original messages before tool execution
            interactive: Whether in interactive mode
            streaming_mode: Whether in streaming mode
            accumulated_content: Accumulated response content (for streaming)

        Returns:
            - Updated messages list
            - Continuation message for subagent coordination (if any)
            - Whether tool calls were found and processed
        """
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        # Extract and normalize tool calls
        tool_calls = self._extract_and_normalize_tool_calls(
            response, accumulated_content
        )

        logger.debug(f"Extracted tool calls in centralized processing: {tool_calls}")
        logger.debug(
            f"Number of tool calls found: {len(tool_calls) if tool_calls else 0}"
        )

        if not tool_calls:
            logger.debug("No tool calls found - returning early")
            return current_messages, None, False

        # Display tool execution info
        self._display_tool_execution_info(tool_calls, interactive, streaming_mode)

        # Validate and convert tool calls
        streaming_error_handler = None
        if streaming_mode:
            # Create error handler for streaming mode
            def error_handler(error_text):
                print(error_text, flush=True)

            streaming_error_handler = error_handler

        function_calls, error_messages = self._validate_and_convert_tool_calls(
            tool_calls, streaming_mode, streaming_error_handler
        )

        # Add assistant message with tool calls to conversation
        response_content = self._extract_content_from_response(response)
        if accumulated_content:
            response_content = accumulated_content

        # Extract and display text that appears before tool calls
        if interactive and response_content and function_calls:
            text_before_tools = self._extract_text_before_tool_calls(response_content)
            if text_before_tools:
                formatted_text = self.formatter.format_markdown(text_before_tools)
                print(f"\n{formatted_text}", flush=True)
                # Use only the extracted text for conversation history
                response_content = text_before_tools

        current_messages.append(
            {
                "role": "assistant",
                "content": response_content or "",
                "tool_calls": tool_calls,
            }
        )

        # Add error messages to conversation
        current_messages.extend(error_messages)

        # Execute valid function calls if any
        tool_results = []
        tool_output = []

        if function_calls:
            try:
                tool_results, tool_output = await self.execute_function_calls(
                    function_calls,
                    interactive=interactive,
                    streaming_mode=streaming_mode,
                )

                # Display tool output in streaming mode
                if streaming_mode and interactive:
                    for output in tool_output:
                        print(f"{output}\\n", flush=True)

            except ToolDeniedReturnToPrompt:
                # Re-raise permission denials to exit immediately
                raise

        # Handle subagent coordination using existing centralized logic
        continuation_message = (
            await self.subagent_coordinator.handle_subagent_coordination(
                tool_calls,
                original_messages,
                interactive=interactive,
                streaming_mode=streaming_mode,
            )
        )

        if continuation_message:
            return current_messages, continuation_message, True

        # Add tool results to conversation
        if tool_results:
            tool_result_messages = self._build_tool_result_messages(
                [
                    tc
                    for tc in tool_calls
                    if tc["function"]["name"] in [fc.name for fc in function_calls]
                ],
                tool_results,
            )
            current_messages.extend(tool_result_messages)

        return current_messages, None, True
