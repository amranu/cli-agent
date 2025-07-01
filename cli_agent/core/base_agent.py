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

from cli_agent.core.api_client_manager import APIClientManager
from cli_agent.core.builtin_tool_executor import BuiltinToolExecutor
from cli_agent.core.chat_interface import ChatInterface
from cli_agent.core.formatting import ResponseFormatter
from cli_agent.core.formatting_utils import FormattingUtils
from cli_agent.core.message_processor import MessageProcessor
from cli_agent.core.response_handler import ResponseHandler
from cli_agent.core.slash_commands import SlashCommandManager
from cli_agent.core.subagent_coordinator import SubagentCoordinator
from cli_agent.core.system_prompt_builder import SystemPromptBuilder
from cli_agent.core.token_manager import TokenManager
from cli_agent.core.tool_call_processor import ToolCallProcessor
from cli_agent.core.tool_execution_engine import ToolExecutionEngine
from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt
from cli_agent.core.tool_schema import ToolSchemaManager
from cli_agent.tools.builtin_tools import get_all_builtin_tools
from cli_agent.utils.tool_name_utils import ToolNameUtils
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

        # Centralized subagent management system
        if not is_subagent:
            try:
                # Import SubagentManager from the reorganized location
                from cli_agent.subagents.subagent import SubagentManager

                self.subagent_manager = SubagentManager(config, agent=self)

                # Event-driven message handling
                self.subagent_message_queue = asyncio.Queue()

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
            # Set session file in config directory for persistent approvals
            permission_config = ToolPermissionConfig()
            from config import get_config_dir
            config_dir = get_config_dir()
            sessions_dir = config_dir / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            permission_config.session_permissions_file = str(sessions_dir / ".tool_permissions.json")
            self.permission_manager = ToolPermissionManager(permission_config)
            logger.info(
                f"Initialized tool permission manager with session file: '{permission_config.session_permissions_file}'"
            )
        except ImportError as e:
            logger.warning(f"Failed to import tool permission manager: {e}")
            self.permission_manager = None

        # Initialize token manager (model name will be set after subclass initialization)
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

        # Initialize event system first
        from cli_agent.core.display_manager import DisplayManager
        from cli_agent.core.event_system import EventBus, EventEmitter

        self.event_bus = EventBus()
        self.event_emitter = EventEmitter(self.event_bus)

        # Event bus will be started when interactive_chat() is called

        # Initialize subagent coordinator after event system is available
        self.subagent_coordinator = SubagentCoordinator(self)

        # Register subagent message callback after coordinator is initialized
        if (
            not is_subagent
            and hasattr(self, "subagent_manager")
            and self.subagent_manager
        ):
            self.subagent_manager.add_message_callback(
                self.subagent_coordinator.on_subagent_message
            )

        # Initialize display manager for interactive mode
        # For subagents, use non-interactive mode to avoid double display
        self.display_manager = DisplayManager(
            self.event_bus, interactive=not is_subagent
        )

        logger.debug("Initialized event system with display manager")

        # Initialize modular components
        self.message_processor = MessageProcessor(self)
        self.tool_call_processor = ToolCallProcessor(self)
        self.api_client_manager = APIClientManager(self)
        self.response_handler = ResponseHandler(self)
        self.tool_execution_engine = ToolExecutionEngine(self)
        self.chat_interface = ChatInterface(self)
        logger.debug("Initialized modular components")
        
        # Initialize hooks system (only for main agents, not subagents)
        self.hook_manager = None
        if not is_subagent:
            try:
                hook_config = config.load_hook_config()
                if hook_config:
                    from cli_agent.core.hooks.hook_manager import HookManager
                    self.hook_manager = HookManager(hook_config, self.event_bus)
                    # Connect hook manager to event bus for notification hooks
                    self.event_bus.set_hook_manager(self.hook_manager)
                    
                    # Show user-visible confirmation of hooks loaded
                    hook_summary = self.hook_manager.get_hook_summary()
                    total_hooks = hook_summary["total_hooks"]
                    hook_types = list(hook_summary["hook_types"].keys())
                    print(f"🪝 Hooks enabled: {total_hooks} hooks loaded ({', '.join(hook_types)})")
                    logger.info("Initialized hooks system with event bus integration")
                else:
                    logger.debug("No hook configuration found, hooks disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize hooks system: {e}")
                self.hook_manager = None
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
            # Ensure emit_result is available for subagents
            logger.info(
                f"Subagent has emit_result tool: {'builtin:emit_result' in builtin_tools}"
            )
        else:
            # Remove emit_result tool for main agents (subagents only)
            if "builtin:emit_result" in builtin_tools:
                del builtin_tools["builtin:emit_result"]
                logger.info("Removed emit_result from main agent tools")

            # Remove background subagent tools if background subagents are not enabled
            if not self.config.background_subagents:
                background_tools = [
                    "builtin:task_status",
                    "builtin:task_results",
                ]
                for tool_key in background_tools:
                    if tool_key in builtin_tools:
                        del builtin_tools[tool_key]
                        logger.info(
                            f"Removed {tool_key} - background subagents disabled"
                        )

        self.available_tools.update(builtin_tools)

        # Debug: Log available tools for subagents
        if self.is_subagent:
            logger.info(
                f"Subagent initialized with {len(self.available_tools)} tools: {list(self.available_tools.keys())}"
            )

    def _normalize_tool_name(self, tool_name: str) -> str:
        """Ensure tool name is fully qualified with prefix (e.g., builtin:)."""
        if not tool_name.startswith("builtin:"):
            return f"builtin:{tool_name}"
        return tool_name

    async def _execute_builtin_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a built-in tool."""
        tool_name = self._normalize_tool_name(tool_name)
        if tool_name == "builtin:bash_execute":
            return await self.builtin_executor.bash_execute(args)
        elif tool_name == "builtin:read_file":
            return self.builtin_executor.read_file(args)
        elif tool_name == "builtin:write_file":
            return self.builtin_executor.write_file(args)
        elif tool_name == "builtin:list_directory":
            return self.builtin_executor.list_directory(args)
        elif tool_name == "builtin:get_current_directory":
            return self.builtin_executor.get_current_directory(args)
        elif tool_name == "builtin:todo_read":
            return self.builtin_executor.todo_read(args)
        elif tool_name == "builtin:todo_write":
            return self.builtin_executor.todo_write(args)
        elif tool_name == "builtin:replace_in_file":
            return self.builtin_executor.replace_in_file(args)
        elif tool_name == "builtin:multiedit":
            return self.builtin_executor.multiedit(args)
        elif tool_name == "builtin:webfetch":
            return self.builtin_executor.webfetch(args)
        elif tool_name == "builtin:websearch":
            return self.builtin_executor.websearch(args)
        elif tool_name == "builtin:glob":
            return self.builtin_executor.glob(args)
        elif tool_name == "builtin:grep":
            return self.builtin_executor.grep(args)
        elif tool_name == "builtin:task":
            return await self.builtin_executor.task(args)
        elif tool_name == "builtin:task_status":
            return self.builtin_executor.task_status(args)
        elif tool_name == "builtin:task_results":
            return self.builtin_executor.task_results(args)
        elif tool_name == "builtin:emit_result":
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
            logger.debug(f"Starting MCP server: {server_name}")

            # Construct command and args for FastMCP client
            command = server_config.command[0]
            args = server_config.command[1:] + server_config.args

            # Process environment variables to expand ${VAR} syntax
            processed_env = {}
            if server_config.env:
                import os
                import re

                for key, value in server_config.env.items():
                    if isinstance(value, str):
                        # Expand ${VAR} and $VAR patterns
                        def replace_var(match):
                            var_name = match.group(1) or match.group(2)
                            return os.environ.get(var_name, match.group(0))

                        # Handle both ${VAR} and $VAR patterns
                        expanded_value = re.sub(
                            r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                            replace_var,
                            value,
                        )
                        processed_env[key] = expanded_value
                    else:
                        processed_env[key] = value

            # Create FastMCP client with stdio transport
            transport = StdioTransport(command=command, args=args, env=processed_env)
            client = FastMCPClient(transport=transport)

            # Enter the context manager and store it for cleanup
            context_manager = client.__aenter__()
            await context_manager

            # Store the client and context manager
            self.mcp_clients[server_name] = client
            self._mcp_contexts = getattr(self, "_mcp_contexts", {})
            self._mcp_contexts[server_name] = client

            # Send initialized notification (required by MCP protocol)
            try:
                await client.send_notification("notifications/initialized")
                logger.debug(f"Sent initialized notification to {server_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to send initialized notification to {server_name}: {e}"
                )

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

        # Stop event bus processing
        if hasattr(self, "event_bus") and self.event_bus:
            await self.event_bus.stop_processing()
            logger.info("Event bus stopped")

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

            if self.config.background_subagents:
                # Background mode: minimal display
                if message.type == "result":
                    formatted = f"🔔 [BACKGROUND-{task_id}] Task completed: {message.content[:50]}..."
                elif message.type == "error":
                    formatted = f"❌ [BACKGROUND-{task_id}] Error: {message.content}"
                else:
                    # Suppress output/status messages in background mode
                    return
            else:
                # Foreground mode: full display (current behavior)
                if message.type == "output":
                    formatted = f"🤖 [SUBAGENT-{task_id}] {message.content}"
                elif message.type == "status":
                    formatted = f"📋 [SUBAGENT-{task_id}] {message.content}"
                elif message.type == "error":
                    formatted = f"❌ [SUBAGENT-{task_id}] {message.content}"
                elif message.type == "result":
                    formatted = f"✅ [SUBAGENT-{task_id}] Result: {message.content}"
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
                logger.debug(
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
        from cli_agent.core.global_interrupt import get_global_interrupt_manager

        global_interrupt_manager = get_global_interrupt_manager()

        if global_interrupt_manager.is_interrupted():
            all_tool_output.append("🛑 Tool execution interrupted by global interrupt")
            return function_results, all_tool_output

        if input_handler:
            # Check for both existing interrupted state and new interrupts
            if input_handler.interrupted or input_handler.check_for_interrupt():
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

            # Emit tool call event
            if hasattr(self, "event_emitter"):
                await self.event_emitter.emit_tool_call(
                    tool_name=tool_name,
                    tool_id=f"toolu_{i}_{tool_name}",
                    arguments=arguments,
                )

            # Generate tool use ID for tracking
            import uuid

            tool_use_id = f"toolu_{i}_{uuid.uuid4().hex[:16]}"

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

                    # Tool result handling via events (unified for all modes)

                    # Emit tool result event
                    if hasattr(self, "event_emitter"):
                        await self.event_emitter.emit_tool_result(
                            tool_name=tool_name,
                            tool_id=tool_use_id,
                            result=str(tool_result),
                            is_error=not tool_success,
                        )

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
        """Execute an MCP tool - delegates to ToolExecutionEngine."""
        return await self.tool_execution_engine.execute_mcp_tool(tool_key, arguments)

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
                # Normalize tool name for OpenAI API compatibility (replace colons with underscores)
                normalized_name = call["name"].replace(":", "_")
                openai_tool_calls.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": normalized_name,
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
            # Normalize tool name for OpenAI API compatibility (replace colons with underscores)
            normalized_name = call["name"].replace(":", "_")
            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": normalized_name,
                    "content": result,
                }
            )

        return updated_messages

    # Centralized Claude Code Format Support
    # ====================================

    def _send_claude_code_assistant_message(
        self, response_content: str, tool_calls: List[Dict[str, Any]]
    ):
        """Send assistant message in Claude Code format with combined text and tool calls."""
        import uuid

        # Create Claude Code style content array with text + tool_use blocks
        content_blocks = []

        # Add text content if present (including empty content)
        if response_content is not None:
            content_blocks.append({"type": "text", "text": response_content})

        # Add tool use blocks
        for tool_call in tool_calls:
            # Extract tool info from different formats
            if hasattr(tool_call, "get"):
                # Check if it's a parsed tool call format (direct name/arguments keys)
                if "name" in tool_call and "arguments" in tool_call:
                    # Parsed tool call format from parse_tool_calls
                    tool_name = tool_call.get("name", "")
                    tool_input = tool_call.get("arguments", {})
                    tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:20]}")
                else:
                    # Dictionary format (DeepSeek original format)
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_input = tool_call.get("function", {}).get("arguments", {})
                    tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:20]}")
            else:
                # SimpleNamespace format (DeepSeek/Gemini)
                tool_name = (
                    getattr(tool_call.function, "name", "")
                    if hasattr(tool_call, "function")
                    else getattr(tool_call, "name", "")
                )
                tool_input = (
                    getattr(tool_call.function, "arguments", {})
                    if hasattr(tool_call, "function")
                    else getattr(tool_call, "args", {})
                )
                tool_id = getattr(tool_call, "id", f"toolu_{uuid.uuid4().hex[:20]}")

            if isinstance(tool_input, str):
                import json

                try:
                    tool_input = json.loads(tool_input)
                except:
                    pass

            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            )

        # Send combined message in Claude Code format
        if content_blocks:
            message = {
                "id": f"msg_{uuid.uuid4().hex[:20]}",
                "type": "message",
                "role": "assistant",
                "model": getattr(self, "_current_model", "unknown"),
                "content": content_blocks,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "service_tier": "standard",
                },
            }

            # Send as assistant message
            from streaming_json import AssistantMessage

            session_id = getattr(self, "_current_session_id", str(uuid.uuid4()))
            msg = AssistantMessage(session_id=session_id, message=message)
            print(msg.to_json(), flush=True)

    def _send_claude_code_tool_results(
        self,
        tool_calls: List[Dict[str, Any]],
        string_results: List[str],
        tool_results: List,
    ):
        """Send tool results in Claude Code format."""
        import uuid

        for i, (tool_call, result) in enumerate(zip(tool_calls, string_results)):
            # Extract tool ID from different formats
            if hasattr(tool_call, "get"):
                tool_id = tool_call.get("id", "")
            else:
                tool_id = getattr(tool_call, "id", "")

            is_error = isinstance(tool_results[i], Exception)

            # Send tool result as user message in Claude Code format
            from streaming_json import UserMessage

            message = {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": tool_id,
                        "type": "tool_result",
                        "content": result,
                        "is_error": is_error,
                    }
                ],
            }

            session_id = getattr(self, "_current_session_id", str(uuid.uuid4()))
            msg = UserMessage(session_id=session_id, message=message)
            print(msg.to_json(), flush=True)

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
        logger.info(f"generate_response called with {len(messages)} messages")
        # For subagents, use interactive=False to avoid terminal formatting issues
        interactive = not self.is_subagent

        # Use provided stream parameter, or fall back to instance/default behavior
        # Subagents should not stream to avoid generator issues
        # Also check if the model supports streaming (e.g., o1 models don't)
        model_supports_streaming = True
        if hasattr(self, "model") and hasattr(self.model, "supports_streaming"):
            model_supports_streaming = self.model.supports_streaming

        if stream is not None:
            # Use explicitly provided stream parameter, but respect model limitations
            use_stream = stream and not self.is_subagent and model_supports_streaming
        else:
            # Fall back to instance attribute or default, but respect model limitations
            use_stream = (
                getattr(self, "stream", True)
                and not self.is_subagent
                and model_supports_streaming
            )

        # Store the modify_messages_in_place flag for use by implementations
        self._modify_messages_in_place = modify_messages_in_place

        logger.info(
            f"generate_response: model_supports_streaming={model_supports_streaming}, use_stream={use_stream}"
        )

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


    # ============================================================================
    # CENTRALIZED MESSAGE PROCESSING FRAMEWORK
    # ============================================================================

    def _preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Centralized message preprocessing pipeline - delegates to MessageProcessor."""
        return self.message_processor.preprocess_messages(messages)

    def _clean_message_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Clean messages - override in subclass for provider-specific cleaning."""
        # Default implementation: minimal cleaning
        cleaned_messages = []
        for message in messages:
            # Ensure content is always a string
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            cleaned_msg = {"role": message.get("role", "user"), "content": content}

            # Preserve tool calls if present
            if message.get("tool_calls"):
                cleaned_msg["tool_calls"] = message["tool_calls"]
            if message.get("tool_call_id"):
                cleaned_msg["tool_call_id"] = message["tool_call_id"]

            cleaned_messages.append(cleaned_msg)

        return cleaned_messages

    def _enhance_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance messages for model-specific requirements - override in subclass."""
        # Default implementation: no enhancement
        return messages

    def _convert_to_provider_format(self, messages: List[Dict[str, Any]]) -> Any:
        """Convert messages to provider format - override in subclass if needed."""
        # Default implementation: return as-is
        return messages

    # ============================================================================
    # CENTRALIZED TOOL CALL PARSING FRAMEWORK
    # ============================================================================

    def _parse_tool_calls_generic(
        self, response: Any, text_content: str = ""
    ) -> List[Dict[str, Any]]:
        """Unified tool call parsing pipeline - delegates to ToolCallProcessor."""
        return self.tool_call_processor.parse_tool_calls_generic(response, text_content)

    def _extract_structured_calls(self, response: Any) -> List[Any]:
        """Extract structured tool calls from response - override in subclass."""
        # Default implementation: empty list
        return []

    def _parse_text_based_calls(self, text_content: str) -> List[Any]:
        """Parse text-based tool calls using provider patterns - override in subclass."""
        # Default implementation: empty list
        return []

    def _normalize_tool_calls_unified(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Unified tool call normalization."""
        normalized_calls = []

        for i, tool_call in enumerate(tool_calls):
            if hasattr(tool_call, "name") and hasattr(tool_call, "args"):
                # SimpleNamespace or object format
                normalized_calls.append(
                    {
                        "id": getattr(tool_call, "id", f"call_{i}"),
                        "name": tool_call.name,
                        "arguments": tool_call.args,
                    }
                )
            elif isinstance(tool_call, dict):
                if "function" in tool_call:
                    # OpenAI-style format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call["function"].get(
                                "name", f"<missing_function_name_{i}>"
                            ),
                            "arguments": tool_call["function"].get("arguments", {}),
                        }
                    )
                else:
                    # Simple dict format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call.get("name", f"<missing_dict_name_{i}>"),
                            "arguments": tool_call.get("arguments", {}),
                        }
                    )
            else:
                # Fallback
                normalized_calls.append(
                    {
                        "id": f"call_{i}",
                        "name": str(tool_call),
                        "arguments": {},
                    }
                )

        return normalized_calls

    # ============================================================================
    # CENTRALIZED RETRY LOGIC FRAMEWORK
    # ============================================================================

    async def _make_api_request_with_retry(
        self, request_func, max_retries: int = 3, base_delay: float = 1.0
    ):
        """Generic API request with exponential backoff retry logic - delegates to APIClientManager."""
        return await self.api_client_manager.make_api_request_with_retry(
            request_func, max_retries, base_delay
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable - override in subclass for provider specifics."""
        error_str = str(error).lower()
        # Generic retryable conditions
        return (
            "timeout" in error_str
            or "network" in error_str
            or "connection" in error_str
            or "rate limit" in error_str
            or "429" in error_str
            or "502" in error_str
            or "503" in error_str
            or "504" in error_str
        )

    def _calculate_retry_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate retry delay - can be overridden for custom strategies."""
        import random

        return base_delay * (2**attempt) + random.uniform(0, 1)

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
                # Tools will be executed by ResponseHandler - just return content
                return full_content
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
            # Tools will be executed by ResponseHandler - just return content
            return self._extract_content_from_response(response)
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

    async def interactive_chat(
        self, input_handler, existing_messages: List[Dict[str, Any]] = None
    ):
        """Interactive chat session - delegates to ChatInterface."""
        return await self.chat_interface.interactive_chat(
            input_handler, existing_messages
        )

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
        logger.info(f"parse_tool_calls returned: {tool_calls}")

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
                elif "name" in call and ("args" in call or "arguments" in call):
                    # Convert from simple format to standard format
                    arguments = call.get("args") or call.get("arguments", {})
                    normalized_calls.append(
                        {
                            "id": call.get("id", f"call_{i}_{int(time.time())}"),
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": arguments,
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

        # Use event system if available
        if hasattr(self, "event_bus") and self.event_bus:
            from cli_agent.core.event_system import ToolExecutionStartEvent

            for tc in tool_calls:
                tool_name = tc["function"]["name"].replace("_", ":", 1)
                try:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    event = ToolExecutionStartEvent(
                        tool_name=tool_name,
                        tool_id=tc.get("id", "unknown"),
                        arguments=args,
                        streaming_mode=streaming_mode,
                    )
                    asyncio.create_task(self.event_bus.emit(event))
                except Exception:
                    event = ToolExecutionStartEvent(
                        tool_name=tool_name,
                        tool_id=tc.get("id", "unknown"),
                        arguments={},
                        streaming_mode=streaming_mode,
                    )
                    asyncio.create_task(self.event_bus.emit(event))
            return

        # Fallback to direct printing
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
                    # Show full arguments instead of truncating
                    print(f"   {i}. {tool_name}")
                    print(f"      Parameters: {json.dumps(args, indent=6)}", flush=True)
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

        logger.info(f"Extracted tool calls in centralized processing: {tool_calls}")
        logger.info(
            f"Number of tool calls found: {len(tool_calls) if tool_calls else 0}"
        )

        if not tool_calls:
            logger.info("No tool calls found - returning early")
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

        # Convert tool calls to proper OpenAI API format for conversation history
        api_formatted_tool_calls = []
        for tc in tool_calls:
            # Use centralized tool name extraction
            original_name = ToolNameUtils.extract_tool_name_from_call(tc)

            # Normalize tool name for OpenAI API compatibility
            normalized_name = ToolNameUtils.normalize_tool_name(original_name)

            formatted_tc = {
                "id": tc.get("id", f"call_{len(api_formatted_tool_calls)}"),
                "type": "function",
                "function": {
                    "name": normalized_name,
                    "arguments": (
                        tc.get("arguments")
                        if isinstance(tc.get("arguments"), str)
                        else json.dumps(tc.get("arguments", {}))
                    ),
                },
            }
            api_formatted_tool_calls.append(formatted_tc)

        current_messages.append(
            {
                "role": "assistant",
                "content": response_content or "",
                "tool_calls": api_formatted_tool_calls,
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
                    input_handler=getattr(self, "_input_handler", None),
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
        subagent_result = await self.subagent_coordinator.handle_subagent_coordination(
            tool_calls,
            original_messages,
            interactive=interactive,
            streaming_mode=streaming_mode,
        )

        if subagent_result:
            # Check if subagent was cancelled due to tool denial
            if subagent_result.get("cancelled"):
                from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

                # Check if we're in stream-json mode for cancellation messages too
                if (hasattr(self, 'display_manager') 
                    and hasattr(self.display_manager, 'json_handler') 
                    and self.display_manager.json_handler):
                    # For stream-json mode, send cancellation messages as assistant text
                    json_handler = self.display_manager.json_handler
                    interrupt_text = subagent_result["interrupt_msg"].strip()
                    completion_text = subagent_result["completion_msg"].strip()
                    
                    if interrupt_text:
                        json_handler.send_assistant_text(interrupt_text)
                    if completion_text:
                        json_handler.send_assistant_text(completion_text)
                else:
                    print(subagent_result["interrupt_msg"], flush=True)
                    print(subagent_result["completion_msg"], flush=True)
                # Raise exception to return to prompt immediately
                raise ToolDeniedReturnToPrompt("Subagent cancelled due to tool denial")

            # Handle normal subagent completion status messages centrally based on mode
            if streaming_mode:
                # For streaming mode, store messages for the caller to yield
                subagent_result["_should_yield_messages"] = True
            elif interactive:
                # Check if we're in stream-json mode
                if (hasattr(self, 'display_manager') 
                    and hasattr(self.display_manager, 'json_handler') 
                    and self.display_manager.json_handler):
                    # For stream-json mode, send messages as assistant text
                    json_handler = self.display_manager.json_handler
                    # Strip the \r\n from messages for clean JSON output
                    interrupt_text = subagent_result["interrupt_msg"].strip()
                    completion_text = subagent_result["completion_msg"].strip()
                    restart_text = subagent_result["restart_msg"].strip()
                    
                    if interrupt_text:
                        json_handler.send_assistant_text(interrupt_text)
                    if completion_text:
                        json_handler.send_assistant_text(completion_text)
                    if restart_text:
                        json_handler.send_assistant_text(restart_text)
                else:
                    # For non-streaming interactive mode, print messages immediately
                    print(subagent_result["interrupt_msg"], flush=True)
                    print(subagent_result["completion_msg"], flush=True)
                    print(subagent_result["restart_msg"], flush=True)

            return current_messages, subagent_result, True

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

    # ============================================================================
    # CENTRALIZED RESPONSE HANDLING METHODS
    # ============================================================================

    async def _handle_complete_response_generic(
        self,
        response: Any,
        original_messages: List[Dict[str, str]],
        interactive: bool = True,
    ) -> Union[str, Any]:
        """Generic handler for non-streaming responses - delegates to ResponseHandler."""
        return await self.response_handler.handle_complete_response_generic(
            response, original_messages, interactive
        )

    def _handle_streaming_response_generic(
        self,
        response,
        original_messages: List[Dict[str, str]] = None,
        interactive: bool = True,
    ):
        """Generic handler for streaming responses - delegates to ResponseHandler."""
        return self.response_handler.handle_streaming_response_generic(
            response, original_messages, interactive
        )

    @abstractmethod
    def _extract_response_content(
        self, response: Any
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Extract text content, tool calls, and provider-specific data from response.

        Returns:
            tuple: (text_content, tool_calls, provider_specific_data)
        """
        pass

    @abstractmethod
    async def _process_streaming_chunks(
        self, response
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Process streaming response chunks.

        Returns:
            tuple: (accumulated_content, tool_calls, provider_specific_data)
        """
        pass

    @abstractmethod
    async def _make_api_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
        stream: bool = True,
    ) -> Any:
        """Make an API request to the provider.

        Returns:
            Provider-specific response object
        """
        pass

    @abstractmethod
    def _create_mock_response(self, content: str, tool_calls: List[Any]) -> Any:
        """Create a mock response object for centralized processing."""
        pass

    # ============================================================================
    # PROVIDER-SPECIFIC FEATURE HANDLERS (with default implementations)
    # ============================================================================

    def _handle_provider_specific_features(self, provider_data: Dict[str, Any]) -> None:
        """Handle provider-specific features like reasoning content. Override in subclasses."""
        pass

    def _format_provider_specific_content(self, provider_data: Dict[str, Any]) -> str:
        """Format provider-specific content for output. Override in subclasses."""
        return ""

    def _extract_continuation_message(self, continuation_message) -> Dict[str, Any]:
        """Extract the actual continuation message from structured response."""
        if (
            isinstance(continuation_message, dict)
            and "continuation_message" in continuation_message
        ):
            return continuation_message["continuation_message"]
        return continuation_message
