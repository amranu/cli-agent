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

from cli_agent.core.formatting import ResponseFormatter
from cli_agent.core.slash_commands import SlashCommandManager
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

        # Centralized subagent management system
        if not is_subagent:
            try:
                import os
                import sys

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
                logger.info("Initialized centralized subagent management system")
            except ImportError as e:
                logger.warning(f"Failed to import subagent manager: {e}")
                self.subagent_manager = None
                self.subagent_message_queue = None
        else:
            self.subagent_manager = None
            self.subagent_message_queue = None

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
            return self._bash_execute(args)
        elif tool_name == "read_file":
            return self._read_file(args)
        elif tool_name == "write_file":
            return self._write_file(args)
        elif tool_name == "list_directory":
            return self._list_directory(args)
        elif tool_name == "get_current_directory":
            return self._get_current_directory(args)
        elif tool_name == "todo_read":
            return self._todo_read(args)
        elif tool_name == "todo_write":
            return self._todo_write(args)
        elif tool_name == "replace_in_file":
            return self._replace_in_file(args)
        elif tool_name == "webfetch":
            return self._webfetch(args)
        elif tool_name == "task":
            return await self._task(args)
        elif tool_name == "task_status":
            return self._task_status(args)
        elif tool_name == "task_results":
            return self._task_results(args)
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
            import sys

            sys.exit(0)

        except Exception as e:
            return f"Error emitting result: {str(e)}"

    def _bash_execute(self, args: Dict[str, Any]) -> str:
        """Execute a bash command and return the output."""
        command = args.get("command", "")
        timeout = args.get("timeout", 120)

        if not command:
            return "Error: No command provided"

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}"
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nReturn code: {result.returncode}"

            return output if output else "Command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _read_file(self, args: Dict[str, Any]) -> str:
        """Read contents of a file with line numbers."""
        file_path = args.get("file_path", "")
        offset = args.get("offset", 1)
        limit = args.get("limit", None)

        if not file_path:
            return "Error: No file path provided"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            start_idx = max(0, offset - 1)  # Convert to 0-based index
            end_idx = (
                len(lines) if limit is None else min(len(lines), start_idx + limit)
            )

            result = []
            for i in range(start_idx, end_idx):
                result.append(f"{i + 1:6d}→{lines[i].rstrip()}")

            return "\n".join(result)

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _write_file(self, args: Dict[str, Any]) -> str:
        """Write content to a file."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            return "Error: No file path provided"

        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if (
                dir_path
            ):  # Only create directory if it's not empty (i.e., file is not in current dir)
                os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully wrote {len(content)} characters to {file_path}"

        except Exception as e:
            return f"Error writing file: {str(e)}"

    def _list_directory(self, args: Dict[str, Any]) -> str:
        """List contents of a directory."""
        directory_path = args.get("directory_path", ".")

        try:
            path = Path(directory_path)
            if not path.exists():
                return f"Error: Directory does not exist: {directory_path}"

            if not path.is_dir():
                return f"Error: Path is not a directory: {directory_path}"

            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")

            return "\n".join(items) if items else "Directory is empty"

        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def _get_current_directory(self, args: Dict[str, Any]) -> str:
        """Get the current working directory."""
        try:
            return os.getcwd()
        except Exception as e:
            return f"Error getting current directory: {str(e)}"

    def _todo_read(self, args: Dict[str, Any]) -> str:
        """Read the current todo list."""
        todo_file = "todo.json"

        try:
            if not os.path.exists(todo_file):
                return "[]"  # Empty todo list

            with open(todo_file, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            return f"Error reading todo list: {str(e)}"

    def _todo_write(self, args: Dict[str, Any]) -> str:
        """Write/update the todo list."""
        todos = args.get("todos", [])
        todo_file = "todo.json"

        try:
            with open(todo_file, "w", encoding="utf-8") as f:
                json.dump(todos, f, indent=2)

            # Return the actual todo list data to the LLM for proper feedback
            return f"Successfully updated todo list with {len(todos)} items. Current todo list:\n{json.dumps(todos, indent=2)}"

        except Exception as e:
            return f"Error writing todo list: {str(e)}"

    def _replace_in_file(self, args: Dict[str, Any]) -> str:
        """Replace text in a file."""
        file_path = args.get("file_path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")

        if not file_path:
            return "Error: No file path provided"
        if not old_text:
            return "Error: No old text provided"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_text not in content:
                return f"Error: Text not found in file: {old_text}"

            new_content = content.replace(old_text, new_text)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            count = content.count(old_text)
            return f"Successfully replaced {count} occurrence(s) of text in {file_path}"

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error replacing text in file: {str(e)}"

    def _webfetch(self, args: Dict[str, Any]) -> str:
        """Fetch a webpage using curl and return its content."""
        url = args.get("url", "")
        limit = args.get("limit", 1000)  # Default to 1000 lines

        if not url:
            return "Error: No URL provided"

        # Use curl to fetch the webpage with a timeout, capturing raw output
        result = subprocess.run(
            ["curl", "-L", "--max-time", "30", url],
            capture_output=True,
            timeout=35,  # Slightly longer than curl timeout
        )

        # Try to decode with utf-8, then fall back to latin-1 (which rarely fails)
        try:
            content = result.stdout.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {url}. Falling back to latin-1.")
            content = result.stdout.decode("latin-1", errors="replace")

        if result.returncode != 0:
            # Try to decode stderr for a better error message
            try:
                stderr = result.stderr.decode("utf-8", errors="replace")
            except:
                stderr = repr(result.stderr)

            error_msg = (
                f"Error fetching URL (curl return code {result.returncode}): {stderr}"
            )

            # If we have content despite the error, include it
            if content.strip():
                return f"{error_msg}\n\nContent retrieved:\n{content}"
            else:
                return error_msg

        # Truncate the content by lines if limit is specified
        if limit is not None and isinstance(limit, int) and limit > 0:
            lines = content.split("\n")
            if len(lines) > limit:
                content = "\n".join(lines[:limit])
                content += f"\n\n[Content truncated at {limit} lines. Original had {len(lines)} lines.]"

        # Return the content (truncated if limit was provided)
        return content

    async def _task(self, args: Dict[str, Any]) -> str:
        """Spawn a new subagent task using the centralized SubagentManager."""
        if not self.subagent_manager:
            return "Error: Subagent management not available"

        description = args.get("description", "Investigation task")
        prompt = args.get("prompt", "")
        context = args.get("context", "")

        if not prompt:
            return "Error: prompt is required"

        # Add context to prompt if provided
        full_prompt = prompt
        if context:
            full_prompt += f"\n\nAdditional context: {context}"

        try:
            # Track active count before and after spawning
            initial_count = self.subagent_manager.get_active_count()
            task_id = await self.subagent_manager.spawn_subagent(
                description, full_prompt
            )
            final_count = self.subagent_manager.get_active_count()

            # Display spawn confirmation with active count
            active_info = (
                f" (Now {final_count} active subagents)" if final_count > 1 else ""
            )
            return f"Spawned subagent task: {task_id}\nDescription: {description}{active_info}\nTask is running in the background - output will appear in the chat as it becomes available."
        except Exception as e:
            return f"Error spawning subagent: {e}"

    def _task_status(self, args: Dict[str, Any]) -> str:
        """Check the status of running subagent tasks using SubagentManager."""
        if not self.subagent_manager:
            return "Subagent management not available"

        task_id = args.get("task_id", None)

        if task_id:
            # Check specific task
            if task_id not in self.subagent_manager.subagents:
                return f"Task {task_id} not found."

            subagent = self.subagent_manager.subagents[task_id]
            status = "Completed" if subagent.completed else "Running"
            runtime = time.time() - subagent.start_time

            result = f"""Task Status: {task_id}
Description: {subagent.description}
Status: {status}
Runtime: {runtime:.2f} seconds"""

            if subagent.completed:
                result += f"\nResult available: {subagent.result is not None}"

            return result
        else:
            # Check all tasks
            subagents = self.subagent_manager.subagents
            if not subagents:
                return "No tasks are currently running."

            result = f"Task Status Summary ({len(subagents)} tasks):\n"
            for tid, subagent in subagents.items():
                status = "Completed" if subagent.completed else "Running"
                runtime = time.time() - subagent.start_time
                result += f"\n{tid}: {subagent.description} - {status} ({runtime:.1f}s)"

            return result

    def _task_results(self, args: Dict[str, Any]) -> str:
        """Retrieve the results and summaries from completed subagent tasks using SubagentManager."""
        try:
            if not self.subagent_manager:
                return "Subagent management not available"

            include_running = args.get("include_running", False)
            clear_after_retrieval = args.get("clear_after_retrieval", True)

            subagents = self.subagent_manager.subagents
            if not subagents:
                return "No tasks found."

            # Count tasks
            task_count = len(subagents)
            completed_count = sum(
                1 for subagent in subagents.values() if subagent.completed
            )
            running_count = task_count - completed_count

            result_parts = [
                f"=== TASK RESULTS SUMMARY ===",
                f"Total tasks: {task_count}",
                f"Completed: {completed_count}",
                f"Running: {running_count}",
                "",
            ]

            # Show completed tasks
            if completed_count > 0:
                result_parts.append("=== COMPLETED TASKS ===")
                for task_id, subagent in subagents.items():
                    if not subagent.completed:
                        continue

                    runtime = time.time() - subagent.start_time
                    result_parts.append(
                        f"\n{task_id}: {subagent.description} - ✅ Completed ({runtime:.2f}s)"
                    )

                    if subagent.result:
                        result_parts.append(f"Result: {subagent.result}")

            # Show running tasks if requested
            if include_running and running_count > 0:
                result_parts.append("\n=== RUNNING TASKS ===")
                for task_id, subagent in subagents.items():
                    if subagent.completed:
                        continue

                    runtime = time.time() - subagent.start_time
                    result_parts.append(
                        f"\n{task_id}: {subagent.description} - ⏳ Running ({runtime:.2f}s)"
                    )

            # Clear completed tasks if requested
            if clear_after_retrieval and completed_count > 0:
                completed_task_ids = [
                    tid for tid, subagent in subagents.items() if subagent.completed
                ]
                for task_id in completed_task_ids:
                    del self.subagent_manager.subagents[task_id]
                result_parts.append(
                    f"\n--- {len(completed_task_ids)} completed tasks cleared from memory ---"
                )

            return "\n".join(result_parts)

        except Exception as e:
            logger.error(f"Error in _task_results: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error retrieving task results: {str(e)}"

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
    def _handle_subagent_permission_request(self, message, task_id):
        """Handle permission request from subagent."""
        try:
            import asyncio
            import threading

            # Get request details from message data
            request_id = message.data.get("request_id")
            response_file = message.data.get("response_file")
            prompt_text = message.content

            if not request_id or not response_file:
                logger.error(
                    "Invalid permission request from subagent: missing request_id or response_file"
                )
                return

            # Display the permission request to user
            print(f"\n🤖 [SUBAGENT-{task_id}] {prompt_text}", end="", flush=True)

            # Get user input
            input_handler = getattr(self, "_input_handler", None)
            if input_handler:
                try:
                    # Get user response
                    response = input_handler.get_input("")
                    if response is None:
                        response = "n"  # Default to deny if interrupted
                except Exception as e:
                    logger.error(
                        f"Error getting user input for subagent permission: {e}"
                    )
                    response = "n"  # Default to deny on error
            else:
                # No input handler, default to deny
                response = "n"

            # Write response to temp file for subagent to read
            try:
                with open(response_file, "w") as f:
                    f.write(response)
            except Exception as e:
                logger.error(f"Error writing permission response: {e}")

        except Exception as e:
            logger.error(f"Error handling subagent permission request: {e}")

    def _on_subagent_message(self, message):
        """Callback for when a subagent message is received - display during yield period."""
        try:
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
                self._handle_subagent_permission_request(message, task_id)
                return  # Don't display permission requests, handle them directly
            else:
                formatted = f"[SUBAGENT-{task_id} {message.type}] {message.content}"

            # Only display immediately if we're in yielding mode (subagents active)
            # This ensures clean separation between main agent and subagent output
            if self.subagent_manager and self.subagent_manager.get_active_count() > 0:
                # Subagents are active - display immediately during yield period
                self._display_subagent_message_immediately(formatted, message.type)
            else:
                # No active subagents - just log for now (main agent controls display)
                logger.info(
                    f"Subagent message logged: {message.type} - {message.content[:50]}"
                )

        except Exception as e:
            logger.error(f"Error handling subagent message: {e}")

    def _display_subagent_message_immediately(self, formatted: str, message_type: str):
        """Display subagent message immediately with proper line endings."""
        try:
            import sys
            import termios

            try:
                # Check if we're in raw terminal mode
                stdin_fd = sys.stdin.fileno()
                current_attrs = termios.tcgetattr(stdin_fd)
                in_raw_mode = not (current_attrs[3] & termios.ECHO) and not (
                    current_attrs[3] & termios.ICANON
                )

                if in_raw_mode:
                    # In raw mode, we need \r\n for proper line breaks
                    # Move to beginning of line and add proper line endings
                    formatted_with_crlf = formatted.replace("\n", "\r\n")
                    output = f"\r\n{formatted_with_crlf}\r\n"
                    sys.stderr.write(output)
                    sys.stderr.flush()
                else:
                    # Normal mode - regular print is fine
                    print(formatted, file=sys.stderr, flush=True)

            except (OSError, termios.error):
                # Terminal control not available - use regular print
                print(formatted, file=sys.stderr, flush=True)

            logger.debug(f"Displayed subagent message: {message_type}")
        except Exception as e:
            logger.error(f"Error displaying subagent message: {e}")

    async def _collect_subagent_results(self):
        """Wait for all subagents to complete and collect their results."""
        if not self.subagent_manager:
            return []

        import time

        results = []
        max_wait_time = 300  # 5 minutes max wait
        start_time = time.time()

        # Wait for all active subagents to complete
        while self.subagent_manager.get_active_count() > 0:
            if time.time() - start_time > max_wait_time:
                logger.error("Timeout waiting for subagents to complete")
                break
            await asyncio.sleep(0.5)

        # Collect results from completed subagents
        logger.info(
            f"Checking {len(self.subagent_manager.subagents)} subagents for results"
        )
        for task_id, subagent in self.subagent_manager.subagents.items():
            logger.info(
                f"Subagent {task_id}: completed={subagent.completed}, has_result={subagent.result is not None}"
            )
            if subagent.completed and subagent.result:
                results.append(
                    {
                        "task_id": task_id,
                        "description": subagent.description,
                        "content": subagent.result,
                        "runtime": time.time() - subagent.start_time,
                    }
                )
                logger.info(
                    f"Collected result from {task_id}: {len(subagent.result)} chars"
                )
            elif subagent.completed:
                logger.warning(f"Subagent {task_id} completed but has no result stored")
            else:
                logger.info(f"Subagent {task_id} not yet completed")

        logger.info(
            f"Collected {len(results)} results from {len(self.subagent_manager.subagents)} subagents"
        )

        # Clean up completed subagents that provided results
        if results:
            completed_task_ids = [result["task_id"] for result in results]
            for task_id in completed_task_ids:
                if task_id in self.subagent_manager.subagents:
                    del self.subagent_manager.subagents[task_id]
                    logger.info(f"Cleaned up completed subagent: {task_id}")

            # Also clear any accumulated messages in the queue since they've been integrated
            if self.subagent_message_queue:
                # Clear any remaining messages in the queue
                try:
                    while True:
                        self.subagent_message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                logger.info("Cleared subagent message queue after collecting results")

        return results

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
            formatted_response = self.format_markdown(text_buffer)
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
            tool_execution_msg = self.display_tool_execution_step(
                i, tool_name, arguments, self.is_subagent, interactive=interactive
            )
            if interactive and not streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            elif interactive and streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            else:
                all_tool_output.append(tool_execution_msg)

            # Create coroutine for parallel execution
            tool_coroutines.append(self._execute_mcp_tool(tool_name, arguments))

        # Execute all tools in parallel
        if tool_coroutines:
            try:
                # Execute all tool calls concurrently
                tool_results = await asyncio.gather(
                    *tool_coroutines, return_exceptions=True
                )

                # Process results in order
                for (i, tool_name, arguments), tool_result in zip(
                    tool_info_list, tool_results
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

                    # Use unified tool result display
                    tool_result_msg = self.display_tool_execution_result(
                        tool_result,
                        not tool_success,
                        self.is_subagent,
                        interactive=interactive,
                    )
                    if interactive and not streaming_mode:
                        print(f"\r\x1b[K{tool_result_msg}", flush=True)
                    elif interactive and streaming_mode:
                        print(f"\r\x1b[K{tool_result_msg}", flush=True)
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
            import sys

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
        """Create a basic system prompt that includes tool information."""
        tool_descriptions = []

        for tool_key, tool_info in self.available_tools.items():
            # Use the converted name format (with underscores)
            converted_tool_name = tool_key.replace(":", "_")
            description = tool_info["description"]
            tool_descriptions.append(f"- **{converted_tool_name}**: {description}")

        tools_text = (
            "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        )

        # Customize prompt based on whether this is a subagent
        if self.is_subagent:
            agent_role = "You are a focused subagent responsible for executing a specific task efficiently."
            subagent_strategy = "**SUBAGENT FOCUS:** You are a subagent with a specific task. Complete your assigned task using the available tools and provide clear results. You cannot spawn other subagents."
        else:
            agent_role = "You are a top-tier autonomous software development agent. You are in control and responsible for completing the user's request."
            subagent_strategy = """**Context Management & Subagent Strategy:**
- **Preserve your context:** Your context window is precious - don't waste it on tasks that can be delegated.
- **Delegate context-heavy tasks:** Use `builtin_task` to spawn subagents for tasks that would consume significant context:
  - Large file analysis or searches across multiple files
  - Complex investigations requiring reading many files
  - Running multiple commands or gathering system information
  - Any task that involves reading >200 lines of code
- **Parallel execution:** For complex investigations requiring multiple independent tasks, spawn multiple subagents simultaneously by making multiple `builtin_task` calls in the same response.
- **Stay focused:** Keep your main context for planning, coordination, and final synthesis of results.
- **Automatic coordination:** After spawning subagents, the main agent automatically pauses, waits for all subagents to complete, then restarts with their combined results.
- **Do not poll status:** Avoid calling `builtin_task_status` repeatedly - the system handles coordination automatically.
- **Single response spawning:** To spawn multiple subagents, include all `builtin_task` calls in one response, not across multiple responses.

**When to Use Subagents:**
✅ **DO delegate:** File searches, large code analysis, running commands, gathering information
❌ **DON'T delegate:** Simple edits, single file reads <50 lines, quick tool calls"""

        # Basic system prompt - subclasses can override this
        system_prompt = f"""{agent_role}

**Mission:** Use the available tools to solve the user's request.

**Guiding Principles:**
- **Ponder, then proceed:** Briefly outline your plan before you act. State your assumptions.
- **Bias for action:** You are empowered to take initiative. Do not ask for permission, just do the work.
- **Problem-solve:** If a tool fails, analyze the error and try a different approach.
- **Break large changes into smaller chunks:** For large code changes, divide the work into smaller, manageable tasks to ensure clarity and reduce errors.

**File Reading Strategy:**
- **Be surgical:** Do not read entire files at once. It is a waste of your context window.
- **Locate, then read:** Use tools like `grep` or `find` to locate the specific line numbers or functions you need to inspect.
- **Read in chunks:** Read files in smaller, targeted chunks of 50-100 lines using the `offset` and `limit` parameters in the `read_file` tool.
- **Full reads as a last resort:** Only read a full file if you have no other way to find what you are looking for.

**File Editing Workflow:**
1.  **Read first:** Always read a file before you try to edit it, following the file reading strategy above.
2.  **Greedy Grepping:** Always `grep` or look for a small section around where you want to do an edit. This is faster and more reliable than reading the whole file.
3.  **Use `replace_in_file`:** For all file changes, use `builtin_replace_in_file` to replace text in files.
4.  **Chunk changes:** Break large edits into smaller, incremental changes to maintain control and clarity.

**Todo List Workflow:**
- **Use the Todo list:** Use `builtin_todo_read` and `builtin_todo_write` to manage your tasks.
- **Start with a plan:** At the beginning of your session, create a todo list to outline your steps.
- **Update as you go:** As you complete tasks, update the todo list to reflect your progress.

{subagent_strategy}

**Workflow:**
1.  **Reason:** Outline your plan.
2.  **Act:** Use one or more tool calls to execute your plan. Use parallel tool calls when it makes sense.
3.  **Respond:** When you have completed the request, provide the final answer to the user.

**Available Tools:**
{tools_text}

You are the expert. Complete the task."""

        return system_prompt

    def _read_agent_md(self) -> str:
        """Read the AGENT.md file for prepending to first messages."""
        try:
            import os

            # Get the project root directory (where AGENT.md should be)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            agent_md_path = os.path.join(project_root, "AGENT.md")

            if os.path.exists(agent_md_path):
                with open(agent_md_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            else:
                logger.warning(f"AGENT.md not found at {agent_md_path}")
                return ""
        except Exception as e:
            logger.error(f"Error reading AGENT.md: {e}")
            return ""

    def _prepend_agent_md_to_first_message(
        self, messages: List[Dict[str, str]], is_first_message: bool = False
    ) -> List[Dict[str, str]]:
        """Prepend AGENT.md content to the first user message if this is the first message of the session."""
        if not is_first_message or not messages:
            return messages

        # Only prepend to the first user message
        first_user_msg_index = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                first_user_msg_index = i
                break

        if first_user_msg_index is not None:
            agent_md_content = self._read_agent_md()
            if agent_md_content:
                # Create a copy of messages to avoid modifying the original
                messages_copy = messages.copy()
                original_content = messages_copy[first_user_msg_index]["content"]

                # Prepend AGENT.md with a clear separator
                enhanced_content = f"""<AGENT_ARCHITECTURE_CONTEXT>
{agent_md_content}
</AGENT_ARCHITECTURE_CONTEXT>

{original_content}"""

                messages_copy[first_user_msg_index] = {
                    **messages_copy[first_user_msg_index],
                    "content": enhanced_content,
                }

                logger.info("Prepended AGENT.md to first user message")
                return messages_copy

        return messages

    def _format_chunk_safely(self, chunk: str) -> str:
        """Apply basic formatting to streaming chunks without losing spaces."""
        return self.formatter.format_chunk_safely(chunk)

    def _apply_simple_formatting(self, chunk: str) -> str:
        """Apply simple visible formatting that definitely works."""
        return self.formatter._apply_simple_formatting(chunk)

    def _apply_rich_console_formatting(self, chunk: str) -> str:
        """Use Rich Console to apply formatting directly to stdout."""
        return self.formatter._apply_rich_console_formatting(chunk)

    def _apply_direct_ansi_formatting(self, chunk: str) -> str:
        """Apply ANSI formatting with proper terminal handling."""
        return self.formatter._apply_direct_ansi_formatting(chunk)

    def _chunk_has_complete_markdown(self, chunk: str) -> bool:
        """Check if chunk contains complete markdown patterns that are safe to format."""
        return self.formatter._chunk_has_complete_markdown(chunk)

    def _apply_basic_markdown_to_text(self, text, chunk: str):
        """Apply basic markdown styling to Rich Text object."""
        return self.formatter._apply_basic_markdown_to_text(text, chunk)

    def _find_safe_markdown_boundary(self, text: str, start_pos: int) -> int:
        """Find a safe position to apply markdown formatting without breaking syntax."""
        return self.formatter._find_safe_markdown_boundary(text, start_pos)

    def format_markdown(self, text: str) -> str:
        """Format markdown text for terminal display using Rich."""
        return self.formatter.format_markdown(text)

    def _basic_markdown_format(self, text: str) -> str:
        """Basic fallback markdown formatting."""
        return self.formatter._basic_markdown_format(text)

    def display_tool_execution_start(
        self, tool_count: int, is_subagent: bool = False, interactive: bool = True
    ) -> str:
        """Display tool execution start message."""
        return self.formatter.display_tool_execution_start(
            tool_count, is_subagent, interactive
        )

    def display_tool_execution_step(
        self,
        step_num: int,
        tool_name: str,
        arguments: dict,
        is_subagent: bool = False,
        interactive: bool = True,
    ) -> str:
        """Display individual tool execution step."""
        return self.formatter.display_tool_execution_step(
            step_num, tool_name, arguments, is_subagent, interactive
        )

    def display_tool_execution_result(
        self,
        result: str,
        is_error: bool = False,
        is_subagent: bool = False,
        interactive: bool = True,
    ) -> str:
        """Display tool execution result."""
        return self.formatter.display_tool_execution_result(
            result, is_error, is_subagent, interactive
        )

    def display_tool_processing(
        self, is_subagent: bool = False, interactive: bool = True
    ) -> str:
        """Display tool processing message."""
        return self.formatter.display_tool_processing(is_subagent, interactive)

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters for most models)."""
        return self.token_manager.estimate_tokens(text)

    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count estimated tokens in the conversation."""
        return self.token_manager.count_conversation_tokens(messages)

    def get_token_limit(self) -> int:
        """Get the context token limit for the current model."""
        return self.token_manager.get_token_limit()

    def should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if conversation should be compacted."""
        return self.token_manager.should_compact(messages)

    async def compact_conversation(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create a compact summary of the conversation to preserve context while reducing tokens."""
        return await self.token_manager.compact_conversation(
            messages, generate_response_func=self.generate_response
        )

    async def generate_response(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None
    ) -> Union[str, Any]:
        """Generate a response using the specific LLM. Centralized implementation with subagent yielding."""
        # For subagents, use interactive=False to avoid terminal formatting issues
        interactive = not self.is_subagent

        # Default to streaming behavior, but allow subclasses to override
        # Subagents should not stream to avoid generator issues
        stream = getattr(self, "stream", True) and not self.is_subagent

        # Call the concrete implementation's _generate_completion method
        tools_list = (
            self.convert_tools_to_llm_format() if self.available_tools else None
        )
        return await self._generate_completion(
            messages, tools_list, stream, interactive
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
    def _add_tool_results_to_conversation(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
        tool_results: List[str],
    ) -> List[Dict[str, Any]]:
        """Add tool results to conversation in model-specific format. Must be implemented by subclasses."""
        pass

    def _prepend_agent_md_to_first_message(
        self, messages: List[Dict[str, Any]], is_first_message: bool
    ) -> List[Dict[str, Any]]:
        """Prepend AGENT.md content to first message."""
        if not is_first_message or not messages:
            return messages

        # Read AGENT.md file and prepend to first user message
        import os

        agent_md_path = "AGENT.md"
        agent_content = ""

        if os.path.exists(agent_md_path):
            try:
                with open(agent_md_path, "r", encoding="utf-8") as f:
                    agent_content = f.read().strip()
            except Exception as e:
                logger.warning(f"Could not read AGENT.md: {e}")

        enhanced_messages = messages.copy()
        if (
            agent_content
            and enhanced_messages
            and enhanced_messages[0].get("role") == "user"
        ):
            original_content = enhanced_messages[0]["content"]
            enhanced_messages[0][
                "content"
            ] = f"{agent_content}\n\n---\n\n{original_content}"

        return enhanced_messages

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
                    formatted_response = self.format_markdown(text_buffer)
                    formatted_response = formatted_response.replace("\n", "\r\n")
                    print(f"\r\x1b[K\r\nAssistant: {formatted_response}")
                    self._text_buffer = ""

            # Display tool execution start
            if interactive and not self.is_subagent:
                print(
                    f"\r\n{self.display_tool_execution_start(len(tool_calls), self.is_subagent, interactive=True)}"
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
        import re

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
        import sys

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

                # Get user input with smart multiline detection
                user_input = input_handler.get_multiline_input("You: ")

                if user_input is None:  # Interrupted
                    if current_task and not current_task.done():
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

                # Check if this is the first message and prepend AGENT.md if so
                is_first_message = len(messages) == 1
                enhanced_messages = self._prepend_agent_md_to_first_message(
                    messages, is_first_message
                )

                # Reset tool execution state for new message
                self._tool_execution_started = False
                self._post_tool_buffer = ""
                self._text_buffer = ""

                # Show thinking message
                print("\nThinking...")

                # Create response task
                tools_list = self.convert_tools_to_llm_format()
                current_task = asyncio.create_task(
                    self.generate_response(enhanced_messages, tools_list)
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
                                )

                                if tool_start and not tool_execution_started:
                                    # Tool execution starting - format and display any buffered content first
                                    tool_execution_started = True
                                    self._tool_execution_started = True

                                    # Get the most current buffer content
                                    current_buffer = getattr(self, "_text_buffer", "")

                                    # Display buffered pre-tool text if any
                                    if current_buffer.strip():
                                        formatted_response = self.format_markdown(
                                            current_buffer
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

                                # Output behavior based on current mode
                                if tool_execution_started:
                                    # In tool execution mode - stream directly
                                    display_chunk = chunk.replace("\n", "\r\n")
                                    print(display_chunk, end="", flush=True)
                                    full_response += chunk
                                else:
                                    # In text mode (pre-tool or post-tool) - buffer for markdown formatting
                                    self._text_buffer = (
                                        getattr(self, "_text_buffer", "") + chunk
                                    )
                                    full_response += chunk
                            else:
                                # Handle any non-string chunks if needed
                                chunk_str = str(chunk)
                                formatted_chunk = self._format_chunk_safely(chunk_str)
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
                                # Format and display the remaining buffered text
                                formatted_response = self.format_markdown(
                                    remaining_text_buffer
                                )
                                display_response = formatted_response.replace(
                                    "\n", "\r\n"
                                )
                                print(display_response, end="", flush=True)
                                print()  # Final newline
                            elif full_response and not tool_execution_started:
                                # No tools were executed and no separate buffer - format the entire response
                                print()  # New line after streaming
                                print(
                                    f"\033[1A\033[KAssistant: ", end=""
                                )  # Move up one line, clear it, write "Assistant: "
                                formatted_response = self.format_markdown(full_response)
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
                    formatted_response = self.format_markdown(str(response))
                    print(f"\nAssistant: {formatted_response}")
                    messages.append({"role": "assistant", "content": str(response)})

            except KeyboardInterrupt:
                # Move to beginning of line and clear, then print exit message
                sys.stdout.write("\r\x1b[KExiting...\n")
                sys.stdout.flush()
                break
            except Exception as e:
                print(f"\nError: {e}")
