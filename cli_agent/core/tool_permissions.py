#!/usr/bin/env python3
"""
Tool permission system for MCP Agent.

Provides user prompts for tool execution with configurable allowed/disallowed tools,
session-based permission tracking, and pattern matching similar to Claude Code.
"""

import asyncio
import fnmatch
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from cli_agent.core.permission_display import get_clean_permission_display, format_tool_description

logger = logging.getLogger(__name__)


class ToolDeniedReturnToPrompt(Exception):
    """Exception raised when a tool is denied and should return to user prompt."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class ToolPermissionConfig:
    """Configuration for tool permissions."""

    allowed_tools: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)
    auto_approve_session: bool = (
        False  # Global setting to auto-approve all tools for session
    )
    session_permissions_file: str = "sessions/.tool_permissions.json"


@dataclass
class ToolPermissionResult:
    """Result of a tool permission check."""

    allowed: bool
    reason: str
    skip_prompt: bool = False
    return_to_prompt: bool = (
        False  # If True, should return to user prompt instead of sending to LLM
    )


class ToolPermissionManager:
    """Manages tool permissions with user prompts and session tracking."""

    def __init__(self, config: ToolPermissionConfig):
        self.config = config
        self.session_approvals: Set[str] = set()  # Tools approved for this session
        self.session_denials: Set[str] = set()  # Tools denied for this session
        self.session_auto_approve: bool = False  # Auto-approve all for this session
        
        # Get clean permission display manager
        self.clean_permission_display = get_clean_permission_display()

        # Load persistent session permissions
        self._load_session_permissions()

    def _load_session_permissions(self):
        """Load persistent session permissions from file."""
        # Skip loading if session permissions file is disabled
        if not self.config.session_permissions_file:
            logger.info(
                "Session permissions file disabled - starting with empty approvals"
            )
            return

        try:
            permissions_file = Path(self.config.session_permissions_file)
            if permissions_file.exists():
                with open(permissions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.session_approvals = set(data.get("approvals", []))
                    self.session_denials = set(data.get("denials", []))
                    self.session_auto_approve = data.get("auto_approve", False)
        except Exception as e:
            logger.warning(f"Could not load session permissions: {e}")

    def _save_session_permissions(self):
        """Save persistent session permissions to file."""
        # Skip saving if session permissions file is disabled
        if not self.config.session_permissions_file:
            logger.info("Session permissions file disabled - not saving approvals")
            return

        try:
            permissions_file = Path(self.config.session_permissions_file)
            permissions_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "approvals": list(self.session_approvals),
                "denials": list(self.session_denials),
                "auto_approve": self.session_auto_approve,
            }

            with open(permissions_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save session permissions: {e}")

    def _matches_pattern(self, tool_name: str, pattern: str) -> bool:
        """
        Check if tool name matches a pattern.

        Supports Claude Code-style patterns:
        - "Bash" or "bash_execute" - exact tool name match
        - "Bash(*)" - any bash command
        - "Bash(git:*)" - bash commands starting with "git:"
        - "*" - wildcard for any tool
        """
        # Handle exact matches first
        if pattern == tool_name:
            return True

        # Handle wildcard patterns
        if pattern == "*":
            return True

        # Convert friendly names to internal tool names
        tool_mapping = {
            "Bash": "bash_execute",
            "Edit": "replace_in_file",
            "Read": "read_file",
            "Write": "write_file",
            "List": "list_directory",
            "WebFetch": "webfetch",
            "Task": "task",
            "Todo": "todo_read",  # Could match todo_read or todo_write
            "Directory": "get_current_directory",
        }

        # Check if pattern is a friendly name that maps to this tool
        if pattern in tool_mapping and tool_mapping[pattern] == tool_name:
            return True

        # Handle tool-specific patterns like "Bash(*)" or "Bash(git:*)"
        tool_pattern_match = re.match(r"^(\w+)\(([^)]+)\)$", pattern)
        if tool_pattern_match:
            tool_type, command_pattern = tool_pattern_match.groups()

            expected_tool = tool_mapping.get(tool_type, tool_type.lower())
            if tool_name != expected_tool:
                return False

            # For bash commands, we'd need to check the actual command
            # This would require passing the arguments, but for now we'll
            # match any command for that tool type
            return True

        # Handle MCP server patterns like "mcp:*" or "server_name:*"
        if ":" in pattern:
            if pattern.endswith(":*"):
                # Pattern like "mcp:*" should match any tool starting with "mcp:"
                prefix = pattern[:-1]  # Remove the '*'
                return tool_name.startswith(prefix)
            elif ":" in tool_name:
                # Both pattern and tool_name have colons, try exact match
                return tool_name == pattern

        # Handle simple glob patterns
        return fnmatch.fnmatch(tool_name, pattern)

    def _is_tool_allowed_by_config(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolPermissionResult:
        """Check if tool is allowed/disallowed by configuration."""

        # Always allow core agent management tools without prompting
        core_tools = {
            "task",
            "task_status",
            "task_results",
            "emit_result",
            "emit_output",
            "emit_error",
            "emit_status",
        }
        if tool_name in core_tools:
            return ToolPermissionResult(
                allowed=True,
                reason=f"Tool '{tool_name}' is a core agent management tool",
                skip_prompt=True,
            )

        # Check disallowed tools first (takes precedence)
        for pattern in self.config.disallowed_tools:
            if self._matches_pattern(tool_name, pattern):
                return ToolPermissionResult(
                    allowed=False,
                    reason=f"Tool '{tool_name}' is disallowed by configuration pattern '{pattern}'",
                    skip_prompt=True,
                )

        # Check allowed tools
        if self.config.allowed_tools:
            for pattern in self.config.allowed_tools:
                if self._matches_pattern(tool_name, pattern):
                    return ToolPermissionResult(
                        allowed=True,
                        reason=f"Tool '{tool_name}' is allowed by configuration pattern '{pattern}'",
                        skip_prompt=True,
                    )

            # If allowed tools are specified but no match found, prompt user
            return ToolPermissionResult(
                allowed=True,  # Default to allow but require prompt
                reason=f"Tool '{tool_name}' is not in the allowed tools list",
                skip_prompt=False,  # Require user prompt
            )

        # No specific configuration, allow with prompt
        return ToolPermissionResult(
            allowed=True,
            reason="No specific configuration, allowing with prompt",
            skip_prompt=False,
        )

    def _format_tool_description(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Format a user-friendly description of what the tool will do."""
        descriptions = {
            "bash_execute": lambda args: f"Execute bash command: {args.get('command', 'N/A')}",
            "read_file": lambda args: f"Read file: {args.get('file_path', 'N/A')}",
            "write_file": lambda args: f"Write to file: {args.get('file_path', 'N/A')}",
            "replace_in_file": lambda args: f"Edit file: {args.get('file_path', 'N/A')}",
            "list_directory": lambda args: f"List directory: {args.get('path', 'current directory')}",
            "webfetch": lambda args: f"Fetch URL: {args.get('url', 'N/A')}",
            "todo_read": lambda args: "Read todo list",
            "todo_write": lambda args: f"Update todo list with {len(args.get('todos', []))} items",
            "task": lambda args: f"Spawn subagent task: {args.get('description', 'N/A')}",
            "task_status": lambda args: f"Check task status: {args.get('task_id', 'N/A')}",
            "task_results": lambda args: f"Get task results: {args.get('task_id', 'N/A')}",
            "get_current_directory": lambda args: "Get current directory",
        }

        formatter = descriptions.get(tool_name)
        if formatter:
            try:
                return formatter(arguments)
            except Exception:
                pass

        return f"Execute tool: {tool_name}"

    async def check_tool_permission(
        self, tool_name: str, arguments: Dict[str, Any], input_handler=None
    ) -> ToolPermissionResult:
        """
        Check if a tool should be allowed to execute.

        Returns ToolPermissionResult with decision and whether to skip prompting.
        """

        # For subagents, ALWAYS set the tool information on the input handler
        # This must happen before any early returns so the info is available for display
        if input_handler is not None and hasattr(input_handler, "subagent_context"):
            tool_description = format_tool_description(tool_name, arguments)
            # Use the attribute names that SubagentInputHandler actually reads
            input_handler.current_tool_name = tool_name
            input_handler.current_tool_arguments = arguments
            # Also set the permission-specific attributes for consistency
            input_handler._permission_tool_name = tool_name
            input_handler._permission_arguments = arguments
            input_handler._permission_description = tool_description
            input_handler._permission_full_prompt = f"Allow {tool_name}? {tool_description}"

        # Check session-level auto-approve
        if self.session_auto_approve or self.config.auto_approve_session:
            return ToolPermissionResult(
                allowed=True, reason="Auto-approved for session", skip_prompt=True
            )

        # CONFIG-BASED PERMISSION: CHECK FIRST!
        config_result = self._is_tool_allowed_by_config(tool_name, arguments)
        if config_result.skip_prompt:
            return config_result

        # Check session-level approvals/denials (only if not globally allowed)
        if tool_name in self.session_approvals:
            return ToolPermissionResult(
                allowed=True, reason="Previously approved for session", skip_prompt=True
            )

        if tool_name in self.session_denials:
            return ToolPermissionResult(
                allowed=False, reason="Previously denied for session", skip_prompt=True
            )

        # Need to prompt user
        if input_handler is None:
            # No input handler available, default to allow
            logger.warning(
                "No input handler available for tool permission prompt, defaulting to allow"
            )
            return ToolPermissionResult(
                allowed=True, reason="No input handler available"
            )

        return await self._prompt_user_for_permission(
            tool_name, arguments, input_handler
        )

    async def _prompt_user_for_permission(
        self, tool_name: str, arguments: Dict[str, Any], input_handler
    ) -> ToolPermissionResult:
        """Prompt the user for tool execution permission."""

        tool_description = format_tool_description(tool_name, arguments)
        use_clean_display = False

        # Try to use clean permission display first
        try:
            if self.clean_permission_display.can_use_clean_display():
                use_clean_display = self.clean_permission_display.show_permission_request(
                    tool_name=tool_name,
                    tool_description=tool_description,
                    arguments=arguments,
                    task_id="main-process",
                    queue_position=1,
                    total_requests=1,
                )
                
                if use_clean_display:
                    logger.debug(f"Using clean display for main process permission request: {tool_name}")
                else:
                    logger.debug(f"Clean display failed, falling back to traditional for: {tool_name}")
                    
        except Exception as e:
            logger.warning(f"Clean permission display failed: {e}, falling back to traditional")
            use_clean_display = False

        # Build complete permission prompt as a single message (for both clean and fallback)
        prompt_lines = [
            "🔧 Tool Execution Request:",
            f"Tool: {tool_name}",
            f"Action: {tool_description}",
        ]

        # Show arguments if they're not sensitive
        if tool_name != "bash_execute" or len(str(arguments)) < 100:
            prompt_lines.append(f"Arguments: {arguments}")

        prompt_lines.extend(
            [
                "",  # Empty line
                "Allow this tool to execute?",
                "[y] Yes, execute once",
                f"[a] Yes, and allow '{tool_name}' for the rest of this session",
                "[A] Yes, and auto-approve ALL tools for this session",
                "[n] No, deny this execution",
                f"[d] No, and deny '{tool_name}' for the rest of this session",
            ]
        )

        # Send as single consolidated message
        full_prompt = "\n".join(prompt_lines)

        try:
            # For subagents, pass the full prompt to the input handler
            # For regular agents, check if we should use the permission queue
            if hasattr(input_handler, "subagent_context"):
                # Subagent: don't display prompt immediately, let main process queue handle it
                # Store the prompt details for the permission request using correct attribute names
                input_handler.current_tool_name = tool_name
                input_handler.current_tool_arguments = arguments
                # Also set permission-specific attributes for consistency
                input_handler._permission_tool_name = tool_name
                input_handler._permission_arguments = arguments
                input_handler._permission_description = tool_description
                input_handler._permission_full_prompt = full_prompt
                
                # Send just the choice request, not the full prompt
                response = input_handler.get_input("Choice [y/a/A/n/d]: ")
            else:
                # Check if we have access to the subagent coordinator permission queue
                # If so, use it for consistent queuing of all permission requests
                agent = getattr(input_handler, '_agent', None)
                if (agent and hasattr(agent, 'subagent_coordinator') and 
                    hasattr(agent.subagent_coordinator, '_permission_queue') and
                    agent.subagent_coordinator._permission_queue is not None):
                    # Use the shared permission queue for consistent handling
                    import tempfile
                    import uuid
                    import time
                    import os
                    
                    # Create a mock subagent message for queue processing
                    request_id = str(uuid.uuid4())
                    temp_dir = tempfile.gettempdir()
                    response_file = os.path.join(temp_dir, f"main_process_response_{request_id}.txt")
                    
                    # Create a mock message similar to subagent permission requests
                    class MockMessage:
                        def __init__(self, content, data):
                            self.type = "permission_request"
                            self.content = content
                            self.data = data
                    
                    mock_message = MockMessage(
                        full_prompt,
                        {
                            "request_id": request_id,
                            "response_file": response_file,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "description": tool_description
                        }
                    )
                    
                    # Queue the permission request
                    await agent.subagent_coordinator._permission_queue.put((mock_message, "main-process"))
                    
                    # Wait for response file
                    timeout = 60
                    start_time = time.time()
                    while not os.path.exists(response_file):
                        if time.time() - start_time > timeout:
                            response = "n"  # Default to deny on timeout
                            break
                        await asyncio.sleep(0.1)
                    else:
                        # Read response from file
                        with open(response_file, "r") as f:
                            response = f.read().strip()
                        # Clean up temp file
                        try:
                            os.remove(response_file)
                        except:
                            pass
                else:
                    # Fallback to direct prompt display for cases without queue
                    if not use_clean_display:
                        print(full_prompt)
                        print()  # Add newline to fix cursor position
                    response = input_handler.get_input("Choice [y/a/A/n/d]: ")
                    
            # Get the user response
            if response is None:
                # Restore terminal state if using clean display
                if use_clean_display:
                    try:
                        self.clean_permission_display.restore_terminal()
                    except Exception as e:
                        logger.warning(f"Failed to restore terminal state after interrupt: {e}")
                        
                return ToolPermissionResult(
                    allowed=False, reason="User interrupted input"
                )
            response = response.strip().lower()
            
            # Restore terminal state if using clean display
            if use_clean_display:
                try:
                    self.clean_permission_display.restore_terminal()
                except Exception as e:
                    logger.warning(f"Failed to restore terminal state: {e}")

            if response in ["y", "yes"]:
                return ToolPermissionResult(allowed=True, reason="User approved once")

            elif response in ["a", "allow"]:
                self.session_approvals.add(tool_name)
                self._save_session_permissions()
                return ToolPermissionResult(
                    allowed=True, reason="User approved for session"
                )

            elif response in ["A", "auto"]:
                self.session_auto_approve = True
                self._save_session_permissions()
                return ToolPermissionResult(
                    allowed=True, reason="User enabled auto-approval for session"
                )

            elif response in ["n", "no"]:
                print(f"\r\n🚫 Tool execution denied.")
                return ToolPermissionResult(
                    allowed=False, reason="User denied once", return_to_prompt=True
                )

            elif response in ["d", "deny"]:
                self.session_denials.add(tool_name)
                self._save_session_permissions()
                print(f"\r\n🚫 Tool '{tool_name}' denied for this session.")
                return ToolPermissionResult(
                    allowed=False,
                    reason="User denied for session",
                    return_to_prompt=True,
                )

            else:
                print(f"Invalid response '{response}', defaulting to deny")
                return ToolPermissionResult(
                    allowed=False, reason="Invalid user response"
                )

        except Exception as e:
            logger.error(f"Error getting user permission: {e}")
            return ToolPermissionResult(
                allowed=False, reason=f"Error getting user input: {e}"
            )

    def reset_session_permissions(self):
        """Reset all session-based permissions."""
        self.session_approvals.clear()
        self.session_denials.clear()
        self.session_auto_approve = False
        self._save_session_permissions()

    def add_session_approval(self, tool_name: str):
        """Add a tool to session approvals."""
        self.session_approvals.add(tool_name)
        self.session_denials.discard(tool_name)  # Remove from denials if present
        self._save_session_permissions()

    def add_session_denial(self, tool_name: str):
        """Add a tool to session denials."""
        self.session_denials.add(tool_name)
        self.session_approvals.discard(tool_name)  # Remove from approvals if present
        self._save_session_permissions()

    def get_session_status(self) -> Dict[str, Any]:
        """Get current session permission status."""
        return {
            "auto_approve": self.session_auto_approve,
            "approved_tools": list(self.session_approvals),
            "denied_tools": list(self.session_denials),
            "config_allowed": self.config.allowed_tools,
            "config_disallowed": self.config.disallowed_tools,
        }
