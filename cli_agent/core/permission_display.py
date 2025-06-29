"""Clean permission display manager for terminal-based permission prompts.

This module provides a clean, focused permission prompt system that clears
the terminal, shows only the permission request, and restores the terminal
state after the user responds.
"""

import logging
import sys
from typing import Dict, Any, Optional

from cli_agent.core.terminal_state import get_terminal_state

logger = logging.getLogger(__name__)


class CleanPermissionDisplay:
    """Manages clean, focused permission prompts that clear the terminal."""

    def __init__(self):
        self.terminal_state = get_terminal_state()
        self.in_clean_mode = False

    def can_use_clean_display(self) -> bool:
        """Check if we can use clean terminal display."""
        return self.terminal_state.can_clear_screen()

    def show_permission_request(
        self,
        tool_name: str,
        tool_description: str,
        arguments: Dict[str, Any],
        task_id: str = "unknown",
        queue_position: int = 1,
        total_requests: int = 1,
    ) -> bool:
        """Show a clean permission request prompt.
        
        Args:
            tool_name: Name of the tool requesting permission
            tool_description: Human-readable description of what the tool will do
            arguments: Tool arguments (may be truncated for display)
            task_id: ID of the requesting task/subagent
            queue_position: Current position in permission queue
            total_requests: Total number of requests in queue
            
        Returns:
            bool: True if clean display was used, False if fallback needed
        """
        if not self.can_use_clean_display():
            return False

        # Clear the screen and enter clean mode
        if not self.terminal_state.clear_screen():
            return False

        self.in_clean_mode = True

        try:
            # Create the permission prompt content
            content = self._create_permission_content(
                tool_name=tool_name,
                tool_description=tool_description,
                arguments=arguments,
                task_id=task_id,
                queue_position=queue_position,
                total_requests=total_requests,
            )

            # Position and display the content
            centered_content = self.terminal_state.position_vertically_centered(content)
            print(centered_content, end='', flush=True)
            
            return True

        except Exception as e:
            logger.error(f"Error showing clean permission display: {e}")
            return False

    def restore_terminal(self) -> bool:
        """Restore the terminal to its previous state.
        
        Returns:
            bool: True if restoration was successful, False otherwise
        """
        if not self.in_clean_mode:
            return True

        try:
            success = self.terminal_state.restore_screen()
            self.in_clean_mode = False
            return success
        except Exception as e:
            logger.error(f"Error restoring terminal: {e}")
            self.in_clean_mode = False
            return False

    def _create_permission_content(
        self,
        tool_name: str,
        tool_description: str,
        arguments: Dict[str, Any],
        task_id: str,
        queue_position: int,
        total_requests: int,
    ) -> str:
        """Create the formatted permission prompt content.
        
        Returns:
            Formatted permission prompt as a string
        """
        # Determine source description
        if task_id == "main-process":
            source_description = "Main process is requesting tool permission:"
        else:
            source_description = f"Subagent {task_id} is requesting tool permission:"

        # Build content lines
        content_lines = [
            "",  # Top padding
            source_description,
            "",
            f"🔧 Tool: {tool_name}",
            f"📋 Action: {tool_description}",
        ]

        # Show arguments if they're not too large or sensitive
        if tool_name != "bash_execute" or len(str(arguments)) < 100:
            args_str = str(arguments)
            if len(args_str) > 200:
                args_str = args_str[:197] + "..."
            content_lines.append(f"Arguments: {args_str}")

        content_lines.extend([
            "",
            "Allow this tool to execute?",
            "",
            "[y] Yes, execute once",
            f"[a] Yes, and allow '{tool_name}' for the rest of this session",
            "[A] Yes, and auto-approve ALL tools for this session",
            "[n] No, deny this execution",
            f"[d] No, and deny '{tool_name}' for the rest of this session",
            "",
        ])

        # Add queue status if multiple requests
        if total_requests > 1:
            content_lines.append(f"📋 Permission Request {queue_position} of {total_requests}")
            content_lines.append("")

        content_lines.append("Choice [y/a/A/n/d]: ")

        # Join content lines
        content = "\n".join(content_lines)
        
        # Position vertically but don't center horizontally
        positioned_content = self.terminal_state.position_vertically_centered(content)

        return positioned_content

    def show_queue_update(self, remaining_requests: int) -> bool:
        """Show a brief queue update message.
        
        Args:
            remaining_requests: Number of requests remaining in queue
            
        Returns:
            bool: True if update was shown, False if not in clean mode
        """
        if not self.in_clean_mode:
            return False

        try:
            # Clear screen and show update
            self.terminal_state.clear_screen()
            
            update_content = f"""
✅ Permission handled.

Processing next request ({remaining_requests} remaining)...
"""
            
            # Position the update vertically but don't center horizontally
            positioned_update = self.terminal_state.position_vertically_centered(update_content.strip())
            
            print(positioned_update, end='', flush=True)
            
            # Brief pause to show the update
            import time
            time.sleep(1.0)
            
            return True

        except Exception as e:
            logger.error(f"Error showing queue update: {e}")
            return False

    def show_error_message(self, error_message: str) -> bool:
        """Show an error message in clean display mode.
        
        Args:
            error_message: Error message to display
            
        Returns:
            bool: True if error was shown, False if not in clean mode
        """
        if not self.in_clean_mode:
            return False

        try:
            error_content = f"❌ Error: {error_message}"
            
            positioned_error = self.terminal_state.position_vertically_centered(error_content)
            
            print(positioned_error, end='', flush=True)
            
            return True

        except Exception as e:
            logger.error(f"Error showing error message: {e}")
            return False

    def is_in_clean_mode(self) -> bool:
        """Check if currently in clean display mode."""
        return self.in_clean_mode

    def force_exit_clean_mode(self):
        """Force exit from clean mode (for error recovery)."""
        if self.in_clean_mode:
            try:
                self.terminal_state.restore_screen()
            except Exception:
                pass  # Best effort
            finally:
                self.in_clean_mode = False


# Global clean permission display instance
_clean_permission_display = None


def get_clean_permission_display() -> CleanPermissionDisplay:
    """Get the global clean permission display instance."""
    global _clean_permission_display
    if _clean_permission_display is None:
        _clean_permission_display = CleanPermissionDisplay()
    return _clean_permission_display


def format_tool_description(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Format a user-friendly description of what the tool will do.
    
    This is a utility function that can be used by permission systems
    to create consistent tool descriptions.
    """
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
        "glob": lambda args: f"Find files matching pattern: {args.get('pattern', 'N/A')}",
        "grep": lambda args: f"Search for text pattern: {args.get('pattern', 'N/A')}",
        "multiedit": lambda args: f"Make multiple edits to file: {args.get('file_path', 'N/A')}",
    }

    formatter = descriptions.get(tool_name)
    if formatter:
        try:
            return formatter(arguments)
        except Exception:
            pass

    return f"Execute tool: {tool_name}"