#!/usr/bin/env python3
# This script implements the main command-line interface for the MCP Agent.
"""Base MCP Agent implementation with shared functionality."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import signal
import termios
import tty
import select
import time
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

import click

from fastmcp.client import Client as FastMCPClient, StdioTransport
import subprocess
import json

from config import HostConfig, load_config

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Suppress WARNING messages during interactive chat
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SlashCommandManager:
    """Manages slash commands similar to Claude Code's system."""
    
    def __init__(self, agent: 'BaseMCPAgent'):
        self.agent = agent
        self.custom_commands = {}
        self.load_custom_commands()
    
    def load_custom_commands(self):
        """Load custom commands from .claude/commands/ and ~/.claude/commands/"""
        # Project-specific commands
        project_commands_dir = Path(".claude/commands")
        if project_commands_dir.exists():
            self._load_commands_from_dir(project_commands_dir, "project")
        
        # Personal commands
        personal_commands_dir = Path.home() / ".claude/commands"
        if personal_commands_dir.exists():
            self._load_commands_from_dir(personal_commands_dir, "personal")
    
    def _load_commands_from_dir(self, commands_dir: Path, command_type: str):
        """Load commands from a directory."""
        for command_file in commands_dir.glob("*.md"):
            try:
                with open(command_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                command_name = command_file.stem
                self.custom_commands[command_name] = {
                    "content": content,
                    "type": command_type,
                    "file": str(command_file)
                }
                logger.debug(f"Loaded {command_type} command: {command_name}")
            except Exception as e:
                logger.warning(f"Failed to load command {command_file}: {e}")
    
    async def handle_slash_command(self, command_line: str) -> Optional[str]:
        """Handle a slash command and return response if handled."""
        if not command_line.startswith('/'):
            return None
        
        # Parse command and arguments
        parts = command_line[1:].split(' ', 1)
        command = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        # Handle built-in commands
        if command == "help":
            return self._handle_help()
        elif command == "clear":
            return self._handle_clear()
        elif command == "model":
            return self._handle_model(args)
        elif command == "review":
            return self._handle_review(args)
        elif command.startswith("mcp__"):
            return await self._handle_mcp_command(command, args)
        elif ":" in command:
            # Custom namespaced command
            return await self._handle_custom_command(command, args)
        elif command in self.custom_commands:
            # Simple custom command
            return await self._handle_custom_command(command, args)
        else:
            return f"Unknown command: /{command}. Type /help for available commands."
    
    def _handle_help(self) -> str:
        """Handle /help command."""
        help_text = """Available Commands:

Built-in Commands:
  /help           - Show this help message
  /clear          - Clear conversation history
  /model [name]   - Show current model or set model
  /review [file]  - Request code review

Custom Commands:"""
        
        if self.custom_commands:
            for cmd_name, cmd_info in self.custom_commands.items():
                help_text += f"\n  /{cmd_name}         - {cmd_info['type']} command"
        else:
            help_text += "\n  (No custom commands found)"
        
        # Add MCP commands if available
        mcp_commands = self._get_mcp_commands()
        if mcp_commands:
            help_text += "\n\nMCP Commands:"
            for cmd in mcp_commands:
                help_text += f"\n  /{cmd}"
        
        return help_text
    
    def _handle_clear(self) -> str:
        """Handle /clear command."""
        if hasattr(self.agent, 'conversation_history'):
            self.agent.conversation_history.clear()
            return "Conversation history cleared."
        return "No conversation history to clear."
    
    def _handle_model(self, args: str) -> str:
        """Handle /model command."""
        if not args.strip():
            # Show current model
            if hasattr(self.agent, 'config'):
                if hasattr(self.agent.config, 'get_deepseek_config'):
                    return f"Current model: {self.agent.config.get_deepseek_config().model}"
                elif hasattr(self.agent.config, 'get_gemini_config'):
                    return f"Current model: {self.agent.config.get_gemini_config().model}"
            return "Current model: Unknown"
        else:
            return "Model switching not implemented yet. Use environment variables to change models."
    
    def _handle_review(self, args: str) -> str:
        """Handle /review command."""
        if args.strip():
            file_path = args.strip()
            return f"Code review requested for: {file_path}\n\nNote: Automated code review not implemented yet. Please use the agent's normal chat to request code review."
        else:
            return "Please specify a file to review: /review <file_path>"
    
    async def _handle_mcp_command(self, command: str, args: str) -> str:
        """Handle MCP slash commands."""
        # Parse MCP command: mcp__<server-name>__<prompt-name>
        parts = command.split('__')
        if len(parts) != 3 or parts[0] != "mcp":
            return f"Invalid MCP command format: /{command}"
        
        server_name = parts[1]
        prompt_name = parts[2]
        
        # Check if we have this MCP server
        if hasattr(self.agent, 'available_tools'):
            # Look for matching tools from this server
            matching_tools = [tool for tool in self.agent.available_tools.keys() 
                             if tool.startswith(f"{server_name}:")]
            if not matching_tools:
                return f"MCP server '{server_name}' not found or has no available tools."
        
        return f"MCP command execution not fully implemented yet.\nServer: {server_name}\nPrompt: {prompt_name}\nArgs: {args}"
    
    async def _handle_custom_command(self, command: str, args: str) -> str:
        """Handle custom commands."""
        # Handle namespaced commands (prefix:command)
        if ":" in command:
            prefix, cmd_name = command.split(":", 1)
            full_command = command
        else:
            cmd_name = command
            full_command = command
        
        if cmd_name not in self.custom_commands:
            return f"Custom command not found: /{full_command}"
        
        cmd_info = self.custom_commands[cmd_name]
        content = cmd_info["content"]
        
        # Replace $ARGUMENTS placeholder
        if args:
            content = content.replace("$ARGUMENTS", args)
        else:
            content = content.replace("$ARGUMENTS", "")
        
        return f"Executing custom command '{cmd_name}':\n\n{content}"
    
    def _get_mcp_commands(self) -> List[str]:
        """Get available MCP commands."""
        mcp_commands = []
        if hasattr(self.agent, 'available_tools'):
            # Group tools by server and create MCP commands
            servers = set()
            for tool_name in self.agent.available_tools.keys():
                if ":" in tool_name and not tool_name.startswith("builtin:"):
                    server_name = tool_name.split(":")[0]
                    servers.add(server_name)
            
            for server in servers:
                mcp_commands.append(f"mcp__{server}__<prompt-name>")
        
        return mcp_commands


class InterruptibleInput:
    """Enhanced input handler with cursor movement and escape key interrupt support."""
    
    def __init__(self):
        self.interrupted = False
        self.old_settings = None
        
    def setup_terminal(self):
        """Setup terminal for raw input."""
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())
            # Enable bracketed paste mode
            sys.stdout.write('\x1b[?2004h')
            sys.stdout.flush()
    
    def restore_terminal(self):
        """Restore terminal settings."""
        if self.old_settings and sys.stdin.isatty():
            # Disable bracketed paste mode
            sys.stdout.write('\x1b[?2004l')
            sys.stdout.flush()
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)
    
    def get_input(self, prompt: str, multiline_mode: bool = False) -> Optional[str]:
        """Get input with cursor movement support and escape interrupt.
        
        Args:
            prompt: The prompt to display
            multiline_mode: If True, requires empty line to send. If False, sends on Enter.
        """
        # Ensure we start at the beginning of the line
        sys.stdout.write('\r' + prompt)
        sys.stdout.flush()
        
        if not sys.stdin.isatty():
            # Fallback for non-tty environments
            try:
                return input()
            except KeyboardInterrupt:
                self.interrupted = True
                return None
        
        self.setup_terminal()
        try:
            line = ""
            cursor_pos = 0
            rapid_input_buffer = []
            last_input_time = 0
            
            while True:
                # Check if data is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    current_time = time.time()
                    
                    if not char:  # EOF
                        break
                    
                    # Check if this is rapid input (potential paste)
                    is_rapid_input = last_input_time > 0 and (current_time - last_input_time) < 0.01
                    
                    if is_rapid_input:
                        # Accumulate rapid input
                        rapid_input_buffer.append(char)
                        last_input_time = current_time
                        continue
                    elif rapid_input_buffer:
                        # Process accumulated rapid input as paste
                        pasted_text = ''.join(rapid_input_buffer)
                        rapid_input_buffer = []
                        
                        # Add current char to the paste
                        pasted_text += char
                        
                        # Check for more rapid input with longer timeout to capture full paste
                        while select.select([sys.stdin], [], [], 0.05)[0]:  # Longer timeout for full paste
                            next_char = sys.stdin.read(1)
                            if next_char:
                                pasted_text += next_char
                            else:
                                break
                        
                        # Process the paste - don't treat any characters as special keys
                        if '\n' in pasted_text:
                            print("(Looks like multiline content - processing immediately)")
                            return pasted_text.rstrip('\n')  # Remove trailing newline
                        else:
                            # Single line paste - add to current line and continue
                            line = line[:cursor_pos] + pasted_text + line[cursor_pos:]
                            cursor_pos += len(pasted_text)
                            # Use the same display logic as regular characters
                            import shutil
                            try:
                                term_width = shutil.get_terminal_size().columns
                            except:
                                term_width = 80
                            
                            # Calculate display line with cursor position indicator
                            prompt_len = len(prompt)
                            available_width = term_width - prompt_len - 1
                            
                            if len(line) <= available_width:
                                # Line fits, display normally
                                display_line = line
                                display_cursor = cursor_pos
                            else:
                                # Line is too long, show window around cursor
                                if cursor_pos < available_width // 2:
                                    # Cursor near start
                                    display_line = line[:available_width-3] + "..."
                                    display_cursor = cursor_pos
                                elif cursor_pos > len(line) - available_width // 2:
                                    # Cursor near end
                                    display_line = "..." + line[-(available_width-3):]
                                    display_cursor = available_width - (len(line) - cursor_pos)
                                else:
                                    # Cursor in middle
                                    start = cursor_pos - available_width // 2 + 3
                                    display_line = "..." + line[start:start + available_width - 6] + "..."
                                    display_cursor = available_width // 2
                            
                            # Clear line and redraw
                            sys.stdout.write('\r\x1b[K' + prompt + display_line)
                            if display_cursor < len(display_line):
                                sys.stdout.write('\x1b[{}G'.format(prompt_len + display_cursor + 1))
                            sys.stdout.flush()
                            # Reset timing to prevent next character from being treated as rapid input
                            last_input_time = 0
                            continue
                    
                    last_input_time = current_time
                        
                    # Handle special keys (only process escape if NOT rapid input)
                    if char == '\x1b':  # Escape sequence
                        # Check if this is just an escape key or arrow key sequence
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            seq = sys.stdin.read(1)
                            if seq == '[':
                                next_seq = sys.stdin.read(1)
                                if next_seq == 'D':  # Left arrow
                                    if cursor_pos > 0:
                                        cursor_pos -= 1
                                        sys.stdout.write('\x1b[D')
                                        sys.stdout.flush()
                                elif next_seq == 'C':  # Right arrow
                                    if cursor_pos < len(line):
                                        cursor_pos += 1
                                        sys.stdout.write('\x1b[C')
                                        sys.stdout.flush()
                                elif next_seq == 'A' or next_seq == 'B':  # Up/Down arrow
                                    pass  # Ignore for now
                        else:
                            # Just escape key - interrupt
                            self.interrupted = True
                            sys.stdout.write('\r\x1b[K[Interrupted by Escape key]\n')
                            sys.stdout.flush()
                            return None
                    elif char == '\x03':  # Ctrl+C
                        self.interrupted = True
                        # Move to beginning of line and clear, then print message
                        sys.stdout.write('\r\x1b[K[Interrupted by Ctrl+C]\n')
                        sys.stdout.flush()
                        return None
                    elif char.isprintable():  # Regular character
                        line = line[:cursor_pos] + char + line[cursor_pos:]
                        cursor_pos += 1
                        # Use a simpler approach - get terminal width and handle display
                        import shutil
                        try:
                            term_width = shutil.get_terminal_size().columns
                        except:
                            term_width = 80
                        
                        # Calculate display line with cursor position indicator
                        prompt_len = len(prompt)
                        available_width = term_width - prompt_len - 1
                        
                        if len(line) <= available_width:
                            # Line fits, display normally
                            display_line = line
                            display_cursor = cursor_pos
                        else:
                            # Line is too long, show window around cursor
                            if cursor_pos < available_width // 2:
                                # Cursor near start
                                display_line = line[:available_width-3] + "..."
                                display_cursor = cursor_pos
                            elif cursor_pos > len(line) - available_width // 2:
                                # Cursor near end
                                display_line = "..." + line[-(available_width-3):]
                                display_cursor = available_width - (len(line) - cursor_pos)
                            else:
                                # Cursor in middle
                                start = cursor_pos - available_width // 2 + 3
                                display_line = "..." + line[start:start + available_width - 6] + "..."
                                display_cursor = available_width // 2
                        
                        # Clear line and redraw
                        sys.stdout.write('\r\x1b[K' + prompt + display_line)
                        if display_cursor < len(display_line):
                            sys.stdout.write('\x1b[{}G'.format(prompt_len + display_cursor + 1))
                        sys.stdout.flush()
                    elif char == '\x7f':  # Backspace
                        if cursor_pos > 0:
                            line = line[:cursor_pos-1] + line[cursor_pos:]
                            cursor_pos -= 1
                            # Use the same display logic as regular characters
                            import shutil
                            try:
                                term_width = shutil.get_terminal_size().columns
                            except:
                                term_width = 80
                            
                            # Calculate display line with cursor position indicator
                            prompt_len = len(prompt)
                            available_width = term_width - prompt_len - 1
                            
                            if len(line) <= available_width:
                                # Line fits, display normally
                                display_line = line
                                display_cursor = cursor_pos
                            else:
                                # Line is too long, show window around cursor
                                if cursor_pos < available_width // 2:
                                    # Cursor near start
                                    display_line = line[:available_width-3] + "..."
                                    display_cursor = cursor_pos
                                elif cursor_pos > len(line) - available_width // 2:
                                    # Cursor near end
                                    display_line = "..." + line[-(available_width-3):]
                                    display_cursor = available_width - (len(line) - cursor_pos)
                                else:
                                    # Cursor in middle
                                    start = cursor_pos - available_width // 2 + 3
                                    display_line = "..." + line[start:start + available_width - 6] + "..."
                                    display_cursor = available_width // 2
                            
                            # Clear line and redraw
                            sys.stdout.write('\r\x1b[K' + prompt + display_line)
                            if display_cursor < len(display_line):
                                sys.stdout.write('\x1b[{}G'.format(prompt_len + display_cursor + 1))
                            sys.stdout.flush()
                    elif char == '\r' or char == '\n':  # Enter
                        print()
                        if multiline_mode and line.strip() == "":
                            # In multiline mode, empty line means send
                            return line.rstrip()
                        else:
                            # Normal mode or non-empty line
                            return line
                    # Ignore other control characters
                        
        finally:
            self.restore_terminal()
        
        return line if line else None
    
    def get_multiline_input(self, initial_prompt: str) -> Optional[str]:
        """Get input with smart paste detection."""
        # Get input - bracketed paste will be detected automatically
        user_input = self.get_input(initial_prompt, multiline_mode=False)
        if user_input is None:
            return None
        
        # Check if this looks like it might be incomplete multiline content
        # (for cases where paste doesn't use bracketed paste mode)
        # Be very conservative to avoid false positives on normal questions
        is_likely_incomplete = (
            len(user_input) > 300 or  # Very long line (further increased)
            (user_input.startswith('def ') or user_input.startswith('class ')) or  # Actual code definitions
            '```' in user_input or  # Code blocks
            (user_input.endswith(':') and len(user_input) > 80) or  # Long lines ending with colon
            (user_input.endswith('{') and len(user_input) > 30) or  # Likely actual code, not just mentioning braces
            user_input.endswith('\\') or  # Backslash continuation
            user_input.count('\n') > 0  # Already contains newlines
        )
        
        if is_likely_incomplete and '\n' not in user_input:
            print("(Looks like multiline content - press Enter on empty line to send, or continue typing)")
            
            # Switch to multiline mode
            lines = [user_input]
            while True:
                line = self.get_input("... ", multiline_mode=True)
                if line is None:  # Interrupted
                    return None
                if line.strip() == "":
                    break
                lines.append(line)
            
            return '\n'.join(lines)
        
        return user_input


class BaseMCPAgent(ABC):
    """Base class for MCP agents with shared functionality."""
    
    def __init__(self, config: HostConfig):
        self.config = config
        self.mcp_clients: Dict[str, ClientSession] = {}
        self.available_tools: Dict[str, Dict] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Add built-in tools
        self._add_builtin_tools()
        
        # Initialize slash command manager
        self.slash_commands = SlashCommandManager(self)
        
        logger.info(f"Initialized Base MCP Agent with {len(self.available_tools)} built-in tools")
    
    def _add_builtin_tools(self):
        """Add built-in tools to the available tools."""
        builtin_tools = {
            "builtin:bash_execute": {
                "server": "builtin",
                "name": "bash_execute",
                "description": "Execute a bash command and return the output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"},
                        "timeout": {"type": "integer", "default": 120, "description": "Timeout in seconds"}
                    },
                    "required": ["command"]
                },
                "client": None
            },
            "builtin:read_file": {
                "server": "builtin",
                "name": "read_file",
                "description": "Read contents of a file with line numbers",
                "schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to read"},
                        "offset": {"type": "integer", "description": "Line number to start from"},
                        "limit": {"type": "integer", "description": "Number of lines to read"}
                    },
                    "required": ["file_path"]
                },
                "client": None
            },
            "builtin:write_file": {
                "server": "builtin",
                "name": "write_file",
                "description": "Write content to a file",
                "schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["file_path", "content"]
                },
                "client": None
            },
            "builtin:list_directory": {
                "server": "builtin",
                "name": "list_directory",
                "description": "List contents of a directory",
                "schema": {
                    "type": "object",
                    "properties": {
                        "directory_path": {"type": "string", "description": "Path to the directory to list"}
                    },
                    "required": ["directory_path"]
                },
                "client": None
            },
            "builtin:get_current_directory": {
                "server": "builtin",
                "name": "get_current_directory",
                "description": "Get the current working directory",
                "schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "client": None
            },
            "builtin:todo_read": {
                "server": "builtin",
                "name": "todo_read",
                "description": "Read the current todo list",
                "schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "client": None
            },
            "builtin:todo_write": {
                "server": "builtin",
                "name": "todo_write",
                "description": "Write/update the todo list",
                "schema": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "content": {"type": "string"},
                                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                                },
                                "required": ["id", "content", "status", "priority"]
                            }
                        }
                    },
                    "required": ["todos"]
                },
                "client": None
            },
            "builtin:replace_in_file": {
                "server": "builtin",
                "name": "replace_in_file",
                "description": "Replace text in a file",
                "schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file"},
                        "old_text": {"type": "string", "description": "Text to replace"},
                        "new_text": {"type": "string", "description": "New text to replace with"}
                    },
                    "required": ["file_path", "old_text", "new_text"]
                },
                "client": None
            },
            "builtin:edit_file": {
                "server": "builtin",
                "name": "edit_file",
                "description": "Edit a file using unified diff format patches",
                "schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to edit"},
                        "diff": {"type": "string", "description": "Unified diff format patch to apply to the file"}
                    },
                    "required": ["file_path", "diff"]
                },
                "client": None
            },
            "builtin:webfetch": {
                "server": "builtin",
                "name": "webfetch",
                "description": "Fetch content from a webpage",
                "schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "limit": {"type": "integer", "description": "Optional limit to truncate the HTML response by this number of lines (default: 1000)"}
                    },
                    "required": ["url"]
                },
                "client": None
            }
        }
        
        self.available_tools.update(builtin_tools)
    
    def _execute_builtin_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
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
        elif tool_name == "edit_file":
            return self._edit_file(args)
        elif tool_name == "webfetch":
            return self._webfetch(args)
        else:
            return f"Unknown built-in tool: {tool_name}"
    
    def _bash_execute(self, args: Dict[str, Any]) -> str:
        """Execute a bash command and return the output."""
        command = args.get("command", "")
        timeout = args.get("timeout", 120)
        
        if not command:
            return "Error: No command provided"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
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
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_idx = max(0, offset - 1)  # Convert to 0-based index
            end_idx = len(lines) if limit is None else min(len(lines), start_idx + limit)
            
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
            if dir_path:  # Only create directory if it's not empty (i.e., file is not in current dir)
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
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
            
            with open(todo_file, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            return f"Error reading todo list: {str(e)}"
    
    def _todo_write(self, args: Dict[str, Any]) -> str:
        """Write/update the todo list."""
        todos = args.get("todos", [])
        todo_file = "todo.json"
        
        try:
            with open(todo_file, 'w', encoding='utf-8') as f:
                json.dump(todos, f, indent=2)
            
            return f"Successfully updated todo list with {len(todos)} items"
            
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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_text not in content:
                return f"Error: Text not found in file: {old_text}"
            
            new_content = content.replace(old_text, new_text)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            count = content.count(old_text)
            return f"Successfully replaced {count} occurrence(s) of text in {file_path}"
            
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error replacing text in file: {str(e)}"
    
    def _edit_file(self, args: Dict[str, Any]) -> str:
        """Edit a file using unified diff format."""
        file_path = Path(args["file_path"]).resolve()
        diff_content = args["diff"]
        
        try:
            if not file_path.exists():
                return f"Error: File does not exist: {file_path}"
            
            # Read the original file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                original_lines = f.readlines()
            
            # Unescape JSON escape sequences in the diff content
            # This handles cases where Deepseek escapes \n, \", etc.
            try:
                # Try to decode JSON escape sequences
                import codecs
                unescaped_diff = codecs.decode(diff_content, 'unicode_escape')
            except:
                # If that fails, just use the original content
                unescaped_diff = diff_content
            
            # Debug: log the diff content and first few lines
            logger.warning(f"Applying diff to {file_path}")
            logger.warning(f"Original diff content: {repr(diff_content[:200])}")
            logger.warning(f"Unescaped diff content: {repr(unescaped_diff[:200])}")
            logger.warning(f"First 3 original lines: {[repr(line) for line in original_lines[:3]]}")
            
            # Parse and apply the diff
            modified_lines = self._apply_diff(original_lines, unescaped_diff)
            
            if modified_lines is None:
                return "Error: Failed to apply diff - invalid diff format or patch doesn't match file content"
            
            # Write the modified content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)
            
            return f"Successfully applied diff to {file_path}. Modified {len(modified_lines)} lines."
            
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    def _apply_diff(self, original_lines: List[str], diff_content: str) -> Optional[List[str]]:
        """Apply a unified diff to the original lines."""
        import re
        
        # Parse the diff into hunks
        hunks = []
        current_hunk = None
        
        for line in diff_content.split('\n'):
            line = line.rstrip('\r\n')
            
            # Look for hunk headers (@@)
            hunk_match = re.match(r'^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@', line)
            if hunk_match:
                if current_hunk:
                    hunks.append(current_hunk)
                
                old_start = int(hunk_match.group(1)) - 1  # Convert to 0-based indexing
                old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                new_start = int(hunk_match.group(3)) - 1  # Convert to 0-based indexing
                new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1
                
                current_hunk = {
                    'old_start': old_start,
                    'old_count': old_count,
                    'new_start': new_start,
                    'new_count': new_count,
                    'lines': []
                }
            elif current_hunk is not None:
                # Process diff lines
                if line.startswith(' '):
                    current_hunk['lines'].append(('context', line[1:] + '\n'))
                elif line.startswith('-'):
                    current_hunk['lines'].append(('remove', line[1:] + '\n'))
                elif line.startswith('+'):
                    current_hunk['lines'].append(('add', line[1:] + '\n'))
                elif line.startswith('\\'):
                    # Handle "No newline at end of file" markers
                    continue
        
        if current_hunk:
            hunks.append(current_hunk)
        
        if not hunks:
            return None
        
        # Apply hunks in reverse order to preserve line numbers
        result_lines = original_lines.copy()
        
        for hunk in reversed(hunks):
            old_start = hunk['old_start']
            old_count = hunk['old_count']
            
            # Verify the context matches
            context_check = []
            add_lines = []
            remove_count = 0
            
            for action, content in hunk['lines']:
                if action == 'context':
                    context_check.append(content)
                elif action == 'remove':
                    context_check.append(content)
                    remove_count += 1
                elif action == 'add':
                    add_lines.append(content)
            
            # Check if the original content matches what we expect to remove
            try:
                original_section = result_lines[old_start:old_start + old_count]
                context_index = 0
                
                for action, content in hunk['lines']:
                    if action in ['context', 'remove']:
                        if context_index >= len(original_section):
                            logger.warning(f"Diff context mismatch at line {old_start + context_index + 1}: reached end of file")
                            logger.warning(f"Expected: {repr(content)}")
                            return None
                        elif original_section[context_index] != content:
                            logger.warning(f"Diff context mismatch at line {old_start + context_index + 1}")
                            logger.warning(f"Expected: {repr(content)}")
                            logger.warning(f"Found: {repr(original_section[context_index])}")
                            return None
                        context_index += 1
                
                # Apply the changes
                new_section = []
                for action, content in hunk['lines']:
                    if action in ['context', 'add']:
                        new_section.append(content)
                
                # Replace the section
                result_lines[old_start:old_start + old_count] = new_section
                
            except IndexError:
                return None
        
        return result_lines
    
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
            timeout=35  # Slightly longer than curl timeout
        )

        # Try to decode with utf-8, then fall back to latin-1 (which rarely fails)
        try:
            content = result.stdout.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {url}. Falling back to latin-1.")
            content = result.stdout.decode('latin-1', errors='replace')

        if result.returncode != 0:
            # Try to decode stderr for a better error message
            try:
                stderr = result.stderr.decode('utf-8', errors='replace')
            except:
                stderr = repr(result.stderr)
            
            error_msg = f"Error fetching URL (curl return code {result.returncode}): {stderr}"
            
            # If we have content despite the error, include it
            if content.strip():
                return f"{error_msg}\n\nContent retrieved:\n{content}"
            else:
                return error_msg

        # Truncate the content by lines if limit is specified
        if limit is not None and isinstance(limit, int) and limit > 0:
            lines = content.split('\n')
            if len(lines) > limit:
                content = '\n'.join(lines[:limit])
                content += f"\n\n[Content truncated at {limit} lines. Original had {len(lines)} lines.]"

        # Return the content (truncated if limit was provided)
        return content
    
    async def start_mcp_server(self, server_name: str, server_config) -> bool:
        """Start and connect to an MCP server using FastMCP."""
        try:
            logger.info(f"Starting MCP server: {server_name}")
            
            # Construct command and args for FastMCP client
            command = server_config.command[0]
            args = server_config.command[1:] + server_config.args
            
            # Create FastMCP client with stdio transport
            transport = StdioTransport(command=command, args=args, env=server_config.env)
            client = FastMCPClient(transport=transport)
            
            # Enter the context manager and store it for cleanup
            context_manager = client.__aenter__()
            await context_manager
            
            # Store the client and context manager
            self.mcp_clients[server_name] = client
            self._mcp_contexts = getattr(self, '_mcp_contexts', {})
            self._mcp_contexts[server_name] = client
            
            # Get available tools from this server
            tools_result = await client.list_tools()
            if tools_result and hasattr(tools_result, 'tools'):
                for tool in tools_result.tools:
                    tool_key = f"{server_name}:{tool.name}"
                    self.available_tools[tool_key] = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        "client": client
                    }
                    logger.info(f"Registered tool: {tool_key}")
            elif hasattr(tools_result, '__len__'):
                # Handle list format
                for tool in tools_result:
                    tool_key = f"{server_name}:{tool.name}"
                    self.available_tools[tool_key] = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        "client": client
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
        if hasattr(self, '_mcp_contexts'):
            self._mcp_contexts.clear()
    
    async def _execute_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool (built-in or external) and return the result."""
        try:
            if tool_key not in self.available_tools:
                return f"Error: Tool {tool_key} not found"
            
            tool_info = self.available_tools[tool_key]
            tool_name = tool_info["name"]
            
            # Check if it's a built-in tool
            if tool_info["server"] == "builtin":
                logger.info(f"Executing built-in tool: {tool_name}")
                return self._execute_builtin_tool(tool_name, arguments)
            
            # Handle external MCP tools with FastMCP
            client = tool_info["client"]
            if client is None:
                return f"Error: No client session for tool {tool_key}"
            
            logger.info(f"Executing MCP tool: {tool_name}")
            result = await client.call_tool(tool_name, arguments)
            
            # Format the result for FastMCP
            if hasattr(result, 'content') and result.content:
                content_parts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        content_parts.append(content.text)
                    elif hasattr(content, 'data'):
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
            logger.error(f"Error executing tool {tool_key}: {e}")
            return f"Error executing tool {tool_key}: {str(e)}"

    async def _execute_mcp_tool_with_keepalive(self, tool_key: str, arguments: Dict[str, Any], input_handler=None, keepalive_interval: float = 5.0) -> tuple:
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
                await asyncio.wait_for(asyncio.shield(tool_task), timeout=keepalive_interval)
                break  # Task completed
            except asyncio.TimeoutError:
                # Task is still running, send keep-alive message
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                # Create a keep-alive message
                keepalive_msg = f"⏳ Tool {tool_key} still running... ({elapsed:.1f}s elapsed)"
                if input_handler:
                    keepalive_msg += ", press ESC to cancel"
                keepalive_messages.append(keepalive_msg)
                logger.debug(f"Keep-alive: {keepalive_msg}")
                continue
        
        # Get the final result
        result = await tool_task
        return result, keepalive_messages
    
    def _create_system_prompt(self, for_first_message: bool = False) -> str:
        """Create a basic system prompt that includes tool information."""
        tool_descriptions = []
        
        for tool_key, tool_info in self.available_tools.items():
            # Use the converted name format (with underscores)
            converted_tool_name = tool_key.replace(":", "_")
            description = tool_info["description"]
            tool_descriptions.append(f"- **{converted_tool_name}**: {description}")
        
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        
        # Basic system prompt - subclasses can override this
        system_prompt = f"""You are a top-tier autonomous software development agent. You are in control and responsible for completing the user's request.

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
2.  **Prefer `replace_in_file`:** For simple changes, `builtin_replace_in_file` is the best tool.
3.  **Use `edit_file` for complexity:** For multi-line or complex changes, use `builtin_edit_file` with unified diff format.
4.  **Chunk changes:** Break large edits into smaller, incremental changes to maintain control and clarity.

**Todo List Workflow:**
- **Use the Todo list:** Use `builtin_todo_read` and `builtin_todo_write` to manage your tasks.
- **Start with a plan:** At the beginning of your session, create a todo list to outline your steps.
- **Update as you go:** As you complete tasks, update the todo list to reflect your progress.

**Workflow:**
1.  **Reason:** Outline your plan.
2.  **Act:** Use one or more tool calls to execute your plan. Use parallel tool calls when it makes sense.
3.  **Respond:** When you have completed the request, provide the final answer to the user.

**Available Tools:**
{tools_text}

You are the expert. Complete the task."""

        return system_prompt
    
    def format_markdown(self, text: str) -> str:
        """Format markdown text for terminal display."""
        if not text:
            return text
            
        # Simple terminal-friendly markdown formatting
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Headers
            if line.startswith('# '):
                formatted_lines.append(f"\n\033[1m\033[4m{line[2:]}\033[0m")  # Bold + underline
            elif line.startswith('## '):
                formatted_lines.append(f"\n\033[1m{line[3:]}\033[0m")  # Bold
            elif line.startswith('### '):
                formatted_lines.append(f"\n\033[1m{line[4:]}\033[0m")  # Bold
            
            # Code blocks
            elif line.strip().startswith('```'):
                if line.strip() == '```':
                    formatted_lines.append("\033[2m" + line + "\033[0m")  # Dim
                else:
                    formatted_lines.append("\033[2m" + line + "\033[0m")  # Dim
            
            # Lists
            elif re.match(r'^\s*[-*+]\s', line):
                formatted_lines.append(f"\033[36m•\033[0m{line[line.index(' ', line.index('-') if '-' in line else line.index('*') if '*' in line else line.index('+')):]}")
            elif re.match(r'^\s*\d+\.\s', line):
                formatted_lines.append(f"\033[36m{line.split('.')[0]}.\033[0m{line[line.index('.') + 1:]}")
            
            # Regular line - process inline formatting
            else:
                # Bold
                line = re.sub(r'\*\*(.*?)\*\*', r'\033[1m\1\033[0m', line)
                # Italic (using dim since true italic isn't widely supported)
                line = re.sub(r'\*(.*?)\*', r'\033[3m\1\033[0m', line)
                # Inline code
                line = re.sub(r'`(.*?)`', r'\033[47m\033[30m\1\033[0m', line)
                
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters for most models)."""
        return len(text) // 4
    
    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count estimated tokens in the conversation."""
        total_tokens = 0
        for message in messages:
            if isinstance(message.get('content'), str):
                total_tokens += self.estimate_tokens(message['content'])
            # Add small overhead for role and structure
            total_tokens += 10
        return total_tokens
    
    def get_token_limit(self) -> int:
        """Get the context token limit for the current model."""
        # Default limits - subclasses can override
        return 32000  # Conservative estimate
    
    def should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if conversation should be compacted."""
        current_tokens = self.count_conversation_tokens(messages)
        limit = self.get_token_limit()
        # Compact when we're at 80% of the limit
        return current_tokens > (limit * 0.8)
    
    async def compact_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a compact summary of the conversation to preserve context while reducing tokens."""
        if len(messages) <= 3:  # Keep conversations that are already short
            return messages
        
        # Always keep the first message (system prompt) and last 2 messages
        system_message = messages[0] if messages[0].get('role') == 'system' else None
        recent_messages = messages[-2:]
        
        # Messages to summarize (everything except system and last 2)
        start_idx = 1 if system_message else 0
        messages_to_summarize = messages[start_idx:-2]
        
        if not messages_to_summarize:
            return messages
        
        # Create summary prompt
        conversation_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages_to_summarize
        ])
        
        summary_prompt = f"""Please create a concise summary of this conversation that preserves:
1. Key decisions and actions taken
2. Important file changes or tool usage
3. Current project state and context
4. Any pending tasks or next steps

Conversation to summarize:
{conversation_text}

Provide a brief but comprehensive summary that maintains continuity for ongoing work."""

        try:
            # Use the current model to create summary
            summary_messages = [{"role": "user", "content": summary_prompt}]
            summary_response = await self.generate_response(summary_messages, tools=None)
            
            # Create condensed conversation
            condensed = []
            if system_message:
                condensed.append(system_message)
            
            # Add summary as a system message
            condensed.append({
                "role": "system", 
                "content": f"[CONVERSATION SUMMARY] {summary_response}"
            })
            
            # Add recent messages
            condensed.extend(recent_messages)
            
            print(f"\n🗜️  Conversation compacted: {len(messages)} → {len(condensed)} messages")
            return condensed
            
        except Exception as e:
            print(f"⚠️  Failed to compact conversation: {e}")
            # Fallback: just keep system + last 5 messages
            fallback = []
            if system_message:
                fallback.append(system_message)
            fallback.extend(messages[-5:])
            return fallback
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> Union[str, Any]:
        """Generate a response using the specific LLM. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to the specific LLM's format. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from the LLM response. Must be implemented by subclasses."""
        pass
    
    async def interactive_chat(self, input_handler: InterruptibleInput):
        """Interactive chat session with shared functionality."""
        messages = []
        current_task = None
        
        print("Starting interactive chat. Type 'quit' or 'exit' to end, 'tools' to list available tools.")
        print("Use /help for slash commands. Press ESC at any time to interrupt operations.\n")
        
        while True:
            try:
                # Cancel any pending task if interrupted
                if input_handler.interrupted and current_task and not current_task.done():
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
                
                if user_input.lower().strip() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower().strip() == 'tools':
                    print("\nAvailable tools:")
                    for tool_name, tool_info in self.available_tools.items():
                        print(f"  {tool_name}: {tool_info['description']}")
                    print()
                    continue
                
                # Handle slash commands
                if user_input.strip().startswith('/'):
                    try:
                        slash_response = await self.slash_commands.handle_slash_command(user_input.strip())
                        if slash_response:
                            print(f"\n{slash_response}\n")
                            continue
                    except Exception as e:
                        print(f"\nError handling slash command: {e}\n")
                        continue
                
                if not user_input.strip():
                    # Empty input, just continue
                    continue
                
                # Add user message
                messages.append({"role": "user", "content": user_input})
                
                # Create response task
                tools_list = self.convert_tools_to_llm_format()
                current_task = asyncio.create_task(
                    self.generate_response(messages, tools_list)
                )
                
                # Monitor for interruption while waiting for response
                old_settings = None
                try:
                    while not current_task.done():
                        try:
                            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.1)[0]:
                                # Set up raw mode for a single character read
                                if old_settings is None:
                                    old_settings = termios.tcgetattr(sys.stdin.fileno())
                                    tty.setraw(sys.stdin.fileno())
                                
                                char = sys.stdin.read(1)
                                if char == '\x1b':  # Escape key
                                    current_task.cancel()
                                    input_handler.interrupted = True
                                    break
                        except Exception:
                            pass  # Ignore errors in interrupt monitoring
                    
                finally:
                    # Always restore terminal settings
                    if old_settings is not None:
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
                
                # Handle task cancellation
                if current_task.cancelled():
                    continue
                
                # Wait for any remaining tasks to complete
                pending = [t for t in [current_task] if not t.done()]
                if pending:
                    done, pending = await asyncio.wait(pending, timeout=0.1)
                    for task in pending:
                        task.cancel()
                
                if input_handler.interrupted:
                    print("\n🛑 Request cancelled by user")
                    input_handler.interrupted = False
                    current_task = None
                    continue
                
                if current_task and current_task.done() and not current_task.cancelled():
                    response = current_task.result()
                    current_task = None
                else:
                    continue  # Request was cancelled, go back to input
                
                if hasattr(response, '__aiter__'):
                    # Streaming response
                    print("\nAssistant (press ESC to interrupt):")
                    sys.stdout.flush()
                    full_response = ""
                    
                    # Set up non-blocking input monitoring
                    stdin_fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(stdin_fd)
                    tty.setraw(stdin_fd)
                    
                    interrupted = False
                    try:
                        async for chunk in response:
                            # Check for escape key on each chunk
                            if select.select([sys.stdin], [], [], 0)[0]:  # Non-blocking check
                                char = sys.stdin.read(1)
                                if char == '\x1b':  # Escape key
                                    interrupted = True
                                    break
                            
                            # Check for interruption flag
                            if input_handler.interrupted:
                                interrupted = True
                                input_handler.interrupted = False
                                break
                                
                            if isinstance(chunk, str):
                                print(chunk, end="", flush=True)
                                full_response += chunk
                            else:
                                # Handle any non-string chunks if needed
                                print(str(chunk), end="", flush=True)
                                full_response += str(chunk)
                    finally:
                        # Always restore terminal settings first
                        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
                        
                        # Clean up display if interrupted
                        if interrupted:
                            print("\n🛑 Streaming interrupted by user")
                            sys.stdout.flush()
                        else:
                            print()  # Normal newline after streaming
                    
                    # Add assistant response to messages
                    if full_response:  # Only add if not interrupted
                        messages.append({"role": "assistant", "content": full_response})
                else:
                    # Non-streaming response
                    print(f"\nAssistant: {response}")
                    messages.append({"role": "assistant", "content": str(response)})
                
            except KeyboardInterrupt:
                # Move to beginning of line and clear, then print exit message
                sys.stdout.write('\r\x1b[KExiting...\n')
                sys.stdout.flush()
                break
            except Exception as e:
                print(f"\nError: {e}")


# Standalone interactive chat function - exact copy from mcp_deepseek_host.py
async def interactive_chat(host):
    """Run an interactive chat session with streaming tool execution."""
    # Determine host type and display appropriate info
    if hasattr(host, 'deepseek_config'):
        print(f"MCP Deepseek Host - Interactive Chat")
        print(f"Model: {host.deepseek_config.model}")
    elif hasattr(host, 'gemini_config'):
        print(f"MCP Gemini Host - Interactive Chat") 
        print(f"Model: {host.gemini_config.model}")
    else:
        print(f"MCP Agent - Interactive Chat")
    
    print(f"Available tools: {len(host.available_tools)}")
    print("Commands: 'quit' to exit, 'tools' to list tools, 'ESC' to interrupt")
    print("Model switching: '/switch-chat', '/switch-reason', '/switch-gemini', '/switch-gemini-pro'")
    print("Utility: '/compact' to manually compact conversation, '/tokens' to show token count")
    print("Input: Single Enter to send, paste multiline content automatically detected")
    print("Navigation: Arrow keys for cursor movement, Backspace to delete")
    print("-" * 50)
    
    messages = []
    input_handler = InterruptibleInput()
    current_task = None
    
    while True:
        try:
            # Check if we were interrupted during a previous operation
            if input_handler.interrupted:
                if current_task and not current_task.done():
                    current_task.cancel()
                    sys.stdout.write('\r\x1b[K\n')
                    print("🛑 Operation cancelled by user")
                input_handler.interrupted = False
                current_task = None
                continue
            
            # Get user input with smart multiline detection
            user_input = input_handler.get_multiline_input("You: ")
            
            if user_input is None:  # Interrupted
                if current_task and not current_task.done():
                    current_task.cancel()
                    sys.stdout.write('\r\x1b[K\n')
                    print("🛑 Operation cancelled by user")
                input_handler.interrupted = False
                current_task = None
                continue
            
            if user_input.lower().strip() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower().strip() == 'tools':
                if host.available_tools:
                    print("\nAvailable tools:")
                    for tool_key, tool_info in host.available_tools.items():
                        print(f"  - {tool_key}: {tool_info['description']}")
                else:
                    print("No tools available")
                continue
            elif user_input.lower().strip() == '/switch-chat':
                # Switch to deepseek-chat model
                config = load_config()
                config.deepseek_model = "deepseek-chat"
                config.save()
                # Update the host's config and recreate the deepseek client
                if hasattr(host, 'deepseek_config'):
                    host.config = config
                    host.deepseek_config = config.get_deepseek_config()
                    host.deepseek_client = host.deepseek_client.__class__(
                        api_key=host.deepseek_config.api_key,
                        base_url=host.deepseek_config.base_url
                    )
                print(f"✅ Model switched to: {config.deepseek_model}")
                continue
            elif user_input.lower().strip() == '/switch-reason':
                # Switch to deepseek-reasoner model
                config = load_config()
                config.deepseek_model = "deepseek-reasoner"
                config.save()
                # Update the host's config and recreate the deepseek client
                if hasattr(host, 'deepseek_config'):
                    host.config = config
                    host.deepseek_config = config.get_deepseek_config()
                    host.deepseek_client = host.deepseek_client.__class__(
                        api_key=host.deepseek_config.api_key,
                        base_url=host.deepseek_config.base_url
                    )
                print(f"✅ Model switched to: {config.deepseek_model}")
                continue
            elif user_input.lower().strip() == '/switch-gemini':
                # Switch to Gemini Flash backend
                config = load_config()
                config.deepseek_model = "gemini"  # Use this as a marker
                config.gemini_model = "gemini-2.5-flash"
                config.save()
                print(f"✅ Backend switched to: Gemini Flash 2.5 ({config.gemini_model})")
                print("⚠️  Note: Restart the chat session to use Gemini backend")
                continue
            elif user_input.lower().strip() == '/switch-gemini-pro':
                # Switch to Gemini Pro backend
                config = load_config()
                config.deepseek_model = "gemini"  # Use this as a marker
                config.gemini_model = "gemini-2.5-pro"
                config.save()
                print(f"✅ Backend switched to: Gemini Pro 2.5 ({config.gemini_model})")
                print("⚠️  Note: Restart the chat session to use Gemini backend")
                continue
            elif user_input.lower().strip() == '/compact':
                # Manually compact conversation
                if len(messages) > 3:
                    print(f"\n🗜️  Compacting conversation... ({len(messages)} messages)")
                    try:
                        messages = await host.compact_conversation(messages)
                        tokens = host.count_conversation_tokens(messages)
                        print(f"✅ Conversation compacted. Current tokens: ~{tokens}")
                    except Exception as e:
                        print(f"❌ Failed to compact: {e}")
                else:
                    print("📝 Conversation is already short, no need to compact")
                continue
            elif user_input.lower().strip() == '/tokens':
                # Show token count
                tokens = host.count_conversation_tokens(messages)
                limit = host.get_token_limit()
                percentage = (tokens / limit) * 100
                print(f"\n📊 Token usage: ~{tokens}/{limit} ({percentage:.1f}%)")
                if percentage > 80:
                    print("⚠️  Consider using '/compact' to reduce token usage")
                continue
            
            # Process the user input (no longer need buffer logic)
            if user_input.strip():  # Only process non-empty input
                messages.append({"role": "user", "content": user_input})
                
                # Check if we should auto-compact before making the API call
                if host.should_compact(messages):
                    tokens_before = host.count_conversation_tokens(messages)
                    print(f"\n🗜️  Auto-compacting conversation (was ~{tokens_before} tokens)...")
                    try:
                        messages = await host.compact_conversation(messages)
                        tokens_after = host.count_conversation_tokens(messages)
                        print(f"✅ Compacted to ~{tokens_after} tokens")
                    except Exception as e:
                        print(f"⚠️  Auto-compact failed: {e}")
                
                try:
                    # Make API call interruptible by running in a task
                    print("\n💭 Thinking... (press ESC to interrupt)")
                    current_task = asyncio.create_task(
                        host.chat_completion(messages, stream=True, interactive=True)
                    )
                    
                    # Create a background task to monitor for escape key
                    async def monitor_escape():
                        old_settings = None
                        try:
                            while not current_task.done():
                                try:
                                    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.1)[0]:
                                        # Set up raw mode for a single character read
                                        if old_settings is None:
                                            old_settings = termios.tcgetattr(sys.stdin.fileno())
                                            tty.setraw(sys.stdin.fileno())
                                        
                                        char = sys.stdin.read(1)
                                        if char == '\x1b':  # Escape key
                                            input_handler.interrupted = True
                                            return
                                    await asyncio.sleep(0.1)
                                except Exception as e:
                                    await asyncio.sleep(0.1)
                        finally:
                            if old_settings is not None:
                                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
                    
                    monitor_task = asyncio.create_task(monitor_escape())
                    
                    # Wait for either completion or interruption
                    response = None
                    done, pending = await asyncio.wait(
                        [current_task, monitor_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel any remaining tasks
                    for task in pending:
                        task.cancel()
                    
                    if input_handler.interrupted:
                        sys.stdout.write('\r\x1b[K\n')
                        print("🛑 Request cancelled by user")
                        input_handler.interrupted = False
                        current_task = None
                        continue
                    
                    if current_task and current_task.done() and not current_task.cancelled():
                        response = current_task.result()
                        current_task = None
                    else:
                        continue  # Request was cancelled, go back to input
                    
                    if hasattr(response, '__aiter__'):
                        # Streaming response with potential tool execution
                        # Clear current line and move to fresh line
                        print("\nAssistant (press ESC to interrupt):")
                        sys.stdout.flush()
                        full_response = ""
                        
                        # Set up non-blocking input monitoring
                        stdin_fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(stdin_fd)
                        tty.setraw(stdin_fd)
                        
                        interrupted = False
                        try:
                            async for chunk in response:
                                # Check for escape key on each chunk
                                if select.select([sys.stdin], [], [], 0)[0]:  # Non-blocking check
                                    char = sys.stdin.read(1)
                                    if char == '\x1b':  # Escape key
                                        interrupted = True
                                        break
                                
                                # Check for interruption flag
                                if input_handler.interrupted:
                                    interrupted = True
                                    input_handler.interrupted = False
                                    break
                                    
                                if isinstance(chunk, str):
                                    print(chunk, end="", flush=True)
                                    full_response += chunk
                                else:
                                    # Handle any non-string chunks if needed
                                    print(str(chunk), end="", flush=True)
                                    full_response += str(chunk)
                        finally:
                            # Always restore terminal settings first
                            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
                            
                            # Clean up display if interrupted
                            if interrupted:
                                print("\n🛑 Streaming interrupted by user")
                                sys.stdout.flush()
                            else:
                                print()  # Normal newline after streaming
                        
                        # Add assistant response to messages
                        if full_response:  # Only add if not interrupted
                            messages.append({"role": "assistant", "content": full_response})
                    else:
                        # Non-streaming response (happens when tools are used)
                        # Clear any input line artifacts and start fresh
                        sys.stdout.write('\r\x1b[K\n')
                        print(f"Assistant: {response}")
                        messages.append({"role": "assistant", "content": response})
                        
                except asyncio.CancelledError:
                    sys.stdout.write('\r\x1b[K\n')
                    print("🛑 Request cancelled")
                    current_task = None
                except Exception as e:
                    sys.stdout.write('\r\x1b[K\n')
                    print(f"Error: {e}")
                    current_task = None
            else:
                # Empty input, just continue
                continue
            
        except KeyboardInterrupt:
            # Move to beginning of line and clear, then print exit message
            sys.stdout.write('\r\x1b[KExiting...\n')
            sys.stdout.flush()
            break
        except Exception as e:
            sys.stdout.write('\r\x1b[K\n')
            print(f"Error: {e}")


# CLI functionality
@click.group()
@click.option('--config-file', default=None, help='Path to the configuration file (default: ~/.mcp/config.json)')
@click.pass_context
def cli(ctx, config_file):
    """MCP Agent - Run AI models with MCP tool integration."""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config_file


@cli.command()
def init():
    """Initialize configuration file."""
    from config import create_sample_env
    create_sample_env()


@cli.command('switch-chat')
@click.pass_context
def switch_chat(ctx):
    """Switch the model to deepseek-chat."""
    config = load_config()
    config.deepseek_model = "deepseek-chat"
    click.echo(f"Model switched to: {config.deepseek_model}")
    # Save the updated config
    config.save()


@cli.command('switch-reason')
@click.pass_context
def switch_reason(ctx):
    """Switch the model to deepseek-reasoner."""
    config = load_config()
    config.deepseek_model = "deepseek-reasoner"
    click.echo(f"Model switched to: {config.deepseek_model}")
    # Save the updated config
    config.save()


@cli.command('switch-gemini')
@click.pass_context
def switch_gemini(ctx):
    """Switch to use Gemini Flash 2.5 as the backend model."""
    config = load_config()
    # Set Gemini Flash as the model and switch backend
    config.deepseek_model = "gemini"  # Use this as a marker
    config.gemini_model = "gemini-2.5-flash"
    click.echo(f"Backend switched to: Gemini Flash 2.5 ({config.gemini_model})")
    # Save the updated config
    config.save()


@cli.command('switch-gemini-pro')
@click.pass_context
def switch_gemini_pro(ctx):
    """Switch to use Gemini Pro 2.5 as the backend model."""
    config = load_config()
    # Set Gemini Pro as the model and switch backend
    config.deepseek_model = "gemini"  # Use this as a marker
    config.gemini_model = "gemini-2.5-pro"
    click.echo(f"Backend switched to: Gemini Pro 2.5 ({config.gemini_model})")
    # Save the updated config
    config.save()


@cli.command()
@click.option('--server', multiple=True, help='MCP server to connect to (format: name:command:arg1:arg2)')
@click.pass_context
async def chat(ctx, server):
    """Start interactive chat session."""
    try:
        # Load configuration
        config = load_config()
        
        # Check if Gemini backend should be used
        if config.deepseek_model == "gemini":
            if not config.gemini_api_key:
                click.echo("Error: GEMINI_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            # Import and create Gemini host
            from mcp_gemini_host import MCPGeminiHost
            host = MCPGeminiHost(config)
        else:
            if not config.deepseek_api_key:
                click.echo("Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            # Create Deepseek host
            from mcp_deepseek_host import MCPDeepseekHost
            host = MCPDeepseekHost(config)
        
        # Connect to specified MCP servers
        for server_spec in server:
            parts = server_spec.split(':')
            if len(parts) < 2:
                click.echo(f"Invalid server spec: {server_spec}")
                continue
            
            server_name = parts[0]
            command = parts[1:]
            
            config.add_mcp_server(server_name, command)
            
            success = await host.start_mcp_server(server_name, config.mcp_servers[server_name])
            if not success:
                click.echo(f"Failed to start server: {server_name}")
        
        # Start interactive chat using the host-specific function
        if hasattr(host, 'deepseek_config'):
            # Use the standalone interactive_chat function from deepseek host
            from mcp_deepseek_host import interactive_chat as deepseek_interactive_chat
            await deepseek_interactive_chat(host)
        elif hasattr(host, 'gemini_config'):
            # Use the standalone interactive_chat function from gemini host if available
            try:
                from mcp_gemini_host import interactive_chat_gemini
                await interactive_chat_gemini(host)
            except ImportError:
                # Fallback to our interactive chat
                await interactive_chat(host)
        else:
            # Fallback to our interactive chat
            await interactive_chat(host)
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'host' in locals():
            if hasattr(host.shutdown, '__call__') and asyncio.iscoroutinefunction(host.shutdown):
                await host.shutdown()
            else:
                host.shutdown()


@cli.command()
@click.argument('message')
@click.option('--server', multiple=True, help='MCP server to connect to')
@click.pass_context
async def ask(ctx, message, server):
    """Ask a single question."""
    try:
        config = load_config()
        
        # Check if Gemini backend should be used
        if config.deepseek_model == "gemini":
            if not config.gemini_api_key:
                click.echo("Error: GEMINI_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            # Import and create Gemini host
            from mcp_gemini_host import MCPGeminiHost
            host = MCPGeminiHost(config)
        else:
            if not config.deepseek_api_key:
                click.echo("Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            from mcp_deepseek_host import MCPDeepseekHost
            host = MCPDeepseekHost(config)
        
        # Connect to servers
        for server_spec in server:
            parts = server_spec.split(':')
            if len(parts) < 2:
                continue
            
            server_name = parts[0]
            command = parts[1:]
            config.add_mcp_server(server_name, command)
            success = await host.start_mcp_server(server_name, config.mcp_servers[server_name])
            if not success:
                click.echo(f"Warning: Failed to connect to MCP server '{server_name}', continuing without it...")
        
        # Get response
        messages = [{"role": "user", "content": message}]
        response = await host.chat_completion(messages, stream=False)
        
        click.echo(response)
        
    finally:
        if 'host' in locals():
            if hasattr(host.shutdown, '__call__') and asyncio.iscoroutinefunction(host.shutdown):
                await host.shutdown()
            else:
                host.shutdown()


@cli.command()
@click.pass_context
async def compact(ctx):
    """Show conversation token usage and compacting options."""
    click.echo("Compact functionality is only available in interactive chat mode.")
    click.echo("Use 'python agent.py chat' and then '/tokens' or '/compact' commands.")


def main():
    """Main entry point."""
    # Store original async callbacks
    original_chat = chat.callback
    original_ask = ask.callback
    
    # Convert async commands to sync
    def sync_chat(**kwargs):
        asyncio.run(original_chat(**kwargs))
    
    def sync_ask(**kwargs):
        asyncio.run(original_ask(**kwargs))
    
    # Replace command callbacks
    chat.callback = sync_chat
    ask.callback = sync_ask
    
    cli()


if __name__ == "__main__":
    main()
