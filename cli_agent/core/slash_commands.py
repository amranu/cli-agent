"""Slash command management system for CLI agents."""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# Configure logging
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
    
    async def handle_slash_command(self, command_line: str, messages: List[Dict[str, Any]] = None) -> Optional[str]:
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
        elif command == "compact":
            return await self._handle_compact(messages)
        elif command == "model":
            return self._handle_model(args)
        elif command == "review":
            return self._handle_review(args)
        elif command == "tokens":
            return self._handle_tokens(messages)
        elif command in ["quit", "exit"]:
            return self._handle_quit()
        elif command == "tools":
            return self._handle_tools()
        elif command == "switch-chat":
            return self._handle_switch_chat()
        elif command == "switch-reason":
            return self._handle_switch_reason()
        elif command == "switch-gemini":
            return self._handle_switch_gemini()
        elif command == "switch-gemini-pro":
            return self._handle_switch_gemini_pro()
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
  /compact        - Compact conversation history into a summary
  /tokens         - Show current token usage statistics
  /model [name]   - Show current model or set model
  /review [file]  - Request code review
  /tools          - List all available tools
  /quit, /exit    - Exit the interactive chat

Model Switching:
  /switch-chat    - Switch to deepseek-chat model
  /switch-reason  - Switch to deepseek-reasoner model
  /switch-gemini  - Switch to Gemini Flash 2.5 backend
  /switch-gemini-pro - Switch to Gemini Pro 2.5 backend

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
    
    def _handle_clear(self) -> Dict[str, Any]:
        """Handle /clear command."""
        if hasattr(self.agent, 'conversation_history'):
            self.agent.conversation_history.clear()
        return {"status": "Conversation history cleared.", "clear_messages": True}
    
    def _handle_quit(self) -> Dict[str, Any]:
        """Handle /quit and /exit commands."""
        return {"status": "Goodbye!", "quit": True}
    
    def _handle_tools(self) -> str:
        """Handle /tools command."""
        if not self.agent.available_tools:
            return "No tools available."
        
        tools_text = "Available tools:\n"
        for tool_name, tool_info in self.agent.available_tools.items():
            tools_text += f"  {tool_name}: {tool_info['description']}\n"
        
        return tools_text.rstrip()
    
    async def _handle_compact(self, messages: List[Dict[str, Any]] = None) -> str:
        """Handle /compact command."""
        if messages is None:
            return "No conversation history provided to compact."
        
        if len(messages) <= 3:
            return "Conversation is too short to compact (3 messages or fewer)."
        
        # Get token count before compacting
        if hasattr(self.agent, 'count_conversation_tokens'):
            tokens_before = self.agent.count_conversation_tokens(messages)
        else:
            tokens_before = "unknown"
        
        try:
            # Use the agent's compact_conversation method
            if hasattr(self.agent, 'compact_conversation'):
                compacted_messages = await self.agent.compact_conversation(messages)
                
                # Get token count after compacting
                if hasattr(self.agent, 'count_conversation_tokens'):
                    tokens_after = self.agent.count_conversation_tokens(compacted_messages)
                    result = f"✅ Conversation compacted: {len(messages)} → {len(compacted_messages)} messages\n📊 Token usage: ~{tokens_before} → ~{tokens_after} tokens"
                else:
                    result = f"✅ Conversation compacted: {len(messages)} → {len(compacted_messages)} messages"
                
                # Return both the result message and the compacted messages
                # The interactive chat will need to update its messages list
                return {"status": result, "compacted_messages": compacted_messages}
            else:
                return "❌ Conversation compacting not available for this agent type."
                
        except Exception as e:
            return f"❌ Failed to compact conversation: {str(e)}"
    
    def _handle_tokens(self, messages: List[Dict[str, Any]] = None) -> str:
        """Handle /tokens command."""
        if not hasattr(self.agent, 'count_conversation_tokens'):
            return "❌ Token counting not available for this agent type."
        
        if messages is None or len(messages) == 0:
            return "No conversation history to analyze."
        
        tokens = self.agent.count_conversation_tokens(messages)
        limit = self.agent.get_token_limit() if hasattr(self.agent, 'get_token_limit') else 32000
        percentage = (tokens / limit) * 100
        
        result = f"📊 Token usage: ~{tokens}/{limit} ({percentage:.1f}%)"
        if percentage > 80:
            result += "\n⚠️  Consider using '/compact' to reduce token usage"
        
        return result
    
    def _handle_switch_chat(self) -> Dict[str, Any]:
        """Handle /switch-chat command."""
        try:
            from config import load_config
            config = load_config()
            config.deepseek_model = "deepseek-chat"
            config.save()
            return {"status": f"✅ Model switched to: {config.deepseek_model}", "reload_host": "deepseek"}
        except Exception as e:
            return f"❌ Failed to switch model: {str(e)}"
    
    def _handle_switch_reason(self) -> Dict[str, Any]:
        """Handle /switch-reason command."""
        try:
            from config import load_config
            config = load_config()
            config.deepseek_model = "deepseek-reasoner"
            config.save()
            return {"status": f"✅ Model switched to: {config.deepseek_model}", "reload_host": "deepseek"}
        except Exception as e:
            return f"❌ Failed to switch model: {str(e)}"
    
    def _handle_switch_gemini(self) -> Dict[str, Any]:
        """Handle /switch-gemini command."""
        try:
            from config import load_config
            config = load_config()
            config.deepseek_model = "gemini"
            config.gemini_model = "gemini-2.5-flash"
            config.save()
            return {"status": f"✅ Backend switched to: Gemini Flash 2.5 ({config.gemini_model})", "reload_host": "gemini"}
        except Exception as e:
            return f"❌ Failed to switch backend: {str(e)}"
    
    def _handle_switch_gemini_pro(self) -> Dict[str, Any]:
        """Handle /switch-gemini-pro command."""
        try:
            from config import load_config
            config = load_config()
            config.deepseek_model = "gemini"
            config.gemini_model = "gemini-2.5-pro"
            config.save()
            return {"status": f"✅ Backend switched to: Gemini Pro 2.5 ({config.gemini_model})", "reload_host": "gemini"}
        except Exception as e:
            return f"❌ Failed to switch backend: {str(e)}"
    
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