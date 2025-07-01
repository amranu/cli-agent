"""Slash command management system for CLI agents."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cli_agent.core.base_agent import BaseMCPAgent

# Import config migration function
from config import get_config_dir

# Configure logging
logger = logging.getLogger(__name__)


class SlashCommandManager:
    """Manages slash commands similar to Claude Code's system."""

    def __init__(self, agent: "BaseMCPAgent"):
        self.agent = agent
        self.custom_commands = {}
        self.load_custom_commands()

    def load_custom_commands(self):
        """Load custom commands from .claude/commands/ and ~/.config/agent/commands/"""
        # Project-specific commands
        project_commands_dir = Path(".claude/commands")
        if project_commands_dir.exists():
            self._load_commands_from_dir(project_commands_dir, "project")

        # Personal commands
        personal_commands_dir = get_config_dir() / "commands"
        if personal_commands_dir.exists():
            self._load_commands_from_dir(personal_commands_dir, "personal")

    def _load_commands_from_dir(self, commands_dir: Path, command_type: str):
        """Load commands from a directory."""
        for command_file in commands_dir.glob("*.md"):
            try:
                with open(command_file, "r", encoding="utf-8") as f:
                    content = f.read()

                command_name = command_file.stem
                self.custom_commands[command_name] = {
                    "content": content,
                    "type": command_type,
                    "file": str(command_file),
                }
                logger.debug(f"Loaded {command_type} command: {command_name}")
            except Exception as e:
                logger.warning(f"Failed to load command {command_file}: {e}")

    async def handle_slash_command(
        self, command_line: str, messages: List[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Handle a slash command and return response if handled."""
        if not command_line.startswith("/"):
            return None

        # Parse command and arguments
        parts = command_line[1:].split(" ", 1)
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
        elif command == "models":
            return self._handle_models(args)
        elif command == "provider":
            return self._handle_provider(args)
        elif command == "review":
            return self._handle_review(args)
        elif command == "tokens":
            return self._handle_tokens(messages)
        elif command in ["quit", "exit"]:
            return self._handle_quit()
        elif command == "tools":
            return self._handle_tools()
        elif command == "hooks":
            return self._handle_hooks(args)
        elif command == "permissions":
            return self._handle_permissions(args)
        elif command == "truncate":
            return self._handle_truncate(args)
        elif command == "refresh-models":
            return self._handle_refresh_models()
        elif command == "switch":
            return self._handle_switch(args)
        elif command == "switch-deepseek" or command == "switch-chat":
            return self._handle_switch_deepseek()
        elif command == "switch-reason":
            return self._handle_switch_reason()
        elif command == "switch-gemini-flash" or command == "switch-gemini":
            return self._handle_switch_gemini_flash()
        elif command == "switch-gemini-pro":
            return self._handle_switch_gemini_pro()
        elif command == "init":
            return await self._handle_init()
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
  /model [name]   - Show current model or switch to model (e.g. /model deepseek-reasoner)
  /models [provider] - List all available models, optionally filtered by provider (e.g. /models anthropic)
  /switch <provider>:<model> - Switch to any provider-model combination (e.g. /switch anthropic:claude-3.5-sonnet)
  /provider [name]- Show current provider or switch provider (e.g. /provider openrouter)
  /review [file]  - Request code review
  /tools          - List all available tools
  /hooks [status|disable|enable] - Show hooks status or control hooks
  /permissions    - Manage tool execution permissions
  /truncate [on|off|length] - Toggle or configure tool result truncation
  /refresh-models - Clear model cache and fetch fresh model lists
  /init           - Analyze codebase and generate AGENT.md
  /quit, /exit    - Exit the interactive chat

Legacy Model Switching (deprecated):
  /switch-deepseek - Switch to deepseek-chat model
  /switch-reason  - Switch to deepseek-reasoner model
  /switch-gemini-flash - Switch to Gemini Flash 2.5 backend
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
        if hasattr(self.agent, "conversation_history"):
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

    def _handle_hooks(self, args: str) -> str:
        """Handle /hooks command."""
        if not hasattr(self.agent, 'hook_manager') or not self.agent.hook_manager:
            return "🚫 Hooks system is not enabled. Create .config/agent/settings.json to configure hooks."
        
        hook_manager = self.agent.hook_manager
        
        # Parse arguments
        args = args.strip().lower()
        
        if args == "disable":
            hook_manager.disable_hooks()
            return "🪝 Hooks temporarily disabled for this session."
        elif args == "enable":
            hook_manager.enable_hooks()
            return "🪝 Hooks re-enabled for this session."
        elif args == "status" or args == "":
            # Show hooks status
            summary = hook_manager.get_hook_summary()
            
            status_text = "🪝 Hooks System Status:\n"
            status_text += f"  Enabled: {'✅ Yes' if summary['enabled'] else '❌ No'}\n"
            status_text += f"  Total hooks: {summary['total_hooks']}\n\n"
            
            if summary['hook_types']:
                status_text += "Hook Types:\n"
                for hook_type, info in summary['hook_types'].items():
                    status_text += f"  {hook_type}: {info['hooks']} hooks ({info['matchers']} matchers)\n"
                    
                status_text += "\nConfiguration sources checked:\n"
                status_text += "  Files:\n"
                status_text += "    • ~/.config/agent/settings.json\n"
                status_text += "    • ~/.config/agent/settings.local.json\n"
                status_text += "  Directories:\n"
                status_text += "    • ~/.config/agent/hooks/*.json\n"
                
                status_text += "\nUse '/hooks disable' or '/hooks enable' to control execution."
            else:
                status_text += "No hook types configured."
                
            return status_text
        else:
            return "Usage: /hooks [status|disable|enable]"

    async def _handle_compact(self, messages: List[Dict[str, Any]] = None) -> str:
        """Handle /compact command."""
        if messages is None:
            return "No conversation history provided to compact."

        if len(messages) <= 3:
            return "Conversation is too short to compact (3 messages or fewer)."

        # Get token count before compacting
        if hasattr(self.agent, "count_conversation_tokens"):
            tokens_before = self.agent.count_conversation_tokens(messages)
        else:
            tokens_before = "unknown"

        try:
            # Use the agent's compact_conversation method
            if hasattr(self.agent, "compact_conversation"):
                compacted_messages = await self.agent.compact_conversation(messages)

                # Get token count after compacting
                if hasattr(self.agent, "count_conversation_tokens"):
                    tokens_after = self.agent.count_conversation_tokens(
                        compacted_messages
                    )
                    result = (
                        f"✅ Conversation compacted: {len(messages)} → {len(compacted_messages)} messages\n"
                        f"📊 Token usage: ~{tokens_before} → ~{tokens_after} tokens"
                    )
                else:
                    result = (
                        f"✅ Conversation compacted: {len(messages)} → "
                        f"{len(compacted_messages)} messages"
                    )

                # Return both the result message and the compacted messages
                # The interactive chat will need to update its messages list
                return {"status": result, "compacted_messages": compacted_messages}
            else:
                return "❌ Conversation compacting not available for this agent type."

        except Exception as e:
            return f"❌ Failed to compact conversation: {str(e)}"

    def _handle_tokens(self, messages: List[Dict[str, Any]] = None) -> str:
        """Handle /tokens command."""
        if not hasattr(self.agent, "count_conversation_tokens"):
            return "❌ Token counting not available for this agent type."

        if messages is None or len(messages) == 0:
            return "No conversation history to analyze."

        # Check if we have reliable token information
        if hasattr(self.agent, "token_manager") and hasattr(self.agent.token_manager, "has_reliable_token_info"):
            if not self.agent.token_manager.has_reliable_token_info():
                return "⚠️  Token information not available for this model. Accurate token counting requires known model limits."

        tokens = self.agent.count_conversation_tokens(messages)
        limit = (
            self.agent.get_token_limit()
            if hasattr(self.agent, "get_token_limit")
            else 32000
        )
        percentage = (tokens / limit) * 100

        result = f"📊 Token usage: ~{tokens}/{limit} ({percentage:.1f}%)"
        if percentage > 80:
            result += "\n⚠️  Consider using '/compact' to reduce token usage"

        return result

    def _handle_switch(self, args: str) -> Dict[str, Any]:
        """Handle /switch command for provider-model combinations.

        Args:
            args: Provider-model combination in format "provider:model"
        """
        if not args.strip():
            return "❌ Usage: /switch <provider>:<model>\nExample: /switch anthropic:claude-3.5-sonnet\nUse /models to see available options."

        provider_model = args.strip()

        # Validate format
        if ":" not in provider_model:
            return "❌ Invalid format. Use: /switch <provider>:<model>\nExample: /switch anthropic:claude-3.5-sonnet"

        try:
            from config import load_config

            config = load_config()

            # Parse and validate the provider-model combination
            try:
                provider_name, model_name = config.parse_provider_model_string(
                    provider_model
                )
            except Exception as e:
                return f"❌ Invalid provider-model format: {str(e)}\nExample: /switch anthropic:claude-3.5-sonnet"

            # Check if the provider-model combination is available
            available_models = config.get_available_provider_models()

            if provider_name not in available_models:
                available_providers = ", ".join(available_models.keys())
                return f"❌ Provider '{provider_name}' not available. Available providers: {available_providers}"

            if model_name not in available_models[provider_name]:
                available_models_for_provider = ", ".join(
                    available_models[provider_name][:10]
                )  # Show first 10
                more_models = (
                    f" (and {len(available_models[provider_name]) - 10} more)"
                    if len(available_models[provider_name]) > 10
                    else ""
                )
                return f"❌ Model '{model_name}' not available for provider '{provider_name}'.\nAvailable models: {available_models_for_provider}{more_models}\nUse '/models {provider_name}' to see all options."

            # Update configuration
            config.default_provider_model = provider_model
            config.save_persistent_config()

            return {
                "status": f"✅ Switched to: {provider_model} (Provider: {provider_name}, Model: {model_name})",
                "reload_host": "provider-model",
            }

        except Exception as e:
            return f"❌ Failed to switch model: {str(e)}"

    def _handle_switch_deepseek(self) -> Dict[str, Any]:
        """Handle /switch-deepseek command."""
        try:
            from config import load_config

            config = load_config()
            config.default_provider_model = "deepseek:deepseek-chat"
            config.model_type = "deepseek"
            config.save_persistent_config()
            return {
                "status": f"⚠️  DEPRECATED: Use '/model deepseek-chat' instead.\n✅ Switched to: {config.default_provider_model}",
                "reload_host": "provider-model",
            }
        except Exception as e:
            return f"❌ Failed to switch model: {str(e)}"

    def _handle_switch_reason(self) -> Dict[str, Any]:
        """Handle /switch-reason command."""
        try:
            from config import load_config

            config = load_config()
            config.default_provider_model = "deepseek:deepseek-reasoner"
            config.model_type = "deepseek"
            config.save_persistent_config()
            return {
                "status": f"⚠️  DEPRECATED: Use '/model deepseek-reasoner' instead.\n✅ Switched to: {config.default_provider_model}",
                "reload_host": "provider-model",
            }
        except Exception as e:
            return f"❌ Failed to switch model: {str(e)}"

    def _handle_switch_gemini_flash(self) -> Dict[str, Any]:
        """Handle /switch-gemini-flash command."""
        try:
            from config import load_config

            config = load_config()
            config.default_provider_model = "google:gemini-2.5-flash"
            config.model_type = "gemini"
            config.save_persistent_config()
            return {
                "status": f"⚠️  DEPRECATED: Use '/provider google' or '/model gemini-2.5-flash' instead.\n✅ Switched to: {config.default_provider_model}",
                "reload_host": "provider-model",
            }
        except Exception as e:
            return f"❌ Failed to switch backend: {str(e)}"

    def _handle_switch_gemini_pro(self) -> Dict[str, Any]:
        """Handle /switch-gemini-pro command."""
        try:
            from config import load_config

            config = load_config()
            config.default_provider_model = "google:gemini-2.5-pro"
            config.model_type = "gemini"
            config.save_persistent_config()
            return {
                "status": f"⚠️  DEPRECATED: Use '/provider google' or '/model gemini-2.5-pro' instead.\n✅ Switched to: {config.default_provider_model}",
                "reload_host": "provider-model",
            }
        except Exception as e:
            return f"❌ Failed to switch backend: {str(e)}"

    def _handle_model(self, args: str) -> Dict[str, Any]:
        """Handle /model command."""
        if not args.strip():
            # Show current model
            if hasattr(self.agent, "config") and hasattr(
                self.agent.config, "default_provider_model"
            ):
                provider_name, model_name = (
                    self.agent.config.parse_provider_model_string(
                        self.agent.config.default_provider_model
                    )
                )
                return f"Current: {self.agent.config.default_provider_model} (Provider: {provider_name}, Model: {model_name})"
            return "Current model: Unknown"
        else:
            # Switch to new model with appropriate provider
            model_name = args.strip()
            try:
                from config import load_config

                config = load_config()

                # Determine the appropriate provider for this model
                default_provider = config.get_default_provider_for_model(model_name)
                if default_provider:
                    provider_to_use = default_provider
                else:
                    # Fallback to current provider if model not found in any provider
                    current_provider, _ = config.parse_provider_model_string(
                        config.default_provider_model
                    )
                    provider_to_use = current_provider

                new_provider_model = f"{provider_to_use}:{model_name}"

                config.default_provider_model = new_provider_model
                config.save_persistent_config()

                return {
                    "status": f"✅ Switched to model: {model_name} (Provider: {provider_to_use})",
                    "reload_host": "provider-model",
                }
            except Exception as e:
                return f"❌ Failed to switch model: {str(e)}"

    def _handle_models(self, args: str = "") -> str:
        """Handle /models command - list all available models with their providers.

        Args:
            args: Optional provider name to filter by (e.g., "anthropic", "openai")
        """
        try:
            from config import load_config

            config = load_config()
            available_models = config.get_available_provider_models()

            if not available_models:
                return "❌ No models available. Configure API keys via environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY)."

            # Filter by provider if specified
            filter_provider = args.strip().lower() if args.strip() else None
            if filter_provider:
                # Check if the provider exists
                if filter_provider not in available_models:
                    available_providers = ", ".join(available_models.keys())
                    return f"❌ Provider '{filter_provider}' not found. Available providers: {available_providers}"

                # Filter to only the specified provider
                available_models = {filter_provider: available_models[filter_provider]}

            output = []
            if filter_provider:
                output.append(
                    f"📋 **Available Models for {filter_provider.upper()}:**\n"
                )
            else:
                output.append("📋 **Available Models:**\n")

            total_models = 0
            for provider, models in available_models.items():
                output.append(f"**{provider.upper()}** ({len(models)} models):")
                for model in models:
                    total_models += 1
                    # Show usage format for switching
                    usage_format = f"{provider}:{model}"
                    output.append(f"  • `{model}` → `/switch {usage_format}`")
                output.append("")  # Empty line between providers

            if filter_provider:
                output.append(f"**Total:** {total_models} models for {filter_provider}")
            else:
                output.append(
                    f"**Total:** {total_models} models across {len(available_models)} providers"
                )

            output.append("\n💡 Use `/switch <provider>:<model>` to switch models")
            output.append("💡 Use `/models <provider>` to filter by provider")
            output.append("💡 Use `/model` to see current model")

            return "\n".join(output)

        except Exception as e:
            return f"❌ Failed to list models: {str(e)}"

    def _handle_provider(self, args: str) -> Dict[str, Any]:
        """Handle /provider command."""
        if not args.strip():
            # Show current provider
            if hasattr(self.agent, "config") and hasattr(
                self.agent.config, "default_provider_model"
            ):
                provider_name, model_name = (
                    self.agent.config.parse_provider_model_string(
                        self.agent.config.default_provider_model
                    )
                )
                return f"Current: {self.agent.config.default_provider_model} (Provider: {provider_name}, Model: {model_name})"
            return "Current provider: Unknown"
        else:
            # Switch to new provider with current model
            provider_name = args.strip()
            try:
                from config import load_config

                config = load_config()
                _, current_model = config.parse_provider_model_string(
                    config.default_provider_model
                )
                new_provider_model = f"{provider_name}:{current_model}"

                config.default_provider_model = new_provider_model
                config.save_persistent_config()

                return {
                    "status": f"✅ Switched to provider: {provider_name} (Model: {current_model})",
                    "reload_host": "provider-model",
                }
            except Exception as e:
                return f"❌ Failed to switch provider: {str(e)}"

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
        parts = command.split("__")
        if len(parts) != 3 or parts[0] != "mcp":
            return f"Invalid MCP command format: /{command}"

        server_name = parts[1]
        prompt_name = parts[2]

        # Check if we have this MCP server
        if hasattr(self.agent, "available_tools"):
            # Look for matching tools from this server
            matching_tools = [
                tool
                for tool in self.agent.available_tools.keys()
                if tool.startswith(f"{server_name}:")
            ]
            if not matching_tools:
                return (
                    f"MCP server '{server_name}' not found or has no available tools."
                )

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

    async def _handle_init(self) -> str:
        """Handle /init command - prompt LLM to analyze codebase and create AGENT.md."""
        try:
            import os

            # Get project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))

            # Key files to analyze for comprehensive understanding
            analysis_files = [
                "agent.py",
                "config.py",
                "mcp_deepseek_host.py",
                "mcp_gemini_host.py",
                "subagent.py",
                "cli_agent/core/base_agent.py",
                "cli_agent/core/slash_commands.py",
                "cli_agent/core/input_handler.py",
                "cli_agent/tools/builtin_tools.py",
                "cli_agent/utils/tool_conversion.py",
                "cli_agent/utils/tool_parsing.py",
            ]

            # Check which files exist
            existing_files = []
            total_lines = 0

            for file_path in analysis_files:
                full_path = os.path.join(project_root, file_path)
                if os.path.exists(full_path):
                    existing_files.append(file_path)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            total_lines += len(content.split("\n"))
                    except Exception:
                        pass

            # Create the prompt for the LLM to analyze and create AGENT.md
            analysis_prompt = f"""Please comprehensively analyze this MCP Agent codebase and create a detailed AGENT.md file that documents the entire system architecture for other AI coding agents.

## Task: Analyze Codebase and Write AGENT.md

You should read and analyze these {len(existing_files)} key files (approximately {total_lines} lines total):
{chr(10).join([f"- {path}" for path in existing_files])}

Create a comprehensive AGENT.md document by:

1. **Reading each file** using the read_file tool to understand the codebase structure
2. **Analyzing the architecture** - understand how modules interact and the overall design
3. **Writing AGENT.md** using the write_file tool with comprehensive documentation

The AGENT.md should include:

**1. Overview**
- High-level description of the MCP Agent system and its capabilities
- Core system features and benefits

**2. Modular Architecture**
- Detailed breakdown of the file structure and module responsibilities
- How the modular design improves maintainability

**3. Execution Flow**
- Step-by-step explanation of how the system processes requests
- Key execution paths and decision points

**4. Tool Integration**
- How built-in and external tools work together
- Tool conversion and execution pipeline

**5. Subagent System**
- The sophisticated process-isolated subagent architecture
- Communication protocols and lifecycle management

**6. Design Patterns**
- Key architectural decisions and patterns used
- Why these patterns were chosen

**7. Development Guidelines**
- How to extend and work with the system
- Best practices for adding new components

Focus on making this documentation extremely helpful for AI coding agents who need to quickly understand and work with this sophisticated codebase. Include specific file paths, class names, and method references where relevant.

Please start by reading the key files to understand the architecture, then write a comprehensive AGENT.md file."""

            # Return the analysis prompt to be sent to LLM
            return {
                "status": f"🔍 Initiating codebase analysis of {len(existing_files)} files (~{total_lines} lines total)...",
                "send_to_llm": analysis_prompt,
            }

        except Exception as e:
            logger.error(f"Failed to create init prompt: {e}")
            return f"❌ Failed to create codebase analysis prompt: {str(e)}"

    def _get_mcp_commands(self) -> List[str]:
        """Get available MCP commands."""
        mcp_commands = []
        if hasattr(self.agent, "available_tools"):
            # Group tools by server and create MCP commands
            servers = set()
            for tool_name in self.agent.available_tools.keys():
                if ":" in tool_name and not tool_name.startswith("builtin:"):
                    server_name = tool_name.split(":")[0]
                    servers.add(server_name)

            for server in servers:
                mcp_commands.append(f"mcp__{server}__<prompt-name>")

        return mcp_commands

    def _handle_permissions(self, args: str) -> str:
        """Handle tool permissions command."""
        if not hasattr(self.agent, "permission_manager"):
            return "❌ Tool permission system not available."

        permission_manager = self.agent.permission_manager

        parts = args.split()
        if not parts:
            # Show current status
            status = permission_manager.get_session_status()

            result = "🔧 Tool Permission Status:\n\n"

            if status["auto_approve"]:
                result += "🟢 Auto-approval: ENABLED for all tools\n\n"
            else:
                result += "🔴 Auto-approval: DISABLED\n\n"

            if status["config_allowed"]:
                result += f"✅ Configuration allowed tools: {', '.join(status['config_allowed'])}\n"

            if status["config_disallowed"]:
                result += f"❌ Configuration disallowed tools: {', '.join(status['config_disallowed'])}\n"

            if status["approved_tools"]:
                result += f"✅ Session approved tools: {', '.join(status['approved_tools'])}\n"

            if status["denied_tools"]:
                result += (
                    f"❌ Session denied tools: {', '.join(status['denied_tools'])}\n"
                )

            if not any(
                [
                    status["config_allowed"],
                    status["config_disallowed"],
                    status["approved_tools"],
                    status["denied_tools"],
                ]
            ):
                result += "ℹ️ No specific tool permissions configured.\n"

            result += "\nCommands:\n"
            result += "  /permissions reset     - Reset session permissions\n"
            result += "  /permissions allow <tool>  - Allow tool for session\n"
            result += "  /permissions deny <tool>   - Deny tool for session\n"
            result += "  /permissions auto      - Enable auto-approval for session\n"

            return result

        command = parts[0].lower()

        if command == "reset":
            permission_manager.reset_session_permissions()
            return "✅ Session permissions reset."

        elif command == "allow" and len(parts) > 1:
            tool_name = parts[1]
            permission_manager.add_session_approval(tool_name)
            return f"✅ Tool '{tool_name}' approved for this session."

        elif command == "deny" and len(parts) > 1:
            tool_name = parts[1]
            permission_manager.add_session_denial(tool_name)
            return f"❌ Tool '{tool_name}' denied for this session."

        elif command == "auto":
            permission_manager.session_auto_approve = True
            permission_manager._save_session_permissions()
            return "✅ Auto-approval enabled for all tools in this session."

        else:
            return "❌ Invalid permissions command. Use: reset, allow <tool>, deny <tool>, or auto"

    def _handle_truncate(self, args: str) -> str:
        """Handle /truncate command to configure tool result truncation."""
        if not hasattr(self.agent, "config"):
            return "❌ Configuration not available for this agent type."

        config = self.agent.config

        if not args:
            # Show current status
            status = "enabled" if config.truncate_tool_results else "disabled"
            return f"🔧 Tool result truncation: {status} (max length: {config.tool_result_max_length} chars)"

        args = args.lower().strip()

        if args in ["on", "enable", "true"]:
            config.truncate_tool_results = True
            return "✅ Tool result truncation enabled"

        elif args in ["off", "disable", "false"]:
            config.truncate_tool_results = False
            return "✅ Tool result truncation disabled"

        elif args.isdigit():
            # Set max length
            length = int(args)
            if length < 10:
                return "❌ Minimum length is 10 characters"
            elif length > 10000:
                return "❌ Maximum length is 10000 characters"
            else:
                config.tool_result_max_length = length
                config.truncate_tool_results = True  # Enable when setting length
                return f"✅ Tool result max length set to {length} characters (truncation enabled)"

        else:
            return "❌ Usage: /truncate [on|off|<length>]\n  Examples: /truncate on, /truncate off, /truncate 500"

    def _handle_refresh_models(self) -> str:
        """Handle /refresh-models command to clear cache and fetch fresh models."""
        try:
            from config import load_config

            config = load_config()
            config.clear_model_cache()

            # Fetch fresh models to populate cache
            available_models = config.get_available_provider_models()

            total_models = sum(len(models) for models in available_models.values())
            providers = list(available_models.keys())

            return f"✅ Model cache cleared and refreshed!\n📋 Found {total_models} models across {len(providers)} providers: {', '.join(providers)}"

        except Exception as e:
            return f"❌ Failed to refresh models: {str(e)}"
