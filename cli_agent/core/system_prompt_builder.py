"""System prompt construction for BaseMCPAgent."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class SystemPromptBuilder:
    """Builds system prompts for different agent types and contexts."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent
        self._role_cache = {}  # Cache loaded roles

    def create_system_prompt(self, for_first_message: bool = False, role: Optional[str] = None) -> str:
        """Create system prompt based on agent type and context.
        
        Args:
            for_first_message: Whether this is for the first message
            role: Optional role name to use (e.g., 'security-expert')
        """
        # Check for model-specific override prompt file first (highest priority)
        import os
        provider = getattr(self.agent.model, "provider_model_name", None)
        model = getattr(self.agent.model, "name", None)
        
        override_content = None
        override_path = None
        
        # 1. Check for model-specific override (highest priority)
        if provider and model:
            model_prompt_path = os.path.expanduser(f"~/.config/agent/prompts/{provider}_{model}.txt")
            if os.path.exists(model_prompt_path):
                override_path = model_prompt_path
                
        # 2. Check for general override (fallback)
        if not override_path:
            general_prompt_path = os.path.expanduser("~/.config/agent/prompts/system_prompt.txt")
            if os.path.exists(general_prompt_path):
                override_path = general_prompt_path
        
        # Process override file if found
        if override_path:
            logger.info(f"Using system prompt override: {override_path}")
            with open(override_path, "r", encoding="utf-8") as f:
                override_content = f.read().strip()
                
                # Support dynamic placeholders in override files
                if "{{TOOLS}}" in override_content:
                    # Build tool list
                    available_tools = []
                    for tool_key, tool_info in self.agent.available_tools.items():
                        tool_name = tool_info.get("name", tool_key.split(":")[-1])
                        description = tool_info.get("description", "No description available")
                        available_tools.append(f"- **{tool_name}**: {description}")
                    tools_section = "\n" + "\n".join(available_tools)
                    override_content = override_content.replace("{{TOOLS}}", tools_section)
                
                if "{{LLM_INSTRUCTIONS}}" in override_content:
                    llm_instructions = self.agent._get_llm_specific_instructions()
                    override_content = override_content.replace("{{LLM_INSTRUCTIONS}}", llm_instructions or "")
                
                return override_content

        # Get LLM-specific instructions
        llm_instructions = self.agent._get_llm_specific_instructions()

        # Build base system prompt with optional role
        base_prompt = self.build_base_system_prompt(role=role)

        # Combine with LLM-specific instructions
        if llm_instructions:
            return f"{base_prompt}\n\n{llm_instructions}"
        else:
            return base_prompt

    def build_base_system_prompt(self, role: Optional[str] = None) -> str:
        """Build the base system prompt with role definition and instructions."""
        # Load role template if specified
        role_config = None
        if role:
            role_config = self.load_role(role)
        
        # Determine agent role and instructions based on type and role
        if self.agent.is_subagent:
            if role_config:
                agent_role = role_config["agent_role"]
            else:
                agent_role = "You are a focused autonomous subagent. You are in control and responsible for completing a specific delegated task."
            subagent_strategy = """**Critical Subagent Instructions:**
1. **Focus:** You are executing a specific task - stay focused and complete it thoroughly.
2. **Use tools:** You have access to the same tools as the main agent - use them extensively.
3. **Investigate thoroughly:** Read files, run commands, analyze code - gather comprehensive information.
4. **Emit summary:** Call `emit_result` with a comprehensive summary of your findings, conclusions, and any recommendations"""
        else:
            if role_config:
                agent_role = role_config["agent_role"]
            else:
                agent_role = "You are a top-tier autonomous software development agent. You are in control and responsible for completing the user's request. Take initiative and ownership of the solution."
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

        # Add available roles information for main agents (not subagents)
        roles_info = ""
        if not self.agent.is_subagent:
            roles_info = self.get_roles_description()

        # Base system prompt template
        base_prompt = f"""{agent_role}

IMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.

# Tone and Style
You should be concise, direct, and to the point. Minimize output tokens while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical.

IMPORTANT: Keep responses short - answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail. Answer the user's question directly, without elaboration, explanation, or details. One word answers are best when appropriate. Avoid introductions, conclusions, and explanations.

# Code Conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- NEVER assume that a given library is available, even if it is well known. Check that the codebase already uses the library first.
- When creating new components, look at existing components to understand framework choice, naming conventions, typing, and other conventions.
- Always follow security best practices. Never introduce code that exposes or logs secrets and keys.
- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked

# Following Conventions
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it.
- When you run non-trivial commands, explain what the command does and why you are running it.
- Do not ask for permission - take initiative and execute your plan autonomously.
- You are empowered to make decisions about code structure, architecture, and implementation approaches.

# Task Management
You have access to todo management tools that help you organize and track tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.

When to use TodoWrite:
- Complex multi-step tasks requiring 3 or more distinct steps
- Non-trivial and complex tasks requiring careful planning
- When user provides multiple tasks
- After receiving new instructions - immediately capture requirements as todos
- When starting work on a task - mark it as in_progress BEFORE beginning work
- After completing a task - mark it as completed and add any follow-up tasks

Task Management Rules:
- Only have ONE task in_progress at any time
- Mark tasks as completed IMMEDIATELY after finishing (don't batch completions)
- ONLY mark a task as completed when you have FULLY accomplished it
- If you encounter errors or cannot finish, keep the task as in_progress
- Create specific, actionable items and break complex tasks into smaller steps

# Tool Usage Guidelines
- When doing file search, prefer to use the task tool to reduce context usage
- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance
- Use tools efficiently and appropriately for each task
- Read files before editing them to understand context
- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands if they were provided to ensure your code is correct
- Handle errors gracefully and provide helpful feedback
- Use built-in tools for common operations
- Leverage MCP tools for specialized functionality
- NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked
- Use TodoRead and TodoWrite to keep track of tasks
- While working on a task, avoid prompting the user unless you DESPERATELY need clarification
{subagent_strategy}
{roles_info}

# Resource Self-Management
- Your context window is precious - do not waste it on tasks that can be delegated.
- Be strategic about context usage: delegate context-heavy tasks to subagents.
- Preserve your context for planning, coordination, and final synthesis of results.
- Self-optimize for efficiency and resource utilization.

# File Reading Strategy
- Be surgical: Do not read entire files at once. It is a waste of your context window.
- Locate, then read: Use tools like `grep` or `find` to locate the specific line numbers or functions you need to inspect.
- Read in chunks: Read files in smaller, targeted chunks of 50-100 lines using the `offset` and `limit` parameters.
- Full reads as a last resort: Only read a full file if you have no other way to find what you are looking for.

# File Editing Workflow
1. Read first: Always read a file before you try to edit it, following the file reading strategy above.
2. Greedy Grepping: Always `grep` or look for a small section around where you want to do an edit.
3. Use `replace_in_file`: For all file changes, use `builtin_replace_in_file` to replace text in files.
4. Chunk changes: Break large edits into smaller, incremental changes to maintain control and clarity.

**Available Tools:**"""

        # Add tool descriptions (role-specific if role is specified)
        if role:
            # Use role-specific tools
            available_tools_dict = self._filter_tools_for_role(role)
        else:
            # Use all available tools
            available_tools_dict = self.agent.available_tools
            
        available_tools = []
        for tool_key, tool_info in available_tools_dict.items():
            tool_name = tool_info.get("name", tool_key.split(":")[-1])
            description = tool_info.get("description", "No description available")
            available_tools.append(f"- **{tool_name}**: {description}")

        base_prompt += "\n" + "\n".join(available_tools)
        
        # Add role-specific instructions if role is specified
        if role_config and "instructions" in role_config:
            base_prompt += f"\n\n{role_config['instructions']}"
        
        base_prompt += "\n\nAnswer the user's request using the relevant tool(s), if they are available. Be concise and direct."

        return base_prompt

    def load_role(self, role_name: str) -> Optional[Dict]:
        """Load a role configuration from YAML file.
        
        Args:
            role_name: Name of the role (e.g., 'security-expert')
            
        Returns:
            Role configuration dict or None if not found
        """
        if role_name in self._role_cache:
            return self._role_cache[role_name]
        
        # Try user config directory first (user roles take precedence)
        user_config_dir = Path.home() / ".config" / "agent" / "roles"
        user_role_path = user_config_dir / f"{role_name}.yaml"
        
        # Fallback to default roles shipped with the project
        default_role_path = Path(__file__).parent / "default_roles" / f"{role_name}.yaml"
        
        role_path = None
        if user_role_path.exists():
            role_path = user_role_path
        elif default_role_path.exists():
            role_path = default_role_path
        
        if not role_path:
            logger.warning(f"Role '{role_name}' not found in user config or default roles")
            return None
        
        try:
            with open(role_path, 'r', encoding='utf-8') as f:
                role_config = yaml.safe_load(f)
                
            # Validate required fields
            required_fields = ["name", "agent_role"]
            for field in required_fields:
                if field not in role_config:
                    logger.error(f"Role '{role_name}' missing required field: {field}")
                    return None
            
            # Process dynamic placeholders in role content
            role_config = self._process_role_placeholders(role_config)
            
            logger.info(f"Loaded role '{role_name}': {role_config['name']}")
            self._role_cache[role_name] = role_config
            return role_config
            
        except Exception as e:
            logger.error(f"Error loading role '{role_name}': {e}")
            return None

    def _process_role_placeholders(self, role_config: Dict) -> Dict:
        """Process dynamic placeholders in role configuration.
        
        Args:
            role_config: Role configuration dictionary
            
        Returns:
            Role configuration with placeholders replaced
        """
        # Create a copy to avoid modifying the original
        processed_config = role_config.copy()
        
        # Get role name for tool filtering (convert display name to file name format)
        role_display_name = role_config.get('name', '')
        role_name = role_display_name.lower().replace(' ', '-')
        
        # Process placeholders in all string fields
        for field_name, field_value in processed_config.items():
            if isinstance(field_value, str):
                processed_config[field_name] = self._replace_placeholders_in_text(field_value, role_name)
        
        return processed_config
    
    def _replace_placeholders_in_text(self, text: str, role_name: str = None) -> str:
        """Replace dynamic placeholders in text.
        
        Args:
            text: Text that may contain placeholders
            role_name: Optional role name for role-specific tool filtering
            
        Returns:
            Text with placeholders replaced
        """
        result = text
        
        # Replace {{TOOLS}} with available tools list (role-specific if role_name provided)
        if "{{TOOLS}}" in result:
            if role_name:
                # Use role-specific tools
                available_tools_dict = self._filter_tools_for_role(role_name)
            else:
                # Use all available tools
                available_tools_dict = self.agent.available_tools
                
            available_tools = []
            for tool_key, tool_info in available_tools_dict.items():
                tool_name = tool_info.get("name", tool_key.split(":")[-1])
                description = tool_info.get("description", "No description available")
                available_tools.append(f"- **{tool_name}**: {description}")
            tools_section = "\n" + "\n".join(available_tools)
            result = result.replace("{{TOOLS}}", tools_section)
        
        # Replace {{LLM_INSTRUCTIONS}} with LLM-specific instructions
        if "{{LLM_INSTRUCTIONS}}" in result:
            llm_instructions = self.agent._get_llm_specific_instructions()
            result = result.replace("{{LLM_INSTRUCTIONS}}", llm_instructions or "")
        
        # Replace {{AGENT_TYPE}} with agent type (main or subagent)
        if "{{AGENT_TYPE}}" in result:
            agent_type = "subagent" if self.agent.is_subagent else "main"
            result = result.replace("{{AGENT_TYPE}}", agent_type)
        
        # Replace {{MODEL_NAME}} with current model name
        if "{{MODEL_NAME}}" in result:
            try:
                model_name = self.agent._get_current_runtime_model() or "unknown"
                result = result.replace("{{MODEL_NAME}}", model_name)
            except Exception:
                result = result.replace("{{MODEL_NAME}}", "unknown")
        
        return result

    def _get_role_specific_tools(self, role_name: str) -> Optional[List[str]]:
        """Load role-specific tool list from {role}_tools file.
        
        Args:
            role_name: Name of the role
            
        Returns:
            List of tool names/patterns for this role, or None for all tools
        """
        if not role_name:
            return None
            
        # Try user config directory first (user tool files take precedence)
        user_config_dir = Path.home() / ".config" / "agent" / "roles"
        user_tools_path = user_config_dir / f"{role_name}_tools"
        
        # Fallback to default role tools shipped with the project
        default_tools_path = Path(__file__).parent / "default_roles" / f"{role_name}_tools"
        
        tools_path = None
        if user_tools_path.exists():
            tools_path = user_tools_path
        elif default_tools_path.exists():
            tools_path = default_tools_path
        
        if not tools_path:
            return None
            
        try:
            with open(tools_path, 'r', encoding='utf-8') as f:
                # Read tool names/patterns, one per line, ignore empty lines and comments
                tool_patterns = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        tool_patterns.append(line)
                logger.info(f"Loaded {len(tool_patterns)} tool patterns for role '{role_name}'")
                return tool_patterns
                
        except Exception as e:
            logger.error(f"Error loading tools for role '{role_name}': {e}")
            return None

    def _filter_tools_for_role(self, role_name: str) -> Dict[str, Dict]:
        """Filter available tools based on role-specific tool patterns.
        
        Args:
            role_name: Name of the role
            
        Returns:
            Filtered tools dictionary, or all tools if no role-specific filtering
        """
        tool_patterns = self._get_role_specific_tools(role_name)
        if not tool_patterns:
            return self.agent.available_tools
            
        import fnmatch
        filtered_tools = {}
        
        for tool_key, tool_info in self.agent.available_tools.items():
            tool_name = tool_info.get("name", tool_key.split(":")[-1])
            
            # Check if tool matches any pattern
            for pattern in tool_patterns:
                if (fnmatch.fnmatch(tool_key, pattern) or 
                    fnmatch.fnmatch(tool_name, pattern) or
                    pattern in tool_key or 
                    pattern in tool_name):
                    filtered_tools[tool_key] = tool_info
                    break
                    
        logger.info(f"Role '{role_name}' has access to {len(filtered_tools)} of {len(self.agent.available_tools)} tools")
        return filtered_tools

    def get_agent_md_content(self) -> str:
        """Get Agent.md content from project directory."""
        try:
            # Look for Agent.md in the current working directory
            import os

            current_dir = os.getcwd()
            agent_md_path = os.path.join(current_dir, "AGENT.md")

            if os.path.exists(agent_md_path):
                with open(agent_md_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.debug(f"Found AGENT.md with {len(content)} characters")
                return content
            else:
                logger.debug("No AGENT.md file found in current directory")
                return ""
        except Exception as e:
            logger.debug(f"Error reading AGENT.md: {e}")
            return ""

    def enhance_first_message_with_agent_md(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Enhance the first user message with Agent.md content if available."""
        if not messages:
            return messages

        # Only enhance the first message
        first_message = messages[0]
        if first_message.get("role") != "user":
            return messages

        # Get Agent.md content
        agent_md_content = self.get_agent_md_content()
        if not agent_md_content:
            return messages

        # Create enhanced messages
        enhanced_messages = messages.copy()
        enhanced_messages[0] = self.prepend_agent_md_to_first_message(
            first_message, agent_md_content
        )

        logger.info("Enhanced first message with Agent.md content")
        return enhanced_messages

    def prepend_agent_md_to_first_message(
        self, first_message: Dict[str, str], agent_md_content: str
    ) -> Dict[str, str]:
        """Prepend Agent.md content to the first user message."""
        original_content = first_message["content"]
        enhanced_content = f"""# Project Context and Instructions (For Reference Only)

The following information is provided for context and reference purposes only. Please respond to the user's actual request below.

{agent_md_content}

---

# User Request

{original_content}"""

        return {"role": "user", "content": enhanced_content}
    
    def get_available_roles(self) -> List[str]:
        """Get list of available role names for subagent assignment.
        
        Returns:
            List of available role names
        """
        available_roles = []
        
        # Check default roles directory
        default_roles_dir = Path(__file__).parent / "default_roles"
        if default_roles_dir.exists():
            for yaml_file in default_roles_dir.glob("*.yaml"):
                role_name = yaml_file.stem
                # Skip example roles
                if not role_name.startswith("example"):
                    available_roles.append(role_name)
        
        # Check user config directory for custom roles
        config_dir = Path.home() / ".config" / "agent" / "roles"
        if config_dir.exists():
            for yaml_file in config_dir.glob("*.yaml"):
                role_name = yaml_file.stem
                if role_name not in available_roles:
                    available_roles.append(role_name)
        
        return sorted(available_roles)
    
    def get_roles_description(self) -> str:
        """Get formatted description of available roles for system prompt.
        
        Returns:
            Formatted string describing available roles
        """
        available_roles = self.get_available_roles()
        
        if not available_roles:
            return "No specialized roles available."
        
        roles_info = []
        for role_name in available_roles:
            role_config = self.load_role(role_name)
            if role_config:
                display_name = role_config.get("name", role_name)
                description = role_config.get("description", "No description available")
                roles_info.append(f"  - **{display_name}** (`{role_name}`): {description}")
        
        if roles_info:
            return f"""
**Available Subagent Roles:**
{chr(10).join(roles_info)}

Use these roles when spawning subagents with the `builtin_task` tool by setting the `role` parameter. For example:
- `builtin_task(description="Analyze security vulnerabilities", prompt="...", role="security-expert")`
- `builtin_task(description="Research information", prompt="...", role="researcher")`"""
        else:
            return "**Available Subagent Roles:** None configured"
