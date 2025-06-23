"""System prompt construction for BaseMCPAgent."""

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class SystemPromptBuilder:
    """Builds system prompts for different agent types and contexts."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    def create_system_prompt(self, for_first_message: bool = False) -> str:
        """Create system prompt based on agent type and context."""
        # Get LLM-specific instructions
        llm_instructions = self.agent._get_llm_specific_instructions()

        # Build base system prompt
        base_prompt = self.build_base_system_prompt()

        # Combine with LLM-specific instructions
        if llm_instructions:
            return f"{base_prompt}\n\n{llm_instructions}"
        else:
            return base_prompt

    def build_base_system_prompt(self) -> str:
        """Build the base system prompt with role definition and instructions."""
        # Determine agent role and instructions based on type
        if self.agent.is_subagent:
            agent_role = "You are a focused subagent responsible for completing a specific delegated task."
            subagent_strategy = """**Critical Subagent Instructions:**
1. **Focus:** You are executing a specific task - stay focused and complete it thoroughly.
2. **Use tools:** You have access to the same tools as the main agent - use them extensively.
3. **Investigate thoroughly:** Read files, run commands, analyze code - gather comprehensive information.
4. **Emit summary:** Call `emit_result` with a comprehensive summary of your findings, conclusions, and any recommendations"""
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

        # Base system prompt template
        base_prompt = f"""{agent_role}

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

**Available Tools:**"""

        # Add tool descriptions
        available_tools = []
        for tool_key, tool_info in self.agent.available_tools.items():
            tool_name = tool_info.get("name", tool_key.split(":")[-1])
            description = tool_info.get("description", "No description available")
            available_tools.append(f"- **{tool_name}**: {description}")

        base_prompt += "\n" + "\n".join(available_tools)
        base_prompt += "\n\nYou are the expert. Complete the task."

        return base_prompt

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
