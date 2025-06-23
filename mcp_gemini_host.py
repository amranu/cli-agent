#!/usr/bin/env python3
"""This is the MCP host implementation that integrates with Google Gemini's API."""

import asyncio
import json
import logging
import select
import sys
import termios
import time
import tty
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from cli_agent.utils.tool_conversion import GeminiToolConverter
from cli_agent.utils.tool_parsing import GeminiToolCallParser
from config import GeminiConfig, HostConfig, create_sample_env, load_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for comprehensive logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MCPGeminiHost(BaseMCPAgent):
    """MCP Host that uses Google Gemini as the language model backend."""

    def __init__(self, config: HostConfig, is_subagent: bool = False):
        super().__init__(config, is_subagent)
        self.gemini_config = config.get_gemini_config()

        # Initialize Gemini client with timeout configuration
        try:
            import httpx

            # Configure timeout for Gemini requests (longer for tool-heavy conversations)
            timeout_seconds = 120.0  # 2 minutes for Gemini requests

            # Create HTTP client with custom timeout
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

            self.gemini_client = genai.Client(
                api_key=self.gemini_config.api_key, http_client=http_client
            )
            self.http_client = http_client  # Store reference for cleanup
            logger.debug(f"Gemini client initialized with {timeout_seconds}s timeout")

        except Exception as e:
            import traceback

            logger.warning(f"Failed to create custom HTTP client: {e}")
            logger.debug(f"HTTP client creation traceback: {traceback.format_exc()}")
            # Fallback to default client
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_config.api_key)
                self.http_client = None
            except Exception as fallback_error:
                logger.error(
                    f"Failed to create even default Gemini client: {fallback_error}"
                )
                raise

        # Set streaming preference for centralized generate_response method
        self.stream = True  # Gemini always uses streaming

        logger.info(
            f"Initialized MCP Gemini Host with model: {self.gemini_config.model}"
        )

    def convert_tools_to_llm_format(self) -> List[types.Tool]:
        """Convert tools to Gemini format using shared utilities."""
        converter = GeminiToolConverter()
        function_declarations = converter.convert_tools(self.available_tools)
        return [types.Tool(function_declarations=function_declarations)]

    def parse_tool_calls(self, response: Any) -> List:
        """Parse tool calls from Gemini response using shared utilities."""
        import json
        from types import SimpleNamespace

        # Extract text content from response for text-based tool call parsing
        text_content = ""
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_content += part.text
        elif hasattr(response, "text") and response.text:
            text_content = response.text
        elif (
            response.candidates
            and hasattr(response.candidates[0], "text")
            and response.candidates[0].text
        ):
            text_content = response.candidates[0].text

        # Parse both structured and text-based tool calls
        tool_calls = GeminiToolCallParser.parse_all_formats(response, text_content)

        # Convert to SimpleNamespace format for compatibility with _execute_function_calls
        converted_calls = []
        for tc in tool_calls:
            function_call = SimpleNamespace()
            function_call.name = tc.function.name

            # Parse arguments from JSON string to dict for proper argument extraction
            try:
                if isinstance(tc.function.arguments, str):
                    function_call.args = json.loads(tc.function.arguments)
                else:
                    function_call.args = tc.function.arguments
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse tool call arguments for {tc.function.name}: {e}"
                )
                function_call.args = {}

            converted_calls.append(function_call)

        return converted_calls

    def _create_system_prompt(self) -> str:
        """Create Gemini-specific system prompt that emphasizes tool usage."""
        # Use the non-abstract method with default parameter
        # Create base prompt directly without calling super()
        tool_descriptions = []

        for tool_key, tool_info in self.available_tools.items():
            # Use the converted name format (with underscores)
            converted_tool_name = tool_key.replace(":", "_")
            description = tool_info["description"]
            tool_descriptions.append(f"- **{converted_tool_name}**: {description}")

        tools_text = (
            "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        )

        # Customize base prompt based on whether this is a subagent
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

        # Create base prompt
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

**Available Tools:**
{tools_text}

You are the expert. Complete the task."""

        # Add Gemini-specific instructions based on agent type
        if self.is_subagent:
            gemini_addendum = """

**CRITICAL FOR GEMINI SUBAGENTS: FOCUS & EXECUTION**
- You are a SUBAGENT - execute your specific task efficiently and provide clear results
- When a task requires action (running commands, reading files, etc.), you MUST use the appropriate tools
- Do NOT just describe what you would do - actually DO IT using the tools
- Take action FIRST, then provide analysis based on the actual results
- You CANNOT spawn other subagents - focus on your assigned task only

Example: If asked to "run uname -a", do NOT respond with "I will run uname -a command" - instead immediately use builtin_bash_execute with the command and show the actual output."""
        else:
            gemini_addendum = """

**CRITICAL FOR GEMINI: TOOL USAGE REQUIREMENTS**
- When a task requires action (running commands, reading files, etc.), you MUST use the appropriate tools
- Do NOT just describe what you would do - actually DO IT using the tools
- If you need to run a command like `uname -a`, use builtin_bash_execute immediately
- If you need to read a file, use builtin_read_file immediately  
- Take action FIRST, then provide analysis based on the actual results
- Your response should show tool execution results, not just intentions

**CRITICAL FOR GEMINI: CONTEXT PRESERVATION**
- Your context is LIMITED and VALUABLE - protect it aggressively
- Before reading large files or doing complex analysis, ask: "Should I delegate this to a subagent?"
- IMMEDIATELY use builtin_task for ANY task that involves:
  - Reading multiple files (>2 files)
  - Analyzing large codebases (>200 lines total)
  - Running multiple commands in sequence
  - Complex investigations or searches
- Example: Instead of reading 10 files yourself, spawn a subagent: "Analyze all Python files in src/ directory for imports"

**DELEGATION EXAMPLES:**
❌ Bad: Reading entire file, then editing it yourself (wastes context)
✅ Good: Spawn subagent to "Find the login function in auth.py and report its structure"

❌ Bad: Running 5 commands yourself to gather system info
✅ Good: Spawn subagent to "Gather complete system information: OS, memory, disk, processes"

Example: If asked to "run uname -a", do NOT respond with "I will run uname -a command" - instead immediately use builtin_bash_execute with the command and show the actual output."""

        return base_prompt + gemini_addendum

    async def _handle_non_streaming_response(
        self, response, original_messages: List[Dict[str, Any]], **kwargs
    ) -> str:
        """Handle non-streaming response from Gemini, processing tool calls if needed."""
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        current_messages = original_messages.copy()
        max_rounds = 10  # Prevent infinite loops

        for round_num in range(max_rounds):
            # Extract text and function calls from response
            text_response = ""
            function_calls = []

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text_response += part.text
                        elif hasattr(part, "function_call") and part.function_call:
                            function_calls.append(part.function_call)

            # If we have function calls, execute them and continue
            if function_calls:
                # Execute function calls
                function_results = []
                for fc in function_calls:
                    try:
                        # Convert function call to proper format
                        tool_name = fc.name.replace("_", ":", 1)
                        arguments = dict(fc.args) if hasattr(fc, "args") else {}

                        # Execute the tool
                        result = await self._execute_mcp_tool(tool_name, arguments)
                        function_results.append(str(result))
                    except ToolDeniedReturnToPrompt as e:
                        # Tool permission denied - exit immediately without making API call
                        raise e  # Exit immediately
                    except Exception as e:
                        function_results.append(f"Error executing {fc.name}: {str(e)}")

                # Instead of adding tool messages to conversation, create a clean prompt
                # that focuses on the original request and tool results
                original_request = (
                    original_messages[-1]["content"]
                    if original_messages
                    else "Continue processing"
                )
                tool_results_text = "\n".join(function_results)

                # Create a focused prompt that avoids re-execution
                gemini_prompt = f"""Original request: {original_request}

Tool execution completed successfully. Results:
{tool_results_text}

Based on these tool results, please provide your final response. Do not re-execute any tools unless specifically needed for a different purpose."""
                config = types.GenerateContentConfig(
                    tools=(
                        self.convert_tools_to_llm_format()
                        if self.available_tools
                        else None
                    ),
                    temperature=self.gemini_config.temperature,
                    max_output_tokens=self.gemini_config.max_output_tokens,
                )

                # Make another request
                response = await self._make_gemini_request_with_retry(
                    lambda: self.gemini_client.models.generate_content(
                        model=self.gemini_config.model,
                        contents=gemini_prompt,
                        config=config,
                    )
                )
                continue
            else:
                # No function calls, return the final text response
                return text_response

        # If we hit max rounds, return what we have
        return text_response

    def _prepend_agent_md_to_first_message(
        self, messages: List[Dict[str, Any]], is_first_message: bool
    ) -> List[Dict[str, Any]]:
        """Prepend AGENT.md content to first message. Gemini uses the base implementation."""
        # Implement directly to avoid issues with abstract vs concrete methods in base class
        if not is_first_message or not messages:
            return messages

        # Only prepend to the first user message
        first_user_msg_index = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                first_user_msg_index = i
                break

        if first_user_msg_index is not None:
            # Try to read AGENT.md - implement similar to base class
            agent_md_content = ""
            try:
                import os

                # Get the project root directory (where AGENT.md should be)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                agent_md_path = os.path.join(current_dir, "AGENT.md")

                if os.path.exists(agent_md_path):
                    with open(agent_md_path, "r", encoding="utf-8") as f:
                        agent_md_content = f.read()

            except Exception as e:
                logger.error(f"Error reading AGENT.md: {e}")
                agent_md_content = ""

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

    async def _make_gemini_request_with_retry(
        self, request_func, max_retries: int = 3, base_delay: float = 1.0
    ):
        """Make a Gemini API request with exponential backoff retry logic."""
        import asyncio
        import random

        for attempt in range(max_retries + 1):
            try:
                # Execute the request function
                if asyncio.iscoroutinefunction(request_func):
                    return await request_func()
                else:
                    return request_func()

            except Exception as e:
                error_str = str(e)
                logger.error(
                    f"Gemini API request failed (attempt {attempt+1}/{max_retries+1}): {error_str}"
                )

                # Try to extract more details from the exception
                if hasattr(e, "response"):
                    try:
                        response_text = (
                            await e.response.aread()
                            if hasattr(e.response, "aread")
                            else e.response.text
                        )
                        logger.error(f"Response body: {response_text}")
                    except:
                        logger.error(f"Could not read response body from exception")

                if hasattr(e, "__cause__") and e.__cause__:
                    logger.error(f"Root cause: {e.__cause__}")

                # Check for 500 Internal Server Error specifically
                if "500" in error_str or "Internal Server Error" in error_str:
                    logger.error(
                        "Gemini API returned 500 Internal Server Error - likely prompt/content issue"
                    )

                is_retryable = (
                    "RetryError" in error_str
                    or "timeout" in error_str.lower()
                    or "network" in error_str.lower()
                    or "connection" in error_str.lower()
                    or "rate limit" in error_str.lower()
                    or "429" in error_str
                    or "502" in error_str
                    or "503" in error_str
                    or "504" in error_str
                    or "500" in error_str  # Add 500 as retryable for now
                )

                if attempt == max_retries or not is_retryable:
                    # Last attempt or non-retryable error
                    raise e

                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Gemini API request failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                logger.warning(f"Retrying in {delay:.2f} seconds...")

                await asyncio.sleep(delay)

    def _parse_python_style_function_calls(self, text: str) -> List:
        """Parse Python-style function calls that Gemini sometimes generates."""
        import re
        from types import SimpleNamespace

        function_calls = []

        # Look for patterns like: function_name('arg1', 'arg2', key='value')
        # This is a simplified parser - may need enhancement for complex cases
        python_call_pattern = r"(\w+)\s*\(\s*([^)]*)\s*\)"

        matches = re.finditer(python_call_pattern, text)

        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)

            # Check if this looks like one of our tools
            tool_key = f"builtin:{func_name}"
            if tool_key not in self.available_tools:
                continue

            # Try to parse the arguments
            try:
                # This is a very basic parser - for production you'd want something more robust
                arguments = {}

                # Simple parsing for common cases
                if args_str.strip():
                    # Remove quotes and split by commas (very basic)
                    args_parts = [part.strip() for part in args_str.split(",")]

                    for i, part in enumerate(args_parts):
                        if "=" in part:
                            # Keyword argument
                            key, value = part.split("=", 1)
                            key = key.strip().strip("'\"")
                            value = value.strip().strip("'\"")
                            # Try to convert to appropriate type
                            try:
                                if value.isdigit():
                                    value = int(value)
                                elif value.lower() in ["true", "false"]:
                                    value = value.lower() == "true"
                            except:
                                pass
                            arguments[key] = value
                        else:
                            # Positional argument - map to expected parameter
                            value = part.strip().strip("'\"")
                            # Try to convert to appropriate type
                            try:
                                # Try integer first
                                if value.isdigit():
                                    value = int(value)
                            except:
                                pass

                            # Map to common parameter names based on function
                            if func_name == "edit_file" and i == 0:
                                arguments["file_path"] = value
                            elif func_name == "read_file" and i == 0:
                                arguments["file_path"] = value
                            elif func_name == "read_file" and i == 1:
                                arguments["limit"] = value
                            elif func_name == "write_file" and i == 0:
                                arguments["file_path"] = value
                            elif func_name == "list_directory" and i == 0:
                                arguments["path"] = value
                            elif func_name == "bash_execute" and i == 0:
                                arguments["command"] = value

                # Create a mock function call object
                function_call = SimpleNamespace()
                function_call.name = f"builtin_{func_name}"
                function_call.args = arguments
                function_calls.append(function_call)

                logger.info(
                    f"Parsed Python-style function call: {func_name} with args: {arguments}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to parse Python-style function call {func_name}: {e}"
                )
                continue

        return function_calls

    def _parse_xml_style_tool_calls(self, content: str) -> List:
        """Parse XML-style tool calls from content supporting multiple formats."""
        import json
        import re
        from types import SimpleNamespace

        function_calls = []

        # Pattern 1: Complex format with tool_name and parameters
        # <execute_tool>{"tool_name": "builtin:bash_execute", "parameters": {...}}</execute_tool>
        complex_pattern = r'<execute_tool>\s*\{\s*"tool_name":\s*"([^"]+)"\s*,\s*"parameters":\s*(\{.*?\})\s*\}\s*</execute_tool>'

        # Pattern 2: Simple format with tool name and direct args
        # <execute_tool>builtin:bash_execute{"command": "..."}</execute_tool>
        simple_pattern = r"<execute_tool>\s*(\w+:\w+)\s*(\{[^}]*\})\s*</execute_tool>"

        # Pattern 3: Inline tool format
        # Tool: builtin:replace_in_file
        # Tool Input:
        # ```json
        # {"arg": "value"}
        # ```
        inline_pattern = (
            r"Tool:\s*(\w+:\w+)\s*\n\s*Tool Input:\s*\n\s*```json\s*\n(.*?)\n\s*```"
        )

        # Try all patterns to catch mixed formats and parallel calls

        # Try complex pattern first
        complex_matches = re.findall(complex_pattern, content, re.DOTALL)
        for match in complex_matches:
            try:
                tool_name = match[0]  # e.g., "builtin:bash_execute"
                parameters_json = match[1].strip()

                # Convert tool name format (builtin:bash_execute -> builtin_bash_execute)
                gemini_tool_name = tool_name.replace(":", "_")

                # Validate JSON
                try:
                    json.loads(parameters_json)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in complex XML tool call: {parameters_json}"
                    )
                    continue

                # Create function call object
                function_call = SimpleNamespace()
                function_call.name = gemini_tool_name
                function_call.args = parameters_json

                function_calls.append(function_call)
                logger.info(
                    f"Parsed complex XML-style tool call: {gemini_tool_name} with args: {function_call.args}"
                )

            except Exception as e:
                logger.warning(f"Failed to parse complex XML-style tool call: {e}")
                continue

        # Try simple pattern (always try, don't skip based on previous matches)
        simple_matches = re.findall(simple_pattern, content, re.DOTALL)
        for match in simple_matches:
            try:
                tool_name = match[0]  # e.g., "builtin:bash_execute"
                args_json = match[1].strip()

                # Convert tool name format (builtin:bash_execute -> builtin_bash_execute)
                gemini_tool_name = tool_name.replace(":", "_")

                # Validate JSON
                try:
                    json.loads(args_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in simple XML tool call: {args_json}")
                    continue

                # Create function call object
                function_call = SimpleNamespace()
                function_call.name = gemini_tool_name
                function_call.args = args_json

                function_calls.append(function_call)
                logger.info(
                    f"Parsed simple XML-style tool call: {gemini_tool_name} with args: {function_call.args}"
                )

            except Exception as e:
                logger.warning(f"Failed to parse simple XML-style tool call: {e}")
                continue

        # Try inline pattern (always try, don't skip based on previous matches)
        inline_matches = re.findall(inline_pattern, content, re.DOTALL)
        for match in inline_matches:
            try:
                tool_name = match[0]  # e.g., "builtin:replace_in_file"
                args_json = match[1].strip()

                # Convert tool name format (builtin:replace_in_file -> builtin_replace_in_file)
                gemini_tool_name = tool_name.replace(":", "_")

                # Validate JSON
                try:
                    json.loads(args_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in inline tool call: {args_json}")
                    continue

                # Create function call object
                function_call = SimpleNamespace()
                function_call.name = gemini_tool_name
                function_call.args = args_json

                function_calls.append(function_call)
                logger.info(
                    f"Parsed inline tool call: {gemini_tool_name} with args: {function_call.args}"
                )

            except Exception as e:
                logger.warning(f"Failed to parse inline tool call: {e}")
                continue

        # Log summary of parsed tool calls
        if function_calls:
            logger.info(
                f"Successfully parsed {len(function_calls)} tool calls from Gemini response"
            )
            for i, call in enumerate(function_calls, 1):
                logger.info(f"  {i}. {call.name}")

        return function_calls

    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract any text that appears before tool calls in the response."""
        import re

        # Pattern to find text before various tool call formats
        patterns = [
            r"^(.*?)(?=<execute_tool>)",  # XML-style tool calls
            r"^(.*?)(?=\w+\s*\()",  # Python-style function calls
            r"^(.*?)(?=Tool:\s*\w+:\w+)",  # Inline tool calls
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                text_before = match.group(1).strip()
                if text_before:  # Only return if there's actual content
                    return text_before

        return ""

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Gemini format."""
        # Gemini expects a single string prompt, so we combine all messages
        gemini_prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                gemini_prompt_parts.append(f"System: {content}")
            elif role == "user":
                gemini_prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                gemini_prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                # Handle tool results
                gemini_prompt_parts.append(f"Tool Result: {content}")

        return "\n\n".join(gemini_prompt_parts)

    def _add_tool_results_to_conversation(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
        tool_results: List[str],
    ) -> List[Dict[str, Any]]:
        """Add tool results to conversation in Gemini format (text-based)."""
        updated_messages = messages.copy()

        # Add tool results as a single user message that provides the results
        tool_results_text = []
        for i, (tool_call, result) in enumerate(zip(tool_calls, tool_results)):
            tool_name = (
                getattr(tool_call, "name", "unknown")
                if hasattr(tool_call, "name")
                else tool_call.get("function", {}).get("name", "unknown")
            )
            tool_results_text.append(
                f"Tool '{tool_name}' executed successfully. Result:\n{result}"
            )

        if tool_results_text:
            tool_summary = "\n\n".join(tool_results_text)
            updated_messages.append(
                {
                    "role": "user",
                    "content": f"The requested tools have been executed. Here are the results:\n\n{tool_summary}\n\nNow please provide your response based on these tool execution results.",
                }
            )

        return updated_messages

    async def _generate_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        interactive: bool = True,
    ) -> Any:
        """Generate completion using Gemini API."""
        # Use the stream parameter passed in (centralized logic decides streaming behavior)

        # Convert messages to Gemini format
        gemini_prompt = self._convert_messages_to_gemini_format(messages)

        # Configure tool calling behavior
        tool_config = None
        if tools:
            try:
                # Configure function calling mode for compositional function calling
                mode_map = {
                    "AUTO": types.FunctionCallingConfigMode.AUTO,
                    "ANY": types.FunctionCallingConfigMode.ANY,
                    "NONE": types.FunctionCallingConfigMode.NONE,
                }

                # Use configured mode, or fall back to legacy force_function_calling setting
                if self.gemini_config.function_calling_mode in mode_map:
                    mode = mode_map[self.gemini_config.function_calling_mode]
                elif self.gemini_config.force_function_calling:
                    mode = types.FunctionCallingConfigMode.ANY
                else:
                    mode = types.FunctionCallingConfigMode.AUTO

                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=mode)
                )
                logger.debug(f"Configured function calling mode: {mode}")

            except Exception as e:
                logger.warning(f"Could not configure function calling: {e}")

        config = types.GenerateContentConfig(
            temperature=self.gemini_config.temperature,
            max_output_tokens=self.gemini_config.max_output_tokens,
            top_p=self.gemini_config.top_p,
            top_k=self.gemini_config.top_k,
            tools=tools if tools else None,
            tool_config=tool_config,
        )

        try:
            if stream:
                return await self._handle_streaming_response(
                    gemini_prompt, config, messages
                )
            else:
                # Non-streaming (for subagents) - need to process tool calls and return final text
                response = await self._make_gemini_request_with_retry(
                    lambda: self.gemini_client.models.generate_content(
                        model=self.gemini_config.model,
                        contents=gemini_prompt,
                        config=config,
                    )
                )
                return await self._handle_non_streaming_response(
                    response, messages, interactive=interactive
                )
        except Exception as e:
            # Re-raise tool permission denials so they can be handled at the chat level
            from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

            if isinstance(e, ToolDeniedReturnToPrompt):
                raise  # Re-raise the exception to bubble up to interactive chat

            import traceback

            logger.error(f"Error in Gemini completion: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _execute_function_calls(
        self, function_calls: List, streaming_mode=False
    ) -> tuple:
        """Execute a list of function calls and return results and output."""
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        function_results = []
        all_tool_output = (
            []
        )  # Collect all tool execution output for non-interactive mode

        # Prepare tool info for parallel execution
        tool_info_list = []
        tool_coroutines = []

        # Check for interruption before starting any tool execution - removed input_handler dependency

        for i, function_call in enumerate(function_calls, 1):
            tool_name = function_call.name.replace(
                "_", ":", 1
            )  # Convert back to MCP format

            # Parse arguments from function call
            arguments = {}
            if hasattr(function_call, "args") and function_call.args:
                try:
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
                i,
                tool_name,
                arguments,
                self.is_subagent,
                interactive=not self.is_subagent,
            )
            if not self.is_subagent and not streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            elif not self.is_subagent and streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            else:
                all_tool_output.append(tool_execution_msg)

            # Create coroutine for parallel execution
            tool_coroutines.append(self._execute_mcp_tool(tool_name, arguments))

        # Execute all tools in parallel
        if tool_coroutines:
            try:
                # Execute all tool calls concurrently like DeepSeek
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
                        interactive=not self.is_subagent,
                    )

                    # Fix newlines in tool result messages to have proper cursor positioning
                    if not self.is_subagent and (not streaming_mode or streaming_mode):
                        # Replace any bare newlines with \n\r to ensure proper cursor positioning
                        formatted_result_msg = tool_result_msg.replace("\n", "\n\r")
                        print(f"\r\x1b[K{formatted_result_msg}", flush=True)
                    else:
                        # Only add to tool output for non-interactive mode
                        all_tool_output.append(tool_result_msg)

            except Exception as e:
                # Re-raise tool permission denials without adding to results
                if isinstance(e, ToolDeniedReturnToPrompt):
                    raise e  # Re-raise without adding to function_results

                # Handle any unexpected errors during parallel execution
                error_msg = f"Error during parallel tool execution: {str(e)}"
                all_tool_output.append(error_msg)
                function_results.append(f"PARALLEL EXECUTION FAILED: {error_msg}")

        return function_results, all_tool_output

    async def _handle_complete_response(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        original_messages: List[Dict[str, str]],
    ) -> str:
        """Handle complete response from Gemini with iterative tool calling."""
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        current_prompt = prompt
        all_accumulated_output = []

        try:
            while True:

                # Make API call to Gemini
                response = await self._make_gemini_request_with_retry(
                    lambda: self.gemini_client.models.generate_content(
                        model=self.gemini_config.model,
                        contents=current_prompt,
                        config=config,
                    )
                )

                # Parse response content
                function_calls = self.parse_tool_calls(response)
                # Extract text response from response
                text_response = ""
                if (
                    response.candidates
                    and response.candidates[0].content
                    and response.candidates[0].content.parts
                ):
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            text_response += part.text
                elif hasattr(response, "text") and response.text:
                    text_response = response.text
                elif (
                    response.candidates
                    and hasattr(response.candidates[0], "text")
                    and response.candidates[0].text
                ):
                    text_response = response.candidates[0].text

                # Debug logging
                logger.debug(
                    f"Parsed {len(function_calls)} function calls and {len(text_response)} chars of text"
                )
                logger.debug(f"Full text response: {repr(text_response)}")
                if function_calls:
                    logger.debug(
                        f"Function calls: {[fc.name for fc in function_calls]}"
                    )

                # Accumulate text response for non-interactive mode only
                # Interactive mode printing is handled by the chat loop to avoid duplication
                if text_response and self.is_subagent:
                    if function_calls:
                        # Text with function calls - add to accumulated output
                        all_accumulated_output.append(f"Assistant: {text_response}")
                    else:
                        # Text without function calls - this is the final response
                        all_accumulated_output.append(f"Assistant: {text_response}")

                if function_calls:
                    # Handle function calls
                    if not self.is_subagent:
                        # Check if there's any buffered text that needs to be displayed before tool execution
                        text_buffer = getattr(self, "_text_buffer", "")
                        if text_buffer.strip():
                            # Format and display the buffered text with Assistant prefix
                            formatted_response = self.format_markdown(text_buffer)
                            # Replace newlines with \r\n for proper terminal handling
                            formatted_response = formatted_response.replace(
                                "\n", "\r\n"
                            )
                            print(f"\r\x1b[K\r\nAssistant: {formatted_response}")
                            # Clear the buffer
                            self._text_buffer = ""

                        print(
                            f"\r\n{self.display_tool_execution_start(len(function_calls), self.is_subagent, interactive=not self.is_subagent)}",
                            flush=True,
                        )

                    # Execute function calls using centralized method
                    try:
                        function_results, tool_output = (
                            await self._execute_function_calls(function_calls)
                        )
                    except ToolDeniedReturnToPrompt as e:
                        # Tool permission denied - exit immediately without making API call
                        raise e  # Exit immediately

                    # Note: We don't add tool execution status messages to accumulated output
                    # as they are only for user feedback and cause LLM hallucinations

                    # Create a clean follow-up prompt to avoid re-execution loops
                    tool_results_text = "\n".join(function_results)
                    original_request = (
                        original_messages[-1]["content"]
                        if original_messages
                        else "Continue processing"
                    )

                    current_prompt = f"""Original request: {original_request}

Tool execution completed successfully. Results:
{tool_results_text}

Based on these tool results, please provide your final response. Do not re-execute any tools unless specifically needed for a different purpose."""

                    # Continue the loop - let Gemini decide if more tools are needed
                    continue
                else:
                    # No function calls - this is the final response
                    if self.is_subagent and all_accumulated_output:
                        # Include all accumulated output (final response already included in loop above)
                        return "\n".join(all_accumulated_output)
                    else:
                        return text_response if text_response else ""

        except Exception as e:
            import traceback

            # Log detailed error information
            logger.error(f"Error in Gemini complete response: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Extract more specific error details if available
            error_details = str(e)
            if hasattr(e, "__cause__") and e.__cause__:
                logger.error(f"Caused by: {e.__cause__}")
                error_details += f" (caused by: {e.__cause__})"

            if hasattr(e, "__context__") and e.__context__:
                logger.error(f"Context: {e.__context__}")

            # Check for common Gemini API error patterns
            if "RetryError" in str(e):
                return f"Error: Gemini API retry failed - likely a network or rate limit issue. Details: {error_details}"
            elif "ClientError" in str(e):
                return f"Error: Gemini API client error - check API key and configuration. Details: {error_details}"
            elif "timeout" in str(e).lower():
                return f"Error: Request timed out. Details: {error_details}"
            else:
                return f"Error: {error_details}"

    async def _handle_streaming_response(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        original_messages: List[Dict[str, str]],
    ):
        """Handle streaming response from Gemini with iterative tool calling."""
        import asyncio

        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        async def async_stream_generator():
            current_prompt = prompt
            consecutive_failures = 0
            max_retries = 3

            try:
                while True:

                    # Stream a single response
                    accumulated_content = ""
                    function_calls = []

                    logger.debug(
                        f"Starting Gemini streaming with prompt length: {len(current_prompt)}"
                    )

                    # Make streaming API call
                    stream_response = None
                    try:
                        logger.debug(
                            f"Making Gemini streaming request with prompt length: {len(current_prompt)}"
                        )
                        logger.debug(f"Model: {self.gemini_config.model}")

                        # Log the last 1000 chars of the prompt to see what's being sent
                        if len(current_prompt) > 1000:
                            logger.debug(
                                f"Prompt excerpt (last 1000 chars): ...{current_prompt[-1000:]}"
                            )
                        else:
                            logger.debug(f"Full prompt: {current_prompt}")

                        # Log the config being used
                        logger.debug(f"Request config: {config}")

                        stream_response = await self._make_gemini_request_with_retry(
                            lambda: self.gemini_client.models.generate_content_stream(
                                model=self.gemini_config.model,
                                contents=current_prompt,
                                config=config,
                            )
                        )

                        if stream_response is None:
                            logger.error("Gemini stream response is None")
                            yield "Error: No response from Gemini (stream is None)"
                            return

                        logger.debug("Successfully created Gemini stream")

                    except Exception as e:
                        logger.error(f"Failed to create Gemini stream: {e}")
                        logger.error(f"Exception type: {type(e)}")
                        logger.error(f"Exception args: {e.args}")

                        # Check for specific error types
                        if "timeout" in str(e).lower():
                            yield f"⏱️ Request timed out. Gemini may be overloaded. Error: {str(e)}"
                        elif "429" in str(e) or "rate limit" in str(e).lower():
                            yield f"🚫 Rate limited by Gemini API. Please wait and try again. Error: {str(e)}"
                        elif "500" in str(e) or "Internal Server Error" in str(e):
                            yield f"🔥 Gemini API Internal Server Error (500). This usually means the prompt is too long or contains problematic content.\n"
                            yield f"💡 Try using '/compact' to reduce conversation length, or start a new conversation.\n"
                            yield f"Error details: {str(e)}"
                        else:
                            yield f"❌ Error creating stream: {str(e)}"
                        return

                    # Process streaming chunks
                    chunk_count = 0
                    has_any_content = False
                    stream_started = False
                    try:
                        logger.debug(
                            f"About to iterate stream_response: {type(stream_response)}"
                        )
                        for chunk in stream_response:
                            try:
                                chunk_count += 1
                                stream_started = True
                                logger.debug(f"Processing chunk {chunk_count}")

                                if chunk is None:
                                    logger.warning(
                                        f"Chunk {chunk_count} is None, skipping"
                                    )
                                    continue

                                if hasattr(chunk, "text") and chunk.text:
                                    accumulated_content += chunk.text
                                    has_any_content = True

                                    # Yield content normally first
                                    yield chunk.text

                                # Check for function calls in chunk
                                if hasattr(chunk, "candidates") and chunk.candidates:
                                    try:
                                        if (
                                            chunk.candidates[0]
                                            and hasattr(chunk.candidates[0], "content")
                                            and chunk.candidates[0].content
                                        ):
                                            if (
                                                hasattr(
                                                    chunk.candidates[0].content, "parts"
                                                )
                                                and chunk.candidates[0].content.parts
                                            ):
                                                for part in chunk.candidates[
                                                    0
                                                ].content.parts:
                                                    if (
                                                        hasattr(part, "function_call")
                                                        and part.function_call
                                                    ):
                                                        function_calls.append(
                                                            part.function_call
                                                        )
                                    except (IndexError, AttributeError) as e:
                                        logger.warning(
                                            f"Error processing chunk {chunk_count} candidates: {e}"
                                        )
                                        continue

                            except Exception as e:
                                logger.error(
                                    f"Error processing chunk {chunk_count}: {e}"
                                )
                                # Don't yield error to user, just log and continue
                                continue
                    except Exception as stream_error:
                        import traceback

                        logger.error(
                            f"Error iterating stream after {chunk_count} chunks: {stream_error}"
                        )
                        logger.error(f"Stream error type: {type(stream_error)}")
                        logger.error(f"Stream error details: {traceback.format_exc()}")
                        if not stream_started:
                            logger.error(
                                "Stream never started - this suggests Gemini API is not responding"
                            )
                            yield f"\n⚠️ Gemini API is not responding. Stream never started. Error: {stream_error}\n"
                            return
                        elif not has_any_content:
                            logger.error("Stream started but produced no content")
                            yield f"\n⚠️ Gemini stream started but failed after {chunk_count} chunks. Error: {stream_error}\n"
                            return

                    # Parse additional function calls from text using shared method BEFORE checking for failure
                    if accumulated_content:
                        python_calls = self._parse_python_style_function_calls(
                            accumulated_content
                        )
                        if python_calls:
                            function_calls.extend(python_calls)

                        xml_calls = self._parse_xml_style_tool_calls(
                            accumulated_content
                        )
                        if xml_calls:
                            function_calls.extend(xml_calls)

                    # Check if we got any meaningful response (text content OR function calls)
                    has_meaningful_response = (
                        has_any_content or accumulated_content or function_calls
                    )

                    if not has_meaningful_response:
                        consecutive_failures += 1
                        if consecutive_failures > max_retries:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for Gemini streaming"
                            )
                            yield f"\n⚠️ Max retries exceeded. Ending conversation after {consecutive_failures} failures.\n"
                            return

                        if not stream_started:
                            logger.warning(
                                f"Gemini stream never started - API may be unresponsive (attempt {consecutive_failures}/{max_retries})"
                            )
                            yield f"\n⚠️ Gemini API unresponsive. Retrying... (attempt {consecutive_failures}/{max_retries})\n"
                        else:
                            logger.warning(
                                f"No meaningful response from Gemini stream after {chunk_count} chunks (attempt {consecutive_failures}/{max_retries})"
                            )
                            yield f"\n⚠️ No meaningful response from Gemini after {chunk_count} chunks. Retrying... (attempt {consecutive_failures}/{max_retries})\n"

                        # Add a brief delay before retrying
                        await asyncio.sleep(1.0)
                        continue

                    # Reset failure counter on successful response
                    consecutive_failures = 0

                    # Check if we have function calls to execute
                    if function_calls:
                        # Check if there's any buffered text that needs to be displayed before tool execution
                        text_buffer = getattr(self, "_text_buffer", "")
                        if text_buffer.strip():
                            # Format and display the buffered text with Assistant prefix
                            formatted_response = self.format_markdown(text_buffer)
                            # Replace newlines with \r\n for proper terminal handling
                            formatted_response = formatted_response.replace(
                                "\n", "\r\n"
                            )
                            print(f"\r\x1b[K\r\nAssistant: {formatted_response}")
                            # Clear the buffer
                            self._text_buffer = ""

                        # Show tool execution indicator to user via print (not yielded to avoid LLM contamination)
                        print(
                            f"\r\n{self.display_tool_execution_start(len(function_calls), self.is_subagent, interactive=not self.is_subagent)}",
                            flush=True,
                        )

                        # Execute function calls using centralized method
                        try:
                            function_results, tool_output = (
                                await self._execute_function_calls(
                                    function_calls, streaming_mode=True
                                )
                            )
                        except ToolDeniedReturnToPrompt as e:
                            # Exit generator immediately - cannot continue
                            yield f"\n🚫 Tool execution denied - returning to prompt.\n"
                            raise e  # Raise the exception after yielding the message

                        # Check if we just spawned subagents and should interrupt immediately
                        if (
                            self.subagent_manager
                            and self.subagent_manager.get_active_count() > 0
                        ):
                            # Check if any of the function calls were "task" tools (subagent spawning)
                            task_tools_executed = any(
                                "task" in fc.name for fc in function_calls
                            )
                            if task_tools_executed:
                                # Interrupt immediately after spawning subagents
                                yield f"\n🔄 Subagents spawned - interrupting main stream to wait for completion...\n"

                                # Wait for all subagents to complete and collect results
                                subagent_results = (
                                    await self._collect_subagent_results()
                                )

                                if subagent_results:
                                    # Add subagent results to the conversation and restart
                                    yield f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...\n"

                                    # Create new message with subagent results
                                    results_summary = "\n".join(
                                        [
                                            f"**Subagent Task: {result['description']}**\n{result['content']}"
                                            for result in subagent_results
                                        ]
                                    )

                                    # Create a new conversation context that includes the original request and subagent results
                                    # but frames it as analysis rather than a new spawning request
                                    continuation_message = {
                                        "role": "user",
                                        "content": f"""I requested: {original_messages[-1]['content']}

You spawned subagents and they have completed their tasks. Here are the results:

{results_summary}

Please provide your final analysis based on these subagent results. Do not spawn any new subagents - just analyze the provided data.""",
                                    }

                                    # Replace conversation with just the continuation context
                                    new_messages = [continuation_message]

                                    # Restart the conversation with subagent results
                                    yield f"\n🔄 Restarting conversation with subagent results...\n"
                                    new_response = await self._generate_completion(
                                        new_messages,
                                        tools=self.convert_tools_to_llm_format(),
                                        stream=True,
                                    )

                                    # Yield the new response (check if it's a generator or string)
                                    if hasattr(new_response, "__aiter__"):
                                        async for new_chunk in new_response:
                                            yield new_chunk
                                    else:
                                        # If it's a string, yield it directly
                                        yield str(new_response)

                                    # Exit since we've restarted
                                    return
                                else:
                                    yield f"\n⚠️ No results collected from subagents.\n"
                                    return

                        # Don't yield tool execution details to avoid LLM hallucinations
                        # The tool results will be included in the next iteration's prompt context

                        # Indicate we're getting the follow-up response (via print to avoid LLM contamination)
                        print(
                            f"\n\r\n\r{self.display_tool_processing(self.is_subagent, interactive=not self.is_subagent)}\n\r",
                            flush=True,
                        )

                        # Create follow-up prompt for next iteration
                        tool_results_text = "\n".join(function_results)

                        # Instead of appending to the growing prompt, create a clean context
                        # with just the original request and tool results to avoid confusion
                        original_request = (
                            original_messages[-1]["content"]
                            if original_messages
                            else "Continue processing"
                        )

                        # Create a focused prompt that prevents re-execution
                        current_prompt = f"""Original request: {original_request}

Tool execution has been completed successfully. Results:
{tool_results_text}

Based on these tool results, please provide your final response. Do not re-execute any tools unless specifically needed for a different purpose."""

                        logger.debug(
                            f"Updated prompt length after tool execution: {len(current_prompt)}"
                        )
                        logger.debug(f"Tool results length: {len(tool_results_text)}")

                        # Continue the loop - let Gemini decide if more tools are needed
                        continue
                    else:
                        # No function calls - this is the final response
                        # Yield any accumulated content before exiting
                        if accumulated_content.strip():
                            yield accumulated_content

                        # Check for subagent interrupts before ending
                        if (
                            self.subagent_manager
                            and self.subagent_manager.get_active_count() > 0
                        ):
                            # INTERRUPT STREAMING - collect subagent results and restart
                            yield f"\n🔄 Subagents active - interrupting main stream to collect results...\n"

                            # Wait for all subagents to complete and collect results
                            subagent_results = await self._collect_subagent_results()

                            if subagent_results:
                                # Add subagent results to the conversation and restart
                                yield f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...\n"

                                # Create new message with subagent results
                                results_summary = "\n".join(
                                    [
                                        f"**Subagent Task: {result['description']}**\n{result['content']}"
                                        for result in subagent_results
                                    ]
                                )

                                # Create a new conversation context that includes the original request and subagent results
                                # but frames it as analysis rather than a new spawning request
                                continuation_message = {
                                    "role": "user",
                                    "content": f"""I requested: {original_messages[-1]['content']}

You spawned subagents and they have completed their tasks. Here are the results:

{results_summary}

Please provide your final analysis based on these subagent results. Do not spawn any new subagents - just analyze the provided data.""",
                                }

                                # Replace conversation with just the continuation context
                                new_messages = [continuation_message]

                                # Restart the conversation with subagent results
                                yield f"\n🔄 Restarting conversation with subagent results...\n"
                                new_response = await self._generate_completion(
                                    new_messages,
                                    tools=self.convert_tools_to_llm_format(),
                                    stream=True,
                                )

                                # Yield the new response (check if it's a generator or string)
                                if hasattr(new_response, "__aiter__"):
                                    async for new_chunk in new_response:
                                        yield new_chunk
                                else:
                                    # If it's a string, yield it directly
                                    yield str(new_response)

                                # Exit since we've restarted
                                return

                        return

            except GeneratorExit:
                logger.debug("Stream generator closed by client")
                return
            except Exception as e:
                # Re-raise tool permission denials so they can be handled at the chat level
                from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

                if isinstance(e, ToolDeniedReturnToPrompt):
                    raise  # Re-raise the exception to bubble up to interactive chat

                error_msg = f"Error in streaming: {str(e)}"
                logger.error(error_msg)
                yield error_msg

        return async_stream_generator()

    async def shutdown(self):
        """Shutdown all MCP connections and HTTP client."""
        # Close HTTP client first
        if hasattr(self, "http_client") and self.http_client:
            try:
                await self.http_client.aclose()
                logger.info("Closed Gemini HTTP client")
            except Exception as e:
                logger.error(f"Error closing Gemini HTTP client: {e}")

        # Call parent shutdown for MCP connections
        await super().shutdown()
