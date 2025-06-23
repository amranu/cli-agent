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
from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt
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
        # Call parent initialization (which will call our abstract methods)
        super().__init__(config, is_subagent)

    # Centralized Client Initialization Implementation
    # ===============================================

    def _get_provider_config(self):
        """Get Gemini-specific configuration."""
        return self.config.get_gemini_config()

    def _get_streaming_preference(self, provider_config) -> bool:
        """Get streaming preference for Gemini."""
        return True  # Gemini always uses streaming

    def _calculate_timeout(self, provider_config) -> float:
        """Calculate timeout based on Gemini model."""
        return 120.0  # 2 minutes for Gemini requests

    def _create_llm_client(self, provider_config, timeout_seconds):
        """Create the Gemini client with HTTP client configuration."""
        self.gemini_config = provider_config  # Store for later use

        try:
            import httpx

            # Create HTTP client with custom timeout
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

            client = genai.Client(
                api_key=provider_config.api_key, http_client=http_client
            )
            self.http_client = http_client  # Store reference for cleanup
            logger.debug(f"Gemini client initialized with {timeout_seconds}s timeout")

        except Exception as e:
            import traceback

            logger.warning(f"Failed to create custom HTTP client: {e}")
            logger.debug(f"HTTP client creation traceback: {traceback.format_exc()}")
            # Fallback to default client
            try:
                client = genai.Client(api_key=provider_config.api_key)
                self.http_client = None
            except Exception as fallback_error:
                logger.error(
                    f"Failed to create even default Gemini client: {fallback_error}"
                )
                raise

        # Store as both _client (from base class) and gemini_client (for compatibility)
        self.gemini_client = client
        return client

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

    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract text that appears before Gemini tool calls."""
        import re

        # Gemini-specific patterns
        patterns = [
            # Gemini XML-style tool calls
            r"^(.*?)(?=<execute_tool>)",
            r"^(.*?)(?=<tool_call>)",
            # Function call patterns
            r"^(.*?)(?=\w+\s*\()",
            # Inline tool calls
            r"^(.*?)(?=Tool:\s*\w+:\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                text_before = match.group(1).strip()
                if text_before:  # Only return if there's actual content
                    # Remove code block markers if present
                    text_before = re.sub(r"^```\w*\s*", "", text_before)
                    text_before = re.sub(r"\s*```$", "", text_before)
                    return text_before

        return ""

    def _get_llm_specific_instructions(self) -> str:
        """Provide Gemini-specific instructions that emphasize tool usage and action."""
        # Add Gemini-specific instructions based on agent type
        if self.is_subagent:
            return """**CRITICAL FOR GEMINI SUBAGENTS: FOCUS & EXECUTION**
- You are a SUBAGENT - execute your specific task efficiently and provide clear results
- When a task requires action (running commands, reading files, etc.), you MUST use the appropriate tools
- Do NOT just describe what you would do - actually DO IT using the tools
- Take action FIRST, then provide analysis based on the actual results
- You CANNOT spawn other subagents - focus on your assigned task only
- **MANDATORY:** You MUST end your response with `emit_result` containing a comprehensive summary of your findings

**Required Completion Pattern:**
1. Use tools to gather information and perform actions
2. Analyze the results from your tool executions
3. Call `emit_result` with a detailed summary of your findings, conclusions, and recommendations

Example: If asked to "run uname -a", do NOT respond with "I will run uname -a command" - instead immediately use bash_execute with the command, then call emit_result with the system information summary."""
        else:
            return """**CRITICAL FOR GEMINI: TOOL USAGE REQUIREMENTS**
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

    async def _handle_non_streaming_response(
        self, response, original_messages: List[Dict[str, Any]], **kwargs
    ) -> str:
        """Handle non-streaming response from Gemini, processing tool calls if needed."""

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

    def _normalize_tool_calls_to_standard_format(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert Gemini tool calls to standardized format."""
        normalized_calls = []

        for i, tool_call in enumerate(tool_calls):
            if hasattr(tool_call, "name"):
                # Gemini function call object
                normalized_calls.append(
                    {
                        "id": getattr(tool_call, "id", f"call_{i}"),
                        "name": tool_call.name,
                        "arguments": getattr(tool_call, "args", {}),
                    }
                )
            elif isinstance(tool_call, dict):
                # Dict format
                if "function" in tool_call:
                    # Structured format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call["function"].get("name", "unknown"),
                            "arguments": tool_call["function"].get("arguments", {}),
                        }
                    )
                else:
                    # Simple dict format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call.get("name", "unknown"),
                            "arguments": tool_call.get("arguments", {}),
                        }
                    )
            else:
                # Fallback for other formats
                normalized_calls.append(
                    {
                        "id": f"call_{i}",
                        "name": str(tool_call),
                        "arguments": {},
                    }
                )

        return normalized_calls

    def _get_current_runtime_model(self) -> str:
        """Get the actual Gemini model being used at runtime."""
        return self.gemini_config.model

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

            if isinstance(e, ToolDeniedReturnToPrompt):
                raise  # Re-raise the exception to bubble up to interactive chat

            import traceback

            logger.error(f"Error in Gemini completion: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _handle_complete_response(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
        original_messages: List[Dict[str, str]],
    ) -> str:
        """Handle complete response from Gemini with iterative tool calling."""

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

                    # Execute function calls using base agent's centralized method
                    try:
                        function_results, tool_output = (
                            await self.execute_function_calls(
                                function_calls,
                                interactive=not self.is_subagent,
                                streaming_mode=False,
                            )
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

                    # Create a mock response for centralized tool processing
                    mock_response = type("MockResponse", (), {})()
                    mock_response.candidates = [type("MockCandidate", (), {})()]
                    mock_response.candidates[0].content = type("MockContent", (), {})()
                    mock_response.candidates[0].content.parts = []

                    # Add function calls as parts
                    for func_call in function_calls:
                        part = type("MockPart", (), {})()
                        part.function_call = func_call
                        mock_response.candidates[0].content.parts.append(part)

                    # Add text content if available
                    if accumulated_content:
                        text_part = type("MockPart", (), {})()
                        text_part.text = accumulated_content
                        mock_response.candidates[0].content.parts.append(text_part)

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

                    # Use centralized tool call processing for streaming
                    current_messages = (
                        []
                    )  # Create empty messages list for centralized processing
                    try:
                        updated_messages, continuation_message, has_tool_calls = (
                            await self._process_tool_calls_centralized(
                                mock_response,
                                current_messages,
                                original_messages,
                                interactive=not self.is_subagent,
                                streaming_mode=True,
                                accumulated_content=accumulated_content,
                            )
                        )

                        if has_tool_calls:
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

                            if continuation_message:
                                # Yield the interrupt and completion messages for streaming
                                yield "\n🔄 Subagents spawned - interrupting main stream to wait for completion...\n"
                                yield "\n📋 Collected subagent result(s). Restarting with results...\n"

                                # Replace conversation with just the continuation context
                                new_messages = [continuation_message]

                                # Restart the conversation with subagent results
                                yield "\n🔄 Restarting conversation with subagent results...\n"
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

                            # Don't yield tool execution details to avoid LLM hallucinations
                            # The tool results will be included in the next iteration's prompt context

                            # Indicate we're getting the follow-up response (via print to avoid LLM contamination)
                            print(
                                f"\n\r\n\r{self.display_tool_processing(self.is_subagent, interactive=not self.is_subagent)}\n\r",
                                flush=True,
                            )

                            # Extract tool results from updated messages for prompt creation
                            tool_results_text = ""
                            for msg in updated_messages:
                                if msg.get("role") == "tool":
                                    tool_results_text += f"{msg.get('content', '')}\n"

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
                            logger.debug(
                                f"Tool results length: {len(tool_results_text)}"
                            )

                            # Continue the loop - let Gemini decide if more tools are needed
                            continue

                    except ToolDeniedReturnToPrompt as e:
                        # Exit generator immediately - cannot continue
                        yield f"\n🚫 Tool execution denied - returning to prompt.\n"
                        raise e  # Raise the exception after yielding the message
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
