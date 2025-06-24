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
        """Parse tool calls using centralized framework with Gemini-specific extraction."""
        from types import SimpleNamespace

        # Handle string responses (no tool calls possible)
        if isinstance(response, str):
            return []

        # Extract text content from response for text-based tool call parsing
        text_content = ""
        if (
            hasattr(response, "candidates")
            and response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_content += part.text
        elif hasattr(response, "text") and response.text:
            text_content = response.text
        elif (
            hasattr(response, "candidates")
            and response.candidates
            and hasattr(response.candidates[0], "text")
            and response.candidates[0].text
        ):
            text_content = response.candidates[0].text

        # Use centralized parsing framework
        unified_calls = self._parse_tool_calls_generic(response, text_content)

        # Convert back to SimpleNamespace format for compatibility
        converted_calls = []
        for call_dict in unified_calls:
            call = SimpleNamespace()
            call.name = call_dict["name"]
            call.args = call_dict["arguments"]
            call.id = call_dict.get("id")
            converted_calls.append(call)

        return converted_calls

    def _extract_structured_calls(self, response: Any) -> List[Any]:
        """Extract structured tool calls from Gemini response."""
        from types import SimpleNamespace

        structured_calls = []

        # Extract function calls from Gemini response
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        # Convert Gemini function call to SimpleNamespace format
                        fc = part.function_call
                        call = SimpleNamespace()
                        call.name = fc.name
                        call.args = dict(fc.args) if fc.args else {}
                        call.id = getattr(fc, "id", None)
                        structured_calls.append(call)

        return structured_calls

    def _parse_text_based_calls(self, text_content: str) -> List[Any]:
        """Parse text-based tool calls using Gemini-specific patterns."""
        text_calls = []

        if text_content:
            # Use existing Gemini parser for text-based calls
            tool_calls = GeminiToolCallParser.parse_all_formats(None, text_content)

            # Convert to SimpleNamespace format for consistency
            import json
            from types import SimpleNamespace

            for tc in tool_calls:
                function_call = SimpleNamespace()
                function_call.name = tc.function.name

                # Parse arguments from JSON string to dict
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

                text_calls.append(function_call)

        return text_calls

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

    async def _make_gemini_request_with_retry(
        self, request_func, max_retries: int = 3, base_delay: float = 1.0
    ):
        """Make a Gemini API request using centralized retry logic."""
        # Use centralized retry logic with Gemini-specific error handling
        return await self._make_api_request_with_retry(
            request_func, max_retries, base_delay
        )

    def _convert_to_provider_format(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to Gemini string format."""
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

    def _enhance_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Gemini-specific message enhancement - add system prompt for subagents/first messages."""
        enhanced_messages = messages
        is_first_message = len(messages) == 1 and messages[0].get("role") == "user"

        if self.is_subagent or is_first_message:
            system_prompt = self._create_system_prompt(for_first_message=True)
            enhanced_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

        return enhanced_messages

    def _is_retryable_error(self, error: Exception) -> bool:
        """Gemini-specific retryable error detection."""
        error_str = str(error).lower()
        # Gemini-specific retryable conditions plus generic ones
        return (
            super()._is_retryable_error(error)
            or "retryerror" in error_str
            or "internal server error" in error_str
            or "500" in error_str
            or "gemini" in error_str
        )

    def _get_current_runtime_model(self) -> str:
        """Get the actual Gemini model being used at runtime."""
        return self.gemini_config.model

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

    # ============================================================================
    # PROVIDER-SPECIFIC ADAPTER METHODS
    # ============================================================================

    def _extract_response_content(
        self, response: Any
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from Gemini response."""
        if not hasattr(response, "candidates") or not response.candidates:
            return "", [], {}

        candidate = response.candidates[0]
        text_content = ""
        tool_calls = []

        if hasattr(candidate, "content") and candidate.content:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_content += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(part.function_call)

        # Gemini doesn't have provider-specific features like reasoning content
        provider_data = {}
        return text_content, tool_calls, provider_data

    async def _process_streaming_chunks(
        self, response
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Process Gemini streaming chunks."""
        accumulated_content = ""
        accumulated_tool_calls = []

        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                accumulated_content += chunk.text

            # Check for function calls in chunk
            if hasattr(chunk, "candidates") and chunk.candidates:
                if (
                    chunk.candidates[0]
                    and hasattr(chunk.candidates[0], "content")
                    and chunk.candidates[0].content
                ):
                    if (
                        hasattr(chunk.candidates[0].content, "parts")
                        and chunk.candidates[0].content.parts
                    ):
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                accumulated_tool_calls.append(part.function_call)

        provider_data = {}
        return accumulated_content, accumulated_tool_calls, provider_data

    async def _make_api_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
        stream: bool = True,
    ) -> Any:
        """Make Gemini API request using centralized retry logic."""
        # Convert messages to Gemini string format
        if isinstance(messages, str):
            gemini_prompt = messages  # Already converted
        else:
            # Convert to Gemini format
            gemini_prompt = self._convert_to_provider_format(messages)

        # Configure tool calling behavior
        tool_config = None
        if tools:
            try:
                mode_map = {
                    "AUTO": types.FunctionCallingConfigMode.AUTO,
                    "ANY": types.FunctionCallingConfigMode.ANY,
                    "NONE": types.FunctionCallingConfigMode.NONE,
                }

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

        # Use centralized retry logic
        if stream:
            return await self._make_api_request_with_retry(
                lambda: self.gemini_client.models.generate_content_stream(
                    model=self.gemini_config.model,
                    contents=gemini_prompt,
                    config=config,
                )
            )
        else:
            return await self._make_api_request_with_retry(
                lambda: self.gemini_client.models.generate_content(
                    model=self.gemini_config.model,
                    contents=gemini_prompt,
                    config=config,
                )
            )

    def _create_mock_response(self, content: str, tool_calls: List[Any]) -> Any:
        """Create mock Gemini response for centralized processing."""
        mock_response = type("MockResponse", (), {})()
        mock_response.candidates = [type("MockCandidate", (), {})()]
        mock_response.candidates[0].content = type("MockContent", (), {})()
        mock_response.candidates[0].content.parts = []

        # Add function calls as parts
        for func_call in tool_calls:
            part = type("MockPart", (), {})()
            part.function_call = func_call
            mock_response.candidates[0].content.parts.append(part)

        # Add text content if available
        if content:
            text_part = type("MockPart", (), {})()
            text_part.text = content
            mock_response.candidates[0].content.parts.append(text_part)

        return mock_response

    def _handle_provider_specific_features(self, provider_data: Dict[str, Any]) -> None:
        """Handle Gemini-specific features (none currently)."""
        pass  # Gemini doesn't have special features like reasoning content

    def _format_provider_specific_content(self, provider_data: Dict[str, Any]) -> str:
        """Format Gemini-specific content for output (none currently)."""
        return ""  # Gemini doesn't have special content formatting

    async def _handle_buffered_streaming_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
        interactive: bool = True,
    ) -> str:
        """Handle streaming response for JSON mode - simplified version."""
        # Process chunks to get content and tool calls
        response = await self._make_api_request(messages, tools, stream=True)
        accumulated_content, tool_calls, provider_data = (
            await self._process_streaming_chunks(response)
        )

        # For streaming JSON mode, trigger the centralized base agent handler
        if self.streaming_json_callback:
            if tool_calls:
                # Create a mock response object for centralized processing
                mock_response = self._create_mock_response(
                    accumulated_content, tool_calls
                )
                return mock_response
            else:
                # Just text content, emit it directly
                self.streaming_json_callback(accumulated_content)

        return accumulated_content

    async def _generate_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        interactive: bool = True,
    ) -> Any:
        """Generate completion using Gemini API - delegate to centralized handlers."""
        # Check if we should use buffering for streaming JSON
        use_buffering = (
            hasattr(self, "streaming_json_callback")
            and self.streaming_json_callback is not None
        )

        try:
            if stream:
                if use_buffering:
                    # Use buffering for streaming JSON mode
                    return await self._handle_buffered_streaming_response(
                        messages, tools, interactive=interactive
                    )
                else:
                    return self._handle_streaming_response_generic(
                        await self._make_api_request(messages, tools, stream=True),
                        messages,
                        interactive=interactive,
                    )
            else:
                # Non-streaming response
                response = await self._make_api_request(messages, tools, stream=False)
                return await self._handle_complete_response_generic(
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
