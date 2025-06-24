#!/usr/bin/env python3
"""MCP Host implementation using Deepseek as the language model backend."""
import asyncio
import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from cli_agent.core.base_llm_provider import BaseLLMProvider
from cli_agent.utils.tool_conversion import OpenAIStyleToolConverter
from cli_agent.utils.tool_parsing import DeepSeekToolCallParser
from config import HostConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to INFO to see subagent messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MCPDeepseekHost(BaseLLMProvider):
    """MCP Host that uses Deepseek as the language model backend."""

    def __init__(self, config: HostConfig, is_subagent: bool = False):
        # Store last reasoning content for deepseek-reasoner (must be before super().__init__)
        self.last_reasoning_content: Optional[str] = None

        # Call parent initialization (which will call our abstract methods)
        super().__init__(config, is_subagent)

    # Centralized Client Initialization Implementation
    # ===============================================

    def _get_provider_config(self):
        """Get DeepSeek-specific configuration."""
        return self.config.get_deepseek_config()

    def _get_streaming_preference(self, provider_config) -> bool:
        """Get streaming preference for DeepSeek."""
        return provider_config.stream

    def _calculate_timeout(self, provider_config) -> float:
        """Calculate timeout based on DeepSeek model."""
        # Use same timeout for all DeepSeek models (reasoner needs longer but 600s works for all)
        return 600.0

    def _create_llm_client(self, provider_config, timeout_seconds):
        """Create the DeepSeek OpenAI client."""
        self.deepseek_config = provider_config  # Store for later use
        client = OpenAI(
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
            timeout=timeout_seconds,
        )
        # Store as both _client (from base class) and deepseek_client (for compatibility)
        self.deepseek_client = client
        return client

    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to Deepseek format using shared utilities."""
        converter = OpenAIStyleToolConverter()
        return converter.convert_tools(self.available_tools)

    # parse_tool_calls is now implemented in BaseLLMProvider

    def _extract_structured_calls_impl(self, response: Any) -> List[Any]:
        """Extract structured tool calls from DeepSeek response."""
        structured_calls = []

        # Handle standard OpenAI-style response objects with tool_calls attribute
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.debug(
                    f"Found {len(message.tool_calls)} tool calls in response object"
                )
                for tc in message.tool_calls:
                    try:
                        # Handle both dict and object arguments
                        if hasattr(tc.function, "arguments"):
                            args = tc.function.arguments
                            if isinstance(args, str):
                                import json

                                args = json.loads(args)
                        else:
                            args = {}

                        # Convert to SimpleNamespace format for compatibility
                        from types import SimpleNamespace

                        call = SimpleNamespace()
                        call.name = tc.function.name
                        call.args = args
                        call.id = getattr(tc, "id", None)
                        structured_calls.append(call)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse tool call {tc.function.name}: {e}"
                        )

        return structured_calls

    def _parse_text_based_calls_impl(self, text_content: str) -> List[Any]:
        """Parse text-based tool calls using DeepSeek-specific patterns."""
        text_calls = []

        if text_content:
            logger.debug("Parsing string response for tool calls")
            tool_calls = DeepSeekToolCallParser.parse_tool_calls(text_content)
            logger.debug(f"DeepSeekToolCallParser found {len(tool_calls)} tool calls")

            # Convert to SimpleNamespace format for consistency
            from types import SimpleNamespace

            for tc in tool_calls:
                call = SimpleNamespace()
                call.name = tc.function.name
                call.args = tc.function.arguments
                call.id = getattr(tc, "id", None)
                text_calls.append(call)

        return text_calls

    def _get_text_extraction_patterns(self) -> List[str]:
        """Get DeepSeek-specific regex patterns for extracting text before tool calls."""
        return [
            # DeepSeek tool call markers
            r"^(.*?)(?=<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>)",
            r"^(.*?)(?=```json\s*\{\s*\"function\")",
            r"^(.*?)(?=```python\s*<｜tool▁calls▁begin｜>)",
            # JSON function calls
            r"^(.*?)(?=\{\s*\"function\":)",
        ]

    def _get_llm_specific_instructions(self) -> str:
        """Provide DeepSeek-specific instructions."""
        if self.is_subagent:
            return """**Special Instructions for DeepSeek Subagents:**
1.  **Reason:** Use the <reasoning> section to outline your plan before taking action.
2.  **Act:** Execute your plan with tool calls. You can execute tool calls during reasoning blocks.
3.  **MANDATORY:** You MUST end your response by calling `emit_result` with a comprehensive summary of your findings.
4.  **Focus:** You are a subagent - you cannot spawn other subagents, only execute tools and provide results.

**Required Pattern:** After completing your investigation/task with tools, always call `emit_result` with detailed findings."""
        else:
            return """**Special Instructions for Deepseek Reasoner:**
1.  **Reason:** Use the <reasoning> section to outline your plan before taking action.
2.  **Act:** Execute your plan with tool calls. You can execute tool calls during reasoning blocks
3.  **Respond:** Provide the final answer to the user."""

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: Optional[bool] = None,
        modify_messages_in_place: bool = False,
    ) -> Union[str, Any]:
        """DeepSeek-specific response generation using centralized preprocessing."""
        # Use centralized message preprocessing
        processed_messages = self._preprocess_messages(messages)

        # Call parent's generate_response with processed messages
        return await super().generate_response(
            processed_messages, tools, stream, modify_messages_in_place
        )

    # _generate_completion is now implemented in BaseLLMProvider

    def _format_messages_for_reasoner(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages specifically for deepseek-reasoner model."""
        # Check if this is the first message in a chat with deepseek-reasoner
        is_first_message = len(messages) == 1 and messages[0].get("role") == "user"
        is_reasoner = self.deepseek_config.model == "deepseek-reasoner"

        enhanced_messages = messages

        # Handle system prompt differently for deepseek-reasoner
        if is_reasoner:
            # For deepseek-reasoner, only add system prompt to first message
            if is_first_message:
                system_prompt = self._create_system_prompt(for_first_message=True)
                if enhanced_messages and enhanced_messages[0].get("role") == "user":
                    # Prepend system prompt to user message
                    user_content = enhanced_messages[0]["content"]
                    enhanced_messages[0][
                        "content"
                    ] = f"{system_prompt}\n\n---\n\nUser Request: {user_content}"
            else:
                # For subsequent messages, prepend last reasoning content to user message
                if (
                    self.last_reasoning_content
                    and enhanced_messages
                    and enhanced_messages[-1].get("role") == "user"
                ):
                    user_content = enhanced_messages[-1]["content"]
                    enhanced_messages[-1][
                        "content"
                    ] = f"Previous reasoning: {self.last_reasoning_content}\n\n---\n\nUser Request: {user_content}"

        return enhanced_messages

    def _clean_message_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """DeepSeek-specific message cleaning to prevent JSON deserialization errors."""
        logger.debug(f"Cleaning {len(messages)} messages for DeepSeek")
        cleaned_messages = []

        for i, message in enumerate(messages):
            logger.debug(
                f"Processing message {i}: {type(message)} with keys: {list(message.keys()) if isinstance(message, dict) else 'not_dict'}"
            )

            # Ensure content is always a string
            content = message.get("content", "")
            if not isinstance(content, str):
                logger.debug(
                    f"Converting non-string content from {type(content)} to string"
                )
                content = str(content)

            cleaned_msg = {"role": message.get("role", "user"), "content": content}

            # Only add tool_calls if DeepSeek supports them in the expected format
            if message.get("tool_calls") and isinstance(
                message.get("tool_calls"), list
            ):
                # Convert to DeepSeek format if needed
                tool_calls = []
                for tc in message["tool_calls"]:
                    if isinstance(tc, dict) and "function" in tc:
                        # Ensure function.arguments is a JSON string, not a dict
                        cleaned_tc = tc.copy()
                        # Ensure type field is present for DeepSeek compatibility
                        if "type" not in cleaned_tc:
                            cleaned_tc["type"] = "function"
                        if (
                            "function" in cleaned_tc
                            and "arguments" in cleaned_tc["function"]
                        ):
                            args = cleaned_tc["function"]["arguments"]
                            if isinstance(args, dict):
                                cleaned_tc["function"]["arguments"] = json.dumps(args)
                            elif not isinstance(args, str):
                                cleaned_tc["function"]["arguments"] = str(args)
                        tool_calls.append(cleaned_tc)
                    elif isinstance(tc, dict) and "name" in tc:
                        # Convert from simplified format
                        tool_calls.append(
                            {
                                "id": tc.get("id", f"call_{len(tool_calls)}"),
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": (
                                        json.dumps(tc.get("args", {}))
                                        if isinstance(tc.get("args"), dict)
                                        else tc.get("args", "{}")
                                    ),
                                },
                            }
                        )

                if tool_calls:
                    cleaned_msg["tool_calls"] = tool_calls

            # Handle tool results/responses
            if message.get("tool_call_id"):
                cleaned_msg["tool_call_id"] = message["tool_call_id"]

            cleaned_messages.append(cleaned_msg)

        logger.debug(f"Cleaned messages result: {len(cleaned_messages)} messages")
        for i, msg in enumerate(cleaned_messages):
            logger.debug(
                f"Message {i}: role={msg.get('role')}, content_len={len(str(msg.get('content', '')))}, has_tool_calls={bool(msg.get('tool_calls'))}"
            )

        return cleaned_messages

    def _enhance_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """DeepSeek-specific message enhancement for deepseek-reasoner."""
        return self._format_messages_for_reasoner(messages)

    def _is_provider_retryable_error(self, error_str: str) -> bool:
        """DeepSeek-specific retryable error detection."""
        # DeepSeek-specific retryable conditions
        return (
            "deepseek" in error_str
            or "model overloaded" in error_str
        )

    def _get_current_runtime_model(self) -> str:
        """Get the actual DeepSeek model being used at runtime."""
        return self.deepseek_config.model

    # _normalize_tool_calls_to_standard_format is now implemented in BaseLLMProvider

    # ============================================================================
    # PROVIDER-SPECIFIC ADAPTER METHODS
    # ============================================================================

    def _extract_response_content(
        self, response: Any
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from DeepSeek response."""
        if not hasattr(response, "choices") or not response.choices:
            logger.info("Response has no choices attribute or empty choices")
            return "", [], {}

        message = response.choices[0].message
        text_content = message.content or ""
        tool_calls = message.tool_calls if hasattr(message, "tool_calls") else []

        logger.info(f"Extracted text_content: {repr(text_content)}")
        logger.info(f"Extracted tool_calls: {len(tool_calls) if tool_calls else 0}")

        # Handle DeepSeek-specific reasoning content
        provider_data = {}
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            provider_data["reasoning_content"] = message.reasoning_content

        return text_content, tool_calls, provider_data

    async def _process_streaming_chunks(
        self, response
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Process DeepSeek streaming chunks."""
        accumulated_content = ""
        accumulated_reasoning_content = ""
        accumulated_tool_calls = []

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta

                # Handle reasoning content (deepseek-reasoner)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    accumulated_reasoning_content += delta.reasoning_content

                # Handle regular content
                if delta.content:
                    accumulated_content += delta.content

                # Handle tool calls in streaming
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if tool_call_delta.index is not None:
                            # Ensure we have enough space in our list
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls.append(
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": None, "arguments": ""},
                                    }
                                )

                            current_tool_call = accumulated_tool_calls[
                                tool_call_delta.index
                            ]

                            # Update tool call data
                            if tool_call_delta.id:
                                current_tool_call["id"] = tool_call_delta.id
                            if tool_call_delta.type:
                                current_tool_call["type"] = tool_call_delta.type
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    current_tool_call["function"][
                                        "name"
                                    ] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    current_tool_call["function"][
                                        "arguments"
                                    ] += tool_call_delta.function.arguments

        # Filter out incomplete tool calls
        complete_tool_calls = [
            tc for tc in accumulated_tool_calls if tc["function"]["name"] is not None
        ]

        # Convert to SimpleNamespace format for compatibility
        from types import SimpleNamespace

        tool_calls_objs = []
        for tc in complete_tool_calls:
            tool_call = SimpleNamespace()
            tool_call.id = tc["id"] or f"stream_call_{len(tool_calls_objs)}"
            tool_call.type = tc["type"]
            tool_call.function = SimpleNamespace()
            tool_call.function.name = tc["function"]["name"]
            tool_call.function.arguments = tc["function"]["arguments"]
            tool_calls_objs.append(tool_call)

        provider_data = {}
        if accumulated_reasoning_content:
            provider_data["reasoning_content"] = accumulated_reasoning_content

        return accumulated_content, tool_calls_objs, provider_data

    async def _make_api_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
        stream: bool = True,
    ) -> Any:
        """Make DeepSeek API request using centralized retry logic."""
        # Use centralized retry logic
        return await self._make_api_request_with_retry(
            lambda: self.deepseek_client.chat.completions.create(
                model=self.deepseek_config.model,
                messages=messages,  # Messages already processed by centralized pipeline
                tools=tools,
                temperature=self.deepseek_config.temperature,
                max_tokens=self.deepseek_config.max_tokens,
                stream=stream,
            )
        )

    def _create_mock_response(self, content: str, tool_calls: List[Any]) -> Any:
        """Create mock DeepSeek response for centralized processing."""
        mock_response = type("MockResponse", (), {})()
        mock_response.choices = [type("MockChoice", (), {})()]
        mock_response.choices[0].message = type("MockMessage", (), {})()
        mock_response.choices[0].message.content = content
        mock_response.choices[0].message.tool_calls = tool_calls
        return mock_response

    def _handle_provider_specific_features(self, provider_data: Dict[str, Any]) -> None:
        """Handle DeepSeek-specific features like reasoning content."""
        if "reasoning_content" in provider_data:
            reasoning_content = provider_data["reasoning_content"]
            # Store reasoning content for next message if using reasoner
            if self.deepseek_config.model == "deepseek-reasoner":
                self.last_reasoning_content = reasoning_content
            print(f"\n<reasoning>{reasoning_content}</reasoning>", flush=True)

    def _format_provider_specific_content(self, provider_data: Dict[str, Any]) -> str:
        """Format DeepSeek-specific content for output."""
        if "reasoning_content" in provider_data:
            return f"<reasoning>{provider_data['reasoning_content']}</reasoning>\n\n"
        return ""

    async def _handle_complete_response(
        self,
        response,
        original_messages: List[Dict[str, str]],
        interactive: bool = True,
    ) -> Union[str, Any]:
        """Handle non-streaming response - delegate to centralized handler."""
        return await self._handle_complete_response_generic(
            response, original_messages, interactive
        )

    def _handle_streaming_response(
        self,
        response,
        original_messages: List[Dict[str, str]] = None,
        interactive: bool = True,
    ):
        """Handle streaming response - delegate to centralized handler."""
        return self._handle_streaming_response_generic(
            response, original_messages, interactive
        )

    # _handle_buffered_streaming_response is now implemented in BaseLLMProvider
