"""Unified MCP Host that combines Provider with Model."""

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from cli_agent.core.base_llm_provider import BaseLLMProvider
from cli_agent.core.base_provider import BaseProvider
from cli_agent.core.model_config import ModelConfig
from cli_agent.utils.tool_conversion import (
    AnthropicToolConverter,
    GeminiToolConverter,
    OpenAIStyleToolConverter,
)
from config import HostConfig

logger = logging.getLogger(__name__)


class MCPHost(BaseLLMProvider):
    """Unified MCP Host that combines a Provider with a Model.

    This class implements the new provider-model architecture where:
    - Provider handles API integration (Anthropic, OpenRouter, OpenAI, etc.)
    - Model handles LLM characteristics (Claude, GPT, Gemini, etc.)
    - MCPHost orchestrates them together while inheriting from BaseLLMProvider
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: ModelConfig,
        config: HostConfig,
        is_subagent: bool = False,
    ):
        """Initialize MCPHost with provider and model.

        Args:
            provider: API provider (AnthropicProvider, OpenRouterProvider, etc.)
            model: Model configuration (ClaudeModel, GPTModel, etc.)
            config: Host configuration
            is_subagent: Whether this is a subagent instance
        """

        self.provider = provider
        self.model = model

        logger.info(f"Initializing MCPHost: {provider.name} + {model}")

        # Call parent initialization (which will call our abstract methods)
        super().__init__(config, is_subagent)

    # ============================================================================
    # REQUIRED BASELLMPROVIDER METHODS - Delegate to Provider/Model
    # ============================================================================

    def convert_tools_to_llm_format(self) -> List[Any]:
        """Convert tools to the model's expected format."""
        tool_format = self.model.get_tool_format()

        if tool_format == "openai":
            converter = OpenAIStyleToolConverter()
        elif tool_format == "anthropic":
            converter = AnthropicToolConverter()
        elif tool_format == "gemini":
            converter = GeminiToolConverter()
        else:
            raise ValueError(f"Unknown tool format: {tool_format}")

        return converter.convert_tools(self.available_tools)

    def _extract_structured_calls_impl(self, response: Any) -> List[Any]:
        """Extract structured tool calls from provider's response format."""
        # Delegate to provider to extract tool calls
        _, tool_calls, _ = self.provider.extract_response_content(response)

        # Convert to standard SimpleNamespace format
        calls = []
        for tc in tool_calls:
            call = SimpleNamespace()

            if hasattr(tc, "function"):  # OpenAI format
                call.name = tc.function.name
                try:
                    call.args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse tool call arguments: {tc.function.arguments}"
                    )
                    call.args = {}
                call.id = getattr(tc, "id", None)

            elif hasattr(tc, "input"):  # Anthropic format
                call.name = tc.name
                call.args = tc.input
                call.id = getattr(tc, "id", None)

            elif hasattr(tc, "args"):  # Gemini format
                call.name = tc.name
                call.args = tc.args
                call.id = getattr(tc, "id", None)

            else:  # Generic format
                # Debug the tool call structure
                logger.debug(
                    f"Unknown tool call format: {type(tc)}, attributes: {dir(tc)}"
                )
                logger.debug(f"Tool call repr: {repr(tc)}")
                call.name = getattr(tc, "name", f"<missing_tc_name_{id(tc)}>")
                call.args = getattr(tc, "args", {})
                call.id = getattr(tc, "id", None)

            calls.append(call)

        logger.debug(f"Extracted {len(calls)} structured tool calls")
        return calls

    def _parse_text_based_calls_impl(self, text_content: str) -> List[Any]:
        """Parse text-based tool calls using model-specific patterns."""
        # For now, return empty list as most providers handle structured calls
        # This could be extended to parse text-based calls for specific models
        return []

    def _get_text_extraction_patterns(self) -> List[str]:
        """Get regex patterns for extracting text before tool calls."""
        # Model-specific patterns could be added here
        return [
            r"^(.*?)(?=<tool_call>)",  # XML-style
            r"^(.*?)(?=```json\s*\{)",  # JSON code blocks
            r"^(.*?)(?=\w+\s*\()",  # Function call style
        ]

    def _is_provider_retryable_error(self, error_str: str) -> bool:
        """Check if error is retryable according to provider-specific rules."""
        # Delegate to provider
        # Note: provider.is_retryable_error expects Exception, but we have str
        # For now, do basic string matching
        return (
            "rate limit" in error_str
            or "overloaded" in error_str
            or "timeout" in error_str
            or "5xx" in error_str
        )

    def _extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract text content, tool calls, and provider-specific data from response."""
        # Delegate to provider
        text, tool_calls, metadata = self.provider.extract_response_content(response)

        # Parse model-specific content
        special_content = self.model.parse_special_content(text)
        metadata.update(special_content)

        return text, tool_calls, metadata

    async def _process_streaming_chunks(
        self, response
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process provider's streaming response chunks."""
        # Always use event-driven streaming when event system is available
        if (
            hasattr(self, "event_bus")
            and self.event_bus
            and hasattr(self, "event_emitter")
        ):
            return await self._process_streaming_chunks_with_events(response)
        else:
            # No event system - delegate to provider for basic processing
            return await self.provider.process_streaming_response(response)


    async def _process_streaming_chunks_with_events(
        self, response
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """
        Process streaming response chunks while emitting events for complete responses.

        This is the single source of truth for all response display via the event system.
        Buffers full response for proper markdown formatting, handles tool execution, and status updates.
        """
        from cli_agent.core.event_system import StatusEvent, TextEvent, ToolCallEvent

        accumulated_content = ""
        accumulated_reasoning_content = ""
        accumulated_tool_calls = []
        metadata = {}
        content_emitted = False  # Track if we've already emitted the text content
        original_content = ""  # Keep original content for conversation history
        is_thinking_content = False  # Track if reasoning content came from thinking tags

        logger.debug(
            "Starting comprehensive event-driven streaming response processing"
        )

        # Process chunks and accumulate content for proper markdown formatting
        # Handle both async and sync iterators (Gemini returns sync generator)
        if hasattr(response, "__aiter__"):
            # Async iterator (OpenAI, Anthropic, etc.)
            # Wrap with interrupt checking
            from cli_agent.core.interrupt_aware_streaming import make_interruptible

            chunks_iter = make_interruptible(
                response, f"{self.provider.name} event streaming"
            )
        elif hasattr(response, "__iter__"):
            # Sync iterator (Gemini) - convert to async with yield points for event processing
            # and wrap with interrupt checking
            from cli_agent.core.interrupt_aware_streaming import make_sync_interruptible

            async def async_generator():
                # Wrap sync iterator with interrupt checking
                interruptible_iter = make_sync_interruptible(
                    response, f"{self.provider.name} event streaming"
                )
                for chunk in interruptible_iter:
                    yield chunk
                    # Yield control to allow immediate event processing
                    await asyncio.sleep(0)

            chunks_iter = async_generator()
        else:
            # Response is not iterable - fall back to provider's process_streaming_response
            logger.warning(
                f"Response object {type(response)} is not iterable, falling back to provider processing"
            )
            return await self.provider.process_streaming_response(response)

        async for chunk in chunks_iter:
            # Handle different provider chunk formats
            if hasattr(chunk, "choices") and chunk.choices:
                # OpenAI/Anthropic format
                delta = chunk.choices[0].delta

                # Handle reasoning content (deepseek-reasoner)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    accumulated_reasoning_content += delta.reasoning_content

                # Handle regular content - accumulate for final buffered emission
                if hasattr(delta, "content") and delta.content:
                    # Check if this chunk contains thinking content for qwen models and extract it
                    if not accumulated_reasoning_content:
                        chunk_special_content = self.model.parse_special_content(delta.content)
                        if "thinking_content" in chunk_special_content:
                            # Extract thinking content and treat it like reasoning content
                            accumulated_reasoning_content = chunk_special_content["thinking_content"]
                            is_thinking_content = True  # Mark this as thinking content for proper tag formatting
                            # Remove thinking tags from the content stream
                            import re
                            thinking_pattern = r'<think>(.*?)</think>'
                            cleaned_content = re.sub(thinking_pattern, '', delta.content, flags=re.DOTALL).strip()
                            if cleaned_content:  # Only add if there's content left after removing thinking tags
                                accumulated_content += cleaned_content
                            original_content += delta.content  # Keep original with tags for conversation history
                        else:
                            accumulated_content += delta.content
                            original_content += delta.content  # Keep for conversation history
                    else:
                        accumulated_content += delta.content
                        original_content += delta.content  # Keep for conversation history

                # Handle tool calls in streaming with events
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    # Apply model-specific tool call formatting if needed
                    tool_calls_to_process = delta.tool_calls
                    if hasattr(self.model, 'format_streaming_tool_calls'):
                        formatted_calls = self.model.format_streaming_tool_calls(delta.tool_calls)
                        if formatted_calls is not None:
                            tool_calls_to_process = formatted_calls
                    
                    # First, detect if we have any new tool calls and emit buffered content
                    for tool_call_delta in tool_calls_to_process:
                        if (
                            hasattr(tool_call_delta, "function")
                            and tool_call_delta.function
                            and hasattr(tool_call_delta.function, "name")
                            and tool_call_delta.function.name
                            and not content_emitted
                            and accumulated_content
                        ):
                            # Tool call detected! Check for thinking content and emit accumulated content immediately
                            content_to_emit = accumulated_content
                            
                            # Prepend reasoning/thinking content if present
                            if accumulated_reasoning_content:
                                if is_thinking_content:
                                    reasoning_with_tags = f"<think>\n{accumulated_reasoning_content}\n</think>\n\n"
                                else:
                                    reasoning_with_tags = f"<reasoning>\n{accumulated_reasoning_content}\n</reasoning>\n\n"
                                content_to_emit = reasoning_with_tags + content_to_emit

                            await self.event_emitter.emit_text(
                                content=content_to_emit + "\n",
                                is_streaming=False,
                                is_markdown=True,
                            )
                            content_emitted = True
                            # Clear accumulated content since it was already emitted
                            accumulated_content = ""
                            accumulated_reasoning_content = ""
                            logger.debug(
                                f"Emitted accumulated content before tool call processing: {len(content_to_emit)} characters"
                            )
                            break  # Only emit once

                    # Now process the tool calls
                    for tool_call_delta in tool_calls_to_process:
                        if (
                            hasattr(tool_call_delta, "index")
                            and tool_call_delta.index is not None
                        ):
                            # Ensure we have enough space in our list
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls.append(
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": None, "arguments": ""},
                                    }
                                )

                            # Update the tool call at this index
                            if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                                accumulated_tool_calls[tool_call_delta.index][
                                    "id"
                                ] = tool_call_delta.id

                            if (
                                hasattr(tool_call_delta, "function")
                                and tool_call_delta.function
                            ):
                                func = tool_call_delta.function
                                if hasattr(func, "name") and func.name:
                                    # Only emit detection event if this is the first time we see this tool name
                                    current_name = accumulated_tool_calls[tool_call_delta.index]["function"]["name"]
                                    if current_name is None:
                                        # This is the first time we're seeing this tool call
                                        # Reduce verbosity for common tools
                                        common_tools = {"builtin_todo_read", "builtin_todo_write", "todo_read", "todo_write"}
                                        if func.name not in common_tools:
                                            await self.event_emitter.emit_status(
                                                f"Tool call detected: {func.name}", level="info"
                                            )
                                    
                                    accumulated_tool_calls[tool_call_delta.index][
                                        "function"
                                    ]["name"] = func.name
                                if hasattr(func, "arguments") and func.arguments:
                                    accumulated_tool_calls[tool_call_delta.index][
                                        "function"
                                    ]["arguments"] += func.arguments

            elif hasattr(chunk, "candidates") and chunk.candidates:
                # Gemini format - only process through candidates to avoid .text warnings
                logger.debug(
                    f"Gemini chunk type: {type(chunk)}, candidates: {len(chunk.candidates)}"
                )
                logger.debug(f"Gemini chunk repr: {repr(chunk)}")
                chunk_text = ""
                candidate = chunk.candidates[0] if chunk.candidates else None
                if candidate and hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        logger.debug(
                            f"Gemini chunk has {len(candidate.content.parts)} parts"
                        )
                        for i, part in enumerate(candidate.content.parts):
                            if hasattr(part, "function_call") and part.function_call:
                                logger.debug(
                                    f"Gemini part {i}: function call '{part.function_call.name}'"
                                )
                                fc = part.function_call

                                # Emit accumulated content before first tool call if not already emitted
                                if not content_emitted and accumulated_content:
                                    # Check for thinking content and prepend
                                    content_to_emit = accumulated_content
                                    
                                    # Prepend reasoning/thinking content if present
                                    if accumulated_reasoning_content:
                                        if is_thinking_content:
                                            reasoning_with_tags = f"<think>\n{accumulated_reasoning_content}\n</think>\n\n"
                                        else:
                                            reasoning_with_tags = f"<reasoning>\n{accumulated_reasoning_content}\n</reasoning>\n\n"
                                        content_to_emit = reasoning_with_tags + content_to_emit

                                    await self.event_emitter.emit_text(
                                        content=content_to_emit + "\n",
                                        is_streaming=False,
                                        is_markdown=True,
                                    )
                                    content_emitted = True
                                    # Clear accumulated content since it was already emitted
                                    accumulated_content = ""
                                    accumulated_reasoning_content = ""
                                    logger.debug(
                                        f"Emitted accumulated content before Gemini tool call: {len(content_to_emit)} characters"
                                    )

                                # Convert Gemini function call to standard format
                                tool_call = {
                                    "id": getattr(fc, "id", None)
                                    or f"call_{len(accumulated_tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": fc.name,
                                        "arguments": json.dumps(
                                            dict(fc.args) if fc.args else {}
                                        ),
                                    },
                                }
                                accumulated_tool_calls.append(tool_call)

                                # Emit tool call discovered event (reduce verbosity for common tools)
                                common_tools = {"builtin_todo_read", "builtin_todo_write", "todo_read", "todo_write"}
                                if fc.name not in common_tools:
                                    await self.event_emitter.emit_status(
                                        f"Tool call detected: {fc.name}", level="info"
                                    )
                            elif hasattr(part, "text") and part.text:
                                logger.debug(
                                    f"Gemini part {i}: text content '{part.text[:100]}...' (len={len(part.text)})"
                                )
                                chunk_text += part.text
                            else:
                                logger.debug(
                                    f"Gemini part {i}: unknown type {type(part)} with attrs {dir(part)}"
                                )
                                # Try to inspect the part more deeply
                                if hasattr(part, "__dict__"):
                                    logger.debug(
                                        f"Gemini part {i} dict: {part.__dict__}"
                                    )
                                logger.debug(f"Gemini part {i} repr: {repr(part)}")
                        logger.debug(
                            f"Gemini chunk total text accumulated: '{chunk_text[:100]}...' (len={len(chunk_text)})"
                        )
                        # Check if chunk text contains thinking content and extract it
                        if chunk_text and not accumulated_reasoning_content:
                            chunk_special_content = self.model.parse_special_content(chunk_text)
                            if "thinking_content" in chunk_special_content:
                                # Extract thinking content and treat it like reasoning content
                                accumulated_reasoning_content = chunk_special_content["thinking_content"]
                                is_thinking_content = True  # Mark this as thinking content for proper tag formatting
                                # Remove thinking tags from the content stream
                                import re
                                thinking_pattern = r'<think>(.*?)</think>'
                                cleaned_chunk_text = re.sub(thinking_pattern, '', chunk_text, flags=re.DOTALL).strip()
                                if cleaned_chunk_text:  # Only add if there's content left after removing thinking tags
                                    accumulated_content += cleaned_chunk_text
                            else:
                                accumulated_content += chunk_text
                        else:
                            accumulated_content += chunk_text
                        
                        original_content += chunk_text  # Keep original for conversation history
                    else:
                        logger.debug("Gemini chunk candidate has no parts")
                else:
                    logger.debug("Gemini chunk has no candidate content")

                # Don't emit text immediately - buffer for proper markdown formatting

        # Only emit content if it wasn't already emitted during tool call processing
        if not content_emitted:
            # Prepend reasoning/thinking content with appropriate tags
            if accumulated_reasoning_content:
                if is_thinking_content:
                    reasoning_with_tags = f"<think>\n{accumulated_reasoning_content}\n</think>\n\n"
                    logger.debug(f"Prepended thinking content with tags to main content")
                else:
                    reasoning_with_tags = f"<reasoning>\n{accumulated_reasoning_content}\n</reasoning>\n\n"
                    logger.debug(f"Prepended reasoning content with tags to main content")
                accumulated_content = reasoning_with_tags + accumulated_content

            # Emit accumulated content as properly formatted markdown
            logger.debug(
                f"Final accumulated_content length: {len(accumulated_content)}, tool_calls: {len(accumulated_tool_calls)}"
            )
            if accumulated_content:
                logger.debug(
                    f"Emitting buffered content: {len(accumulated_content)} characters - '{accumulated_content[:200]}...'"
                )
                # Emit the complete response with markdown formatting and newline
                await self.event_emitter.emit_text(
                    content=accumulated_content + "\n", is_streaming=False, is_markdown=True
                )
            else:
                logger.debug(
                    "No accumulated content to emit, but streaming processing completed"
                )
        else:
            logger.debug(
                "Content was already emitted during tool call processing - skipping final emission"
            )

        # Content accumulation complete - ready for final formatting

        # Handle accumulated reasoning content
        if accumulated_reasoning_content:
            if is_thinking_content:
                metadata["thinking_content"] = accumulated_reasoning_content
            else:
                metadata["reasoning_content"] = accumulated_reasoning_content

        # Response already emitted during streaming loop above

        # Emit tool call events for any complete tool calls
        for tool_call in accumulated_tool_calls:
            if tool_call.get("id") and tool_call.get("function", {}).get("name"):
                try:
                    arguments = json.loads(tool_call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": tool_call["function"]["arguments"]}

                await self.event_emitter.emit_tool_call(
                    tool_name=tool_call["function"]["name"],
                    tool_id=tool_call["id"],
                    arguments=arguments,
                )

        # Response processing complete - no status message needed

        logger.debug(
            f"Streaming processing complete. Content: {len(accumulated_content)} chars, Tool calls: {len(accumulated_tool_calls)}"
        )

        # Return original content for conversation history, not the cleared accumulated_content
        final_content = original_content
        if accumulated_reasoning_content and not content_emitted:
            # Only add tags if content wasn't already emitted with them
            if is_thinking_content:
                final_content = f"<think>\n{accumulated_reasoning_content}\n</think>\n\n{final_content}"
            else:
                final_content = f"<reasoning>\n{accumulated_reasoning_content}\n</reasoning>\n\n{final_content}"

        return final_content, accumulated_tool_calls, metadata

    async def _make_api_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
        stream: bool = True,
    ) -> Any:
        """Make an API request to the provider."""
        # First enhance messages with system prompts and model-specific formatting
        enhanced_messages = self._enhance_messages_for_model(messages)

        # Then format messages for the specific model
        formatted_messages = self.model.format_messages_for_model(enhanced_messages)

        # Get model parameters and validate them
        model_params = self.model.get_default_parameters()
        model_params = self.model.validate_parameters(**model_params)

        # Make request through provider
        return await self.provider.make_request(
            messages=formatted_messages,
            model_name=self.model.provider_model_name,
            tools=tools,
            stream=stream,
            **model_params,
        )

    def _create_mock_response(self, content: str, tool_calls: List[Any]) -> Any:
        """Create a mock response object for centralized processing."""
        # Create a generic mock response that works with the provider
        mock_response = type("MockResponse", (), {})()

        # Use provider's expected format
        if self.provider.name == "anthropic":
            mock_response.content = []
            if content:
                text_block = type("TextBlock", (), {"type": "text", "text": content})()
                mock_response.content.append(text_block)
            for tc in tool_calls:
                # Handle different tool call formats
                if isinstance(tc, dict):
                    tool_id = tc.get("id", "mock_id")
                    tool_name = tc.get("function", {}).get("name", "mock_tool")
                    tool_input = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(tool_input, str):
                        import json

                        try:
                            tool_input = json.loads(tool_input)
                        except:
                            tool_input = {}
                else:
                    tool_id = getattr(tc, "id", "mock_id")
                    tool_name = getattr(tc, "name", "mock_tool")
                    tool_input = getattr(tc, "args", {})

                tool_block = type(
                    "ToolBlock",
                    (),
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input,
                    },
                )()
                mock_response.content.append(tool_block)
        else:  # OpenAI-compatible format
            mock_response.choices = [type("MockChoice", (), {})()]
            mock_response.choices[0].message = type("MockMessage", (), {})()
            mock_response.choices[0].message.content = content

            # Convert tool calls to proper object format for parsing
            mock_tool_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    # Convert dict to object format expected by parsing
                    mock_tc = type("MockToolCall", (), {})()
                    mock_tc.id = tc.get("id")
                    mock_tc.type = tc.get("type", "function")

                    # Create function object
                    mock_tc.function = type("MockFunction", (), {})()
                    mock_tc.function.name = tc.get("function", {}).get("name")
                    mock_tc.function.arguments = tc.get("function", {}).get(
                        "arguments", "{}"
                    )

                    mock_tool_calls.append(mock_tc)
                else:
                    mock_tool_calls.append(tc)

            mock_response.choices[0].message.tool_calls = mock_tool_calls

        return mock_response

    # ============================================================================
    # CONFIGURATION METHODS - From Original Architecture
    # ============================================================================

    def _get_provider_config(self):
        """Get provider-specific configuration."""
        # This method is from the original architecture
        # For now, return the model config
        return self.model

    def _get_streaming_preference(self, provider_config) -> bool:
        """Get streaming preference."""
        return self.provider.supports_streaming() and self.model.supports_streaming

    def _calculate_timeout(self, provider_config) -> float:
        """Calculate timeout based on provider and model characteristics."""
        # Different providers/models may need different timeouts
        base_timeout = 120.0

        if self.model.model_family == "deepseek":
            base_timeout = 600.0  # DeepSeek can be slower
        elif self.provider.name == "anthropic":
            base_timeout = 180.0  # Anthropic can be slower for large contexts

        return base_timeout

    def _create_llm_client(self, provider_config, timeout_seconds):
        """Create LLM client - already handled by provider."""
        # The provider already has its client created
        return self.provider.client

    def _get_current_runtime_model(self) -> str:
        """Get the actual model being used at runtime."""
        return f"{self.provider.name}:{self.model.name}"

    # ============================================================================
    # MODEL-SPECIFIC FEATURES
    # ============================================================================

    def _handle_provider_specific_features(self, provider_data: Dict[str, Any]) -> None:
        """Handle provider-specific features like reasoning content."""
        # Reasoning content is now handled in the main streaming flow
        # so we don't need to emit it separately here

        # Handle thinking content from Claude
        if "thinking" in provider_data:
            thinking_content = provider_data["thinking"]
            if hasattr(self, "event_emitter") and self.event_emitter:
                import asyncio

                asyncio.create_task(
                    self.event_emitter.emit_text(
                        f"\n<thinking>{thinking_content}</thinking>", is_markdown=False
                    )
                )

    def _format_provider_specific_content(self, provider_data: Dict[str, Any]) -> str:
        """Format provider-specific content for output."""
        formatted_parts = []

        if "reasoning_content" in provider_data:
            formatted_parts.append(
                f"<reasoning>{provider_data['reasoning_content']}</reasoning>"
            )

        if "thinking" in provider_data:
            formatted_parts.append(f"<thinking>{provider_data['thinking']}</thinking>")

        if formatted_parts:
            return "\n".join(formatted_parts) + "\n\n"

        return ""

    def _get_llm_specific_instructions(self) -> str:
        """Get model-specific instructions."""
        return self.model.get_model_specific_instructions(self.is_subagent)

    def _enhance_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Model-specific message enhancement (system prompt style handling)."""
        # First apply generic LLM enhancement
        enhanced_messages = self._enhance_messages_generic(messages)

        # Then apply model-specific system prompt handling
        if self._should_add_system_prompt(messages):
            system_prompt = self._create_system_prompt(for_first_message=True)
            enhanced_messages = self._apply_model_specific_system_prompt(
                enhanced_messages, system_prompt
            )

        return enhanced_messages

    def _apply_model_specific_system_prompt(
        self, messages: List[Dict[str, Any]], system_prompt: str
    ) -> List[Dict[str, Any]]:
        """Apply system prompt based on model's style preference."""
        system_style = self.model.get_system_prompt_style()

        if system_style == "message":
            # Add as system message
            return [{"role": "system", "content": system_prompt}] + messages
        elif system_style == "parameter":
            # Add as system message for provider to extract and use as system parameter
            return [{"role": "system", "content": system_prompt}] + messages
        elif system_style == "prepend":
            # Prepend to first user message (for models that don't support system messages)
            if messages and messages[0].get("role") == "user":
                user_content = messages[0]["content"]
                enhanced_messages = messages.copy()
                enhanced_messages[0]["content"] = f"{system_prompt}\n\n---\n\n{user_content}"
                return enhanced_messages
            return messages
        elif system_style == "none":
            # Skip system prompt entirely (e.g., for o1 models that don't work well with instructions)
            return messages
        else:
            # Default: add as system message
            return [{"role": "system", "content": system_prompt}] + messages

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def get_token_limit(self) -> int:
        """Get effective token limit for conversations."""
        return self.model.get_token_limit()

    def __str__(self) -> str:
        return f"MCPHost({self.provider.name}:{self.model.name})"

    def __repr__(self) -> str:
        return f"MCPHost(provider={self.provider.__class__.__name__}, model={self.model.__class__.__name__})"
