"""Unified MCP Host that combines Provider with Model."""

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
                logger.debug(f"Unknown tool call format: {type(tc)}, attributes: {dir(tc)}")
                logger.debug(f"Tool call repr: {repr(tc)}")
                call.name = getattr(tc, "name", "unknown")
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
        if hasattr(self, 'event_bus') and self.event_bus and hasattr(self, 'event_emitter'):
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
        from cli_agent.core.event_system import TextEvent, ToolCallEvent, StatusEvent
        
        accumulated_content = ""
        accumulated_reasoning_content = ""
        accumulated_tool_calls = []
        metadata = {}

        logger.debug("Starting comprehensive event-driven streaming response processing")

        # Process chunks and accumulate content for proper markdown formatting
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta

                # Handle reasoning content (deepseek-reasoner)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    accumulated_reasoning_content += delta.reasoning_content

                # Handle regular content - accumulate for final buffered emission
                if hasattr(delta, 'content') and delta.content:
                    accumulated_content += delta.content

                # Handle tool calls in streaming with events
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if hasattr(tool_call_delta, 'index') and tool_call_delta.index is not None:
                            # Ensure we have enough space in our list
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls.append({
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""}
                                })

                            # Update the tool call at this index
                            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                accumulated_tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id

                            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                func = tool_call_delta.function
                                if hasattr(func, 'name') and func.name:
                                    accumulated_tool_calls[tool_call_delta.index]["function"]["name"] = func.name
                                    # Emit tool call discovered event
                                    await self.event_emitter.emit_status(
                                        f"Tool call detected: {func.name}", 
                                        level="info"
                                    )
                                if hasattr(func, 'arguments') and func.arguments:
                                    accumulated_tool_calls[tool_call_delta.index]["function"]["arguments"] += func.arguments

        # Content accumulation complete - emit buffered response while event bus is still active
        if accumulated_content:
            logger.debug(f"Emitting buffered response during streaming: {len(accumulated_content)} characters")
            await self.event_emitter.emit_text(
                content=accumulated_content,
                is_streaming=False,
                is_markdown=True
            )
            # Add newline after LLM response
            await self.event_emitter.emit_text(
                content="\n",
                is_streaming=False,
                is_markdown=False
            )

        # Content accumulation complete - ready for final formatting

        # Handle accumulated reasoning content
        if accumulated_reasoning_content:
            metadata["reasoning_content"] = accumulated_reasoning_content
            
        # Parse model-specific content  
        special_content = self.model.parse_special_content(accumulated_content)
        metadata.update(special_content)

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
                    arguments=arguments
                )

        # Response processing complete - no status message needed
        
        logger.debug(f"Streaming processing complete. Content: {len(accumulated_content)} chars, Tool calls: {len(accumulated_tool_calls)}")

        return accumulated_content, accumulated_tool_calls, metadata

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
                tool_block = type(
                    "ToolBlock",
                    (),
                    {
                        "type": "tool_use",
                        "id": getattr(tc, "id", "mock_id"),
                        "name": getattr(tc, "name", "mock_tool"),
                        "input": getattr(tc, "args", {}),
                    },
                )()
                mock_response.content.append(tool_block)
        else:  # OpenAI-compatible format
            mock_response.choices = [type("MockChoice", (), {})()]
            mock_response.choices[0].message = type("MockMessage", (), {})()
            mock_response.choices[0].message.content = content
            mock_response.choices[0].message.tool_calls = tool_calls

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
        # Handle reasoning content from DeepSeek
        if "reasoning_content" in provider_data:
            reasoning_content = provider_data["reasoning_content"]
            if hasattr(self, 'event_emitter') and self.event_emitter:
                import asyncio
                asyncio.create_task(self.event_emitter.emit_text(
                    f"\n<reasoning>{reasoning_content}</reasoning>",
                    is_markdown=False
                ))

        # Handle thinking content from Claude
        if "thinking" in provider_data:
            thinking_content = provider_data["thinking"]
            if hasattr(self, 'event_emitter') and self.event_emitter:
                import asyncio
                asyncio.create_task(self.event_emitter.emit_text(
                    f"\n<thinking>{thinking_content}</thinking>",
                    is_markdown=False
                ))

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
        """Model-specific message enhancement."""
        enhanced_messages = messages.copy()

        # Enhance first message with AGENT.md content if available
        is_first_message = len(messages) == 1 and messages[0].get("role") == "user"
        if is_first_message and not self.is_subagent:
            enhanced_messages = (
                self.system_prompt_builder.enhance_first_message_with_agent_md(
                    enhanced_messages
                )
            )

        # Add system prompt based on model's style
        system_style = self.model.get_system_prompt_style()

        if self.is_subagent or is_first_message:
            system_prompt = self._create_system_prompt(for_first_message=True)

            if system_style == "message":
                # Add as system message
                enhanced_messages = [
                    {"role": "system", "content": system_prompt}
                ] + messages
            elif system_style == "parameter":
                # Add as system message for provider to extract and use as system parameter
                enhanced_messages = [
                    {"role": "system", "content": system_prompt}
                ] + messages
            elif system_style == "prepend":
                # Prepend to first user message (for models that don't support system messages)
                if enhanced_messages and enhanced_messages[0].get("role") == "user":
                    user_content = enhanced_messages[0]["content"]
                    enhanced_messages[0][
                        "content"
                    ] = f"{system_prompt}\n\n---\n\nUser: {user_content}"
            elif system_style == "none":
                # Skip system prompt entirely (e.g., for o1 models that don't work well with instructions)
                pass

        return enhanced_messages

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
