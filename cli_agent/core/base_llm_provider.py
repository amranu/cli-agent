"""Base LLM Provider class that centralizes common functionality across all LLM implementations."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

logger = logging.getLogger(__name__)


class BaseLLMProvider(BaseMCPAgent):
    """Template base class for LLM providers with centralized streaming and tool logic.

    This class eliminates duplicate code between different LLM providers by implementing
    common functionality while requiring providers to only implement their specific
    API integration and response parsing logic.
    """

    # ============================================================================
    # CENTRALIZED COMPLETION GENERATION
    # ============================================================================

    async def _generate_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        interactive: bool = True,
    ) -> Any:
        """Generate completion using provider API - centralized implementation."""
        logger.info(
            f"_generate_completion called with {len(messages)} messages, tools: {tools is not None}, stream: {stream}, interactive: {interactive}"
        )

        # Check for global interrupt before making API request
        from cli_agent.core.global_interrupt import check_interrupt

        try:
            check_interrupt("LLM request interrupted")
        except KeyboardInterrupt:
            logger.info("LLM request interrupted by user")
            raise

        # Check if we should use buffering for streaming JSON
        use_buffering = (
            hasattr(self, "streaming_json_callback")
            and self.streaming_json_callback is not None
        )

        try:
            if stream:
                # Make API request with streaming and aggressive interrupt monitoring
                from cli_agent.core.interrupt_aware_streaming import (
                    run_with_interrupt_monitoring,
                )

                response = await run_with_interrupt_monitoring(
                    self._make_api_request(messages, tools, stream=True),
                    "LLM streaming API request",
                )

                # Check if provider actually disabled streaming (e.g., for o1 models)
                streaming_disabled = getattr(
                    response, "_actual_streaming_disabled", False
                )
                logger.info(
                    f"Response type: {type(response)}, streaming_disabled: {streaming_disabled}"
                )

                if streaming_disabled:
                    # Provider disabled streaming, handle as non-streaming response
                    logger.info(
                        f"Streaming was disabled by provider, handling as non-streaming response"
                    )
                    result = await self._handle_complete_response_generic(
                        response, messages, interactive=interactive
                    )
                    logger.info(f"Handled non-streaming response result: {result}")
                    return result
                else:
                    # Normal streaming response
                    if use_buffering:
                        # Use buffering for streaming JSON mode
                        return await self._handle_buffered_streaming_response(
                            response,
                            messages,
                            interactive=interactive,
                        )
                    else:
                        # Handle streaming response and return final content with interrupt monitoring
                        async def consume_streaming_response():
                            generator = self._handle_streaming_response_generic(
                                response,
                                messages,
                                interactive=interactive,
                            )

                            # Consume the generator and return the final content
                            final_result = None
                            logger.info(
                                f"Consuming streaming generator: {type(generator)}"
                            )

                            try:
                                async for content in generator:
                                    # Check for interrupts during streaming processing
                                    from cli_agent.core.global_interrupt import (
                                        get_global_interrupt_manager,
                                    )

                                    interrupt_manager = get_global_interrupt_manager()
                                    if interrupt_manager.is_interrupted():
                                        logger.info(
                                            "Streaming response interrupted by user"
                                        )
                                        # Always raise FirstInterruptException for graceful handling
                                        from cli_agent.core.interrupt_aware_streaming import FirstInterruptException
                                        raise FirstInterruptException("Streaming response interrupted - first interrupt")

                                    logger.info(
                                        f"Generator yielded: {type(content)} - {repr(content[:100] if isinstance(content, str) else content)}"
                                    )
                                    # Store the final result regardless of type
                                    final_result = content
                            except Exception as e:
                                logger.error(f"Error consuming generator: {e}")
                                final_result = ""

                            logger.info(
                                f"Final result collected: {type(final_result)} - {len(final_result) if isinstance(final_result, str) else final_result}"
                            )
                            return final_result

                        return await run_with_interrupt_monitoring(
                            consume_streaming_response(),
                            "LLM streaming response processing",
                        )
            else:
                # Non-streaming response with interrupt monitoring
                from cli_agent.core.interrupt_aware_streaming import (
                    run_with_interrupt_monitoring,
                )

                response = await run_with_interrupt_monitoring(
                    self._make_api_request(messages, tools, stream=False),
                    "LLM API request",
                )
                logger.info(f"Non-streaming response received: {type(response)}")
                result = await self._handle_complete_response_generic(
                    response, messages, interactive=interactive
                )
                logger.info(f"Handled response result: {result}")
                return result

        except Exception as e:
            # Re-raise tool permission denials so they can be handled at the chat level
            if isinstance(e, ToolDeniedReturnToPrompt):
                raise  # Re-raise the exception to bubble up to interactive chat

            logger.error(f"Error in generate completion: {e}")
            return f"Error: {str(e)}"

    async def _handle_complete_response_generic(
        self,
        response: Any,
        original_messages: List[Dict[str, str]] = None,
        interactive: bool = True,
    ) -> Any:
        """Handle complete response - centralized implementation."""
        return await self.response_handler.handle_complete_response_generic(
            response, original_messages, interactive
        )

    async def _handle_buffered_streaming_response(
        self,
        response,
        original_messages: List[Dict[str, str]] = None,
        interactive: bool = True,
    ) -> str:
        """Handle streaming response for JSON mode - centralized implementation."""
        # Process chunks to get content and tool calls
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

    # ============================================================================
    # CENTRALIZED TOOL CALL PARSING FRAMEWORK
    # ============================================================================

    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls using centralized framework with provider-specific extraction."""
        logger.debug(f"parse_tool_calls received response type: {type(response)}")
        logger.debug(f"parse_tool_calls response content: {response}")

        # Extract text content for text-based parsing
        text_content = ""
        if isinstance(response, str):
            text_content = response

        # Use centralized parsing framework
        return self._parse_tool_calls_generic(response, text_content)

    def _extract_structured_calls(self, response: Any) -> List[Any]:
        """Extract structured tool calls from provider response - delegates to implementation."""
        return self._extract_structured_calls_impl(response)

    def _parse_text_based_calls(self, text_content: str) -> List[Any]:
        """Parse text-based tool calls using provider-specific patterns - delegates to implementation."""
        return self._parse_text_based_calls_impl(text_content)

    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract text that appears before tool calls using provider patterns."""
        import re

        # Get provider-specific patterns
        patterns = self._get_text_extraction_patterns()

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

    # ============================================================================
    # CENTRALIZED TOOL CALL NORMALIZATION
    # ============================================================================

    def _normalize_tool_calls_to_standard_format(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert provider tool calls to standardized format - centralized implementation."""
        normalized_calls = []

        for i, tool_call in enumerate(tool_calls):
            if hasattr(tool_call, "get") and callable(getattr(tool_call, "get")):
                # Dict format (already structured)
                if "function" in tool_call:
                    # OpenAI-style format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call["function"].get(
                                "name", f"<missing_function_name_{i}>"
                            ),
                            "arguments": tool_call["function"].get("arguments", {}),
                        }
                    )
                else:
                    # Simple dict format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call.get("name", f"<missing_dict_name_{i}>"),
                            "arguments": tool_call.get("arguments", {}),
                        }
                    )
            elif hasattr(tool_call, "name"):
                # Object format (SimpleNamespace, Gemini function call, etc.)
                normalized_calls.append(
                    {
                        "id": getattr(tool_call, "id", f"call_{i}"),
                        "name": getattr(tool_call, "name", f"<missing_obj_name_{i}>"),
                        "arguments": getattr(
                            tool_call, "args", getattr(tool_call, "arguments", {})
                        ),
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
    # CENTRALIZED ERROR HANDLING
    # ============================================================================

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable - combines generic and provider-specific logic."""
        error_str = str(error).lower()

        # Generic retryable conditions from base class
        generic_retryable = (
            "timeout" in error_str
            or "network" in error_str
            or "connection" in error_str
            or "rate limit" in error_str
            or "429" in error_str
            or "502" in error_str
            or "503" in error_str
            or "504" in error_str
        )

        # Provider-specific retryable conditions
        provider_retryable = self._is_provider_retryable_error(error_str)

        return generic_retryable or provider_retryable

    # ============================================================================
    # ABSTRACT METHODS - MUST BE IMPLEMENTED BY PROVIDERS
    # ============================================================================

    @abstractmethod
    def _extract_structured_calls_impl(self, response: Any) -> List[Any]:
        """Extract structured tool calls from provider-specific response format.

        Args:
            response: Provider's raw response object

        Returns:
            List of tool call objects in provider's format
        """
        pass

    @abstractmethod
    def _parse_text_based_calls_impl(self, text_content: str) -> List[Any]:
        """Parse text-based tool calls using provider-specific text parsing.

        Args:
            text_content: Raw text content from response

        Returns:
            List of tool call objects parsed from text
        """
        pass

    @abstractmethod
    def _get_text_extraction_patterns(self) -> List[str]:
        """Get provider-specific regex patterns for extracting text before tool calls.

        Returns:
            List of regex patterns specific to the provider's tool call format
        """
        pass

    @abstractmethod
    def _is_provider_retryable_error(self, error_str: str) -> bool:
        """Check if error is retryable according to provider-specific rules.

        Args:
            error_str: Lowercase error message string

        Returns:
            True if error should be retried for this provider
        """
        pass

    @abstractmethod
    def _extract_response_content(
        self, response: Any
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Extract text content, tool calls, and provider-specific data from response.

        Args:
            response: Provider's raw response object

        Returns:
            tuple: (text_content, tool_calls, provider_specific_data)
        """
        pass

    @abstractmethod
    async def _process_streaming_chunks(
        self, response
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Process provider's streaming response chunks.

        Args:
            response: Provider's streaming response object

        Returns:
            tuple: (accumulated_content, tool_calls, provider_specific_data)
        """
        pass

    @abstractmethod
    async def _make_api_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List] = None,
        stream: bool = True,
    ) -> Any:
        """Make an API request to the provider.

        Args:
            messages: Processed messages ready for the provider
            tools: Tools formatted for the provider (if any)
            stream: Whether to use streaming

        Returns:
            Provider-specific response object
        """
        pass

    @abstractmethod
    def _create_mock_response(self, content: str, tool_calls: List[Any]) -> Any:
        """Create a mock response object for centralized processing.

        Args:
            content: Text content
            tool_calls: List of tool calls

        Returns:
            Mock response object compatible with provider's format
        """
        pass

    # ============================================================================
    # OPTIONAL OVERRIDE METHODS - HAVE DEFAULT IMPLEMENTATIONS
    # ============================================================================

    def _handle_provider_specific_features(self, provider_data: Dict[str, Any]) -> None:
        """Handle provider-specific features. Override in subclasses if needed.

        Args:
            provider_data: Provider-specific data from response extraction
        """
        pass  # Default: no special features

    def _format_provider_specific_content(self, provider_data: Dict[str, Any]) -> str:
        """Format provider-specific content for output. Override in subclasses if needed.

        Args:
            provider_data: Provider-specific data from response extraction

        Returns:
            Formatted content string
        """
        return ""  # Default: no special content formatting

    # ============================================================================
    # SYSTEM PROMPT MANAGEMENT - Moved from BaseMCPAgent
    # ============================================================================

    def _create_system_prompt(self, for_first_message: bool = False) -> str:
        """Create a centralized system prompt with LLM-specific customization points."""
        # Build the system prompt using centralized template system
        base_prompt = self.system_prompt_builder.build_base_system_prompt()

        # Add LLM-specific customizations
        llm_customizations = self._get_llm_specific_instructions()

        if llm_customizations:
            final_prompt = base_prompt + "\n\n" + llm_customizations
        else:
            final_prompt = base_prompt

        return final_prompt

    def _get_llm_specific_instructions(self) -> str:
        """Override in subclasses to add LLM-specific instructions.

        This is a hook for LLM implementations to add:
        - Model-specific behavior instructions
        - API usage guidelines
        - Tool execution requirements
        - Context management specifics
        """
        return ""

    # ============================================================================
    # TOKEN MANAGEMENT - Moved from BaseMCPAgent
    # ============================================================================

    def _update_token_manager_model(self):
        """Update token manager with current model name for accurate counting."""
        try:
            current_model = self._get_current_runtime_model()
            if current_model:
                self.token_manager.model_name = current_model
                logger.debug(f"Updated token manager model to: {current_model}")
        except Exception as e:
            logger.warning(f"Failed to update token manager model: {e}")

    def _estimate_token_count(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages using accurate tokenization."""
        self._update_token_manager_model()
        return self.token_manager.count_conversation_tokens(messages)

    async def compact_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compact conversation using the token manager's intelligent compaction."""
        self._update_token_manager_model()
        return await self.token_manager.compact_conversation(messages, self.generate_response)

    @abstractmethod
    def _get_current_runtime_model(self) -> str:
        """Get the actual model being used at runtime. Must be implemented by subclasses."""
        pass

    # ============================================================================
    # MESSAGE ENHANCEMENT - Generic LLM operations moved from MCPHost
    # ============================================================================

    def _enhance_messages_generic(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generic LLM message enhancement (AGENT.md, basic processing)."""
        enhanced_messages = messages.copy()

        # Enhance first message with AGENT.md content if available
        is_first_message = len(messages) == 1 and messages[0].get("role") == "user"
        if is_first_message and not self.is_subagent:
            enhanced_messages = (
                self.system_prompt_builder.enhance_first_message_with_agent_md(
                    enhanced_messages
                )
            )

        return enhanced_messages

    def _should_add_system_prompt(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if system prompt should be added based on LLM context."""
        is_first_message = len(messages) == 1 and messages[0].get("role") == "user"
        return self.is_subagent or is_first_message
