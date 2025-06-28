"""Base provider abstraction for API endpoints."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Base class for API providers (Anthropic, OpenAI, OpenRouter, etc.)

    Handles API-specific logic:
    - Authentication methods
    - Request/response formats
    - Streaming protocols
    - Error codes and retry logic
    - Rate limiting
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """Initialize provider with API credentials and settings.

        Args:
            api_key: API key for authentication
            base_url: Override default base URL (optional)
            **kwargs: Provider-specific configuration options
        """
        self.api_key = api_key
        self.base_url = base_url or self.get_default_base_url()
        self.client = self._create_client(**kwargs)
        logger.debug(
            f"Initialized {self.__class__.__name__} with base_url: {self.base_url}"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name for identification."""
        pass

    @abstractmethod
    def get_default_base_url(self) -> str:
        """Get default API base URL for this provider."""
        pass

    @abstractmethod
    def _create_client(self, **kwargs) -> Any:
        """Create provider-specific HTTP client.

        Args:
            **kwargs: Provider-specific client configuration

        Returns:
            Configured client instance for making API requests
        """
        pass

    @abstractmethod
    async def make_request(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
        **model_params,
    ) -> Any:
        """Make API request using provider's format.

        Args:
            messages: Conversation messages in standard format
            model_name: Provider-specific model identifier
            tools: Tools formatted for this provider (optional)
            stream: Whether to use streaming response
            **model_params: Model-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Provider's raw response object
        """
        pass

    @abstractmethod
    def extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from provider's response format.

        Args:
            response: Provider's raw response object

        Returns:
            Tuple of:
            - text_content: Main text response
            - tool_calls: List of tool calls in provider format
            - metadata: Provider-specific metadata (reasoning, usage, etc.)
        """
        pass

    def make_streaming_interruptible(
        self, response: Any, operation_name: str = None
    ) -> Any:
        """Wrap streaming response with interrupt checking.

        Args:
            response: Provider's streaming response object
            operation_name: Name for logging (defaults to provider name)

        Returns:
            InterruptAwareStream wrapper
        """
        from cli_agent.core.interrupt_aware_streaming import make_interruptible

        if operation_name is None:
            operation_name = f"{self.name} streaming response"

        return make_interruptible(response, operation_name)

    def make_sync_streaming_interruptible(
        self, response: Any, operation_name: str = None
    ) -> Any:
        """Wrap synchronous streaming response with interrupt checking.

        Args:
            response: Provider's synchronous streaming response object
            operation_name: Name for logging (defaults to provider name)

        Returns:
            Generator with interrupt checking
        """
        from cli_agent.core.interrupt_aware_streaming import make_sync_interruptible

        if operation_name is None:
            operation_name = f"{self.name} sync streaming response"

        return make_sync_interruptible(response, operation_name)

    @abstractmethod
    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process streaming response from provider.

        Args:
            response: Provider's streaming response object

        Returns:
            Tuple of:
            - accumulated_content: Full text content
            - tool_calls: List of completed tool calls
            - metadata: Provider-specific metadata
        """
        pass

    async def process_streaming_response_with_callbacks(
        self,
        response: Any,
        on_content: callable = None,
        on_tool_call_start: callable = None,
        on_tool_call_progress: callable = None,
        on_reasoning: callable = None,
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process streaming response with real-time callbacks.

        Default implementation calls process_streaming_response.
        Providers can override for real-time event emission.

        Args:
            response: Provider's streaming response object
            on_content: Called when content chunk received: on_content(text_chunk)
            on_tool_call_start: Called when tool call starts: on_tool_call_start(tool_name)
            on_tool_call_progress: Called for tool call progress: on_tool_call_progress(tool_call_data)
            on_reasoning: Called for reasoning content: on_reasoning(reasoning_chunk)

        Returns:
            Tuple of:
            - accumulated_content: Full text content
            - tool_calls: List of completed tool calls
            - metadata: Provider-specific metadata
        """
        # Default implementation: just call the non-callback version
        return await self.process_streaming_response(response)

    @abstractmethod
    def is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable for this provider.

        Args:
            error: Exception from API request

        Returns:
            True if the error should be retried
        """
        pass

    @abstractmethod
    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from provider error.

        Args:
            error: Exception from API request

        Returns:
            Human-readable error message
        """
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses."""
        pass

    @abstractmethod
    def get_rate_limit_info(self, response: Any) -> Dict[str, Any]:
        """Extract rate limit information from response headers.

        Args:
            response: Provider's response object

        Returns:
            Dict with rate limit info (requests_remaining, reset_time, etc.)
        """
        pass

    # Helper methods that providers can use

    def _register_http_client(self, client: Any, client_type: str = None) -> None:
        """Register HTTP client with global manager for centralized cleanup.
        
        Args:
            client: The HTTP client instance to register
            client_type: Optional client type identifier for debugging
        """
        try:
            from cli_agent.utils.http_client import http_client_manager
            
            client_id = f"{self.name}_{id(client)}"
            http_client_manager.register_client(client_id, client)
            
            client_desc = f"{client_type} " if client_type else ""
            logger.debug(f"Registered {self.name} {client_desc}HTTP client with global manager")
        except ImportError:
            logger.warning(
                f"HTTP client manager not available for {self.name} client registration"
            )

    def _extract_headers(self, response: Any) -> Dict[str, str]:
        """Extract headers from response if available.

        Args:
            response: Provider's response object

        Returns:
            Dict of response headers
        """
        if hasattr(response, "headers"):
            return dict(response.headers)
        return {}

    def _log_request(
        self, model_name: str, message_count: int, tool_count: int, stream: bool
    ):
        """Log API request details for debugging.

        Args:
            model_name: Model being used
            message_count: Number of messages in request
            tool_count: Number of tools provided
            stream: Whether streaming is enabled
        """
        logger.debug(
            f"{self.name} API request: model={model_name}, "
            f"messages={message_count}, tools={tool_count}, stream={stream}"
        )

    def _log_response(
        self, text_length: int, tool_call_count: int, metadata: Dict[str, Any]
    ):
        """Log API response details for debugging.

        Args:
            text_length: Length of response text
            tool_call_count: Number of tool calls in response
            metadata: Response metadata
        """
        logger.debug(
            f"{self.name} API response: text_len={text_length}, "
            f"tool_calls={tool_call_count}, metadata_keys={list(metadata.keys())}"
        )

    async def make_request_with_logging(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
        **model_params,
    ) -> Any:
        """Make API request with automatic logging.

        This is a convenience method that providers can use to get automatic
        request/response logging.
        """
        self._log_request(
            model_name=model_name,
            message_count=len(messages),
            tool_count=len(tools) if tools else 0,
            stream=stream,
        )

        response = await self.make_request(
            messages=messages,
            model_name=model_name,
            tools=tools,
            stream=stream,
            **model_params,
        )

        if not stream:
            # Log non-streaming response
            text, tool_calls, metadata = self.extract_response_content(response)
            self._log_response(
                text_length=len(text),
                tool_call_count=len(tool_calls),
                metadata=metadata,
            )

        return response
