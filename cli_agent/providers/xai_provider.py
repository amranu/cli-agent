"""xAI API provider implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class XAIProvider(BaseProvider):
    """xAI API provider for Grok models."""

    @property
    def name(self) -> str:
        return "xai"

    def get_default_base_url(self) -> str:
        return "https://api.x.ai/v1"

    def _create_client(self, **kwargs) -> Any:
        """Create xAI client using OpenAI SDK (xAI is OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI

            timeout = kwargs.get("timeout", 120.0)

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
                **{k: v for k, v in kwargs.items() if k != "timeout"},
            )

            # Register with global HTTP client manager for centralized cleanup
            self._register_http_client(client)

            logger.debug(f"Created xAI client with timeout: {timeout}s")
            return client

        except ImportError:
            raise ImportError(
                "openai package is required for XAIProvider. Install with: pip install openai"
            )

    def supports_streaming(self) -> bool:
        return True

    async def get_available_models(self) -> List[str]:
        """Get list of available models from xAI API."""
        try:
            response = await self.client.models.list()
            # Filter for chat completion models (exclude image generation models)
            chat_models = []
            for model in response.data:
                model_id = model.id
                # Filter for Grok models that support chat completions
                if (
                    model_id.startswith("grok-")
                    and not model_id.startswith("grok-2-image")
                    and "image" not in model_id
                ):
                    chat_models.append(model_id)

            logger.info(
                f"xAI provider found {len(chat_models)} chat models: {chat_models}"
            )
            return sorted(chat_models)
        except Exception as e:
            logger.error(f"Failed to fetch xAI models: {e}")
            # Return empty list if we can't fetch models
            return []

    @staticmethod
    async def fetch_available_models_static(api_key: str) -> List[str]:
        """Static method to fetch models without persisting client state."""
        try:
            from openai import AsyncOpenAI

            # Create a temporary client for this request only
            client = AsyncOpenAI(
                api_key=api_key, 
                base_url="https://api.x.ai/v1",
                timeout=10.0
            )

            try:
                response = await client.models.list()
                # Filter for chat completion models (exclude image generation models)
                chat_models = []
                for model in response.data:
                    model_id = model.id
                    # Filter for Grok models that support chat completions
                    if (
                        model_id.startswith("grok-")
                        and not model_id.startswith("grok-2-image")
                        and "image" not in model_id
                    ):
                        chat_models.append(model_id)

                logger.info(f"xAI static fetch found {len(chat_models)} chat models")
                return sorted(chat_models)

            finally:
                # Always close the client
                await client.close()

        except ImportError:
            logger.error("openai package is required for xAI model discovery")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch xAI models (static): {e}")
            return []

    async def make_request(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
        **model_params,
    ) -> Any:
        """Make request to xAI API."""

        request_params = {
            "model": model_name,
            "messages": messages,  # Already in OpenAI format
            "stream": stream,
            **model_params,
        }

        # Add tools if present
        if tools:
            request_params["tools"] = tools

        logger.debug(
            f"xAI API request: model={model_name}, {len(messages)} messages, tools={len(tools) if tools else 0}"
        )

        try:
            response = await self.client.chat.completions.create(**request_params)
            return response
        except Exception as e:
            logger.error(f"xAI API request failed: {e}")
            raise

    def extract_response_content(
        self, response: Any, requested_model: str = None
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from xAI response."""
        if not hasattr(response, "choices") or not response.choices:
            return "", [], {}

        message = response.choices[0].message
        text_content = message.content or ""
        tool_calls = (
            message.tool_calls
            if hasattr(message, "tool_calls") and message.tool_calls
            else []
        )

        metadata = {}

        # Extract usage information
        if hasattr(response, "usage"):
            metadata["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Extract xAI-specific usage details
            if hasattr(response.usage, "prompt_tokens_details"):
                metadata["usage"]["prompt_tokens_details"] = {
                    "text_tokens": response.usage.prompt_tokens_details.text_tokens,
                    "audio_tokens": response.usage.prompt_tokens_details.audio_tokens,
                    "image_tokens": response.usage.prompt_tokens_details.image_tokens,
                    "cached_tokens": response.usage.prompt_tokens_details.cached_tokens,
                }

            if hasattr(response.usage, "completion_tokens_details"):
                metadata["usage"]["completion_tokens_details"] = {
                    "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
                    "audio_tokens": response.usage.completion_tokens_details.audio_tokens,
                    "accepted_prediction_tokens": response.usage.completion_tokens_details.accepted_prediction_tokens,
                    "rejected_prediction_tokens": response.usage.completion_tokens_details.rejected_prediction_tokens,
                }

        # Extract model information
        if hasattr(response, "model"):
            metadata["model"] = response.model
            logger.info(f"xAI response: actual model used={response.model}")

        # Extract xAI-specific reasoning content if present
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            metadata["reasoning_content"] = message.reasoning_content

        logger.debug(
            f"Extracted xAI response: {len(text_content)} chars, {len(tool_calls)} tool calls"
        )
        return text_content, tool_calls, metadata

    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process xAI streaming response."""
        accumulated_content = ""
        accumulated_tool_calls = []
        metadata = {}
        reasoning_content = ""

        # Wrap response with interrupt checking
        interruptible_response = self.make_streaming_interruptible(
            response, "xAI streaming"
        )

        async for chunk in interruptible_response:
            if chunk.choices:
                delta = chunk.choices[0].delta

                # Handle text content
                if delta.content:
                    accumulated_content += delta.content

                # Handle xAI-specific reasoning content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content

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

            # Extract model info from first chunk
            if hasattr(chunk, "model") and not metadata.get("model"):
                metadata["model"] = chunk.model

        # Add reasoning content to metadata if present
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content

        # Filter out incomplete tool calls and convert to proper format
        complete_tool_calls = []
        for tc in accumulated_tool_calls:
            if tc["function"]["name"] is not None:
                # Convert to tool call object format
                tool_call = type(
                    "ToolCall",
                    (),
                    {
                        "id": tc["id"] or f"stream_call_{len(complete_tool_calls)}",
                        "type": tc["type"],
                        "function": type(
                            "Function",
                            (),
                            {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        )(),
                    },
                )()
                complete_tool_calls.append(tool_call)

        logger.debug(
            f"Processed xAI stream: {len(accumulated_content)} chars, {len(complete_tool_calls)} tool calls"
        )
        return accumulated_content, complete_tool_calls, metadata

    def is_retryable_error(self, error: Exception) -> bool:
        """Check if xAI error is retryable."""
        error_str = str(error).lower()

        # Non-retryable errors - these should be handled by conversation cleanup
        non_retryable_patterns = [
            "tool_call_id",  # tool_call_id reference errors require conversation cleanup
            "invalid parameter",  # Parameter validation errors
        ]
        
        if any(pattern in error_str for pattern in non_retryable_patterns):
            return False

        # xAI-specific retryable errors
        retryable_patterns = [
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "overloaded",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "xai",  # xAI-specific errors that might be retryable
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from xAI error."""
        # Try to extract xAI error message (similar to OpenAI format)
        if hasattr(error, "response") and hasattr(error.response, "json"):
            try:
                error_data = error.response.json()
                if "error" in error_data:
                    return error_data["error"].get("message", str(error))
            except:
                pass

        # Try to get message from error body
        if hasattr(error, "body") and isinstance(error.body, dict):
            if "error" in error.body:
                return error.body["error"].get("message", str(error))

        return str(error)

    def get_rate_limit_info(self, response: Any) -> Dict[str, Any]:
        """Extract rate limit info from xAI response."""
        headers = self._extract_headers(response)

        rate_limit_info = {}

        # xAI rate limit headers (likely similar to OpenAI)
        if "x-ratelimit-remaining-requests" in headers:
            rate_limit_info["requests_remaining"] = int(
                headers["x-ratelimit-remaining-requests"]
            )
        if "x-ratelimit-reset-requests" in headers:
            rate_limit_info["requests_reset"] = headers["x-ratelimit-reset-requests"]
        if "x-ratelimit-remaining-tokens" in headers:
            rate_limit_info["tokens_remaining"] = int(
                headers["x-ratelimit-remaining-tokens"]
            )
        if "x-ratelimit-reset-tokens" in headers:
            rate_limit_info["tokens_reset"] = headers["x-ratelimit-reset-tokens"]

        return rate_limit_info