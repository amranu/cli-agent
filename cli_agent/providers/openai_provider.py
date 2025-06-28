"""OpenAI API provider implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider for GPT models."""

    @property
    def name(self) -> str:
        return "openai"

    def get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def _create_client(self, **kwargs) -> Any:
        """Create OpenAI client."""
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

            logger.debug(f"Created OpenAI client with timeout: {timeout}s")
            return client

        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIProvider. Install with: pip install openai"
            )

    def supports_streaming(self) -> bool:
        return True

    async def get_available_models(self) -> List[str]:
        """Get list of available models from OpenAI API."""
        try:
            response = await self.client.models.list()
            # Filter for chat completion models (exclude embeddings, etc.)
            chat_models = []
            for model in response.data:
                model_id = model.id
                # Filter for GPT models and o1 models that support chat completions
                if (
                    model_id.startswith("gpt-")
                    or model_id.startswith("o1-")
                    or model_id in ["chatgpt-4o-latest"]
                ):
                    # Exclude fine-tuned models (contain colons) and deprecated models
                    if ":" not in model_id and not model_id.endswith("-instruct"):
                        chat_models.append(model_id)

            logger.info(
                f"OpenAI provider found {len(chat_models)} chat models: {chat_models}"
            )
            return sorted(chat_models)
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            # Return empty list if we can't fetch models
            return []

    @staticmethod
    async def fetch_available_models_static(api_key: str) -> List[str]:
        """Static method to fetch models without persisting client state."""
        try:
            from openai import AsyncOpenAI

            # Create a temporary client for this request only
            client = AsyncOpenAI(api_key=api_key, timeout=10.0)

            try:
                response = await client.models.list()
                # Filter for chat completion models (exclude embeddings, etc.)
                chat_models = []
                for model in response.data:
                    model_id = model.id
                    # Filter for GPT models and o1 models that support chat completions
                    if (
                        model_id.startswith("gpt-")
                        or model_id.startswith("o1-")
                        or model_id in ["chatgpt-4o-latest"]
                    ):
                        # Exclude fine-tuned models (contain colons) and deprecated models
                        if ":" not in model_id and not model_id.endswith("-instruct"):
                            chat_models.append(model_id)

                logger.info(f"OpenAI static fetch found {len(chat_models)} chat models")
                return sorted(chat_models)

            finally:
                # Always close the client
                await client.close()

        except ImportError:
            logger.error("openai package is required for OpenAI model discovery")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models (static): {e}")
            return []

    async def make_request(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
        **model_params,
    ) -> Any:
        """Make request to OpenAI API."""

        # Check if this is an o1 model which has different parameter requirements
        is_o1_model = model_name.startswith("o1-")
        # For o1 models, adjust parameter names and restrictions
        if is_o1_model:
            # Convert max_tokens to max_completion_tokens for o1 models
            if "max_tokens" in model_params:
                model_params["max_completion_tokens"] = model_params.pop("max_tokens")

            # o1 models don't support tools, temperature, or streaming
            if tools:
                logger.warning(
                    f"o1 model {model_name} does not support tools, ignoring tool calls"
                )
                tools = None

            # Remove unsupported parameters for o1 models
            unsupported_params = [
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ]
            for param in unsupported_params:
                if param in model_params:
                    logger.debug(
                        f"Removing unsupported parameter '{param}' for o1 model {model_name}"
                    )
                    model_params.pop(param)

            # o1 models don't support streaming
            if stream:
                logger.warning(
                    f"o1 model {model_name} does not support streaming, disabling"
                )
                stream = False

        request_params = {
            "model": model_name,
            "messages": messages,  # Already in OpenAI format
            "stream": stream,
            **model_params,
        }

        # Add tools if present and supported
        if tools and not is_o1_model:
            request_params["tools"] = tools

        logger.debug(
            f"OpenAI API request: model={model_name}, {len(messages)} messages, tools={len(tools) if tools else 0}"
        )

        try:
            response = await self.client.chat.completions.create(**request_params)

            # Add metadata to indicate actual streaming state for o1 models
            if is_o1_model and stream and not request_params.get("stream", False):
                # Mark that streaming was requested but disabled for o1 model
                # Use a wrapper class to add metadata
                class ResponseWrapper:
                    def __init__(self, response):
                        self._response = response
                        self._actual_streaming_disabled = True
                        self._requested_streaming = True

                    def __getattr__(self, name):
                        return getattr(self._response, name)

                    def __setattr__(self, name, value):
                        if name.startswith("_"):
                            super().__setattr__(name, value)
                        else:
                            setattr(self._response, name, value)

                return ResponseWrapper(response)

            return response
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise

    def extract_response_content(
        self, response: Any, requested_model: str = None
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from OpenAI response."""
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

        # Extract model information
        if hasattr(response, "model"):
            metadata["model"] = response.model
            logger.info(f"OpenAI response: actual model used={response.model}")

        logger.debug(
            f"Extracted OpenAI response: {len(text_content)} chars, {len(tool_calls)} tool calls"
        )
        return text_content, tool_calls, metadata

    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process OpenAI streaming response."""
        accumulated_content = ""
        accumulated_tool_calls = []
        metadata = {}

        # Wrap response with interrupt checking
        interruptible_response = self.make_streaming_interruptible(
            response, "OpenAI streaming"
        )

        async for chunk in interruptible_response:
            if chunk.choices:
                delta = chunk.choices[0].delta

                # Handle text content
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

            # Extract model info from first chunk
            if hasattr(chunk, "model") and not metadata.get("model"):
                metadata["model"] = chunk.model

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
            f"Processed OpenAI stream: {len(accumulated_content)} chars, {len(complete_tool_calls)} tool calls"
        )
        return accumulated_content, complete_tool_calls, metadata

    def is_retryable_error(self, error: Exception) -> bool:
        """Check if OpenAI error is retryable."""
        error_str = str(error).lower()

        # OpenAI-specific retryable errors
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
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from OpenAI error."""
        # Try to extract OpenAI error message
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
        """Extract rate limit info from OpenAI response."""
        headers = self._extract_headers(response)

        rate_limit_info = {}

        # OpenAI rate limit headers
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
