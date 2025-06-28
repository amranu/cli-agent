"""DeepSeek API provider implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider (OpenAI-compatible)."""

    @property
    def name(self) -> str:
        return "deepseek"

    def get_default_base_url(self) -> str:
        return "https://api.deepseek.com"

    def _create_client(self, **kwargs) -> Any:
        """Create DeepSeek client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI

            timeout = kwargs.get("timeout", 600.0)  # DeepSeek can be slower

            # DeepSeek uses OpenAI-compatible API
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
                **{k: v for k, v in kwargs.items() if k != "timeout"},
            )

            # Register with global HTTP client manager for centralized cleanup
            self._register_http_client(client)

            logger.debug(f"Created DeepSeek client with timeout: {timeout}s")
            return client

        except ImportError:
            raise ImportError(
                "openai package is required for DeepSeekProvider. Install with: pip install openai"
            )

    def supports_streaming(self) -> bool:
        return True

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from DeepSeek API.

        Note: DeepSeek uses OpenAI-compatible API, so this attempts to use
        the standard /v1/models endpoint. If DeepSeek doesn't support this
        endpoint, it will return a hardcoded list of known models.

        Returns:
            List of model dictionaries with id, name, context_length, and description
        """
        try:
            response = await self.client.models.list()
            # Extract model information
            models = []
            for model in response.data:
                model_id = model.id
                # Filter for DeepSeek models
                if "deepseek" in model_id.lower():
                    model_info = {
                        "id": model_id,
                        "name": model_id.replace("-", " ").title(),
                        "context_length": getattr(
                            model, "context_length", 32000
                        ),  # DeepSeek default
                        "description": f"DeepSeek {model_id.replace('deepseek-', '').replace('-', ' ').title()} model",
                    }
                    models.append(model_info)

            logger.info(f"DeepSeek provider found {len(models)} models")
            return sorted(models, key=lambda x: x["name"])

        except Exception as e:
            logger.warning(
                f"Failed to fetch DeepSeek models (may not support /v1/models endpoint): {e}"
            )
            # Return known DeepSeek models as fallback
            fallback_models = [
                {
                    "id": "deepseek-chat",
                    "name": "DeepSeek Chat",
                    "context_length": 32000,
                    "description": "DeepSeek's general-purpose chat model",
                },
                {
                    "id": "deepseek-reasoner",
                    "name": "DeepSeek Reasoner",
                    "context_length": 32000,
                    "description": "DeepSeek's reasoning-focused model with Chain of Thought",
                },
                {
                    "id": "deepseek-coder",
                    "name": "DeepSeek Coder",
                    "context_length": 32000,
                    "description": "DeepSeek's code-specialized model",
                },
            ]
            logger.info(
                f"Using fallback DeepSeek models: {len(fallback_models)} models"
            )
            return fallback_models

    async def make_request(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
        **model_params,
    ) -> Any:
        """Make request to DeepSeek API."""

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
            f"DeepSeek API request: {len(messages)} messages, tools={len(tools) if tools else 0}"
        )

        # Debug: Print the actual messages being sent
        if logger.isEnabledFor(logging.DEBUG):
            import json

            logger.debug(
                f"Messages being sent to DeepSeek: {json.dumps(messages, indent=2)}"
            )

        try:
            response = await self.client.chat.completions.create(**request_params)
            return response
        except Exception as e:
            logger.error(f"DeepSeek API request failed: {e}")
            # Debug: Print the request params when there's an error
            if logger.isEnabledFor(logging.DEBUG):
                import json

                logger.debug(
                    f"Failed request params: {json.dumps(request_params, indent=2, default=str)}"
                )
            raise

    def extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from DeepSeek response."""
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

        # Handle DeepSeek-specific reasoning content
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            metadata["reasoning_content"] = message.reasoning_content

        # Extract model information
        if hasattr(response, "model"):
            metadata["model"] = response.model

        logger.debug(
            f"Extracted DeepSeek response: {len(text_content)} chars, {len(tool_calls)} tool calls"
        )
        return text_content, tool_calls, metadata

    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process DeepSeek streaming response."""
        accumulated_content = ""
        accumulated_reasoning_content = ""
        accumulated_tool_calls = []
        metadata = {}

        # Wrap response with interrupt checking
        interruptible_response = self.make_streaming_interruptible(
            response, "DeepSeek streaming"
        )

        async for chunk in interruptible_response:
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

            # Extract model info from first chunk
            if hasattr(chunk, "model") and not metadata.get("model"):
                metadata["model"] = chunk.model

        # Add reasoning content to metadata if present
        if accumulated_reasoning_content:
            metadata["reasoning_content"] = accumulated_reasoning_content

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
            f"Processed DeepSeek stream: {len(accumulated_content)} chars, {len(complete_tool_calls)} tool calls"
        )
        return accumulated_content, complete_tool_calls, metadata

    def is_retryable_error(self, error: Exception) -> bool:
        """Check if DeepSeek error is retryable."""
        error_str = str(error).lower()

        # DeepSeek-specific retryable errors
        retryable_patterns = [
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "overloaded",
            "model overloaded",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "deepseek",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from DeepSeek error."""
        # Try to extract OpenAI-style error message (DeepSeek is compatible)
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
        """Extract rate limit info from DeepSeek response."""
        headers = self._extract_headers(response)

        rate_limit_info = {}

        # OpenAI-style rate limit headers (DeepSeek compatibility)
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
