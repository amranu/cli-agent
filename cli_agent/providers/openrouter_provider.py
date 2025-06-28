"""OpenRouter API provider implementation."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider for accessing multiple models through one API."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """Initialize OpenRouter provider with caching for model list."""
        super().__init__(api_key, base_url, **kwargs)
        self._models_cache = None
        self._models_cache_time = 0
        self._cache_duration = 3600  # Cache for 1 hour

    @property
    def name(self) -> str:
        return "openrouter"

    def get_default_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    def _create_client(self, **kwargs) -> Any:
        """Create OpenRouter client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI

            timeout = kwargs.get("timeout", 120.0)

            # OpenRouter uses OpenAI-compatible API
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
                **{k: v for k, v in kwargs.items() if k != "timeout"},
            )

            # Register with global HTTP client manager for centralized cleanup
            self._register_http_client(client)

            logger.debug(f"Created OpenRouter client with timeout: {timeout}s")
            return client

        except ImportError:
            raise ImportError(
                "openai package is required for OpenRouterProvider. Install with: pip install openai"
            )

    def supports_streaming(self) -> bool:
        return True

    async def make_request(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
        **model_params,
    ) -> Any:
        """Make request to OpenRouter API."""

        # OpenRouter uses OpenAI-compatible format
        request_params = {
            "model": model_name,
            "messages": messages,  # Already in OpenAI format
            "stream": stream,
            **model_params,
        }

        # Add tools if present
        if tools:
            request_params["tools"] = tools

        # Add OpenRouter-specific headers if needed
        extra_headers = {}
        if hasattr(self, "app_name"):
            extra_headers["HTTP-Referer"] = self.app_name
        if hasattr(self, "site_url"):
            extra_headers["X-Title"] = self.site_url

        logger.debug(
            f"OpenRouter API request: {len(messages)} messages, tools={len(tools) if tools else 0}"
        )

        try:
            response = await self.client.chat.completions.create(**request_params)
            return response
        except Exception as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise

    def extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from OpenRouter response (OpenAI format)."""
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

        logger.debug(
            f"Extracted OpenRouter response: {len(text_content)} chars, {len(tool_calls)} tool calls"
        )
        return text_content, tool_calls, metadata

    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process OpenRouter streaming response (OpenAI format)."""
        accumulated_content = ""
        accumulated_tool_calls = []
        metadata = {}

        # Wrap response with interrupt checking
        interruptible_response = self.make_streaming_interruptible(
            response, "OpenRouter streaming"
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
            f"Processed OpenRouter stream: {len(accumulated_content)} chars, {len(complete_tool_calls)} tool calls"
        )
        return accumulated_content, complete_tool_calls, metadata

    def is_retryable_error(self, error: Exception) -> bool:
        """Check if OpenRouter error is retryable."""
        error_str = str(error).lower()

        # OpenRouter-specific retryable errors
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
        """Extract meaningful error message from OpenRouter error."""
        # Try to extract OpenAI-style error message
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
        """Extract rate limit info from OpenRouter response."""
        headers = self._extract_headers(response)

        rate_limit_info = {}

        # OpenAI-style rate limit headers
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

    def set_app_info(self, app_name: str, site_url: str):
        """Set application info for OpenRouter attribution.

        Args:
            app_name: Name of your application
            site_url: URL of your application/site
        """
        self.app_name = app_name
        self.site_url = site_url
        logger.debug(f"Set OpenRouter app info: {app_name} - {site_url}")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter API.

        Returns:
            List of model dictionaries with id, name, context_length, and description
        """
        # Check cache first
        current_time = time.time()
        if (
            self._models_cache is not None
            and current_time - self._models_cache_time < self._cache_duration
        ):
            logger.debug("Returning cached OpenRouter models")
            return self._models_cache

        try:
            # Import httpx for direct API call
            import httpx

            # Create a temporary httpx client for this request
            async with httpx.AsyncClient() as http_client:
                # Add retry logic for robustness
                from cli_agent.utils.retry import retry_with_backoff

                async def fetch_models():
                    return await http_client.get(
                        f"{self.base_url}/models",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )

                response = await retry_with_backoff(
                    fetch_models,
                    max_retries=2,
                    base_delay=1.0,
                    retryable_errors=[
                        "timeout",
                        "connection",
                        "500",
                        "502",
                        "503",
                        "504",
                    ],
                )

                if response.status_code != 200:
                    error_msg = (
                        f"OpenRouter models API returned status {response.status_code}"
                    )
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = f"{error_msg}: {error_data['error']}"
                    except:
                        error_msg = f"{error_msg}: {response.text}"
                    logger.error(error_msg)
                    return []

                data = response.json()
                models = data.get("data", [])

                # Process models to extract essential info
                processed_models = []
                for model in models:
                    model_info = {
                        "id": model.get("id"),
                        "name": model.get("name", model.get("id", "Unknown")),
                        "context_length": model.get("context_length", 4096),
                        "description": model.get("description", ""),
                        "pricing": model.get("pricing", {}),
                        "top_provider": model.get("top_provider", {}),
                    }

                    # Only include models that have an ID
                    if model_info["id"]:
                        processed_models.append(model_info)

                # Sort by name for consistency
                processed_models.sort(key=lambda x: x["name"].lower())

                # Update cache
                self._models_cache = processed_models
                self._models_cache_time = current_time

                logger.info(f"OpenRouter provider found {len(processed_models)} models")
                return processed_models

        except ImportError:
            logger.error(
                "httpx package is required for OpenRouter model discovery. Install with: pip install httpx"
            )
            return []
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to OpenRouter API: {e}")
            return []
        except httpx.TimeoutException:
            logger.error("OpenRouter API request timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
            return []

    async def get_available_models_summary(self) -> List[str]:
        """Get a simple list of model IDs from OpenRouter.

        Returns:
            List of model ID strings
        """
        models = await self.get_available_models()
        return [model["id"] for model in models]

    @staticmethod
    async def fetch_available_models_static(api_key: str) -> List[Dict[str, Any]]:
        """Static method to fetch models without persisting client state.

        Args:
            api_key: OpenRouter API key

        Returns:
            List of model dictionaries
        """
        try:
            import httpx

            # Create a temporary httpx client for this request
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"OpenRouter models API returned status {response.status_code}"
                    )
                    return []

                data = response.json()
                models = data.get("data", [])

                # Process models to extract essential info
                processed_models = []
                for model in models:
                    model_info = {
                        "id": model.get("id"),
                        "name": model.get("name", model.get("id", "Unknown")),
                        "context_length": model.get("context_length", 4096),
                        "description": model.get("description", ""),
                    }

                    # Only include models that have an ID
                    if model_info["id"]:
                        processed_models.append(model_info)

                # Sort by name for consistency
                processed_models.sort(key=lambda x: x["name"].lower())

                logger.info(
                    f"OpenRouter static fetch found {len(processed_models)} models"
                )
                return processed_models

        except ImportError:
            logger.error("httpx package is required for OpenRouter model discovery")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models (static): {e}")
            return []
