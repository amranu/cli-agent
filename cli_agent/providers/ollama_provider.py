#!/usr/bin/env python3
"""MCP Host implementation using Ollama as the language model backend."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama API provider for local models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return "ollama"

    def get_default_base_url(self) -> str:
        return "http://localhost:11434"

    def _create_client(self, **kwargs) -> Any:
        """Create HTTP client for Ollama API."""
        timeout = kwargs.get("timeout", 180.0)

        client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

        # Register with global HTTP client manager for centralized cleanup
        self._register_http_client(client)

        logger.debug(f"Created Ollama HTTP client with timeout: {timeout}s")
        return client

    def supports_streaming(self) -> bool:
        return True

    def _model_supports_tools(self, model_name: str) -> bool:
        """Check if a model supports tool calling.
        
        Args:
            model_name: The Ollama model name
            
        Returns:
            True if the model supports tools, False otherwise
        """
        # Models known to not support tools
        unsupported_models = [
            "gemma3:",  # All gemma3 variants don't support tools
            "deepseek-r1:",  # DeepSeek R1 models don't support tools
        ]
        
        # Check if model matches any unsupported pattern
        for pattern in unsupported_models:
            if model_name.startswith(pattern):
                return False
                
        # Default to True for other models
        return True

    def extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from Ollama response."""
        if isinstance(response, dict):
            # Handle dict response format
            choices = response.get("choices", [])
            if not choices:
                return "", [], {}

            message = choices[0].get("message", {})
            text_content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            metadata = {}
            if "model" in response:
                metadata["model"] = response["model"]
            if "usage" in response:
                metadata["usage"] = response["usage"]

            return text_content, tool_calls, metadata
        elif hasattr(response, "choices") and response.choices:
            # Handle object response format
            message = response.choices[0].message
            text_content = message.content or ""
            tool_calls = (
                message.tool_calls
                if hasattr(message, "tool_calls") and message.tool_calls
                else []
            )
            metadata = {}
            if hasattr(response, "model"):
                metadata["model"] = response.model
            if hasattr(response, "usage"):
                metadata["usage"] = response.usage
            return text_content, tool_calls, metadata
        else:
            return "", [], {}

    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process Ollama streaming response."""
        accumulated_content = ""
        accumulated_tool_calls = []
        metadata = {}
        current_tool_call = None

        # Handle both async and sync iterators
        if hasattr(response, "__aiter__"):
            # Async iterator - wrap with interrupt checking
            interruptible_response = self.make_streaming_interruptible(
                response, "Ollama streaming"
            )

            async for chunk in interruptible_response:
                if isinstance(chunk, dict):
                    # Handle dict chunk format
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            accumulated_content += delta["content"]
                        if "tool_calls" in delta and delta["tool_calls"]:
                            for tc in delta["tool_calls"]:
                                if tc.get("index") is not None:
                                    # Handle streaming tool call updates
                                    idx = tc["index"]
                                    while len(accumulated_tool_calls) <= idx:
                                        accumulated_tool_calls.append(
                                            {
                                                "id": None,
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "",
                                                },
                                            }
                                        )

                                    if "id" in tc:
                                        accumulated_tool_calls[idx]["id"] = tc["id"]
                                    if "function" in tc:
                                        func = tc["function"]
                                        if "name" in func:
                                            accumulated_tool_calls[idx]["function"][
                                                "name"
                                            ] = func["name"]
                                        if "arguments" in func:
                                            accumulated_tool_calls[idx]["function"][
                                                "arguments"
                                            ] += func["arguments"]

                    if "model" in chunk and not metadata.get("model"):
                        metadata["model"] = chunk["model"]
                    if "usage" in chunk:
                        metadata["usage"] = chunk["usage"]

                elif hasattr(chunk, "choices") and chunk.choices:
                    # Handle object chunk format
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_content += delta.content
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            if hasattr(tc, "index") and tc.index is not None:
                                idx = tc.index
                                while len(accumulated_tool_calls) <= idx:
                                    accumulated_tool_calls.append(
                                        {
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    )

                                if hasattr(tc, "id"):
                                    accumulated_tool_calls[idx]["id"] = tc.id
                                if hasattr(tc, "function"):
                                    func = tc.function
                                    if hasattr(func, "name") and func.name:
                                        accumulated_tool_calls[idx]["function"][
                                            "name"
                                        ] = func.name
                                    if hasattr(func, "arguments") and func.arguments:
                                        accumulated_tool_calls[idx]["function"][
                                            "arguments"
                                        ] += func.arguments

                    if hasattr(chunk, "model") and not metadata.get("model"):
                        metadata["model"] = chunk.model
                    if hasattr(chunk, "usage"):
                        metadata["usage"] = chunk.usage
        else:
            # Sync iterator - wrap with interrupt checking
            interruptible_response = self.make_sync_streaming_interruptible(
                response, "Ollama streaming"
            )

            for chunk in interruptible_response:
                # Similar processing as above but for sync iteration
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            accumulated_content += delta["content"]
                    if "model" in chunk and not metadata.get("model"):
                        metadata["model"] = chunk["model"]

        logger.debug(
            f"Processed Ollama stream: {len(accumulated_content)} chars, {len(accumulated_tool_calls)} tool calls"
        )
        return accumulated_content, accumulated_tool_calls, metadata


    async def make_request(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
        **model_params,
    ) -> Any:
        """Make request to Ollama API."""

        # Prepare request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
        }

        # Add model parameters
        for key, value in model_params.items():
            if key in [
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "frequency_penalty",
                "presence_penalty",
            ]:
                payload[key] = value

        # Add tools if provided, but skip for models that don't support them
        if tools and self._model_supports_tools(model_name):
            payload["tools"] = tools
        elif tools and not self._model_supports_tools(model_name):
            logger.warning(f"Model {model_name} does not support tools - skipping tool calls")

        logger.debug(
            f"Ollama API request: {len(messages)} messages, tools={len(tools) if tools else 0}"
        )

        try:
            if stream:
                # For streaming, return the response directly
                response = await self.client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers={"accept": "text/event-stream"},
                )
                response.raise_for_status()
                return self._create_streaming_iterator(response)
            else:
                # For non-streaming, return JSON response
                response = await self.client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Ollama API request failed: {e}")
            raise

    def _create_streaming_iterator(self, response):
        """Create an async iterator from httpx.Response for Ollama streaming."""

        async def ollama_stream_iterator():
            """Process Ollama streaming events."""
            # Wrap with interrupt checking
            interruptible_response = self.make_streaming_interruptible(
                response.aiter_lines(), "Ollama stream iterator"
            )

            async for line in interruptible_response:
                if not line.strip():
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        # Convert dict format to object format for MCPHost compatibility
                        yield self._convert_dict_to_object(chunk_data)
                    except json.JSONDecodeError:
                        continue

        return ollama_stream_iterator()

    def _convert_dict_to_object(self, chunk_data):
        """Convert Ollama dict format chunks to object format for MCPHost compatibility."""
        # Create a simple object that mimics the expected structure
        class ChunkObject:
            def __init__(self, data):
                if "choices" in data and data["choices"]:
                    choice_data = data["choices"][0]
                    self.choices = [ChoiceObject(choice_data)]
                else:
                    self.choices = []
                    
                # Copy other metadata
                for key, value in data.items():
                    if key != "choices":
                        setattr(self, key, value)

        class ChoiceObject:
            def __init__(self, choice_data):
                self.index = choice_data.get("index", 0)
                if "delta" in choice_data:
                    self.delta = DeltaObject(choice_data["delta"])
                elif "message" in choice_data:
                    self.message = choice_data["message"]

        class DeltaObject:
            def __init__(self, delta_data):
                # Handle content and separate thinking/reasoning from actual response
                raw_content = delta_data.get("content")
                self.content = self._process_content_for_thinking(raw_content)
                self.role = delta_data.get("role")
                if "tool_calls" in delta_data:
                    self.tool_calls = delta_data["tool_calls"]
                else:
                    self.tool_calls = None
                    
            def _process_content_for_thinking(self, content):
                """Process content to handle thinking tags for models like qwen."""
                if not content:
                    return content
                    
                # For now, return the content as-is but we'll need to add
                # thinking extraction logic here similar to DeepSeek
                return content

        return ChunkObject(chunk_data)

    def is_retryable_error(self, error: Exception) -> bool:
        """Check if Ollama error is retryable."""
        error_str = str(error).lower()

        # Ollama-specific retryable errors
        retryable_patterns = [
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "connection",
            "ollama",
            "model not found",
            "service unavailable",
            "internal server error",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from Ollama error."""
        # Try to extract structured error message from HTTP error
        if hasattr(error, "response"):
            try:
                error_data = error.response.json()
                if "error" in error_data:
                    return error_data["error"].get("message", str(error))
            except:
                pass

        return str(error)

    def get_rate_limit_info(self, response: Any) -> Dict[str, Any]:
        """Extract rate limit info from Ollama response."""
        rate_limit_info = {}

        # Handle both dict response and httpx response
        if hasattr(response, "headers"):
            headers = response.headers
        elif isinstance(response, dict):
            # No headers in JSON response
            return rate_limit_info
        else:
            return rate_limit_info

        # Ollama rate limit headers (if available)
        if "x-ratelimit-remaining" in headers:
            rate_limit_info["requests_remaining"] = int(
                headers["x-ratelimit-remaining"]
            )
        if "x-ratelimit-reset" in headers:
            rate_limit_info["reset_time"] = headers["x-ratelimit-reset"]

        return rate_limit_info
