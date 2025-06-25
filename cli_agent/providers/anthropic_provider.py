"""Anthropic API provider implementation."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic API provider for Claude models."""

    @property
    def name(self) -> str:
        return "anthropic"

    def get_default_base_url(self) -> str:
        return "https://api.anthropic.com"

    def _create_client(self, **kwargs) -> Any:
        """Create HTTP client for Anthropic API."""
        timeout = kwargs.get("timeout", 120.0)

        client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

        logger.debug(f"Created Anthropic HTTP client with timeout: {timeout}s")
        return client

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
        """Make request to Anthropic API."""

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic(messages)

        # Validate that we have messages after conversion
        if not anthropic_messages:
            raise ValueError(
                "No valid messages after filtering system messages for Anthropic API"
            )

        # Extract system message if present
        system_content = self._extract_system_content(messages)

        # Convert tools to Anthropic format
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None

        # Prepare request parameters
        request_params = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": model_params.get("max_tokens", 4096),
            "stream": stream,
        }

        # Add other model parameters
        for key, value in model_params.items():
            if key not in ["max_tokens"]:
                request_params[key] = value

        # Add system content if present
        if system_content:
            request_params["system"] = system_content

        # Add tools if present
        if anthropic_tools:
            request_params["tools"] = anthropic_tools

        logger.debug(
            f"Anthropic API request: {len(anthropic_messages)} messages, tools={len(tools) if tools else 0}"
        )

        try:
            if stream:
                response = await self.client.post(
                    "/v1/messages",
                    json=request_params,
                    headers={"accept": "text/event-stream"},
                )
            else:
                response = await self.client.post("/v1/messages", json=request_params)

            response.raise_for_status()

            if stream:
                # Return an async iterator instead of the raw response
                return self._create_streaming_iterator(response)
            else:
                return response.json()

        except Exception as e:
            # Log detailed error information for billing/quota issues
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_details = e.response.json()
                    if "credit balance" in str(error_details).lower():
                        logger.error(
                            f"Anthropic API billing issue: {error_details.get('error', {}).get('message', 'Unknown billing error')}"
                        )
                    else:
                        logger.debug(f"Anthropic API error details: {error_details}")
                except:
                    pass
            logger.error(f"Anthropic API request failed: {e}")
            raise

    def extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from Anthropic response."""
        if not isinstance(response, dict) or "content" not in response:
            return "", [], {}

        text_parts = []
        tool_calls = []
        metadata = {}

        # Extract usage information
        if "usage" in response:
            metadata["usage"] = {
                "input_tokens": response["usage"].get("input_tokens", 0),
                "output_tokens": response["usage"].get("output_tokens", 0),
            }

        # Process content blocks
        for content_block in response["content"]:
            if content_block.get("type") == "text":
                text_parts.append(content_block.get("text", ""))
            elif content_block.get("type") == "tool_use":
                # Create tool call object compatible with the expected format
                tool_call = type(
                    "ToolUse",
                    (),
                    {
                        "id": content_block.get("id"),
                        "name": content_block.get("name"),
                        "input": content_block.get("input", {}),
                        "type": "tool_use",
                    },
                )()
                tool_calls.append(tool_call)

        text_content = "".join(text_parts)

        logger.debug(
            f"Extracted Anthropic response: {len(text_content)} chars, {len(tool_calls)} tool calls"
        )
        return text_content, tool_calls, metadata

    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process Anthropic streaming response."""
        accumulated_content = ""
        tool_calls = []
        metadata = {}
        current_tool_use = None

        # Process server-sent events
        async for line in response.aiter_lines():
            if not line.strip():
                continue

            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix

                if data_str.strip() == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "content_block_start":
                    content_block = event.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        current_tool_use = {
                            "id": content_block.get("id"),
                            "name": content_block.get("name"),
                            "input": {},
                            "partial_json": "",
                        }

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        accumulated_content += delta.get("text", "")
                    elif delta.get("type") == "input_json_delta" and current_tool_use:
                        current_tool_use["partial_json"] += delta.get(
                            "partial_json", ""
                        )

                elif event_type == "content_block_stop":
                    if current_tool_use:
                        # Parse accumulated JSON
                        try:
                            current_tool_use["input"] = json.loads(
                                current_tool_use["partial_json"]
                            )
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse tool input JSON: {current_tool_use.get('partial_json', '')}"
                            )
                            current_tool_use["input"] = {}

                        # Create tool call object
                        tool_call = type(
                            "ToolUse",
                            (),
                            {
                                "id": current_tool_use["id"],
                                "name": current_tool_use["name"],
                                "input": current_tool_use["input"],
                                "type": "tool_use",
                            },
                        )()

                        tool_calls.append(tool_call)
                        current_tool_use = None

                elif event_type == "message_start":
                    message = event.get("message", {})
                    usage = message.get("usage", {})
                    if usage:
                        metadata["usage"] = {
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        }

        logger.debug(
            f"Processed Anthropic stream: {len(accumulated_content)} chars, {len(tool_calls)} tool calls"
        )
        return accumulated_content, tool_calls, metadata

    def is_retryable_error(self, error: Exception) -> bool:
        """Check if Anthropic error is retryable."""
        error_str = str(error).lower()

        # Anthropic-specific retryable errors
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
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from Anthropic error."""
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
        """Extract rate limit info from Anthropic response."""
        rate_limit_info = {}

        # Handle both dict response and httpx response
        if hasattr(response, "headers"):
            headers = response.headers
        elif isinstance(response, dict):
            # No headers in JSON response
            return rate_limit_info
        else:
            return rate_limit_info

        # Anthropic rate limit headers
        if "anthropic-ratelimit-requests-remaining" in headers:
            rate_limit_info["requests_remaining"] = int(
                headers["anthropic-ratelimit-requests-remaining"]
            )
        if "anthropic-ratelimit-requests-reset" in headers:
            rate_limit_info["requests_reset"] = headers[
                "anthropic-ratelimit-requests-reset"
            ]
        if "anthropic-ratelimit-tokens-remaining" in headers:
            rate_limit_info["tokens_remaining"] = int(
                headers["anthropic-ratelimit-tokens-remaining"]
            )
        if "anthropic-ratelimit-tokens-reset" in headers:
            rate_limit_info["tokens_reset"] = headers[
                "anthropic-ratelimit-tokens-reset"
            ]

        return rate_limit_info

    def _convert_messages_to_anthropic(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert standard messages to Anthropic format."""
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                continue

            # Handle tool results - convert to user message with tool result content
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                tool_name = msg.get("name", "unknown_tool")

                # Ensure content is a string and handle potential encoding issues
                if content is None:
                    content = ""
                content_str = str(content)

                # Format tool result as user message for Anthropic
                formatted_content = f"Tool result for {tool_name} (call_id: {tool_call_id}):\n{content_str}"
                anthropic_messages.append(
                    {"role": "user", "content": formatted_content}
                )
                continue

            # Handle empty content messages
            if not content or content.strip() == "":
                # Skip empty user messages as they violate Anthropic API
                if role != "assistant":
                    continue
                # For assistant messages, only allow empty content if it's the final message
                is_last_message = msg == messages[-1]
                if not is_last_message:
                    # Replace empty assistant content with a minimal placeholder
                    content = "."

            # Convert role names
            if role == "assistant":
                anthropic_role = "assistant"
            else:
                anthropic_role = "user"  # user and other message types

            anthropic_messages.append({"role": anthropic_role, "content": content})

        return anthropic_messages

    def _extract_system_content(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract system message content for Anthropic's system parameter."""
        system_parts = []

        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))

        return "\n\n".join(system_parts) if system_parts else None

    def _convert_tools_to_anthropic(
        self, tools: Optional[List[Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []

        for tool in tools:
            # Assume tools are already in a compatible format
            # This would need to be adapted based on the tool format
            if isinstance(tool, dict):
                anthropic_tools.append(tool)
            else:
                # Convert other tool formats as needed
                logger.warning(f"Unknown tool format: {type(tool)}")

        return anthropic_tools if anthropic_tools else None

    def _create_streaming_iterator(self, response):
        """Create an async iterator from httpx.Response for streaming that yields OpenAI-compatible chunks."""

        async def anthropic_stream_iterator():
            """Convert Anthropic streaming events to OpenAI-compatible format."""
            current_tool_use = None
            tool_index = 0  # Track tool call index for OpenAI compatibility

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "content_block_start":
                        content_block = event.get("content_block", {})
                        if content_block.get("type") == "tool_use":
                            current_tool_use = {
                                "id": content_block.get("id"),
                                "name": content_block.get("name"),
                                "arguments": "",
                                "index": tool_index,
                            }
                            tool_index += 1

                            # Yield tool call start with ID and name
                            chunk = type(
                                "Chunk",
                                (),
                                {
                                    "choices": [
                                        type(
                                            "Choice",
                                            (),
                                            {
                                                "delta": type(
                                                    "Delta",
                                                    (),
                                                    {
                                                        "tool_calls": [
                                                            type(
                                                                "ToolCall",
                                                                (),
                                                                {
                                                                    "index": current_tool_use[
                                                                        "index"
                                                                    ],
                                                                    "id": current_tool_use[
                                                                        "id"
                                                                    ],
                                                                    "type": "function",
                                                                    "function": type(
                                                                        "Function",
                                                                        (),
                                                                        {
                                                                            "name": current_tool_use[
                                                                                "name"
                                                                            ],
                                                                            "arguments": "",
                                                                        },
                                                                    )(),
                                                                },
                                                            )()
                                                        ]
                                                    },
                                                )()
                                            },
                                        )()
                                    ]
                                },
                            )()
                            yield chunk

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            # Yield text content as OpenAI-compatible chunk
                            chunk = type(
                                "Chunk",
                                (),
                                {
                                    "choices": [
                                        type(
                                            "Choice",
                                            (),
                                            {
                                                "delta": type(
                                                    "Delta",
                                                    (),
                                                    {"content": delta.get("text", "")},
                                                )()
                                            },
                                        )()
                                    ]
                                },
                            )()
                            yield chunk

                        elif (
                            delta.get("type") == "input_json_delta" and current_tool_use
                        ):
                            # Yield incremental tool arguments
                            partial_json = delta.get("partial_json", "")
                            current_tool_use["arguments"] += partial_json

                            chunk = type(
                                "Chunk",
                                (),
                                {
                                    "choices": [
                                        type(
                                            "Choice",
                                            (),
                                            {
                                                "delta": type(
                                                    "Delta",
                                                    (),
                                                    {
                                                        "tool_calls": [
                                                            type(
                                                                "ToolCall",
                                                                (),
                                                                {
                                                                    "index": current_tool_use[
                                                                        "index"
                                                                    ],
                                                                    "function": type(
                                                                        "Function",
                                                                        (),
                                                                        {
                                                                            "arguments": partial_json
                                                                        },
                                                                    )(),
                                                                },
                                                            )()
                                                        ]
                                                    },
                                                )()
                                            },
                                        )()
                                    ]
                                },
                            )()
                            yield chunk

                    elif event_type == "content_block_stop":
                        if current_tool_use:
                            # Tool call is complete, but we've already yielded incremental updates
                            current_tool_use = None

        return anthropic_stream_iterator()
