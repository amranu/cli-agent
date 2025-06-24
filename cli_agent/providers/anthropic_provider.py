"""Anthropic API provider implementation."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

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
        """Create Anthropic client."""
        try:
            import anthropic

            timeout = kwargs.get("timeout", 120.0)

            client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
                **{k: v for k, v in kwargs.items() if k != "timeout"},
            )

            logger.debug(f"Created Anthropic client with timeout: {timeout}s")
            return client

        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. Install with: pip install anthropic"
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
        """Make request to Anthropic API."""

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic(messages)

        # Extract system message if present
        system_content = self._extract_system_content(messages)

        # Convert tools to Anthropic format
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None

        # Prepare request parameters
        request_params = {
            "model": model_name,
            "messages": anthropic_messages,
            "stream": stream,
            **model_params,
        }

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
            response = await self.client.messages.create(**request_params)
            return response
        except Exception as e:
            logger.error(f"Anthropic API request failed: {e}")
            raise

    def extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from Anthropic response."""
        if not hasattr(response, "content") or not response.content:
            return "", [], {}

        text_parts = []
        tool_calls = []
        metadata = {}

        # Extract usage information
        if hasattr(response, "usage"):
            metadata["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        # Process content blocks
        for content_block in response.content:
            if hasattr(content_block, "type"):
                if content_block.type == "text":
                    text_parts.append(content_block.text)
                elif content_block.type == "tool_use":
                    tool_calls.append(content_block)

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

        async for event in response:
            if hasattr(event, "type"):
                if event.type == "content_block_start":
                    if (
                        hasattr(event, "content_block")
                        and event.content_block.type == "tool_use"
                    ):
                        current_tool_use = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "input": {},
                        }

                elif event.type == "content_block_delta":
                    if hasattr(event, "delta"):
                        if event.delta.type == "text_delta":
                            accumulated_content += event.delta.text
                        elif (
                            event.delta.type == "input_json_delta" and current_tool_use
                        ):
                            # Accumulate tool input JSON
                            if "partial_json" not in current_tool_use:
                                current_tool_use["partial_json"] = ""
                            current_tool_use["partial_json"] += event.delta.partial_json

                elif event.type == "content_block_stop":
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

                elif event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        metadata["usage"] = {
                            "input_tokens": event.message.usage.input_tokens,
                            "output_tokens": event.message.usage.output_tokens,
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
        # Try to extract structured error message
        if hasattr(error, "response") and hasattr(error.response, "json"):
            try:
                error_data = error.response.json()
                if "error" in error_data:
                    return error_data["error"].get("message", str(error))
            except:
                pass

        return str(error)

    def get_rate_limit_info(self, response: Any) -> Dict[str, Any]:
        """Extract rate limit info from Anthropic response."""
        headers = self._extract_headers(response)

        rate_limit_info = {}

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

            # Convert role names
            if role == "assistant":
                anthropic_role = "assistant"
            else:
                anthropic_role = "user"  # user, tool results, etc.

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
