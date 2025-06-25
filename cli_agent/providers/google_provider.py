"""Google Gemini API provider implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""

    @property
    def name(self) -> str:
        return "google"

    def get_default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    def _create_client(self, **kwargs) -> Any:
        """Create Gemini client."""
        try:
            import httpx
            from google import genai

            timeout = kwargs.get("timeout", 120.0)

            # Create HTTP client with custom timeout
            try:
                http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout),
                    limits=httpx.Limits(
                        max_connections=10, max_keepalive_connections=5
                    ),
                )

                client = genai.Client(api_key=self.api_key)

                # Store reference for cleanup
                self.http_client = http_client

            except Exception as e:
                logger.warning(f"Failed to create custom HTTP client: {e}")
                # Fallback to default client
                client = genai.Client(api_key=self.api_key)
                self.http_client = None

            logger.debug(f"Created Gemini client with timeout: {timeout}s")
            return client

        except ImportError:
            raise ImportError(
                "google-genai package is required for GoogleProvider. Install with: pip install google-genai"
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
        """Make request to Gemini API."""

        # Convert messages to Gemini string format
        gemini_prompt = self._convert_messages_to_gemini_format(messages)

        # Configure tool calling if tools provided
        tool_config = None
        if tools:
            tool_config = self._create_tool_config(
                model_params.get("function_calling_mode", "AUTO")
            )

        # Create generation config
        config = self._create_generation_config(tools, tool_config, **model_params)

        logger.debug(
            f"Gemini API request: {len(messages)} messages, tools={len(tools) if tools else 0}"
        )

        try:
            if stream:
                # Gemini streaming returns an async generator
                response = self.client.models.generate_content_stream(
                    model=model_name,
                    contents=gemini_prompt,
                    config=config,
                )
                # Gemini streaming returns a regular generator, not async
                return response
            else:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=gemini_prompt,
                    config=config,
                )
                return response
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            raise

    def extract_response_content(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from Gemini response."""
        if not hasattr(response, "candidates") or not response.candidates:
            return "", [], {}

        candidate = response.candidates[0]
        text_content = ""
        tool_calls = []
        metadata = {}

        if hasattr(candidate, "content") and candidate.content:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_content += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(part.function_call)

        # Extract usage information if available
        if hasattr(response, "usage_metadata"):
            metadata["usage"] = {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
                "total_token_count": response.usage_metadata.total_token_count,
            }

        logger.debug(
            f"Extracted Gemini response: {len(text_content)} chars, {len(tool_calls)} tool calls"
        )
        return text_content, tool_calls, metadata

    async def process_streaming_response(
        self, response: Any
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Process Gemini streaming response."""
        accumulated_content = ""
        accumulated_tool_calls = []
        metadata = {}

        # Gemini returns a regular generator, not async generator
        for chunk in response:
            # Check for function calls in chunk first to avoid accessing .text when function calls are present
            if hasattr(chunk, "candidates") and chunk.candidates:
                if (
                    chunk.candidates[0]
                    and hasattr(chunk.candidates[0], "content")
                    and chunk.candidates[0].content
                ):

                    if (
                        hasattr(chunk.candidates[0].content, "parts")
                        and chunk.candidates[0].content.parts
                    ):

                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, "text") and part.text:
                                accumulated_content += part.text
                            elif hasattr(part, "function_call") and part.function_call:
                                accumulated_tool_calls.append(part.function_call)
            else:
                # Fallback: if no candidates, try direct text access (for simple text chunks)
                if (
                    hasattr(chunk, "text")
                    and chunk.text
                    and not hasattr(chunk, "candidates")
                ):
                    accumulated_content += chunk.text

            # Extract usage metadata from final chunk
            if hasattr(chunk, "usage_metadata"):
                metadata["usage"] = {
                    "prompt_token_count": chunk.usage_metadata.prompt_token_count,
                    "candidates_token_count": chunk.usage_metadata.candidates_token_count,
                    "total_token_count": chunk.usage_metadata.total_token_count,
                }

        logger.debug(
            f"Processed Gemini stream: {len(accumulated_content)} chars, {len(accumulated_tool_calls)} tool calls"
        )
        return accumulated_content, accumulated_tool_calls, metadata

    def is_retryable_error(self, error: Exception) -> bool:
        """Check if Gemini error is retryable."""
        error_str = str(error).lower()

        # Gemini-specific retryable errors
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
            "retryerror",
            "gemini",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from Gemini error."""
        # Try to extract detailed error message
        if hasattr(error, "message"):
            return error.message

        # Check for nested error details
        if hasattr(error, "details") and error.details:
            return str(error.details)

        return str(error)

    def get_rate_limit_info(self, response: Any) -> Dict[str, Any]:
        """Extract rate limit info from Gemini response."""
        headers = self._extract_headers(response)

        rate_limit_info = {}

        # Gemini rate limit headers (if available)
        if "x-ratelimit-remaining" in headers:
            rate_limit_info["requests_remaining"] = int(
                headers["x-ratelimit-remaining"]
            )
        if "x-ratelimit-reset" in headers:
            rate_limit_info["reset_time"] = headers["x-ratelimit-reset"]

        return rate_limit_info

    async def shutdown(self):
        """Cleanup HTTP client if created."""
        if hasattr(self, "http_client") and self.http_client:
            try:
                await self.http_client.aclose()
                logger.debug("Closed Gemini HTTP client")
            except Exception as e:
                logger.error(f"Error closing Gemini HTTP client: {e}")

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to Gemini string format."""
        gemini_prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                gemini_prompt_parts.append(f"System: {content}")
            elif role == "user":
                gemini_prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                gemini_prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                gemini_prompt_parts.append(f"Tool Result: {content}")

        return "\n\n".join(gemini_prompt_parts)

    def _create_tool_config(self, function_calling_mode: str = "AUTO"):
        """Create tool configuration for Gemini."""
        try:
            from google.genai import types

            mode_map = {
                "AUTO": types.FunctionCallingConfigMode.AUTO,
                "ANY": types.FunctionCallingConfigMode.ANY,
                "NONE": types.FunctionCallingConfigMode.NONE,
            }

            mode = mode_map.get(
                function_calling_mode, types.FunctionCallingConfigMode.AUTO
            )

            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            )
        except Exception as e:
            logger.warning(f"Could not configure function calling: {e}")
            return None

    def _create_generation_config(self, tools, tool_config, **model_params):
        """Create generation configuration for Gemini."""
        try:
            from google.genai import types

            # Convert tool dictionaries to Gemini tool objects
            gemini_tools = None
            if tools:
                gemini_tools = self._convert_tools_to_gemini_objects(tools)

            config = types.GenerateContentConfig(
                temperature=model_params.get("temperature", 0.7),
                max_output_tokens=model_params.get("max_tokens", 4000),
                top_p=model_params.get("top_p", 0.9),
                top_k=model_params.get("top_k", 40),
                tools=gemini_tools,
                tool_config=tool_config,
            )

            return config
        except Exception as e:
            logger.error(f"Failed to create generation config: {e}")
            # Return minimal config
            return {
                "temperature": model_params.get("temperature", 0.7),
                "max_output_tokens": model_params.get("max_tokens", 4000),
            }

    def _convert_tools_to_gemini_objects(self, tools):
        """Convert tool dictionaries to Gemini tool objects."""
        try:
            from google.genai import types

            gemini_tools = []
            for tool in tools:
                # Extract function info from the tool dictionary
                if isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    description = tool.get("description", "")
                    parameters = tool.get("parameters", {})

                    # Create Gemini function declaration
                    function_declaration = types.FunctionDeclaration(
                        name=name, description=description, parameters=parameters
                    )

                    # Create tool with the function
                    gemini_tool = types.Tool(
                        function_declarations=[function_declaration]
                    )

                    gemini_tools.append(gemini_tool)
                else:
                    logger.warning(f"Unexpected tool format: {type(tool)}")

            return gemini_tools
        except Exception as e:
            logger.error(f"Failed to convert tools to Gemini objects: {e}")
            return []
