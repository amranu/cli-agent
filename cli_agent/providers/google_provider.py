"""Google Gemini API provider implementation."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from cli_agent.core.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._models_cache = None
        self._models_cache_time = 0
        self._cache_duration = 3600  # 1 hour cache

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

                # Register with global HTTP client manager for centralized cleanup
                self._register_http_client(http_client)

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

        # Convert messages to Gemini conversation format
        gemini_contents = self._convert_messages_to_gemini_format(messages)

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
                    contents=gemini_contents,
                    config=config,
                )
                # Gemini streaming returns a regular generator, not async
                return response
            else:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=gemini_contents,
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
        # Wrap with interrupt checking
        interruptible_response = self.make_sync_streaming_interruptible(
            response, "Gemini streaming"
        )

        for chunk in interruptible_response:
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

    def _convert_messages_to_gemini_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[Any]:
        """Convert messages to proper Gemini conversation format."""
        try:
            from google.genai import types
        except ImportError:
            # Fallback to string format if types not available
            return self._convert_messages_to_string_format(messages)

        gemini_contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Store system message for later - Gemini handles system instructions separately
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += f"\n{content}"
            elif role == "user":
                gemini_contents.append(
                    types.Content(role="user", parts=[types.Part(text=content)])
                )
            elif role == "assistant":
                # Handle assistant messages
                parts = []
                if content:
                    parts.append(types.Part(text=content))

                # Handle tool calls if present
                tool_calls = msg.get("tool_calls", [])
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        func = tool_call["function"]
                        func_name = func.get("name", "")
                        func_args = func.get("arguments", "{}")

                        # Parse arguments if they're a string
                        if isinstance(func_args, str):
                            try:
                                import json

                                func_args = json.loads(func_args)
                            except:
                                func_args = {}

                        # Create function call part
                        function_call = types.FunctionCall(
                            name=func_name, args=func_args
                        )
                        parts.append(types.Part(function_call=function_call))

                if parts:
                    gemini_contents.append(
                        types.Content(
                            role="model",  # Gemini uses "model" for assistant
                            parts=parts,
                        )
                    )
            elif role == "tool":
                # Handle tool results
                tool_call_id = msg.get("tool_call_id", "")
                tool_name = msg.get("name", "unknown_tool")

                # Create function response
                function_response = types.FunctionResponse(
                    name=tool_name, response={"result": content}
                )

                gemini_contents.append(
                    types.Content(
                        role="function",
                        parts=[types.Part(function_response=function_response)],
                    )
                )

        # If we have a system instruction, we should handle it separately
        # For now, prepend it to the first user message if present
        if system_instruction and gemini_contents:
            first_content = gemini_contents[0]
            if first_content.role == "user" and first_content.parts:
                # Prepend system instruction to first user message
                original_text = (
                    first_content.parts[0].text
                    if hasattr(first_content.parts[0], "text")
                    else ""
                )
                combined_text = f"System: {system_instruction}\n\nUser: {original_text}"
                gemini_contents[0] = types.Content(
                    role="user", parts=[types.Part(text=combined_text)]
                )

        return gemini_contents

    def _convert_messages_to_string_format(self, messages: List[Dict[str, Any]]) -> str:
        """Fallback: Convert messages to Gemini string format."""
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

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from Google Gemini API.

        Returns:
            List of model dictionaries with id, name, context_length, and description
        """
        # Check cache first
        current_time = time.time()
        if (
            self._models_cache is not None
            and current_time - self._models_cache_time < self._cache_duration
        ):
            logger.debug("Returning cached Google models")
            return self._models_cache

        try:
            # Use httpx for the models API call since it's a simple GET request
            async with httpx.AsyncClient() as http_client:
                # Add retry logic for robustness
                from cli_agent.utils.retry import retry_with_backoff

                async def fetch_models():
                    return await http_client.get(
                        f"{self.base_url}/models",
                        params={"key": self.api_key},
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
                        f"Google models API returned status {response.status_code}"
                    )
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = f"{error_msg}: {error_data['error']['message']}"
                    except:
                        error_msg = f"{error_msg}: {response.text}"
                    logger.error(error_msg)
                    return self._get_fallback_models()

                data = response.json()
                models = data.get("models", [])

                # Process models to extract essential info
                processed_models = []
                for model in models:
                    model_name = model.get("name", "")
                    model_id = model_name.replace("models/", "") if model_name else ""

                    # Only include generative models (filter out embedding, etc.)
                    supported_methods = model.get("supportedGenerationMethods", [])
                    if "generateContent" not in supported_methods:
                        continue

                    # Filter for Gemini models
                    if not model_id.lower().startswith("gemini"):
                        continue

                    # Extract context length
                    input_token_limit = model.get("inputTokenLimit", 0)
                    output_token_limit = model.get("outputTokenLimit", 0)
                    context_length = (
                        input_token_limit
                        if input_token_limit > 0
                        else self._estimate_context_length(model_id)
                    )

                    model_info = {
                        "id": model_id,
                        "name": model.get(
                            "displayName", model_id.replace("-", " ").title()
                        ),
                        "context_length": context_length,
                        "description": model.get(
                            "description", f"Google Gemini model: {model_id}"
                        ),
                        "input_token_limit": input_token_limit,
                        "output_token_limit": output_token_limit,
                        "supported_methods": supported_methods,
                        "version": model.get("version", ""),
                    }

                    processed_models.append(model_info)

                # Sort by name for consistency
                processed_models.sort(key=lambda x: x["name"].lower())

                # Update cache
                self._models_cache = processed_models
                self._models_cache_time = current_time

                logger.info(f"Google provider found {len(processed_models)} models")
                return processed_models

        except Exception as e:
            logger.error(f"Failed to fetch Google models: {e}")
            return self._get_fallback_models()

    def _estimate_context_length(self, model_id: str) -> int:
        """Estimate context length based on model ID."""
        model_id_lower = model_id.lower()

        # Estimate based on known Gemini model patterns
        if "2.5" in model_id_lower:
            if "flash" in model_id_lower:
                return 1000000  # Gemini 2.5 Flash typically has 1M context
            elif "pro" in model_id_lower:
                return 2000000  # Gemini 2.5 Pro has 2M context
        elif "1.5" in model_id_lower:
            if "flash" in model_id_lower:
                return 1000000  # Gemini 1.5 Flash has 1M context
            elif "pro" in model_id_lower:
                return 2000000  # Gemini 1.5 Pro has 2M context

        return 32000  # Conservative default

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Return hardcoded fallback models if API fails."""
        fallback_models = [
            {
                "id": "gemini-2.5-flash",
                "name": "Gemini 2.5 Flash",
                "context_length": 1000000,
                "description": "Google's fastest and most cost-effective model",
                "input_token_limit": 1000000,
                "output_token_limit": 8192,
            },
            {
                "id": "gemini-2.5-pro",
                "name": "Gemini 2.5 Pro",
                "context_length": 2000000,
                "description": "Google's most capable model for complex reasoning tasks",
                "input_token_limit": 2000000,
                "output_token_limit": 8192,
            },
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "context_length": 2000000,
                "description": "Google's powerful model with large context window",
                "input_token_limit": 2000000,
                "output_token_limit": 8192,
            },
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "context_length": 1000000,
                "description": "Fast and efficient model for everyday tasks",
                "input_token_limit": 1000000,
                "output_token_limit": 8192,
            },
        ]

        logger.info(f"Using fallback Google models: {len(fallback_models)} models")
        return fallback_models

    async def get_available_models_summary(self) -> List[str]:
        """Get a simple list of model IDs from Google.

        Returns:
            List of model ID strings
        """
        models = await self.get_available_models()
        return [model["id"] for model in models]

    @staticmethod
    async def fetch_available_models_static(api_key: str) -> List[Dict[str, Any]]:
        """Static method to fetch models without persisting client state.

        Args:
            api_key: Google API key

        Returns:
            List of model dictionaries
        """
        try:
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": api_key},
                    timeout=10.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Google models API returned status {response.status_code}"
                    )
                    return []

                data = response.json()
                models = data.get("models", [])

                # Process models to extract essential info
                processed_models = []
                for model in models:
                    model_name = model.get("name", "")
                    model_id = model_name.replace("models/", "") if model_name else ""

                    # Only include generative models and Gemini models
                    supported_methods = model.get("supportedGenerationMethods", [])
                    if (
                        "generateContent" not in supported_methods
                        or not model_id.lower().startswith("gemini")
                    ):
                        continue

                    input_token_limit = model.get("inputTokenLimit", 0)
                    context_length = (
                        input_token_limit if input_token_limit > 0 else 32000
                    )

                    model_info = {
                        "id": model_id,
                        "name": model.get(
                            "displayName", model_id.replace("-", " ").title()
                        ),
                        "context_length": context_length,
                        "description": model.get(
                            "description", f"Google Gemini model: {model_id}"
                        ),
                    }

                    processed_models.append(model_info)

                # Sort by name for consistency
                processed_models.sort(key=lambda x: x["name"].lower())

                logger.info(f"Google static fetch found {len(processed_models)} models")
                return processed_models

        except Exception as e:
            logger.error(f"Failed to fetch Google models (static): {e}")
            return []
