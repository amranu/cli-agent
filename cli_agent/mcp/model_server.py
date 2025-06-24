"""MCP Model Server implementation.

Exposes all available AI models as MCP tools for standardized access.
"""

import asyncio
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except ImportError:
    print(
        "Error: FastMCP not installed. Please install with: pip install fastmcp",
        file=sys.stderr,
    )
    FastMCP = None

from config import load_config


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for use as tool name.

    Converts model names to valid tool names by replacing special characters
    with underscores and removing invalid characters.
    """
    # Replace special characters with underscores
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
    # Remove consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized


def create_model_server() -> FastMCP:
    """Create and configure the MCP model server.

    Returns:
        FastMCP: Configured MCP server with model tools

    Raises:
        ImportError: If FastMCP is not available
        Exception: If configuration loading fails
    """
    if FastMCP is None:
        raise ImportError(
            "FastMCP not installed. Please install with: pip install fastmcp"
        )

    # Create MCP server
    app = FastMCP("AI Models Server")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        raise Exception(f"Failed to load configuration: {e}")

    # Get available models from configuration
    available_models = config.get_available_provider_models()

    if not available_models:
        print(
            "Warning: No models available. Check your API key configuration.",
            file=sys.stderr,
        )

    # Create tools for each available model
    total_tools = 0
    for provider, models in available_models.items():
        for model in models:
            try:
                create_model_tool(app, provider, model, config)
                total_tools += 1
            except Exception as e:
                print(
                    f"Warning: Failed to create tool for {provider}:{model}: {e}",
                    file=sys.stderr,
                )

    print(f"Created MCP server with {total_tools} model tools", file=sys.stderr)
    return app


def create_model_tool(app: FastMCP, provider: str, model: str, config):
    """Dynamically create an MCP tool for a specific model.

    Args:
        app: FastMCP server instance
        provider: Provider name (e.g., 'anthropic', 'openai')
        model: Model name (e.g., 'claude-3.5-sonnet')
        config: Configuration object
    """
    tool_name = f"{provider}_{normalize_model_name(model)}"
    provider_model = f"{provider}:{model}"

    # Create the tool function dynamically
    async def model_tool(
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        f"""Use {model} model via {provider} provider.

        Args:
            messages: Array of conversation messages with 'role' and 'content' keys
            system_prompt: Optional system prompt to override default
            temperature: Temperature parameter (0.0-1.0) for response randomness
            max_tokens: Maximum number of tokens to generate
            stream: Enable streaming response (returns final result)

        Returns:
            Generated response text from the model
        """
        try:
            # Create host for this provider-model combination
            host = config.create_host_from_provider_model(provider_model)

            # Prepare messages
            processed_messages = messages.copy()

            # Add system prompt if provided
            if system_prompt:
                # Check if first message is already a system message
                if processed_messages and processed_messages[0].get("role") == "system":
                    # Replace existing system message
                    processed_messages[0] = {"role": "system", "content": system_prompt}
                else:
                    # Add new system message at the beginning
                    processed_messages = [
                        {"role": "system", "content": system_prompt}
                    ] + processed_messages

            # Set model parameters
            params = {}
            if temperature is not None:
                params["temperature"] = min(
                    max(temperature, 0.0), 1.0
                )  # Clamp to valid range
            if max_tokens is not None:
                params["max_tokens"] = max(
                    1, int(max_tokens)
                )  # Ensure positive integer

            # Generate response
            response = await host.generate_response(
                messages=processed_messages, stream=stream, **params
            )

            # Return the response text
            if isinstance(response, str):
                return response
            else:
                # Handle case where response might be a different format
                return str(response)

        except Exception as e:
            error_msg = f"Error using {provider_model}: {str(e)}"
            print(error_msg, file=sys.stderr)
            return error_msg

    # Set the function name and docstring
    model_tool.__name__ = tool_name
    model_tool.__doc__ = f"""Use {model} model via {provider} provider.

    This tool provides access to the {model} model through the {provider} API.

    Args:
        messages: Array of conversation messages with 'role' and 'content' keys
        system_prompt: Optional system prompt to override default behavior
        temperature: Temperature parameter (0.0-1.0) for controlling response randomness
        max_tokens: Maximum number of tokens to generate in the response
        stream: Enable streaming response (tool still returns final complete result)

    Returns:
        Generated response text from the model

    Example:
        {{"messages": [{{"role": "user", "content": "Hello"}}], "temperature": 0.7}}
    """

    # Register the tool with the MCP server
    app.tool(name=tool_name)(model_tool)


if __name__ == "__main__":
    # Allow running this module directly for testing
    server = create_model_server()
    if "--stdio" in sys.argv:
        asyncio.run(server.run_stdio_async())
    else:
        print("Starting MCP model server on stdio transport", file=sys.stderr)
        asyncio.run(server.run_stdio_async())
