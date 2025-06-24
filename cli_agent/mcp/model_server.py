"""MCP Model Server implementation.

Exposes all available AI models as MCP tools for standardized access.
Supports persistent conversations with each model.
"""

import asyncio
import re
import sys
import uuid
from datetime import datetime
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


class ConversationManager:
    """Manages persistent conversations for each model."""

    def __init__(self):
        """Initialize conversation storage."""
        self.conversations: Dict[str, Dict[str, Any]] = {}

    def create_conversation(self, model_key: str, conversation_id: str = None) -> str:
        """Create a new conversation for a model.

        Args:
            model_key: The model identifier (e.g., "anthropic_claude_3_5_sonnet")
            conversation_id: Optional specific conversation ID

        Returns:
            The conversation ID
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())[:8]

        self.conversations[conversation_id] = {
            "id": conversation_id,
            "model_key": model_key,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["messages"].append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.conversations[conversation_id][
                "updated_at"
            ] = datetime.now().isoformat()

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages from a conversation."""
        if conversation_id in self.conversations:
            # Return messages in format expected by models (without timestamp)
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.conversations[conversation_id]["messages"]
            ]
        return []

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear messages from a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["messages"] = []
            self.conversations[conversation_id][
                "updated_at"
            ] = datetime.now().isoformat()
            return True
        return False

    def list_conversations(self, model_key: str = None) -> List[Dict[str, Any]]:
        """List conversations, optionally filtered by model."""
        conversations = []
        for conv_id, conv_data in self.conversations.items():
            if model_key is None or conv_data["model_key"] == model_key:
                conversations.append(
                    {
                        "id": conv_id,
                        "model_key": conv_data["model_key"],
                        "message_count": len(conv_data["messages"]),
                        "created_at": conv_data["created_at"],
                        "updated_at": conv_data["updated_at"],
                        "last_message": (
                            conv_data["messages"][-1]["content"][:100] + "..."
                            if conv_data["messages"]
                            else ""
                        ),
                    }
                )

        # Sort by most recently updated
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations


# Global conversation manager
conversation_manager = ConversationManager()


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

    # Map config provider names to actual provider names
    provider_name_map = {
        "gemini": "google",  # config uses "gemini" but create_host expects "google"
        "anthropic": "anthropic",
        "openai": "openai",
        "openrouter": "openrouter",
        "deepseek": "deepseek",
    }

    if not available_models:
        print(
            "Warning: No models available. Check your API key configuration.",
            file=sys.stderr,
        )
        print(
            "Configure API keys via environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)",
            file=sys.stderr,
        )
        return app

    # Log which providers are available
    provider_counts = {provider: len(models) for provider, models in available_models.items()}
    print(
        f"Exposing models from {len(available_models)} providers: {', '.join(f'{p} ({c})' for p, c in provider_counts.items())}",
        file=sys.stderr,
    )

    # Create tools for each available model
    total_tools = 0
    for config_provider, models in available_models.items():
        # Map to actual provider name
        actual_provider = provider_name_map.get(config_provider, config_provider)

        for model in models:
            try:
                create_model_tool(app, config_provider, actual_provider, model, config)
                total_tools += 1
            except Exception as e:
                print(
                    f"Warning: Failed to create tool for {actual_provider}:{model}: {e}",
                    file=sys.stderr,
                )

    print(f"Created MCP server with {total_tools} model tools", file=sys.stderr)
    return app


def create_model_tool(
    app: FastMCP, config_provider: str, actual_provider: str, model: str, config
):
    """Dynamically create an MCP tool for a specific model.

    Args:
        app: FastMCP server instance
        config_provider: Provider name from config (e.g., 'gemini', 'anthropic')
        actual_provider: Actual provider name for host creation (e.g., 'google', 'anthropic')
        model: Model name (e.g., 'claude-3.5-sonnet')
        config: Configuration object
    """
    tool_name = f"{config_provider}_{normalize_model_name(model)}"
    provider_model = f"{actual_provider}:{model}"

    # Create the tool function dynamically
    async def model_tool(
        messages: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        new_conversation: bool = False,
        clear_conversation: bool = False,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        f"""Use {model} model via {actual_provider} provider with persistent conversations.

        Args:
            messages: Array of conversation messages (optional if continuing conversation)
            conversation_id: ID of existing conversation to continue
            new_conversation: Force create new conversation (ignores existing ID)
            clear_conversation: Clear conversation history before processing
            system_prompt: Optional system prompt to override default
            temperature: Temperature parameter (0.0-1.0) for response randomness
            max_tokens: Maximum number of tokens to generate
            stream: Enable streaming response (returns final result)

        Returns:
            Dictionary with 'response' text and 'conversation_id'
        """
        try:
            # Handle conversation management
            current_conversation_id = conversation_id

            # Determine conversation behavior
            if new_conversation or (not conversation_id and not messages):
                # Create new conversation
                current_conversation_id = conversation_manager.create_conversation(
                    tool_name
                )
            elif conversation_id and clear_conversation:
                # Clear existing conversation
                conversation_manager.clear_conversation(conversation_id)
                current_conversation_id = conversation_id
            elif conversation_id:
                # Use existing conversation
                current_conversation_id = conversation_id
                # Validate conversation exists
                if not conversation_manager.get_conversation(conversation_id):
                    return {
                        "error": f"Conversation {conversation_id} not found",
                        "conversation_id": None,
                    }
            elif messages:
                # New conversation with initial messages
                current_conversation_id = conversation_manager.create_conversation(
                    tool_name
                )
            else:
                return {
                    "error": "Either messages or conversation_id must be provided",
                    "conversation_id": None,
                }

            # Get existing conversation messages
            existing_messages = conversation_manager.get_messages(
                current_conversation_id
            )

            # Prepare messages for the model
            if messages:
                # Add new messages to conversation
                for msg in messages:
                    conversation_manager.add_message(
                        current_conversation_id, msg["role"], msg["content"]
                    )
                # Combine existing and new messages
                processed_messages = existing_messages + messages
            else:
                # Use existing conversation messages only
                processed_messages = existing_messages

            if not processed_messages:
                return {
                    "error": "No messages to process",
                    "conversation_id": current_conversation_id,
                }

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

            # Create host for this provider-model combination
            host = config.create_host_from_provider_model(provider_model)

            # Generate response
            response = await host.generate_response(
                messages=processed_messages, stream=stream, **params
            )

            # Get response text
            response_text = response if isinstance(response, str) else str(response)

            # Add assistant response to conversation
            conversation_manager.add_message(
                current_conversation_id, "assistant", response_text
            )

            # Return response with conversation ID
            return {
                "response": response_text,
                "conversation_id": current_conversation_id,
            }

        except Exception as e:
            error_msg = f"Error using {provider_model}: {str(e)}"
            print(error_msg, file=sys.stderr)
            return {
                "error": error_msg,
                "conversation_id": (
                    current_conversation_id
                    if "current_conversation_id" in locals()
                    else None
                ),
            }

    # Set the function name and docstring
    model_tool.__name__ = tool_name
    model_tool.__doc__ = f"Use {model} model via {actual_provider} provider with persistent conversations."

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
