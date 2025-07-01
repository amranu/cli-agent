"""MCP Model Server implementation.

Exposes all available AI models as MCP tools for standardized access.
Supports persistent conversations with each model.
"""

import asyncio
import logging
import re
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging to suppress noisy MCP server messages
logging.getLogger("FastMCP.fastmcp.server.server").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

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
        "ollama": "ollama",
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

    # Log which providers are available (debug level)
    provider_counts = {
        provider: len(models) for provider, models in available_models.items()
    }
    # Debug output - only show if debug logging is enabled
    logger.debug(
        f"Exposing models from {len(available_models)} providers: {', '.join(f'{p} ({c})' for p, c in provider_counts.items())}"
    )

    # Create conversation manager
    conversation_manager = ConversationManager()

    # Get all available models for the parameter enum
    all_models = []
    for config_provider, models in available_models.items():
        actual_provider = provider_name_map.get(config_provider, config_provider)
        for model in models:
            all_models.append(f"{actual_provider}:{model}")

    # Create the docstring with available models
    available_models_str = ', '.join(all_models)
    
    @app.tool(description=f"Start or continue a chat with any available AI model. Available models: {available_models_str}")
    async def chat(
        model: str,
        message: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
        new_conversation: bool = False,
        clear_conversation: bool = False,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        f"""Start or continue a chat with any available AI model.

        Available models: {available_models_str}

        Args:
            model: Model to use in format 'provider:model' (e.g., 'anthropic:claude-3.5-sonnet')
            message: Simple string message to send (will be converted to user message)
            messages: List of conversation messages (alternative to message parameter)
            conversation_id: ID for persistent conversation (auto-generated if not provided)
            new_conversation: Start a new conversation (ignores existing conversation_id)
            clear_conversation: Clear the specified conversation
            system_prompt: Override system prompt for this request
            temperature: Temperature parameter (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Dictionary with response, conversation_id, and metadata
        """
        # Validate model format
        if ":" not in model:
            return {
                "error": f"Invalid model format '{model}'. Use 'provider:model' format (e.g., 'anthropic:claude-3.5-sonnet')",
                "available_models": all_models[:10],  # Show first 10 as examples
            }

        if model not in all_models:
            return {
                "error": f"Model '{model}' not available",
                "available_models": all_models,
            }

        # Use the existing model tool logic but with dynamic model selection
        try:
            provider_model = model
            conversation_key = (
                f"{model}_{conversation_id}" if conversation_id else model
            )

            # Handle conversation management
            if clear_conversation and conversation_id:
                conversation_manager.clear_conversation(conversation_id)
                return {
                    "message": f"Cleared conversation {conversation_id} for {model}"
                }

            if new_conversation or not conversation_id:
                conversation_id = conversation_manager.create_conversation(
                    conversation_key
                )

            # Get or create conversation
            conversation = conversation_manager.get_conversation(conversation_id)
            if not conversation:
                conversation_id = conversation_manager.create_conversation(
                    conversation_key, conversation_id
                )
                conversation = conversation_manager.get_conversation(conversation_id)

            # Handle message input - prioritize simple message over messages list
            if message and not messages:
                # Convert simple string message to user message
                messages = [{"role": "user", "content": message}]
            
            # Add new messages to conversation
            if messages:
                for msg in messages:
                    conversation_manager.add_message(
                        conversation_id, msg["role"], msg["content"]
                    )

            # Get current conversation messages
            current_messages = conversation.get("messages", [])

            # Create host and generate response
            host = config.create_host_from_provider_model(provider_model)

            # Override parameters if provided
            if system_prompt:
                # Add system message to beginning
                current_messages.insert(0, {"role": "system", "content": system_prompt})

            # Generate response
            response_content = await host.generate_response(
                current_messages, stream=stream
            )

            # Add response to conversation
            conversation_manager.add_message(
                conversation_id, "assistant", response_content
            )

            return {
                "response": response_content,
                "conversation_id": conversation_id,
                "model": model,
                "message_count": len(conversation.get("messages", [])),
                "created": conversation.get("created"),
            }

        except Exception as e:
            return {
                "error": f"Failed to generate response with {model}: {str(e)}",
                "conversation_id": conversation_id,
                "model": model,
            }

    logger.debug(
        f"Created MCP server with consolidated chat tool supporting {len(all_models)} models"
    )
    return app


if __name__ == "__main__":
    # Allow running this module directly for testing
    server = create_model_server()
    if "--stdio" in sys.argv:
        asyncio.run(server.run_stdio_async())
    else:
        logger.debug("Starting MCP model server on stdio transport")
        asyncio.run(server.run_stdio_async())
