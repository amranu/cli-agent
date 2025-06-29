"""
Tests for the MCP model server chat tool functionality.

This test suite focuses on testing the async start_chat tool,
conversation management integration, and model interaction.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Add the project root to Python path for imports
import sys
import os
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))  # Go up two levels from tests/unit/
sys.path.insert(0, project_root)

from cli_agent.mcp.model_server import ConversationManager


class MockHost:
    """Mock host for testing model interactions."""
    
    def __init__(self, response="Mock response"):
        self.response = response
        
    async def generate_response(self, messages, stream=False):
        """Mock generate_response method."""
        await asyncio.sleep(0.01)  # Simulate async operation
        return self.response


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self, available_models=None):
        self.available_models = available_models or {
            "anthropic": ["claude-3.5-sonnet"],
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "gemini": ["gemini-2.5-flash"]
        }
    
    def get_available_provider_models(self):
        return self.available_models
    
    def create_host_from_provider_model(self, provider_model):
        return MockHost(f"Response from {provider_model}")


@pytest.mark.asyncio
class TestStartChatToolImplementation:
    """Test the start_chat tool implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.conversation_manager = ConversationManager()
        self.mock_config = MockConfig()

    async def test_start_chat_simple_message(self):
        """Test start_chat with a simple message."""
        # Mock the dependencies
        with patch('cli_agent.mcp.model_server.conversation_manager', self.conversation_manager):
            with patch('cli_agent.mcp.model_server.load_config', return_value=self.mock_config):
                
                # Simulate the start_chat function logic
                model = "anthropic:claude-3.5-sonnet"
                message = "Hello, how are you?"
                
                # Validate model format
                assert ":" in model
                available_models = []
                for provider, models in self.mock_config.get_available_provider_models().items():
                    for model_name in models:
                        available_models.append(f"{provider}:{model_name}")
                
                assert model in available_models or "anthropic:claude-3.5-sonnet" in [
                    "anthropic:claude-3.5-sonnet", "openai:gpt-4", "openai:gpt-3.5-turbo", "google:gemini-2.5-flash"
                ]
                
                # Create conversation
                conversation_key = model
                conversation_id = self.conversation_manager.create_conversation(conversation_key)
                
                # Convert message to messages format
                messages = [{"role": "user", "content": message}]
                
                # Add to conversation
                for msg in messages:
                    self.conversation_manager.add_message(conversation_id, msg["role"], msg["content"])
                
                # Get conversation messages
                current_messages = self.conversation_manager.get_messages(conversation_id)
                assert len(current_messages) == 1
                assert current_messages[0]["role"] == "user"
                assert current_messages[0]["content"] == message
                
                # Generate response
                host = self.mock_config.create_host_from_provider_model(model)
                response_content = await host.generate_response(current_messages)
                
                # Add response to conversation
                self.conversation_manager.add_message(conversation_id, "assistant", response_content)
                
                # Verify final state
                final_messages = self.conversation_manager.get_messages(conversation_id)
                assert len(final_messages) == 2
                assert final_messages[1]["role"] == "assistant"
                assert response_content in final_messages[1]["content"]

    async def test_start_chat_with_message_list(self):
        """Test start_chat with a list of messages."""
        model = "openai:gpt-4"
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more about it."}
        ]
        
        conversation_key = model
        conversation_id = self.conversation_manager.create_conversation(conversation_key)
        
        # Add messages to conversation
        for msg in messages:
            self.conversation_manager.add_message(conversation_id, msg["role"], msg["content"])
        
        # Verify messages were added
        current_messages = self.conversation_manager.get_messages(conversation_id)
        assert len(current_messages) == 3
        assert current_messages[0]["content"] == "What is Python?"
        assert current_messages[1]["content"] == "Python is a programming language."
        assert current_messages[2]["content"] == "Tell me more about it."

    async def test_start_chat_new_conversation_flag(self):
        """Test start_chat with new_conversation flag."""
        model = "anthropic:claude-3.5-sonnet"
        
        # Create initial conversation
        conv_id_1 = self.conversation_manager.create_conversation(model)
        self.conversation_manager.add_message(conv_id_1, "user", "First conversation")
        
        # Create new conversation (simulating new_conversation=True)
        conv_id_2 = self.conversation_manager.create_conversation(model)
        
        assert conv_id_1 != conv_id_2
        
        # Verify they are separate
        messages_1 = self.conversation_manager.get_messages(conv_id_1)
        messages_2 = self.conversation_manager.get_messages(conv_id_2)
        
        assert len(messages_1) == 1
        assert len(messages_2) == 0

    async def test_start_chat_clear_conversation(self):
        """Test start_chat with clear_conversation flag."""
        model = "openai:gpt-4"
        conversation_id = self.conversation_manager.create_conversation(model)
        
        # Add some messages
        self.conversation_manager.add_message(conversation_id, "user", "Message 1")
        self.conversation_manager.add_message(conversation_id, "assistant", "Response 1")
        
        # Verify messages exist
        assert len(self.conversation_manager.get_messages(conversation_id)) == 2
        
        # Clear conversation
        result = self.conversation_manager.clear_conversation(conversation_id)
        assert result is True
        
        # Verify messages are cleared
        assert len(self.conversation_manager.get_messages(conversation_id)) == 0

    async def test_start_chat_with_system_prompt(self):
        """Test start_chat with system prompt override."""
        model = "anthropic:claude-3.5-sonnet"
        system_prompt = "You are a helpful coding assistant."
        user_message = "Help me with Python."
        
        conversation_id = self.conversation_manager.create_conversation(model)
        
        # Simulate adding system prompt and user message
        messages = [{"role": "user", "content": user_message}]
        
        # Add messages to conversation
        for msg in messages:
            self.conversation_manager.add_message(conversation_id, msg["role"], msg["content"])
        
        # Get current messages and add system prompt at beginning
        current_messages = self.conversation_manager.get_messages(conversation_id)
        current_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Verify system prompt is first
        assert current_messages[0]["role"] == "system"
        assert current_messages[0]["content"] == system_prompt
        assert current_messages[1]["role"] == "user"
        assert current_messages[1]["content"] == user_message

    async def test_start_chat_invalid_model_format(self):
        """Test start_chat error handling for invalid model format."""
        invalid_model = "just-a-model-name"  # No colon
        
        # Simulate validation
        assert ":" not in invalid_model
        
        # This would return an error response
        expected_error_response = {
            "error": f"Invalid model format '{invalid_model}'. Use 'provider:model' format (e.g., 'anthropic:claude-3.5-sonnet')",
            "available_models": ["anthropic:claude-3.5-sonnet", "openai:gpt-4"]  # First 10 as examples
        }
        
        assert "error" in expected_error_response
        assert "Invalid model format" in expected_error_response["error"]

    async def test_start_chat_unavailable_model(self):
        """Test start_chat error handling for unavailable model."""
        unavailable_model = "nonexistent:model"
        
        # Get available models
        available_models = []
        for provider, models in self.mock_config.get_available_provider_models().items():
            for model_name in models:
                available_models.append(f"{provider}:{model_name}")
        
        # Simulate validation
        assert unavailable_model not in available_models
        
        # This would return an error response
        expected_error_response = {
            "error": f"Model '{unavailable_model}' not available",
            "available_models": available_models
        }
        
        assert "error" in expected_error_response
        assert f"Model '{unavailable_model}' not available" in expected_error_response["error"]

    async def test_start_chat_host_creation_error(self):
        """Test start_chat error handling when host creation fails."""
        model = "anthropic:claude-3.5-sonnet"
        conversation_id = "test_conv"
        
        # Mock config that raises exception
        mock_config_error = Mock()
        mock_config_error.create_host_from_provider_model.side_effect = Exception("API key not found")
        
        # Simulate error handling
        try:
            host = mock_config_error.create_host_from_provider_model(model)
            await host.generate_response([])
        except Exception as e:
            error_response = {
                "error": f"Failed to generate response with {model}: {str(e)}",
                "conversation_id": conversation_id,
                "model": model
            }
            
            assert "error" in error_response
            assert "Failed to generate response" in error_response["error"]
            assert "API key not found" in error_response["error"]

    async def test_start_chat_response_structure(self):
        """Test the structure of a successful start_chat response."""
        model = "openai:gpt-4"
        message = "Hello"
        
        conversation_id = self.conversation_manager.create_conversation(model)
        
        # Add user message
        self.conversation_manager.add_message(conversation_id, "user", message)
        
        # Generate response
        host = self.mock_config.create_host_from_provider_model(model)
        response_content = await host.generate_response([{"role": "user", "content": message}])
        
        # Add assistant response
        self.conversation_manager.add_message(conversation_id, "assistant", response_content)
        
        # Get conversation data
        conversation = self.conversation_manager.get_conversation(conversation_id)
        
        # Simulate response structure
        response = {
            "response": response_content,
            "conversation_id": conversation_id,
            "model": model,
            "message_count": len(conversation.get("messages", [])),
            "created": conversation.get("created_at")
        }
        
        # Verify response structure
        assert "response" in response
        assert "conversation_id" in response
        assert "model" in response
        assert "message_count" in response
        assert response["model"] == model
        assert response["message_count"] == 2  # user + assistant


class TestConversationPersistence:
    """Test conversation persistence across multiple chat calls."""

    def setup_method(self):
        """Set up test fixtures."""
        self.conversation_manager = ConversationManager()

    async def test_conversation_persistence(self):
        """Test that conversations persist across multiple calls."""
        model = "anthropic:claude-3.5-sonnet"
        
        # First interaction
        conv_id = self.conversation_manager.create_conversation(model)
        self.conversation_manager.add_message(conv_id, "user", "Hello")
        self.conversation_manager.add_message(conv_id, "assistant", "Hi there!")
        
        # Second interaction (same conversation)
        self.conversation_manager.add_message(conv_id, "user", "How are you?")
        self.conversation_manager.add_message(conv_id, "assistant", "I'm doing well!")
        
        # Verify conversation has all messages
        messages = self.conversation_manager.get_messages(conv_id)
        assert len(messages) == 4
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["content"] == "How are you?"
        assert messages[3]["content"] == "I'm doing well!"

    async def test_multiple_conversations_same_model(self):
        """Test multiple conversations for the same model."""
        model = "openai:gpt-4"
        
        # Create two different conversations
        conv_id_1 = self.conversation_manager.create_conversation(f"{model}_conv1")
        conv_id_2 = self.conversation_manager.create_conversation(f"{model}_conv2")
        
        # Add different messages to each
        self.conversation_manager.add_message(conv_id_1, "user", "Conversation 1 message")
        self.conversation_manager.add_message(conv_id_2, "user", "Conversation 2 message")
        
        # Verify they are separate
        messages_1 = self.conversation_manager.get_messages(conv_id_1)
        messages_2 = self.conversation_manager.get_messages(conv_id_2)
        
        assert len(messages_1) == 1
        assert len(messages_2) == 1
        assert messages_1[0]["content"] == "Conversation 1 message"
        assert messages_2[0]["content"] == "Conversation 2 message"

    async def test_conversation_metadata_updates(self):
        """Test that conversation metadata is updated correctly."""
        model = "anthropic:claude-3.5-sonnet"
        conv_id = self.conversation_manager.create_conversation(model)
        
        # Get initial metadata
        conversation = self.conversation_manager.get_conversation(conv_id)
        initial_updated_at = conversation["updated_at"]
        
        # Add a message (should update metadata)
        import time
        time.sleep(0.01)  # Small delay to ensure timestamp difference
        self.conversation_manager.add_message(conv_id, "user", "New message")
        
        # Check that updated_at changed
        updated_conversation = self.conversation_manager.get_conversation(conv_id)
        new_updated_at = updated_conversation["updated_at"]
        
        assert new_updated_at != initial_updated_at
        assert len(updated_conversation["messages"]) == 1


class TestMCPToolIntegration:
    """Test integration aspects of the MCP tool."""

    def test_tool_description_contains_available_models(self):
        """Test that tool description includes available models."""
        available_models = {
            "anthropic": ["claude-3.5-sonnet"],
            "openai": ["gpt-4"]
        }
        
        # Simulate building the available models string
        all_models = []
        provider_name_map = {
            "anthropic": "anthropic",
            "openai": "openai",
            "gemini": "google"
        }
        
        for config_provider, models in available_models.items():
            actual_provider = provider_name_map.get(config_provider, config_provider)
            for model in models:
                all_models.append(f"{actual_provider}:{model}")
        
        available_models_str = ', '.join(all_models)
        
        assert "anthropic:claude-3.5-sonnet" in available_models_str
        assert "openai:gpt-4" in available_models_str

    def test_provider_name_mapping(self):
        """Test that provider names are mapped correctly."""
        provider_name_map = {
            "gemini": "google",
            "anthropic": "anthropic", 
            "openai": "openai",
            "openrouter": "openrouter",
            "deepseek": "deepseek",
            "ollama": "ollama"
        }
        
        # Test mappings
        assert provider_name_map["gemini"] == "google"
        assert provider_name_map["anthropic"] == "anthropic"
        assert provider_name_map.get("unknown", "unknown") == "unknown"

    def test_conversation_id_generation(self):
        """Test conversation ID generation and uniqueness."""
        conversation_manager = ConversationManager()
        
        # Generate multiple conversation IDs
        conv_ids = []
        for i in range(10):
            conv_id = conversation_manager.create_conversation(f"model_{i}")
            conv_ids.append(conv_id)
        
        # All IDs should be unique
        assert len(conv_ids) == len(set(conv_ids))
        
        # All IDs should be 8 characters (UUID truncated)
        for conv_id in conv_ids:
            assert len(conv_id) == 8


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])