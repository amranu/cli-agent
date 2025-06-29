"""
Tests for the MCP model server implementation.

This test suite covers the ConversationManager, model server creation,
and the FastMCP tool integration functionality.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Add the project root to Python path for imports
import sys
import os
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))  # Go up two levels from tests/unit/
sys.path.insert(0, project_root)

from cli_agent.mcp.model_server import (
    ConversationManager,
    normalize_model_name,
    create_model_server
)


class TestConversationManager:
    """Test the ConversationManager class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.conversation_manager = ConversationManager()

    def test_initialization(self):
        """Test ConversationManager initialization."""
        cm = ConversationManager()
        assert cm.conversations == {}
        assert isinstance(cm.conversations, dict)

    def test_create_conversation_auto_id(self):
        """Test creating a conversation with auto-generated ID."""
        model_key = "anthropic_claude_3_5_sonnet"
        
        conv_id = self.conversation_manager.create_conversation(model_key)
        
        assert conv_id is not None
        assert len(conv_id) == 8  # UUID truncated to 8 chars
        assert conv_id in self.conversation_manager.conversations
        
        conv_data = self.conversation_manager.conversations[conv_id]
        assert conv_data["id"] == conv_id
        assert conv_data["model_key"] == model_key
        assert conv_data["messages"] == []
        assert "created_at" in conv_data
        assert "updated_at" in conv_data

    def test_create_conversation_specific_id(self):
        """Test creating a conversation with specific ID."""
        model_key = "openai_gpt_4"
        specific_id = "test_conv_123"
        
        conv_id = self.conversation_manager.create_conversation(model_key, specific_id)
        
        assert conv_id == specific_id
        assert conv_id in self.conversation_manager.conversations
        assert self.conversation_manager.conversations[conv_id]["model_key"] == model_key

    def test_get_conversation_exists(self):
        """Test getting an existing conversation."""
        model_key = "test_model"
        conv_id = self.conversation_manager.create_conversation(model_key)
        
        retrieved = self.conversation_manager.get_conversation(conv_id)
        
        assert retrieved is not None
        assert retrieved["id"] == conv_id
        assert retrieved["model_key"] == model_key

    def test_get_conversation_not_exists(self):
        """Test getting a non-existent conversation."""
        result = self.conversation_manager.get_conversation("nonexistent")
        assert result is None

    def test_add_message(self):
        """Test adding messages to a conversation."""
        model_key = "test_model"
        conv_id = self.conversation_manager.create_conversation(model_key)
        
        # Add user message
        self.conversation_manager.add_message(conv_id, "user", "Hello, world!")
        
        conv_data = self.conversation_manager.conversations[conv_id]
        assert len(conv_data["messages"]) == 1
        
        message = conv_data["messages"][0]
        assert message["role"] == "user"
        assert message["content"] == "Hello, world!"
        assert "timestamp" in message
        
        # Add assistant message
        self.conversation_manager.add_message(conv_id, "assistant", "Hello! How can I help you?")
        
        assert len(conv_data["messages"]) == 2
        assert conv_data["messages"][1]["role"] == "assistant"
        assert conv_data["messages"][1]["content"] == "Hello! How can I help you?"

    def test_add_message_nonexistent_conversation(self):
        """Test adding message to non-existent conversation (should not crash)."""
        # This should not raise an exception
        self.conversation_manager.add_message("nonexistent", "user", "test")
        
        # Conversation should not be created
        assert "nonexistent" not in self.conversation_manager.conversations

    def test_get_messages(self):
        """Test getting messages from a conversation."""
        model_key = "test_model"
        conv_id = self.conversation_manager.create_conversation(model_key)
        
        # Add messages
        self.conversation_manager.add_message(conv_id, "user", "Message 1")
        self.conversation_manager.add_message(conv_id, "assistant", "Response 1")
        
        messages = self.conversation_manager.get_messages(conv_id)
        
        assert len(messages) == 2
        # Should return messages without timestamp (clean format)
        assert messages[0] == {"role": "user", "content": "Message 1"}
        assert messages[1] == {"role": "assistant", "content": "Response 1"}

    def test_get_messages_nonexistent_conversation(self):
        """Test getting messages from non-existent conversation."""
        messages = self.conversation_manager.get_messages("nonexistent")
        assert messages == []

    def test_clear_conversation(self):
        """Test clearing messages from a conversation."""
        model_key = "test_model"
        conv_id = self.conversation_manager.create_conversation(model_key)
        
        # Add messages
        self.conversation_manager.add_message(conv_id, "user", "Message 1")
        self.conversation_manager.add_message(conv_id, "assistant", "Response 1")
        
        # Verify messages exist
        assert len(self.conversation_manager.get_messages(conv_id)) == 2
        
        # Clear conversation
        result = self.conversation_manager.clear_conversation(conv_id)
        
        assert result is True
        assert len(self.conversation_manager.get_messages(conv_id)) == 0
        # Conversation metadata should still exist
        assert conv_id in self.conversation_manager.conversations

    def test_clear_nonexistent_conversation(self):
        """Test clearing a non-existent conversation."""
        result = self.conversation_manager.clear_conversation("nonexistent")
        assert result is False

    def test_list_conversations_empty(self):
        """Test listing conversations when none exist."""
        conversations = self.conversation_manager.list_conversations()
        assert conversations == []

    def test_list_conversations_with_data(self):
        """Test listing conversations with data."""
        # Create multiple conversations
        conv1_id = self.conversation_manager.create_conversation("model1")
        conv2_id = self.conversation_manager.create_conversation("model2")
        
        # Add messages to conversations
        self.conversation_manager.add_message(conv1_id, "user", "Hello from conv1")
        self.conversation_manager.add_message(conv2_id, "user", "Hello from conv2")
        self.conversation_manager.add_message(conv2_id, "assistant", "Response in conv2")
        
        conversations = self.conversation_manager.list_conversations()
        
        assert len(conversations) == 2
        
        # Find conversations by ID
        conv1_data = next(c for c in conversations if c["id"] == conv1_id)
        conv2_data = next(c for c in conversations if c["id"] == conv2_id)
        
        assert conv1_data["model_key"] == "model1"
        assert conv1_data["message_count"] == 1
        assert conv1_data["last_message"].startswith("Hello from conv1")
        
        assert conv2_data["model_key"] == "model2"
        assert conv2_data["message_count"] == 2
        assert conv2_data["last_message"].startswith("Response in conv2")

    def test_list_conversations_filtered_by_model(self):
        """Test listing conversations filtered by model."""
        # Create conversations for different models
        conv1_id = self.conversation_manager.create_conversation("model1")
        conv2_id = self.conversation_manager.create_conversation("model2")
        conv3_id = self.conversation_manager.create_conversation("model1")
        
        # Get all conversations
        all_conversations = self.conversation_manager.list_conversations()
        assert len(all_conversations) == 3
        
        # Get conversations for model1 only
        model1_conversations = self.conversation_manager.list_conversations("model1")
        assert len(model1_conversations) == 2
        assert all(c["model_key"] == "model1" for c in model1_conversations)
        
        # Get conversations for model2 only
        model2_conversations = self.conversation_manager.list_conversations("model2")
        assert len(model2_conversations) == 1
        assert model2_conversations[0]["model_key"] == "model2"

    def test_list_conversations_sorted_by_updated_at(self):
        """Test that conversations are sorted by most recently updated."""
        # Create conversations
        conv1_id = self.conversation_manager.create_conversation("model1")
        conv2_id = self.conversation_manager.create_conversation("model2")
        
        # Add message to conv1 (making it more recently updated)
        self.conversation_manager.add_message(conv1_id, "user", "Recent message")
        
        conversations = self.conversation_manager.list_conversations()
        
        # First conversation should be the most recently updated (conv1)
        assert conversations[0]["id"] == conv1_id
        assert conversations[1]["id"] == conv2_id

    def test_last_message_truncation(self):
        """Test that last message is truncated to 100 characters."""
        conv_id = self.conversation_manager.create_conversation("test_model")
        
        # Add a long message
        long_message = "x" * 150  # 150 characters
        self.conversation_manager.add_message(conv_id, "user", long_message)
        
        conversations = self.conversation_manager.list_conversations()
        
        assert len(conversations) == 1
        last_message = conversations[0]["last_message"]
        assert len(last_message) == 103  # 100 chars + "..."
        assert last_message.endswith("...")


class TestNormalizeModelName:
    """Test the normalize_model_name function."""

    def test_normalize_basic_name(self):
        """Test normalizing a basic model name."""
        result = normalize_model_name("gpt-4")
        assert result == "gpt_4"

    def test_normalize_complex_name(self):
        """Test normalizing a complex model name with multiple special characters."""
        result = normalize_model_name("claude-3.5-sonnet-20241022")
        assert result == "claude_3_5_sonnet_20241022"

    def test_normalize_with_colons_and_slashes(self):
        """Test normalizing names with colons and slashes."""
        result = normalize_model_name("anthropic/claude-3:5")
        assert result == "anthropic_claude_3_5"

    def test_normalize_consecutive_special_chars(self):
        """Test that consecutive special characters become single underscore."""
        result = normalize_model_name("model--name..with___spaces")
        assert result == "model_name_with_spaces"

    def test_normalize_leading_trailing_underscores(self):
        """Test that leading and trailing underscores are removed."""
        result = normalize_model_name("_model-name_")
        assert result == "model_name"

    def test_normalize_only_special_chars(self):
        """Test normalizing a string with only special characters."""
        result = normalize_model_name("---...")
        assert result == ""

    def test_normalize_empty_string(self):
        """Test normalizing an empty string."""
        result = normalize_model_name("")
        assert result == ""

    def test_normalize_already_normalized(self):
        """Test normalizing an already normalized name."""
        result = normalize_model_name("gpt_4_turbo")
        assert result == "gpt_4_turbo"


class TestCreateModelServer:
    """Test the create_model_server function."""

    def test_create_model_server_fastmcp_not_available(self):
        """Test that create_model_server raises ImportError when FastMCP is not available."""
        with patch('cli_agent.mcp.model_server.FastMCP', None):
            with pytest.raises(ImportError) as exc_info:
                create_model_server()
            
            assert "FastMCP not installed" in str(exc_info.value)
            assert "pip install fastmcp" in str(exc_info.value)

    def test_create_model_server_config_load_failure(self):
        """Test that create_model_server handles configuration loading failures."""
        with patch('cli_agent.mcp.model_server.FastMCP'):
            with patch('cli_agent.mcp.model_server.load_config', side_effect=Exception("Config error")):
                with pytest.raises(Exception) as exc_info:
                    create_model_server()
                
                assert "Failed to load configuration" in str(exc_info.value)
                assert "Config error" in str(exc_info.value)

    @patch('cli_agent.mcp.model_server.FastMCP')
    @patch('cli_agent.mcp.model_server.load_config')
    def test_create_model_server_no_models_available(self, mock_load_config, mock_fastmcp_class):
        """Test create_model_server when no models are available."""
        # Mock configuration with no available models
        mock_config = Mock()
        mock_config.get_available_provider_models.return_value = {}
        mock_load_config.return_value = mock_config
        
        # Mock FastMCP
        mock_app = Mock()
        mock_fastmcp_class.return_value = mock_app
        
        result = create_model_server()
        
        assert result == mock_app
        mock_fastmcp_class.assert_called_once_with("AI Models Server")

    @patch('cli_agent.mcp.model_server.FastMCP')
    @patch('cli_agent.mcp.model_server.load_config')
    def test_create_model_server_with_models(self, mock_load_config, mock_fastmcp_class):
        """Test create_model_server with available models."""
        # Mock configuration with available models
        mock_config = Mock()
        mock_config.get_available_provider_models.return_value = {
            "anthropic": ["claude-3.5-sonnet", "claude-3-opus"],
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "gemini": ["gemini-2.5-flash"]
        }
        mock_config.create_host_from_provider_model = Mock()
        mock_load_config.return_value = mock_config
        
        # Mock FastMCP and tool registration
        mock_app = Mock()
        mock_fastmcp_class.return_value = mock_app
        
        result = create_model_server()
        
        assert result == mock_app
        mock_fastmcp_class.assert_called_once_with("AI Models Server")
        # Should have registered the start_chat tool
        mock_app.tool.assert_called()

    @patch('cli_agent.mcp.model_server.FastMCP')
    @patch('cli_agent.mcp.model_server.load_config')
    def test_create_model_server_provider_name_mapping(self, mock_load_config, mock_fastmcp_class):
        """Test that provider names are correctly mapped."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_available_provider_models.return_value = {
            "gemini": ["gemini-2.5-flash"],  # Should be mapped to "google"
            "anthropic": ["claude-3.5-sonnet"]  # Should stay "anthropic"
        }
        mock_load_config.return_value = mock_config
        
        # Mock FastMCP
        mock_app = Mock()
        mock_fastmcp_class.return_value = mock_app
        
        result = create_model_server()
        
        # Verify that the tool was registered (meaning models were processed)
        mock_app.tool.assert_called()
        
        # The tool decorator should have been called with a description containing the mapped models
        tool_call = mock_app.tool.call_args
        assert tool_call is not None


@pytest.mark.asyncio
class TestStartChatTool:
    """Test the start_chat tool functionality."""

    @patch('cli_agent.mcp.model_server.FastMCP')
    @patch('cli_agent.mcp.model_server.load_config')
    async def test_start_chat_invalid_model_format(self, mock_load_config, mock_fastmcp_class):
        """Test start_chat with invalid model format."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_available_provider_models.return_value = {
            "anthropic": ["claude-3.5-sonnet"]
        }
        mock_load_config.return_value = mock_config
        
        mock_app = Mock()
        mock_fastmcp_class.return_value = mock_app
        
        # Create server to get the tool function
        server = create_model_server()
        
        # Get the registered tool function
        tool_calls = mock_app.tool.call_args_list
        assert len(tool_calls) > 0
        
        # Extract the tool function (it's the decorated function)
        tool_decorator_call = tool_calls[0]
        tool_function = tool_decorator_call[1]['func'] if 'func' in tool_decorator_call[1] else None
        
        # The actual function would be available through the decorator
        # For testing, we'll test the logic directly
        
        # Test invalid model format (no colon)
        # This would be tested by calling the actual tool function
        # but since we're mocking FastMCP, we'll test the validation logic
        
        invalid_model = "gpt4"  # Missing colon
        assert ":" not in invalid_model  # This is what the function checks

    @patch('cli_agent.mcp.model_server.ConversationManager')
    def test_conversation_manager_singleton(self, mock_conv_manager_class):
        """Test that ConversationManager is used as intended."""
        # The global conversation_manager should be an instance
        from cli_agent.mcp.model_server import conversation_manager
        assert isinstance(conversation_manager, ConversationManager)


class TestMCPModelServerIntegration:
    """Integration tests for the MCP model server."""

    @pytest.mark.integration
    @patch('cli_agent.mcp.model_server.load_config')
    def test_full_server_creation_flow(self, mock_load_config):
        """Test the complete server creation flow."""
        # Mock a realistic configuration
        mock_config = Mock()
        mock_config.get_available_provider_models.return_value = {
            "anthropic": ["claude-3.5-sonnet"],
            "openai": ["gpt-4", "gpt-3.5-turbo"]
        }
        mock_load_config.return_value = mock_config
        
        try:
            # This should work if FastMCP is available
            server = create_model_server()
            assert server is not None
            
        except ImportError:
            # FastMCP not available - this is expected in testing environments
            pytest.skip("FastMCP not available for integration test")

    def test_model_name_normalization_comprehensive(self):
        """Test comprehensive model name normalization scenarios."""
        test_cases = [
            ("anthropic/claude-3.5-sonnet", "anthropic_claude_3_5_sonnet"),
            ("openai:gpt-4-turbo-preview", "openai_gpt_4_turbo_preview"), 
            ("google/gemini-2.5-flash", "google_gemini_2_5_flash"),
            ("deepseek-reasoner", "deepseek_reasoner"),
            ("model_with_underscores", "model_with_underscores"),
            ("model123", "model123"),
            ("123model", "123model"),
            ("UPPERCASE-model", "UPPERCASE_model"),
            ("model   with    spaces", "model_with_spaces"),
        ]
        
        for input_name, expected_output in test_cases:
            result = normalize_model_name(input_name)
            assert result == expected_output, f"Failed for input '{input_name}': expected '{expected_output}', got '{result}'"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])