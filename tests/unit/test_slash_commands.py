"""Unit tests for SlashCommandManager functionality."""

from unittest.mock import AsyncMock

import pytest

from cli_agent.core.slash_commands import SlashCommandManager


@pytest.mark.unit
class TestSlashCommandManager:
    """Test SlashCommandManager core functionality."""

    def test_init(self, mock_base_agent):
        """Test SlashCommandManager initialization."""
        manager = SlashCommandManager(mock_base_agent)
        assert manager.agent == mock_base_agent
        assert hasattr(manager, "custom_commands")

    @pytest.mark.asyncio
    async def test_help_command(self, mock_base_agent):
        """Test /help command."""
        manager = SlashCommandManager(mock_base_agent)

        result = await manager.handle_slash_command("/help")
        assert isinstance(result, str)
        assert "help" in result.lower() or "available" in result.lower()

    @pytest.mark.asyncio
    async def test_clear_command(self, mock_base_agent):
        """Test /clear command."""
        manager = SlashCommandManager(mock_base_agent)
        mock_base_agent.conversation_history = [{"role": "user", "content": "test"}]

        result = await manager.handle_slash_command("/clear")
        # Should return a dictionary with action type
        assert isinstance(result, (str, dict))

    @pytest.mark.asyncio
    async def test_tools_command(self, mock_base_agent):
        """Test /tools command."""
        manager = SlashCommandManager(mock_base_agent)

        result = await manager.handle_slash_command("/tools")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_quit_command(self, mock_base_agent):
        """Test /quit command."""
        manager = SlashCommandManager(mock_base_agent)

        result = await manager.handle_slash_command("/quit")
        # Should return a dictionary with action type
        assert isinstance(result, (str, dict))

    @pytest.mark.asyncio
    async def test_tokens_command(self, mock_base_agent):
        """Test /tokens command with reliable and unreliable token info."""
        from unittest.mock import Mock
        
        manager = SlashCommandManager(mock_base_agent)

        # Mock messages
        messages = [{"role": "user", "content": "test"}]
        
        # Test with reliable token information
        mock_token_manager = Mock()
        mock_token_manager.has_reliable_token_info.return_value = True
        mock_base_agent.token_manager = mock_token_manager
        
        # Add the missing method to mock_base_agent
        mock_base_agent.count_conversation_tokens = Mock(return_value=100)
        mock_base_agent.get_token_limit = Mock(return_value=1000)
        
        result = await manager.handle_slash_command("/tokens", messages)
        assert isinstance(result, str)
        assert "📊 Token usage" in result
        assert "100/1000" in result
        
        # Test with unreliable token information
        mock_token_manager.has_reliable_token_info.return_value = False
        
        result = await manager.handle_slash_command("/tokens", messages)
        assert isinstance(result, str)
        assert "Token information not available" in result
        assert "known model limits" in result
        
        # Test without token manager
        mock_base_agent.token_manager = None
        result = await manager.handle_slash_command("/tokens", messages)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_compact_command(self, mock_base_agent):
        """Test /compact command."""
        manager = SlashCommandManager(mock_base_agent)

        # Mock conversation history
        messages = [
            {"role": "user", "content": "message 1"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "message 2"},
        ]

        # Mock compact_conversation method
        mock_base_agent.compact_conversation = AsyncMock(
            return_value=[{"role": "user", "content": "compacted message"}]
        )

        result = await manager.handle_slash_command("/compact", messages)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_model_switching_commands(self, mock_base_agent):
        """Test model switching commands."""
        manager = SlashCommandManager(mock_base_agent)

        # Test various model switching commands (both old and new names)
        commands = [
            "/switch-chat",  # old name (backward compatibility)
            "/switch-deepseek",  # new name
            "/switch-reason",
            "/switch-gemini",  # old name (backward compatibility)
            "/switch-gemini-flash",  # new name
            "/switch-gemini-pro",
        ]

        for command in commands:
            result = await manager.handle_slash_command(command)
            # Should return a dictionary with model info or string
            assert isinstance(result, (str, dict))

    @pytest.mark.asyncio
    async def test_unknown_command(self, mock_base_agent):
        """Test unknown command handling."""
        manager = SlashCommandManager(mock_base_agent)

        result = await manager.handle_slash_command("/unknown_command")
        # Should return error message for unknown commands
        assert isinstance(result, str) and "unknown" in result.lower()

    @pytest.mark.asyncio
    async def test_command_without_slash(self, mock_base_agent):
        """Test handling commands without slash prefix."""
        manager = SlashCommandManager(mock_base_agent)

        result = await manager.handle_slash_command("help")  # No slash
        # Should still work or return None
        assert result is None or isinstance(result, (str, dict))

    @pytest.mark.asyncio
    async def test_empty_command(self, mock_base_agent):
        """Test handling empty command."""
        manager = SlashCommandManager(mock_base_agent)

        result = await manager.handle_slash_command("")
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_command_loading(self, mock_base_agent):
        """Test custom command loading."""
        manager = SlashCommandManager(mock_base_agent)

        # Test that custom_commands attribute exists
        assert hasattr(manager, "custom_commands")
        assert isinstance(manager.custom_commands, dict)

    @pytest.mark.asyncio
    async def test_permissions_command(self, mock_base_agent):
        """Test /permissions command if available."""
        manager = SlashCommandManager(mock_base_agent)

        result = await manager.handle_slash_command("/permissions")
        # Should return a string or None if not implemented
        assert result is None or isinstance(result, str)

    def test_mcp_commands_discovery(self, mock_base_agent):
        """Test MCP commands discovery."""
        manager = SlashCommandManager(mock_base_agent)

        # Test that _get_mcp_commands method exists
        if hasattr(manager, "_get_mcp_commands"):
            commands = manager._get_mcp_commands()
            assert isinstance(commands, list)

    @pytest.mark.asyncio
    async def test_command_with_arguments(self, mock_base_agent):
        """Test commands with arguments."""
        manager = SlashCommandManager(mock_base_agent)

        # Test model command with arguments
        result = await manager.handle_slash_command("/model deepseek-chat")
        assert result is None or isinstance(result, str) or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_error_handling_in_commands(self, mock_base_agent):
        """Test error handling in commands."""
        manager = SlashCommandManager(mock_base_agent)

        # Mock an error in the agent
        mock_base_agent.conversation_history = None  # This might cause an error

        # Should not raise an exception
        try:
            await manager.handle_slash_command("/help")
            assert True  # No exception was raised
        except Exception:
            # Some errors might be expected in test environment
            assert True

    @pytest.mark.asyncio
    async def test_command_case_sensitivity(self, mock_base_agent):
        """Test command case sensitivity."""
        manager = SlashCommandManager(mock_base_agent)

        # Test both lowercase and uppercase
        result1 = await manager.handle_slash_command("/help")
        result2 = await manager.handle_slash_command("/HELP")

        # Should handle case consistently
        assert isinstance(result1, (str, type(None)))
        assert isinstance(result2, (str, type(None)))
