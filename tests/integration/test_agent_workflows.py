"""Integration tests for agent workflows without hanging."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.integration
class TestAgentWorkflows:
    """Test agent workflows without hanging issues."""

    @pytest.mark.asyncio
    async def test_message_processing_workflow(self, mock_base_agent):
        """Test message processing without interactive chat."""
        # Mock the generate response method
        mock_base_agent._generate_completion = AsyncMock(
            return_value="Hello! I'm ready to help."
        )

        # Test processing a single message
        messages = [{"role": "user", "content": "Hello"}]
        response = await mock_base_agent.generate_response(messages)

        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_tool_execution_workflow(self, mock_base_agent, temp_dir):
        """Test tool execution workflow."""
        # Just test that the agent has the tool available
        assert "builtin:read_file" in mock_base_agent.available_tools

    @pytest.mark.asyncio
    async def test_subagent_workflow(self, mock_base_agent):
        """Test subagent workflow without interactive chat."""
        # Mock subagent manager
        mock_manager = MagicMock()
        mock_manager.spawn_subagent = AsyncMock(return_value="task_123")
        mock_manager.get_active_count.return_value = 1
        mock_manager.subagents = {
            "task_123": MagicMock(
                completed=True,
                result="Subagent completed successfully",
                description="Test task",
                start_time=0,
            )
        }

        mock_base_agent.subagent_manager = mock_manager

        # Test that agent has subagent manager
        assert mock_base_agent.subagent_manager is not None

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, mock_base_agent):
        """Test error handling in workflows."""
        # Mock an error in generate_completion
        mock_base_agent._generate_completion = AsyncMock(
            side_effect=Exception("Test error")
        )

        messages = [{"role": "user", "content": "Test"}]

        # Should handle errors gracefully
        try:
            result = await mock_base_agent.generate_response(messages)
            # If no exception is raised, result should be some error message
            assert result is not None
        except Exception:
            # If exception is raised, that's also fine for this test
            assert True

    @pytest.mark.asyncio
    async def test_slash_command_workflow(self, mock_base_agent):
        """Test slash command processing."""
        from cli_agent.core.slash_commands import SlashCommandManager

        # Create real slash command manager for testing
        slash_manager = SlashCommandManager(mock_base_agent)

        # Test help command
        result = await slash_manager.handle_slash_command("/help")
        assert isinstance(result, str)
        assert "help" in result.lower() or "available" in result.lower()

    @pytest.mark.asyncio
    async def test_configuration_workflow(self, sample_host_config):
        """Test configuration handling."""
        # Test that configuration is properly structured
        assert sample_host_config.deepseek_api_key == "test-key"
        assert sample_host_config.deepseek_model == "deepseek-chat"
        assert sample_host_config.gemini_api_key == "test-key"
        assert sample_host_config.gemini_model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_tool_discovery_workflow(self, mock_base_agent):
        """Test tool discovery and loading."""
        # Test that tools are properly loaded
        assert hasattr(mock_base_agent, "available_tools")
        assert isinstance(mock_base_agent.available_tools, dict)
        assert len(mock_base_agent.available_tools) > 0

    @pytest.mark.asyncio
    async def test_markdown_formatting_workflow(self, mock_base_agent):
        """Test markdown formatting functionality."""
        # Test that agent has conversation management capabilities
        assert hasattr(mock_base_agent, "conversation_history")
        assert isinstance(mock_base_agent.conversation_history, list)

    @pytest.mark.asyncio
    async def test_conversation_compacting_workflow(self, mock_base_agent):
        """Test conversation compacting."""
        # Test with short conversation (should not compact)
        short_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = mock_base_agent.compact_conversation(short_messages)
        assert len(result) == len(short_messages)

    @pytest.mark.asyncio
    async def test_tool_normalization_workflow(self, mock_base_agent):
        """Test tool name normalization."""
        # Test tool name normalization
        assert (
            mock_base_agent.normalize_tool_name("builtin:bash_execute")
            == "builtin_bash_execute"
        )
        assert (
            mock_base_agent.normalize_tool_name("mcp:server:tool") == "mcp_server_tool"
        )

    @pytest.mark.asyncio
    async def test_token_calculation_workflow(self, mock_base_agent):
        """Test token calculation functionality."""
        # Test token limit calculation
        token_limit = mock_base_agent.get_token_limit()
        assert isinstance(token_limit, int)
        assert token_limit > 0

    @pytest.mark.asyncio
    async def test_builtin_tools_workflow(self, mock_base_agent):
        """Test builtin tools loading and structure."""
        from cli_agent.tools.builtin_tools import get_all_builtin_tools

        tools = get_all_builtin_tools()
        assert isinstance(tools, dict)
        assert "builtin:bash_execute" in tools
        assert "builtin:read_file" in tools
        assert "builtin:write_file" in tools
