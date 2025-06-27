"""Unit tests for BaseMCPAgent core functionality."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.tools.builtin_tools import get_all_builtin_tools
from config import load_config


@pytest.mark.unit
class TestBaseMCPAgent:
    """Test BaseMCPAgent core functionality."""

    def test_init_main_agent(self, sample_host_config):
        """Test BaseMCPAgent initialization for main agent."""

        # Create a concrete implementation for testing
        class TestAgent(BaseMCPAgent):
            def convert_tools_to_llm_format(self):
                return []

            def parse_tool_calls(self, response):
                return []

            async def generate_response(self, messages, stream=True):
                return "test response"

            def _normalize_tool_calls_to_standard_format(self, tool_calls):
                return []

            def _extract_text_before_tool_calls(self, content: str) -> str:
                return ""

            def _get_provider_config(self):
                return self.config.get_deepseek_config()

            def _get_streaming_preference(self, provider_config) -> bool:
                return True

            def _calculate_timeout(self, provider_config) -> float:
                return 60.0

            def _create_llm_client(self, provider_config, timeout_seconds):
                return MagicMock()

            async def _generate_completion(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return "test response"

            def _get_current_runtime_model(self) -> str:
                return "test-model"

            def _extract_response_content(self, response):
                return ("test content", [], {})

            async def _process_streaming_chunks(self, response):
                return ("test content", [], {})

            async def _make_api_request(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return MagicMock()

            def _create_mock_response(self, content: str, tool_calls):
                return MagicMock()

        agent = TestAgent(sample_host_config, is_subagent=False)

        assert agent.config == sample_host_config
        assert agent.is_subagent is False
        assert isinstance(agent.available_tools, dict)
        assert len(agent.available_tools) > 0

        # Main agent should not have emit_result tool
        assert "builtin:emit_result" not in agent.available_tools

        # Main agent should have task spawning tool
        assert "builtin:task" in agent.available_tools

        # Background subagent tools should only be available if background_subagents is enabled
        if sample_host_config.background_subagents:
            assert "builtin:task_status" in agent.available_tools
            assert "builtin:task_results" in agent.available_tools
        else:
            assert "builtin:task_status" not in agent.available_tools
            assert "builtin:task_results" not in agent.available_tools

    def test_init_subagent(self, sample_host_config):
        """Test BaseMCPAgent initialization for subagent."""

        class TestAgent(BaseMCPAgent):
            def convert_tools_to_llm_format(self):
                return []

            def parse_tool_calls(self, response):
                return []

            async def generate_response(self, messages, stream=True):
                return "test response"

            def _normalize_tool_calls_to_standard_format(self, tool_calls):
                return []

            def _extract_text_before_tool_calls(self, content: str) -> str:
                return ""

            def _get_provider_config(self):
                return self.config.get_deepseek_config()

            def _get_streaming_preference(self, provider_config) -> bool:
                return True

            def _calculate_timeout(self, provider_config) -> float:
                return 60.0

            def _create_llm_client(self, provider_config, timeout_seconds):
                return MagicMock()

            async def _generate_completion(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return "test response"

            def _get_current_runtime_model(self) -> str:
                return "test-model"

            def _extract_response_content(self, response):
                return ("test content", [], {})

            async def _process_streaming_chunks(self, response):
                return ("test content", [], {})

            async def _make_api_request(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return MagicMock()

            def _create_mock_response(self, content: str, tool_calls):
                return MagicMock()

        agent = TestAgent(sample_host_config, is_subagent=True)

        assert agent.is_subagent is True

        # Subagent should have emit_result tool
        assert "builtin:emit_result" in agent.available_tools

        # Subagent should NOT have task management tools
        assert "builtin:task" not in agent.available_tools
        assert "builtin:task_status" not in agent.available_tools

    def test_available_tools_loading(self, mock_base_agent):
        """Test that builtin tools are properly loaded in agent."""
        # BaseMCPAgent should have tools loaded after initialization
        assert isinstance(mock_base_agent.available_tools, dict)
        assert len(mock_base_agent.available_tools) > 0

    def test_tool_name_normalization(self, mock_base_agent):
        """Test tool name normalization functionality."""
        # Test the normalize_tool_name method
        result = mock_base_agent.normalize_tool_name("builtin:bash_execute")
        assert result == "builtin_bash_execute"

        result = mock_base_agent.normalize_tool_name("mcp:server:tool")
        assert result == "mcp_server_tool"

    @pytest.mark.asyncio
    async def test_generate_response_streaming(self, mock_base_agent):
        """Test generate_response with streaming."""

        # Mock the abstract method to return a generator
        async def mock_generator():
            yield "chunk1"
            yield "chunk2"

        mock_base_agent.generate_response = AsyncMock(return_value=mock_generator())

        result = await mock_base_agent.generate_response(
            [{"role": "user", "content": "test"}]
        )

        # Should return the generator
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    async def test_generate_response_non_streaming(self, mock_base_agent):
        """Test generate_response without streaming."""
        mock_base_agent.generate_response = AsyncMock(return_value="Complete response")

        result = await mock_base_agent.generate_response(
            [{"role": "user", "content": "test"}]
        )

        assert result == "Complete response"

    def test_conversation_history_management(self, mock_base_agent):
        """Test conversation history management."""
        # Test initial state
        assert isinstance(mock_base_agent.conversation_history, list)
        assert len(mock_base_agent.conversation_history) == 0

        # Test adding messages
        mock_base_agent.conversation_history.append({"role": "user", "content": "test"})
        assert len(mock_base_agent.conversation_history) == 1

    def test_mcp_clients_initialization(self, mock_base_agent):
        """Test MCP clients dictionary initialization."""
        assert isinstance(mock_base_agent.mcp_clients, dict)
        # Should start empty
        assert len(mock_base_agent.mcp_clients) == 0

    def test_token_limit_method(self, mock_base_agent):
        """Test get_token_limit method."""
        # Should return a reasonable default
        limit = mock_base_agent.get_token_limit()
        assert isinstance(limit, int)
        assert limit > 0
        assert limit == 4000  # From our mock

    def test_subagent_manager_initialization(self, mock_base_agent):
        """Test subagent manager initialization for main agent."""
        # Main agent should have subagent manager initialized
        assert mock_base_agent.subagent_manager is not None

    def test_subagent_without_manager(self, sample_host_config):
        """Test that subagents don't have subagent manager."""

        class TestAgent(BaseMCPAgent):
            def convert_tools_to_llm_format(self):
                return []

            def parse_tool_calls(self, response):
                return []

            async def generate_response(self, messages, stream=True):
                return "test response"

            def _normalize_tool_calls_to_standard_format(self, tool_calls):
                return []

            def _extract_text_before_tool_calls(self, content: str) -> str:
                return ""

            def _get_provider_config(self):
                return self.config.get_deepseek_config()

            def _get_streaming_preference(self, provider_config) -> bool:
                return True

            def _calculate_timeout(self, provider_config) -> float:
                return 60.0

            def _create_llm_client(self, provider_config, timeout_seconds):
                return MagicMock()

            async def _generate_completion(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return "test response"

            def _get_current_runtime_model(self) -> str:
                return "test-model"

            def _extract_response_content(self, response):
                return ("test content", [], {})

            async def _process_streaming_chunks(self, response):
                return ("test content", [], {})

            async def _make_api_request(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return MagicMock()

            def _create_mock_response(self, content: str, tool_calls):
                return MagicMock()

        subagent = TestAgent(sample_host_config, is_subagent=True)
        assert subagent.subagent_manager is None

    def test_config_management(self, mock_base_agent):
        """Test configuration management."""
        # Agent should have config properly stored
        assert mock_base_agent.config is not None
        assert hasattr(mock_base_agent.config, "get_deepseek_config")

    def test_is_subagent_flag(self, mock_base_agent):
        """Test subagent flag setting."""
        # Mock agent should be main agent (not subagent)
        assert mock_base_agent.is_subagent is False

    def test_background_subagents_config_default(self):
        """Test that background_subagents defaults to False."""
        # Ensure no environment variable is set
        if "BACKGROUND_SUBAGENTS" in os.environ:
            del os.environ["BACKGROUND_SUBAGENTS"]

        config = load_config()
        assert config.background_subagents is False

    def test_background_subagents_config_enabled(self):
        """Test that BACKGROUND_SUBAGENTS=true enables background mode."""
        original_value = os.environ.get("BACKGROUND_SUBAGENTS")

        try:
            os.environ["BACKGROUND_SUBAGENTS"] = "true"
            config = load_config()
            assert config.background_subagents is True
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["BACKGROUND_SUBAGENTS"] = original_value
            elif "BACKGROUND_SUBAGENTS" in os.environ:
                del os.environ["BACKGROUND_SUBAGENTS"]

    def test_background_subagents_config_disabled(self):
        """Test that BACKGROUND_SUBAGENTS=false disables background mode."""
        original_value = os.environ.get("BACKGROUND_SUBAGENTS")

        try:
            os.environ["BACKGROUND_SUBAGENTS"] = "false"
            config = load_config()
            assert config.background_subagents is False
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["BACKGROUND_SUBAGENTS"] = original_value
            elif "BACKGROUND_SUBAGENTS" in os.environ:
                del os.environ["BACKGROUND_SUBAGENTS"]

    def test_background_subagent_tools_availability(self, sample_host_config):
        """Test that background subagent tools are only available when enabled."""

        class TestAgent(BaseMCPAgent):
            def convert_tools_to_llm_format(self):
                return []

            def parse_tool_calls(self, response):
                return []

            async def generate_response(self, messages, stream=True):
                return "test response"

            def _normalize_tool_calls_to_standard_format(self, tool_calls):
                return []

            def _extract_text_before_tool_calls(self, content: str) -> str:
                return ""

            def _get_provider_config(self):
                return self.config.get_deepseek_config()

            def _get_streaming_preference(self, provider_config) -> bool:
                return True

            def _calculate_timeout(self, provider_config) -> float:
                return 60.0

            def _create_llm_client(self, provider_config, timeout_seconds):
                return MagicMock()

            async def _generate_completion(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return "test response"

            def _get_current_runtime_model(self) -> str:
                return "test-model"

            def _extract_response_content(self, response):
                return ("test content", [], {})

            async def _process_streaming_chunks(self, response):
                return ("test content", [], {})

            async def _make_api_request(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return MagicMock()

            def _create_mock_response(self, content: str, tool_calls):
                return MagicMock()

        # Test with background subagents disabled (default)
        sample_host_config.background_subagents = False
        agent_disabled = TestAgent(sample_host_config, is_subagent=False)
        assert "builtin:task" in agent_disabled.available_tools  # Always available
        assert "builtin:task_status" not in agent_disabled.available_tools
        assert "builtin:task_results" not in agent_disabled.available_tools

        # Test with background subagents enabled
        sample_host_config.background_subagents = True
        agent_enabled = TestAgent(sample_host_config, is_subagent=False)
        assert "builtin:task" in agent_enabled.available_tools
        assert "builtin:task_status" in agent_enabled.available_tools
        assert "builtin:task_results" in agent_enabled.available_tools
