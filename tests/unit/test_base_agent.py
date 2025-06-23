"""Unit tests for BaseMCPAgent core functionality."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.tools.builtin_tools import get_all_builtin_tools


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

            async def _generate_completion(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return "test response"

            def _normalize_tool_calls_to_standard_format(self, tool_calls):
                return []

            def _get_current_runtime_model(self):
                return "test-model"

            def _get_provider_config(self):
                return self.config.get_deepseek_config()

            def _get_streaming_preference(self, provider_config) -> bool:
                return True

            def _calculate_timeout(self, provider_config) -> float:
                return 60.0

            def _create_llm_client(self, provider_config, timeout_seconds):
                from unittest.mock import MagicMock

                return MagicMock()

            def _extract_text_before_tool_calls(self, content: str) -> str:
                return ""

        agent = TestAgent(sample_host_config, is_subagent=False)

        assert agent.config == sample_host_config
        assert agent.is_subagent is False
        assert isinstance(agent.available_tools, dict)
        assert len(agent.available_tools) > 0

        # Main agent should not have emit_result tool
        assert "builtin:emit_result" not in agent.available_tools

        # Main agent should have task management tools
        assert "builtin:task" in agent.available_tools
        assert "builtin:task_status" in agent.available_tools

    def test_init_subagent(self, sample_host_config):
        """Test BaseMCPAgent initialization for subagent."""

        class TestAgent(BaseMCPAgent):
            def convert_tools_to_llm_format(self):
                return []

            def parse_tool_calls(self, response):
                return []

            async def _generate_completion(
                self, messages, tools=None, stream=True, interactive=True
            ):
                return "test response"

            def _normalize_tool_calls_to_standard_format(self, tool_calls):
                return []

            def _get_current_runtime_model(self):
                return "test-model"

            def _get_provider_config(self):
                return self.config.get_deepseek_config()

            def _get_streaming_preference(self, provider_config) -> bool:
                return True

            def _calculate_timeout(self, provider_config) -> float:
                return 60.0

            def _create_llm_client(self, provider_config, timeout_seconds):
                from unittest.mock import MagicMock

                return MagicMock()

            def _extract_text_before_tool_calls(self, content: str) -> str:
                return ""

        agent = TestAgent(sample_host_config, is_subagent=True)

        assert agent.is_subagent is True

        # Subagent should have emit_result tool
        assert "builtin:emit_result" in agent.available_tools

        # Subagent should NOT have task management tools
        assert "builtin:task" not in agent.available_tools
        assert "builtin:task_status" not in agent.available_tools

    def test_markdown_formatting(self, mock_base_agent):
        """Test markdown formatting functionality."""
        # Test basic formatting
        text = "This is **bold** text with `code` and *italic*"

        result = mock_base_agent.format_markdown(text)
        # Should contain the formatted text with ANSI codes
        assert "bold" in result
        assert "code" in result
        assert "italic" in result

    def test_markdown_formatting_fallback(self, mock_base_agent):
        """Test markdown formatting fallback when Rich is not available."""
        text = "This is **bold** text with `code`"

        result = mock_base_agent.format_markdown(text)
        # Should contain ANSI escape codes for basic formatting
        assert "\033[1m" in result  # Bold formatting
        assert "\033[96m" in result  # Code formatting (cyan color, not background)

    @pytest.mark.asyncio
    async def test_generate_response_streaming(self, mock_base_agent):
        """Test generate_response with streaming."""
        mock_base_agent.stream = True
        mock_base_agent.is_subagent = False

        # Mock the _generate_completion method
        async def mock_generator():
            yield "chunk1"
            yield "chunk2"

        mock_base_agent._generate_completion = AsyncMock(return_value=mock_generator())

        result = await mock_base_agent.generate_response(
            [{"role": "user", "content": "test"}]
        )

        # Should return the generator
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    async def test_generate_response_non_streaming(self, mock_base_agent):
        """Test generate_response without streaming."""
        mock_base_agent.stream = False

        mock_base_agent._generate_completion = AsyncMock(
            return_value="Complete response"
        )

        result = await mock_base_agent.generate_response(
            [{"role": "user", "content": "test"}]
        )

        assert result == "Complete response"

    @pytest.mark.asyncio
    async def test_execute_builtin_tool_bash(self, mock_base_agent):
        """Test bash_execute builtin tool."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "test output"
            mock_run.return_value.stderr = ""
            mock_run.return_value.returncode = 0

            result = await mock_base_agent._execute_builtin_tool(
                "bash_execute", {"command": "echo test"}
            )

            assert "test output" in result

    @pytest.mark.asyncio
    async def test_execute_builtin_tool_read_file(self, mock_base_agent, temp_dir):
        """Test read_file builtin tool."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        result = await mock_base_agent._execute_builtin_tool(
            "read_file", {"file_path": str(test_file)}
        )

        assert "test content" in result

    @pytest.mark.asyncio
    async def test_execute_builtin_tool_write_file(self, mock_base_agent, temp_dir):
        """Test write_file builtin tool."""
        test_file = temp_dir / "output.txt"

        result = await mock_base_agent._execute_builtin_tool(
            "write_file", {"file_path": str(test_file), "content": "new content"}
        )

        assert "successfully" in result.lower()
        assert test_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_compact_conversation(self, mock_base_agent, sample_messages):
        """Test conversation compacting functionality."""
        # Test with short conversation (should not compact)
        short_messages = sample_messages[:2]
        result = await mock_base_agent.compact_conversation(short_messages)
        assert len(result) == len(short_messages)

        # Test with long conversation
        long_messages = sample_messages * 20  # Create a long conversation
        mock_base_agent.get_token_limit = MagicMock(
            return_value=100
        )  # Low limit to force compacting

        result = await mock_base_agent.compact_conversation(long_messages)
        assert len(result) < len(long_messages)

    def test_normalize_tool_name(self, mock_base_agent):
        """Test tool name normalization."""
        assert (
            mock_base_agent.normalize_tool_name("builtin:bash_execute")
            == "builtin_bash_execute"
        )
        assert (
            mock_base_agent.normalize_tool_name("mcp:server:tool") == "mcp_server_tool"
        )

    @pytest.mark.asyncio
    async def test_subagent_task_spawning(self, mock_base_agent):
        """Test subagent task spawning."""
        mock_base_agent.subagent_manager = MagicMock()
        mock_base_agent.subagent_manager.spawn_subagent = AsyncMock(
            return_value="task_123"
        )
        mock_base_agent.subagent_manager.get_active_count.return_value = 1

        result = await mock_base_agent._task(
            {
                "description": "Test task",
                "prompt": "Do something",
                "context": "Additional context",
            }
        )

        assert "task_123" in result
        assert "Test task" in result
        mock_base_agent.subagent_manager.spawn_subagent.assert_called_once()

    def test_builtin_tools_loading(self, mock_base_agent):
        """Test that builtin tools are properly loaded."""
        tools = get_all_builtin_tools()

        # Check that essential tools are present
        essential_tools = ["bash_execute", "read_file", "write_file", "list_directory"]
        for tool in essential_tools:
            assert f"builtin:{tool}" in tools

        # Check tool structure
        bash_tool = tools["builtin:bash_execute"]
        assert bash_tool["name"] == "bash_execute"
        assert bash_tool["server"] == "builtin"
        assert "description" in bash_tool
        assert "schema" in bash_tool

    @pytest.mark.asyncio
    async def test_error_handling_in_tool_execution(self, mock_base_agent):
        """Test error handling during tool execution."""
        # Test with invalid command
        result = await mock_base_agent._execute_builtin_tool(
            "bash_execute", {"command": "invalid_command_xyz"}
        )
        assert "error" in result.lower() or "not found" in result.lower()

        # Test with missing file
        result = await mock_base_agent._execute_builtin_tool(
            "read_file", {"file_path": "/nonexistent/file.txt"}
        )
        assert "error" in result.lower() or "not found" in result.lower()

    def test_token_limit_calculation(self, mock_base_agent):
        """Test token limit calculation."""
        # Default implementation should return a reasonable limit
        limit = mock_base_agent.get_token_limit()
        assert isinstance(limit, int)
        assert limit > 0
