"""Shared test fixtures and configuration for pytest."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the modules we'll be testing
from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from config import HostConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_host_config(monkeypatch):
    """Create a sample HostConfig for testing."""
    # Set environment variables for the config
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_TEMPERATURE", "0.7")
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "4096")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.7")
    monkeypatch.setenv("GEMINI_MAX_TOKENS", "8192")

    # Create config with environment variables
    return HostConfig()


@pytest.fixture
def mock_tools():
    """Mock tools dictionary for testing."""
    return {
        "builtin:bash_execute": {
            "server": "builtin",
            "name": "bash_execute",
            "description": "Execute bash commands",
            "schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        "builtin:read_file": {
            "server": "builtin",
            "name": "read_file",
            "description": "Read file contents",
            "schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }


@pytest.fixture
def mock_base_agent(sample_host_config, mock_tools):
    """Create a mock BaseMCPAgent for testing."""

    class MockAgent(BaseMCPAgent):
        def convert_tools_to_llm_format(self) -> List[Dict]:
            return []

        def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
            return []

        async def generate_response(
            self, messages: List[Dict[str, Any]], stream=True
        ) -> Any:
            return "Mock response"

        # Implement all abstract methods
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

        # Add missing methods that tests are looking for
        def get_token_limit(self) -> int:
            """Mock token limit for testing."""
            return 4000

        def compact_conversation(
            self, messages: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Mock conversation compacting for testing."""
            # Return a shorter list to simulate compacting
            return messages[: len(messages) // 2] if len(messages) > 2 else messages

        def normalize_tool_name(self, tool_name: str) -> str:
            """Mock tool name normalization for testing."""
            return tool_name.replace(":", "_")

    agent = MockAgent(sample_host_config, is_subagent=False)
    agent.available_tools = mock_tools
    return agent


@pytest.fixture
def mock_deepseek_response():
    """Mock DeepSeek API response."""

    class MockChoice:
        def __init__(self):
            self.message = MagicMock()
            self.message.content = "Test response from DeepSeek"
            self.message.tool_calls = None

    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]

    return MockResponse()


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""

    class MockPart:
        def __init__(self, text=None, function_call=None):
            self.text = text
            self.function_call = function_call

    class MockContent:
        def __init__(self, parts):
            self.parts = parts

    class MockCandidate:
        def __init__(self, content):
            self.content = content

    class MockResponse:
        def __init__(self, text="Test response from Gemini"):
            self.candidates = [MockCandidate(MockContent([MockPart(text=text)]))]

    return MockResponse()


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, please help me"},
        {"role": "assistant", "content": "I'll help you with that."},
        {"role": "user", "content": "List the files in the current directory"},
    ]


@pytest.fixture
def sample_tool_calls():
    """Sample tool calls for testing."""
    return [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "builtin_bash_execute",
                "arguments": json.dumps({"command": "ls -la"}),
            },
        }
    ]


@pytest.fixture
def mock_subagent_manager():
    """Mock SubagentManager for testing."""
    manager = MagicMock()
    manager.get_active_count.return_value = 0
    manager.subagents = {}
    return manager


@pytest.fixture
def mock_input_handler():
    """Mock InterruptibleInput for testing."""
    handler = MagicMock(spec=InterruptibleInput)
    handler.get_multiline_input.return_value = "test input"
    handler.interrupted = False
    return handler


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable logging during tests to reduce noise."""
    import logging

    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_mcp_client():
    """Mock MCP client for testing external tools."""
    client = AsyncMock()
    client.list_tools.return_value = []
    client.call_tool.return_value = MagicMock(content="Mock tool result")
    return client


# Test utilities
def create_test_file(directory: Path, filename: str, content: str) -> Path:
    """Helper to create test files."""
    file_path = directory / filename
    file_path.write_text(content)
    return file_path


def assert_tool_called_with(mock_agent, tool_name: str, args: Dict[str, Any]):
    """Helper to assert a tool was called with specific arguments."""
    # Implementation depends on how we mock tool execution
    pass


# Mock patches that can be used across tests
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for DeepSeek testing."""
    with patch("openai.OpenAI") as mock:
        yield mock


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    with patch("google.genai.Client") as mock:
        yield mock
