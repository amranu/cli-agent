"""Tests for subprocess display coordination functionality."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from cli_agent.core.event_system import EventBus
from cli_agent.core.subprocess_display import SubprocessDisplayCoordinator
from cli_agent.core.terminal_manager import TerminalManager


@pytest.mark.unit
class TestSubprocessDisplay:
    """Test cases for SubprocessDisplayCoordinator class."""

    @pytest.fixture
    def event_bus(self):
        """Create test event bus."""
        return EventBus()

    @pytest.fixture
    def subprocess_display(self, event_bus):
        """Create test subprocess display coordinator."""
        return SubprocessDisplayCoordinator(event_bus)

    @pytest.fixture
    def mock_terminal_manager(self):
        """Create mock terminal manager."""
        mock = MagicMock(spec=TerminalManager)
        mock.allocate_subprocess_lines.return_value = (1, 3)
        mock.lines_per_subprocess = 3
        return mock

    def test_initialization(self, event_bus):
        """Test SubprocessDisplayCoordinator initialization."""
        coordinator = SubprocessDisplayCoordinator(event_bus)
        assert coordinator.event_bus is event_bus
        assert coordinator.active_processes == {}
        assert coordinator.process_line_mapping == {}

    @pytest.mark.asyncio
    async def test_register_subprocess_success(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test successful subprocess registration."""
        subprocess_display.terminal_manager = mock_terminal_manager

        result = await subprocess_display.register_subprocess(
            "task_001", "Test Process"
        )

        assert result is True
        assert "task_001" in subprocess_display.active_processes
        assert (
            subprocess_display.active_processes["task_001"]["process_name"]
            == "Test Process"
        )
        mock_terminal_manager.allocate_subprocess_lines.assert_called_once_with(
            "task_001", "Test Process"
        )

    @pytest.mark.asyncio
    async def test_register_subprocess_no_space(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test subprocess registration when no space available."""
        subprocess_display.terminal_manager = mock_terminal_manager
        mock_terminal_manager.allocate_subprocess_lines.return_value = None

        result = await subprocess_display.register_subprocess(
            "task_001", "Test Process"
        )

        assert result is False
        assert "task_001" not in subprocess_display.active_processes

    @pytest.mark.asyncio
    async def test_register_already_registered(self, subprocess_display):
        """Test registering an already registered subprocess."""
        subprocess_display.active_processes["task_001"] = {"process_name": "Existing"}

        result = await subprocess_display.register_subprocess(
            "task_001", "Test Process"
        )

        assert result is True
        assert (
            subprocess_display.active_processes["task_001"]["process_name"]
            == "Existing"
        )

    @pytest.mark.asyncio
    async def test_unregister_subprocess(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test subprocess unregistration."""
        subprocess_display.terminal_manager = mock_terminal_manager
        subprocess_display.active_processes["task_001"] = {
            "process_name": "Test",
            "current_line_offset": 1,
        }

        # Mock the async sleep to speed up test
        with patch("asyncio.sleep"):
            await subprocess_display.unregister_subprocess("task_001")

        mock_terminal_manager.write_to_subprocess_lines.assert_called()
        mock_terminal_manager.deallocate_subprocess_lines.assert_called_with("task_001")

    @pytest.mark.asyncio
    async def test_display_subprocess_message(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test displaying subprocess message."""
        subprocess_display.terminal_manager = mock_terminal_manager
        subprocess_display.active_processes["task_001"] = {
            "process_name": "Test",
            "current_line_offset": 0,
            "message_count": 0,
            "last_message_time": 0,
        }

        await subprocess_display.display_subprocess_message(
            "task_001", "Test message", "info"
        )

        mock_terminal_manager.write_to_subprocess_lines.assert_called()
        args = mock_terminal_manager.write_to_subprocess_lines.call_args
        assert args[0][0] == "task_001"
        assert "Test message" in args[0][1]
        assert args[0][2] == 0  # line offset

    @pytest.mark.asyncio
    async def test_display_subprocess_message_auto_register(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test displaying message auto-registers subprocess."""
        subprocess_display.terminal_manager = mock_terminal_manager

        await subprocess_display.display_subprocess_message(
            "task_001", "Test message", "info"
        )

        # Should have auto-registered the subprocess
        assert "task_001" in subprocess_display.active_processes
        mock_terminal_manager.allocate_subprocess_lines.assert_called_with(
            "task_001", ""
        )

    @pytest.mark.asyncio
    async def test_display_subprocess_error(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test displaying subprocess error message."""
        subprocess_display.terminal_manager = mock_terminal_manager
        subprocess_display.active_processes["task_001"] = {
            "process_name": "Test",
            "current_line_offset": 0,
            "message_count": 0,
            "last_message_time": 0,
        }

        await subprocess_display.display_subprocess_error("task_001", "Test error")

        mock_terminal_manager.write_to_subprocess_lines.assert_called()
        args = mock_terminal_manager.write_to_subprocess_lines.call_args
        assert "Error: Test error" in args[0][1]

    def test_get_active_subprocess_count(self, subprocess_display):
        """Test getting active subprocess count."""
        subprocess_display.active_processes = {
            "task_001": {"process_name": "Test1"},
            "task_002": {"process_name": "Test2"},
        }

        assert subprocess_display.get_active_subprocess_count() == 2

    def test_get_active_subprocess_ids(self, subprocess_display):
        """Test getting active subprocess IDs."""
        subprocess_display.active_processes = {
            "task_001": {"process_name": "Test1"},
            "task_002": {"process_name": "Test2"},
        }

        ids = subprocess_display.get_active_subprocess_ids()
        assert set(ids) == {"task_001", "task_002"}

    def test_is_subprocess_registered(self, subprocess_display):
        """Test checking if subprocess is registered."""
        subprocess_display.active_processes["task_001"] = {"process_name": "Test"}

        assert subprocess_display.is_subprocess_registered("task_001") is True
        assert subprocess_display.is_subprocess_registered("task_002") is False

    @pytest.mark.asyncio
    async def test_cleanup_inactive_subprocesses(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test cleanup of inactive subprocesses."""
        subprocess_display.terminal_manager = mock_terminal_manager
        subprocess_display.active_processes = {
            "task_001": {"process_name": "Active", "current_line_offset": 0},
            "task_002": {"process_name": "Inactive", "current_line_offset": 1},
        }

        active_task_ids = {"task_001"}  # Only task_001 is still active

        with patch("asyncio.sleep"):
            await subprocess_display.cleanup_inactive_subprocesses(active_task_ids)

        # task_002 should be cleaned up
        mock_terminal_manager.deallocate_subprocess_lines.assert_called_with("task_002")

    @pytest.mark.asyncio
    async def test_sync_with_subagent_manager(
        self, subprocess_display, mock_terminal_manager
    ):
        """Test syncing with subagent manager."""
        subprocess_display.terminal_manager = mock_terminal_manager

        # Mock subagent manager
        mock_manager = MagicMock()
        mock_manager.get_active_task_ids.return_value = ["task_001", "task_002"]
        mock_manager.subagents = {
            "task_001": MagicMock(description="File analysis"),
            "task_002": MagicMock(description="Code generation"),
        }

        await subprocess_display.sync_with_subagent_manager(mock_manager)

        # Should register new subagents
        assert mock_terminal_manager.allocate_subprocess_lines.call_count >= 2
