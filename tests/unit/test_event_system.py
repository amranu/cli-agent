"""Tests for event system functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cli_agent.core.event_system import (
    ErrorEvent,
    Event,
    EventBus,
    EventEmitter,
    EventType,
    InterruptEvent,
    StatusEvent,
    SystemMessageEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)


@pytest.mark.unit
class TestEventSystem:
    """Test cases for event system components."""

    def test_event_types_enum(self):
        """Test EventType enum has all expected values."""
        expected_types = {
            "TEXT",
            "TOOL_CALL",
            "TOOL_EXECUTION_START",
            "TOOL_RESULT",
            "STATUS",
            "ERROR",
            "SYSTEM",
            "SYSTEM_MESSAGE",
            "UI_STATUS",
            "USER_INPUT",
            "INTERRUPT",
        }
        actual_types = {item.name for item in EventType}
        assert expected_types.issubset(actual_types)

    def test_base_event_creation(self):
        """Test basic Event creation and properties."""
        # Event is abstract, use TextEvent instead
        event = TextEvent(content="test")
        assert event.event_type == EventType.TEXT
        assert event.timestamp is not None
        assert isinstance(event.event_id, str)

    def test_text_event_creation(self):
        """Test TextEvent creation with specific attributes."""
        event = TextEvent(content="Hello World", is_streaming=True, is_markdown=False)
        assert event.event_type == EventType.TEXT
        assert event.content == "Hello World"
        assert event.is_streaming is True
        assert event.is_markdown is False

    def test_tool_call_event_creation(self):
        """Test ToolCallEvent creation."""
        event = ToolCallEvent(
            tool_name="test_tool",
            tool_id="call_123",
            arguments={"arg1": "value1"},
            description="Test tool call",
        )
        assert event.event_type == EventType.TOOL_CALL
        assert event.tool_name == "test_tool"
        assert event.tool_id == "call_123"
        assert event.arguments == {"arg1": "value1"}
        assert event.description == "Test tool call"

    def test_tool_result_event_creation(self):
        """Test ToolResultEvent creation."""
        event = ToolResultEvent(
            tool_id="call_123",
            tool_name="test_tool",
            result="Success",
            is_error=False,
            execution_time=1.5,
        )
        assert event.event_type == EventType.TOOL_RESULT
        assert event.tool_id == "call_123"
        assert event.tool_name == "test_tool"
        assert event.result == "Success"
        assert event.is_error is False
        assert event.execution_time == 1.5

    def test_status_event_creation(self):
        """Test StatusEvent creation."""
        event = StatusEvent(
            status="Processing", details="Working on task", level="info"
        )
        assert event.event_type == EventType.STATUS
        assert event.status == "Processing"
        assert event.details == "Working on task"
        assert event.level == "info"

    def test_error_event_creation(self):
        """Test ErrorEvent creation."""
        event = ErrorEvent(
            error_message="Something went wrong",
            error_type="validation_error",
            stack_trace="Traceback...",
        )
        assert event.event_type == EventType.ERROR
        assert event.error_message == "Something went wrong"
        assert event.error_type == "validation_error"
        assert event.stack_trace == "Traceback..."

    def test_system_message_event_creation(self):
        """Test SystemMessageEvent creation."""
        event = SystemMessageEvent(
            message="Welcome to the system", message_type="welcome", emoji="👋"
        )
        assert event.event_type == EventType.SYSTEM_MESSAGE
        assert event.message == "Welcome to the system"
        assert event.message_type == "welcome"
        assert event.emoji == "👋"

    def test_interrupt_event_creation(self):
        """Test InterruptEvent creation."""
        event = InterruptEvent(interrupt_type="user", reason="User pressed Ctrl+C")
        assert event.event_type == EventType.INTERRUPT
        assert event.interrupt_type == "user"
        assert event.reason == "User pressed Ctrl+C"

    def test_event_bus_initialization(self):
        """Test EventBus initialization."""
        bus = EventBus()
        assert isinstance(bus.subscribers, dict)
        assert bus.is_running is False

    def test_event_bus_subscribe(self):
        """Test EventBus subscription."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.TEXT, callback)
        assert EventType.TEXT in bus.subscribers
        assert callback in bus.subscribers[EventType.TEXT]

    def test_event_bus_unsubscribe(self):
        """Test EventBus unsubscription."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.TEXT, callback)
        bus.unsubscribe(EventType.TEXT, callback)

        # Callback should be removed but event type may still exist
        assert callback not in bus.subscribers.get(EventType.TEXT, [])

    @pytest.mark.asyncio
    async def test_event_bus_emit(self):
        """Test EventBus event emission."""
        bus = EventBus()
        callback = AsyncMock()

        bus.subscribe(EventType.TEXT, callback)
        event = TextEvent(content="Test")

        # Start processing to handle events
        await bus.start_processing()
        await bus.emit(event)
        await asyncio.sleep(0.1)  # Let processing happen
        await bus.stop_processing()

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_event_bus_emit_sync(self):
        """Test EventBus synchronous event emission."""
        bus = EventBus()
        callback = MagicMock()

        bus.subscribe(EventType.TEXT, callback)
        event = TextEvent(content="Test")

        # Start processing to handle sync emit
        await bus.start_processing()
        bus.emit_sync(event)
        await asyncio.sleep(0.1)  # Let processing happen
        await bus.stop_processing()

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_event_bus_processing_loop(self):
        """Test EventBus processing loop."""
        bus = EventBus()
        callback = AsyncMock()

        bus.subscribe(EventType.TEXT, callback)

        # Start processing in background
        processing_task = asyncio.create_task(bus.start_processing())
        await asyncio.sleep(0.1)  # Let it start

        # Emit an event
        event = TextEvent(content="Test")
        await bus.emit(event)
        await asyncio.sleep(0.1)  # Let it process

        # Stop processing
        await bus.stop_processing()
        await processing_task

        callback.assert_called_with(event)

    def test_event_emitter_initialization(self):
        """Test EventEmitter initialization."""
        bus = EventBus()
        emitter = EventEmitter(bus)
        assert emitter.event_bus is bus

    @pytest.mark.asyncio
    async def test_event_emitter_emit_text(self):
        """Test EventEmitter text emission."""
        bus = EventBus()
        emitter = EventEmitter(bus)
        callback = AsyncMock()

        bus.subscribe(EventType.TEXT, callback)

        # Start processing to handle events
        await bus.start_processing()
        await emitter.emit_text("Hello", is_streaming=True, is_markdown=False)
        await asyncio.sleep(0.1)  # Let processing happen
        await bus.stop_processing()

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, TextEvent)
        assert event.content == "Hello"
        assert event.is_streaming is True
        assert event.is_markdown is False

    @pytest.mark.asyncio
    async def test_event_emitter_emit_tool_call(self):
        """Test EventEmitter tool call emission."""
        bus = EventBus()
        emitter = EventEmitter(bus)
        callback = AsyncMock()

        bus.subscribe(EventType.TOOL_CALL, callback)

        # Start processing to handle events
        await bus.start_processing()
        await emitter.emit_tool_call(
            tool_name="test", tool_id="123", arguments={"key": "value"}
        )
        await asyncio.sleep(0.1)  # Let processing happen
        await bus.stop_processing()

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, ToolCallEvent)
        assert event.tool_name == "test"

    @pytest.mark.asyncio
    async def test_event_emitter_emit_status(self):
        """Test EventEmitter status emission."""
        bus = EventBus()
        emitter = EventEmitter(bus)
        callback = AsyncMock()

        bus.subscribe(EventType.STATUS, callback)

        # Start processing to handle events
        await bus.start_processing()
        await emitter.emit_status("Processing", details="Working", level="info")
        await asyncio.sleep(0.1)  # Let processing happen
        await bus.stop_processing()

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, StatusEvent)
        assert event.status == "Processing"

    @pytest.mark.asyncio
    async def test_event_emitter_emit_error(self):
        """Test EventEmitter error emission."""
        bus = EventBus()
        emitter = EventEmitter(bus)
        callback = AsyncMock()

        bus.subscribe(EventType.ERROR, callback)

        # Start processing to handle events
        await bus.start_processing()
        await emitter.emit_error("Error occurred", error_type="test_error")
        await asyncio.sleep(0.1)  # Let processing happen
        await bus.stop_processing()

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, ErrorEvent)
        assert event.error_message == "Error occurred"

    @pytest.mark.asyncio
    async def test_event_bus_error_handling(self):
        """Test EventBus handles callback errors gracefully."""
        bus = EventBus()

        # Create a callback that raises an exception
        failing_callback = AsyncMock(side_effect=Exception("Callback failed"))
        working_callback = AsyncMock()

        bus.subscribe(EventType.TEXT, failing_callback)
        bus.subscribe(EventType.TEXT, working_callback)

        event = TextEvent(content="Test")

        # Start processing to handle events
        await bus.start_processing()
        # Should not raise exception despite failing callback
        await bus.emit(event)
        await asyncio.sleep(0.1)  # Let processing happen
        await bus.stop_processing()

        # Working callback should still be called
        working_callback.assert_called_once_with(event)

    def test_event_to_json(self):
        """Test event JSON serialization."""
        event = TextEvent(content="Hello")
        json_str = event.to_json()

        assert isinstance(json_str, str)
        assert "Hello" in json_str
        assert "text" in json_str  # EventType.TEXT.value is "text", not "TEXT"
        assert event.event_id in json_str
