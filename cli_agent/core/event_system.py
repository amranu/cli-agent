"""
Event-driven display system for MCP agents.

This module provides a centralized event bus architecture that transforms
character-by-character streaming into discrete JSON events that drive all
display updates.
"""

import asyncio
import json
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types for the display system."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_RESULT = "tool_result"
    STATUS = "status"
    ERROR = "error"
    SYSTEM = "system"
    SYSTEM_MESSAGE = "system_message"  # Welcome, goodbye, thinking messages
    UI_STATUS = "ui_status"  # Non-critical UI status updates
    USER_INPUT = "user_input"
    INTERRUPT = "interrupt"


@dataclass
class Event(ABC):
    """Base class for all events in the system."""

    event_id: str = ""
    event_type: EventType = None
    timestamp: datetime = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_{uuid.uuid4().hex[:12]}"
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class TextEvent(Event):
    """Text content from LLM response."""

    content: str = ""
    chunk_id: Optional[str] = None
    is_markdown: bool = True
    is_streaming: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.TEXT


@dataclass
class ToolCallEvent(Event):
    """Tool execution request."""

    tool_name: str = ""
    tool_id: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.TOOL_CALL


@dataclass
class ToolExecutionStartEvent(Event):
    """Tool execution start notification."""

    tool_name: str = ""
    tool_id: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    streaming_mode: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.TOOL_EXECUTION_START


@dataclass
class ToolResultEvent(Event):
    """Tool execution result."""

    tool_id: str = ""
    tool_name: str = ""
    result: str = ""
    is_error: bool = False
    execution_time: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.TOOL_RESULT


@dataclass
class StatusEvent(Event):
    """Status updates (subagents, interruptions, etc)."""

    status: str = ""
    details: Optional[str] = None
    level: str = "info"  # info, warning, error

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.STATUS


@dataclass
class ErrorEvent(Event):
    """Error notifications."""

    error_message: str = ""
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.ERROR


@dataclass
class SystemEvent(Event):
    """System-level events (initialization, shutdown, etc)."""

    system_type: str = ""  # init, shutdown, config_change
    data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.SYSTEM


@dataclass
class UserInputEvent(Event):
    """User input events."""

    input_text: str = ""
    input_type: str = "chat"  # chat, command, interrupt

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.USER_INPUT


@dataclass
class InterruptEvent(Event):
    """Interrupt/cancellation events."""

    interrupt_type: str = ""  # user, system, timeout
    reason: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.INTERRUPT


@dataclass
class SystemMessageEvent(Event):
    """System messages like welcome, goodbye, thinking status."""

    message: str = ""
    message_type: str = "info"  # welcome, goodbye, thinking, status
    emoji: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.SYSTEM_MESSAGE


@dataclass
class UIStatusEvent(Event):
    """Non-critical UI status updates."""

    status_text: str = ""
    status_type: str = "info"  # info, progress, completion
    duration: Optional[float] = None  # How long to display (None = permanent)

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.UI_STATUS


class EventBus:
    """Central event dispatcher for display and processing coordination."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._event_history: List[Event] = []
        self.max_history = 1000  # Keep last 1000 events
        self.hook_manager = None  # Will be set by agent

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to specific event types."""
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value} events")

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Unsubscribe from specific event types."""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from {event_type.value} events")

    async def emit(self, event: Event):
        """Emit event to all subscribers."""
        # Set session ID if not already set
        if not event.session_id:
            event.session_id = self.session_id

        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self.max_history:
            self._event_history.pop(0)

        # For critical events that need immediate display (like streaming text),
        # process them immediately instead of queuing
        critical_event_types = {
            EventType.TEXT,
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.ERROR,
            EventType.STATUS,
            EventType.TOOL_EXECUTION_START,
            EventType.SYSTEM_MESSAGE,
            EventType.UI_STATUS,
        }

        if event.event_type in critical_event_types and self.is_running:
            # Process immediately for better streaming responsiveness
            subscribers = self.subscribers.get(event.event_type, [])
            for callback in subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in immediate event callback: {e}")
                    # Emit error event (but queue it to avoid recursion)
                    error_event = ErrorEvent(
                        event_id="",
                        timestamp=datetime.now(),
                        error_message=str(e),
                        error_type="callback_error",
                    )
                    # Queue the error event instead of immediate processing
                    await self.event_queue.put(error_event)
            
            # EXECUTE NOTIFICATION HOOKS for relevant events
            notification_event_types = {EventType.STATUS, EventType.SYSTEM_MESSAGE, EventType.ERROR}
            if (event.event_type in notification_event_types and 
                hasattr(self, 'hook_manager') and self.hook_manager):
                try:
                    from cli_agent.core.hooks.hook_config import HookType
                    
                    context = {
                        "message": self._extract_message_from_event(event),
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "session_id": event.session_id or self.session_id
                    }
                    
                    # Execute notification hooks in background to avoid blocking display
                    asyncio.create_task(
                        self.hook_manager.execute_hooks(HookType.NOTIFICATION, context)
                    )
                except Exception as e:
                    logger.warning(f"Error executing notification hooks: {e}")
        else:
            # Queue for processing (non-critical events or when not running)
            await self.event_queue.put(event)

        logger.debug(f"Emitted event: {event.event_type.value} - {event.event_id}")

    def emit_sync(self, event: Event):
        """Synchronous version of emit for non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.emit(event))
            else:
                loop.run_until_complete(self.emit(event))
        except RuntimeError:
            # No event loop running, create a new one
            asyncio.run(self.emit(event))

    async def start_processing(self):
        """Start the event processing loop."""
        if self.is_running:
            logger.warning("Event bus is already running")
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def stop_processing(self):
        """Stop the event processing loop."""
        if not self.is_running:
            return

        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")

    async def _process_events(self):
        """Main event processing loop."""
        logger.info("Event processing loop started")

        while self.is_running:
            try:
                # Wait for events with timeout to allow for clean shutdown
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                # Notify all subscribers for this event type
                subscribers = self.subscribers.get(event.event_type, [])
                for callback in subscribers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")
                        # Emit error event
                        error_event = ErrorEvent(
                            event_id="",
                            timestamp=datetime.now(),
                            error_message=str(e),
                            error_type="callback_error",
                        )
                        # Don't await this to avoid infinite loops
                        asyncio.create_task(self.emit(error_event))

            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                continue

    def get_event_history(
        self, event_type: Optional[EventType] = None, limit: Optional[int] = None
    ) -> List[Event]:
        """Get event history, optionally filtered by type and limited."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if limit:
            events = events[-limit:]

        return events

    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")
    
    def set_hook_manager(self, hook_manager):
        """Set the hook manager for this event bus."""
        self.hook_manager = hook_manager
        logger.debug("Hook manager set on event bus")
    
    def _extract_message_from_event(self, event: Event) -> str:
        """Extract a displayable message from an event for hooks."""
        if hasattr(event, 'message'):
            return str(event.message)
        elif hasattr(event, 'status'):
            return str(event.status)
        elif hasattr(event, 'error_message'):
            return str(event.error_message)
        elif hasattr(event, 'content'):
            return str(event.content)
        else:
            return f"{event.event_type.value} event"


class EventEmitter:
    """Utility class for emitting common events."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    async def emit_text(
        self,
        content: str,
        is_streaming: bool = False,
        is_markdown: bool = True,
        chunk_id: Optional[str] = None,
    ):
        """Emit a text event."""
        event = TextEvent(
            event_id="",
            timestamp=datetime.now(),
            content=content,
            is_streaming=is_streaming,
            is_markdown=is_markdown,
            chunk_id=chunk_id,
        )
        await self.event_bus.emit(event)

    async def emit_tool_call(
        self,
        tool_name: str,
        tool_id: str,
        arguments: Dict[str, Any],
        description: Optional[str] = None,
    ):
        """Emit a tool call event."""
        event = ToolCallEvent(
            event_id="",
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_id=tool_id,
            arguments=arguments,
            description=description,
        )
        await self.event_bus.emit(event)

    async def emit_tool_result(
        self,
        tool_id: str,
        tool_name: str,
        result: str,
        is_error: bool = False,
        execution_time: Optional[float] = None,
    ):
        """Emit a tool result event."""
        event = ToolResultEvent(
            event_id="",
            timestamp=datetime.now(),
            tool_id=tool_id,
            tool_name=tool_name,
            result=result,
            is_error=is_error,
            execution_time=execution_time,
        )
        await self.event_bus.emit(event)

    async def emit_status(
        self, status: str, details: Optional[str] = None, level: str = "info"
    ):
        """Emit a status event."""
        event = StatusEvent(
            event_id="",
            timestamp=datetime.now(),
            status=status,
            details=details,
            level=level,
        )
        await self.event_bus.emit(event)

    async def emit_error(
        self,
        error_message: str,
        error_type: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ):
        """Emit an error event."""
        event = ErrorEvent(
            event_id="",
            timestamp=datetime.now(),
            error_message=error_message,
            error_type=error_type,
            stack_trace=stack_trace,
        )
        await self.event_bus.emit(event)

    async def emit_interrupt(self, interrupt_type: str, reason: Optional[str] = None):
        """Emit an interrupt event."""
        event = InterruptEvent(
            event_id="",
            timestamp=datetime.now(),
            interrupt_type=interrupt_type,
            reason=reason,
        )
        await self.event_bus.emit(event)

    async def emit_tool_execution_start(
        self,
        tool_name: str,
        tool_id: str,
        arguments: Dict[str, Any],
        streaming_mode: bool = False,
    ):
        """Emit a tool execution start event."""
        event = ToolExecutionStartEvent(
            event_id="",
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_id=tool_id,
            arguments=arguments,
            streaming_mode=streaming_mode,
        )
        await self.event_bus.emit(event)

    async def emit_system_message(
        self, message: str, message_type: str = "info", emoji: Optional[str] = None
    ):
        """Emit a system message event."""
        event = SystemMessageEvent(
            event_id="",
            timestamp=datetime.now(),
            message=message,
            message_type=message_type,
            emoji=emoji,
        )
        await self.event_bus.emit(event)

    async def emit_ui_status(
        self,
        status_text: str,
        status_type: str = "info",
        duration: Optional[float] = None,
    ):
        """Emit a UI status event."""
        event = UIStatusEvent(
            event_id="",
            timestamp=datetime.now(),
            status_text=status_text,
            status_type=status_type,
            duration=duration,
        )
        await self.event_bus.emit(event)
