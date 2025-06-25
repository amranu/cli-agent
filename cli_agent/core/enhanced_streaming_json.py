#!/usr/bin/env python3
"""
Enhanced streaming JSON support that integrates with the event system.
Extends the existing streaming_json.py with event-driven architecture.
"""

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from cli_agent.core.event_system import (
    ErrorEvent,
    EventBus,
    StatusEvent,
    SystemEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class EnhancedStreamingJSONHandler:
    """Enhanced streaming JSON handler that integrates with the event system."""

    def __init__(
        self, session_id: Optional[str] = None, event_bus: Optional[EventBus] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.event_bus = event_bus
        self.output_format = "stream-json"
        self.input_format = "stream-json"
        self._current_model = "deepseek-chat"
        self._message_counter = 0

    def set_event_bus(self, event_bus: EventBus):
        """Set the event bus for this handler."""
        self.event_bus = event_bus

    def set_model(self, model: str):
        """Set the current model name."""
        self._current_model = model

    async def send_system_init(
        self, cwd: str, tools: List[str], mcp_servers: List[str], model: str
    ):
        """Send system initialization message and emit system event."""
        # Emit system event to event bus
        if self.event_bus:
            await self.event_bus.emit(
                SystemEvent(
                    message="System initialized",
                    details={
                        "cwd": cwd,
                        "tools": tools,
                        "mcp_servers": mcp_servers,
                        "model": model,
                        "session_id": self.session_id,
                    },
                )
            )

        # Send JSON message
        msg = {
            "type": "system",
            "session_id": self.session_id,
            "subtype": "init",
            "cwd": cwd,
            "tools": tools,
            "mcp_servers": mcp_servers,
            "model": model,
            "permissionMode": "default",
            "apiKeySource": "none",
        }
        self._output_json(msg)

    async def send_assistant_text(self, text: str, message_id: Optional[str] = None):
        """Send assistant text response and emit text event."""
        msg_id = message_id or self._generate_message_id()

        # Emit text event to event bus
        if self.event_bus:
            await self.event_bus.emit(
                TextEvent(
                    content=text,
                    source="assistant",
                    message_id=msg_id,
                    session_id=self.session_id,
                )
            )

        # Send JSON message
        message = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": self._current_model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": self._get_usage_stats(),
        }

        msg = {
            "type": "assistant",
            "session_id": self.session_id,
            "message": message,
            "parent_tool_use_id": None,
        }
        self._output_json(msg)

    async def send_assistant_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ):
        """Send assistant tool use and emit tool call event."""
        msg_id = message_id or self._generate_message_id()
        tool_id = tool_use_id or self._generate_tool_id()

        # Emit tool call event to event bus
        if self.event_bus:
            await self.event_bus.emit(
                ToolCallEvent(
                    tool_name=tool_name,
                    arguments=tool_input,
                    tool_id=tool_id,
                    message_id=msg_id,
                    session_id=self.session_id,
                )
            )

        # Send JSON message
        message = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": self._current_model,
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            ],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": self._get_usage_stats(),
        }

        msg = {
            "type": "assistant",
            "session_id": self.session_id,
            "message": message,
            "parent_tool_use_id": None,
        }
        self._output_json(msg)
        return tool_id

    async def send_tool_result(
        self, tool_use_id: str, content: str, is_error: bool = False
    ):
        """Send tool execution result and emit tool result event."""
        # Emit tool result event to event bus
        if self.event_bus:
            await self.event_bus.emit(
                ToolResultEvent(
                    tool_id=tool_use_id,
                    result=content,
                    is_error=is_error,
                    session_id=self.session_id,
                )
            )

        # Send JSON message
        message = {
            "role": "user",
            "content": [
                {
                    "tool_use_id": tool_use_id,
                    "type": "tool_result",
                    "content": content,
                    "is_error": is_error,
                }
            ],
        }

        msg = {
            "type": "user",
            "session_id": self.session_id,
            "message": message,
            "parent_tool_use_id": None,
        }
        self._output_json(msg)

    async def send_status_update(self, status: str, details: Optional[str] = None):
        """Send status update and emit status event."""
        # Emit status event to event bus
        if self.event_bus:
            await self.event_bus.emit(
                StatusEvent(
                    status=status,
                    details=details,
                    session_id=self.session_id,
                )
            )

        # Send JSON status message
        msg = {
            "type": "status",
            "session_id": self.session_id,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self._output_json(msg)

    async def send_error(self, error_message: str, error_type: str = "general"):
        """Send error message and emit error event."""
        # Emit error event to event bus
        if self.event_bus:
            await self.event_bus.emit(
                ErrorEvent(
                    error_message=error_message,
                    error_type=error_type,
                    session_id=self.session_id,
                )
            )

        # Send JSON error message
        msg = {
            "type": "error",
            "session_id": self.session_id,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
        }
        self._output_json(msg)

    async def process_streaming_response(self, response_generator):
        """Process streaming response and emit appropriate events."""
        full_response = ""
        tool_calls = []

        async for chunk in response_generator:
            if isinstance(chunk, str):
                full_response += chunk
                # Emit text event for each chunk
                if self.event_bus:
                    await self.event_bus.emit(
                        TextEvent(
                            content=chunk,
                            source="assistant",
                            chunk_id=str(uuid.uuid4())[:8],
                            session_id=self.session_id,
                        )
                    )

                # Send JSON message for chunk
                if not any(
                    marker in chunk
                    for marker in ["<tool_call>", "</tool_call>", "tool_calls"]
                ):
                    await self.send_assistant_text(chunk)

            # Handle tool calls if present
            elif hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls"):
                    if choice.delta.tool_calls:
                        tool_calls.extend(choice.delta.tool_calls)

        return full_response, tool_calls

    async def handle_tool_execution_sequence(
        self, tool_calls: List[Dict[str, Any]], tool_results: List[str]
    ):
        """Handle a complete tool execution sequence with events."""
        for i, tool_call in enumerate(tool_calls):
            # Send tool use
            tool_id = await self.send_assistant_tool_use(
                tool_name=tool_call["function"]["name"],
                tool_input=json.loads(tool_call["function"]["arguments"]),
                tool_use_id=tool_call.get("id"),
            )

            # Send status update
            await self.send_status_update(
                f"Executing {tool_call['function']['name']}",
                f"Tool {i+1} of {len(tool_calls)}",
            )

            # Send tool result if available
            if i < len(tool_results):
                await self.send_tool_result(tool_id, tool_results[i], is_error=False)

    def read_input_json(self) -> Optional[Dict[str, Any]]:
        """Read streaming JSON input from stdin."""
        try:
            line = sys.stdin.readline().strip()
            if line:
                return json.loads(line)
            return None
        except json.JSONDecodeError:
            return None
        except EOFError:
            return None

    def _output_json(self, msg: Dict[str, Any]):
        """Output JSON message to stdout."""
        json_str = json.dumps(msg)
        print(json_str, flush=True)

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        self._message_counter += 1
        return f"msg_{uuid.uuid4().hex[:16]}_{self._message_counter}"

    def _generate_tool_id(self) -> str:
        """Generate a unique tool ID."""
        return f"toolu_{uuid.uuid4().hex[:20]}"

    def _get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return {
            "input_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 1,
            "service_tier": "standard",
        }


class EventDrivenJSONOutput:
    """Helper class to convert events to JSON output format."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def handle_text_event(self, event: TextEvent):
        """Convert text event to JSON output."""
        msg = {
            "type": "assistant",
            "session_id": self.session_id,
            "message": {
                "id": event.message_id or f"msg_{uuid.uuid4().hex[:20]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": event.content}],
                "stop_reason": None,
                "stop_sequence": None,
            },
        }
        print(json.dumps(msg), flush=True)

    async def handle_tool_call_event(self, event: ToolCallEvent):
        """Convert tool call event to JSON output."""
        msg = {
            "type": "assistant",
            "session_id": self.session_id,
            "message": {
                "id": event.message_id or f"msg_{uuid.uuid4().hex[:20]}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": event.tool_id,
                        "name": event.tool_name,
                        "input": event.arguments,
                    }
                ],
                "stop_reason": None,
                "stop_sequence": None,
            },
        }
        print(json.dumps(msg), flush=True)

    async def handle_tool_result_event(self, event: ToolResultEvent):
        """Convert tool result event to JSON output."""
        msg = {
            "type": "user",
            "session_id": self.session_id,
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": event.tool_id,
                        "type": "tool_result",
                        "content": event.result,
                        "is_error": event.is_error,
                    }
                ],
            },
        }
        print(json.dumps(msg), flush=True)

    async def handle_status_event(self, event: StatusEvent):
        """Convert status event to JSON output."""
        msg = {
            "type": "status",
            "session_id": self.session_id,
            "status": event.status,
            "details": event.details,
            "timestamp": event.timestamp.isoformat(),
        }
        print(json.dumps(msg), flush=True)

    async def handle_error_event(self, event: ErrorEvent):
        """Convert error event to JSON output."""
        msg = {
            "type": "error",
            "session_id": self.session_id,
            "error_type": event.error_type,
            "error_message": event.error_message,
            "timestamp": event.timestamp.isoformat(),
        }
        print(json.dumps(msg), flush=True)


async def main():
    """Test the enhanced streaming JSON handler."""
    import os

    from cli_agent.core.display_manager import DisplayManager
    from cli_agent.core.event_system import EventBus

    # Create event bus and display manager
    event_bus = EventBus()
    display_manager = DisplayManager(event_bus)

    # Create enhanced handler
    handler = EnhancedStreamingJSONHandler(event_bus=event_bus)

    # Start event processing
    event_task = asyncio.create_task(event_bus.start_processing())
    display_task = asyncio.create_task(display_manager.start())

    try:
        # Send system init
        await handler.send_system_init(
            cwd=os.getcwd(),
            tools=["Bash", "Read", "Write", "todo_read", "todo_write"],
            mcp_servers=[],
            model="deepseek-chat",
        )

        # Send assistant text
        await handler.send_assistant_text("I'll help you with your request.")

        # Send tool use
        tool_id = await handler.send_assistant_tool_use(
            tool_name="Bash",
            tool_input={"command": "ls -la", "description": "List current directory"},
        )

        # Send tool result
        await handler.send_tool_result(
            tool_id, "total 10\ndrwxr-xr-x 2 user user 4096 Jan 1 12:00 ."
        )

        # Send status update
        await handler.send_status_update(
            "Operation completed", "All tools executed successfully"
        )

        # Wait a bit for processing
        await asyncio.sleep(1)

    finally:
        # Clean shutdown
        event_task.cancel()
        display_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass
        try:
            await display_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
