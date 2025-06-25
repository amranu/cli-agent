"""Subprocess display coordination for clean multi-subagent terminal output."""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Set

from cli_agent.core.event_system import EventBus, StatusEvent, TextEvent
from cli_agent.core.terminal_manager import get_terminal_manager

logger = logging.getLogger(__name__)


class SubprocessDisplayCoordinator:
    """Coordinates display of multiple subprocess outputs using dedicated terminal lines."""

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus
        self.terminal_manager = get_terminal_manager()
        self.active_processes: Dict[str, Dict[str, any]] = {}
        self.process_line_mapping: Dict[str, int] = (
            {}
        )  # process_id -> current line offset

        # Subscribe to relevant events if event bus available
        if self.event_bus:
            from cli_agent.core.event_system import EventType

            self.event_bus.subscribe(EventType.STATUS, self._handle_status_event)

    async def _handle_status_event(self, event: StatusEvent):
        """Handle status events that might be from subprocesses."""
        # Check if this is a subagent message
        if hasattr(event, "status") and "[SUBAGENT-" in event.status:
            await self._process_subagent_message(event.status)

    async def _process_subagent_message(self, message: str):
        """Process and display subagent messages using allocated lines."""
        # Extract task ID from message format: "📨 [SUBAGENT-task_id] type: content"
        match = re.search(r"\[SUBAGENT-([^\]]+)\]", message)
        if not match:
            return

        task_id = match.group(1)

        # Extract message content after the format prefix
        content_match = re.search(r"\[SUBAGENT-[^\]]+\]\s*[^:]*:\s*(.*)", message)
        content = content_match.group(1) if content_match else message

        await self.display_subprocess_message(task_id, content)

    async def register_subprocess(
        self, process_id: str, process_name: str = ""
    ) -> bool:
        """Register a new subprocess for display coordination.

        Returns:
            True if successfully allocated lines, False if no space available
        """
        if process_id in self.active_processes:
            return True  # Already registered

        # Attempt to allocate terminal lines
        allocation = self.terminal_manager.allocate_subprocess_lines(
            process_id, process_name
        )
        if allocation is None:
            logger.warning(f"No terminal space available for subprocess {process_id}")
            return False

        start_line, end_line = allocation

        # Register the process
        self.active_processes[process_id] = {
            "process_name": process_name,
            "start_line": start_line,
            "end_line": end_line,
            "current_line_offset": 0,
            "message_count": 0,
            "last_message_time": asyncio.get_event_loop().time(),
        }

        # Initialize display with process header
        header = f"🤖 {process_name or process_id} - Starting..."
        self.terminal_manager.write_to_subprocess_lines(process_id, header, 0)

        logger.info(
            f"Registered subprocess {process_id} with lines {start_line}-{end_line}"
        )
        return True

    async def unregister_subprocess(self, process_id: str):
        """Unregister a completed subprocess."""
        if process_id in self.active_processes:
            # Show completion message
            completion_msg = f"✅ Completed"
            self.terminal_manager.write_to_subprocess_lines(
                process_id,
                completion_msg,
                self.active_processes[process_id]["current_line_offset"],
            )

            # Brief delay before cleanup
            await asyncio.sleep(1.0)

            # Deallocate terminal lines
            self.terminal_manager.deallocate_subprocess_lines(process_id)

            # Remove from tracking
            del self.active_processes[process_id]
            if process_id in self.process_line_mapping:
                del self.process_line_mapping[process_id]

            logger.info(f"Unregistered subprocess {process_id}")

    def display_subprocess_message_sync(
        self, task_id: str, message: str, message_type: str = "info"
    ):
        """Display a message synchronously for a specific subagent using its allocated lines."""
        if task_id not in self.active_processes:
            # If not registered, try to register now using proper task_id
            allocation = self.terminal_manager.allocate_subprocess_lines(
                task_id, task_id[:20]
            )
            if allocation is None:
                # Fall back to regular display - just print for now
                print(f"[{task_id}] {message}")
                return

            start_line, end_line = allocation
            # Register the process manually
            self.active_processes[task_id] = {
                "process_name": task_id[:20],
                "start_line": start_line,
                "end_line": end_line,
                "current_line_offset": 0,
                "message_count": 0,
                "last_message_time": 0,
            }

            # Initialize display with process header
            header = f"🤖 {task_id[:20]} - Starting..."
            self.terminal_manager.write_to_subprocess_lines(task_id, header, 0)

        process_info = self.active_processes[task_id]

        # Determine which line to use (rotate through available lines)
        lines_available = self.terminal_manager.lines_per_subprocess
        current_offset = process_info["current_line_offset"]

        # Add emoji based on message type
        if "error" in message_type.lower():
            prefix = "❌"
        elif "complete" in message.lower() or "finished" in message.lower():
            prefix = "✅"
        elif "start" in message.lower() or "begin" in message.lower():
            prefix = "🚀"
        else:
            prefix = "💬"

        # Format message with timestamp if it's a status update
        formatted_message = f"{prefix} {message}"

        # Write to the current line
        self.terminal_manager.write_to_subprocess_lines(
            task_id, formatted_message, current_offset
        )

        # Update process info
        process_info["message_count"] += 1
        process_info["last_message_time"] = 0  # Use 0 for sync version

        # Advance to next line, wrapping around
        process_info["current_line_offset"] = (current_offset + 1) % lines_available

    async def display_subprocess_message(
        self, task_id: str, message: str, message_type: str = "info"
    ):
        """Display a message for a specific subagent using its allocated lines."""
        if task_id not in self.active_processes:
            # If not registered, try to register now using proper task_id
            if not await self.register_subprocess(task_id):
                # Fall back to regular display
                if self.event_bus:
                    event = StatusEvent(
                        status=f"[{task_id}] {message}", level=message_type
                    )
                    await self.event_bus.emit(event)
                return

        process_info = self.active_processes[task_id]

        # Determine which line to use (rotate through available lines)
        lines_available = self.terminal_manager.lines_per_subprocess
        current_offset = process_info["current_line_offset"]

        # Add emoji based on message type
        if "error" in message_type.lower():
            prefix = "❌"
        elif "complete" in message.lower() or "finished" in message.lower():
            prefix = "✅"
        elif "start" in message.lower() or "begin" in message.lower():
            prefix = "🚀"
        else:
            prefix = "💬"

        # Format message with timestamp if it's a status update
        formatted_message = f"{prefix} {message}"

        # Write to the current line
        self.terminal_manager.write_to_subprocess_lines(
            task_id, formatted_message, current_offset
        )

        # Update process info
        process_info["message_count"] += 1
        process_info["last_message_time"] = asyncio.get_event_loop().time()

        # Advance to next line, wrapping around
        process_info["current_line_offset"] = (current_offset + 1) % lines_available

    async def display_subprocess_progress(
        self, process_id: str, status: str, details: str = ""
    ):
        """Display progress update for a subprocess."""
        if details:
            message = f"{status}: {details}"
        else:
            message = status

        await self.display_subprocess_message(process_id, message, "progress")

    async def display_subprocess_error(self, process_id: str, error_message: str):
        """Display error message for a subprocess."""
        await self.display_subprocess_message(
            process_id, f"Error: {error_message}", "error"
        )

    def get_active_subprocess_count(self) -> int:
        """Get number of active subprocesses."""
        return len(self.active_processes)

    def get_active_subprocess_ids(self) -> List[str]:
        """Get list of active subprocess IDs."""
        return list(self.active_processes.keys())

    def is_subprocess_registered(self, process_id: str) -> bool:
        """Check if a subprocess is registered."""
        return process_id in self.active_processes

    async def update_subprocess_status(self, process_id: str, status: str):
        """Update the status line (line 0) for a subprocess."""
        if process_id in self.active_processes:
            status_msg = f"📊 Status: {status}"
            self.terminal_manager.write_to_subprocess_lines(process_id, status_msg, 0)

    async def cleanup_inactive_subprocesses(self, active_task_ids: Set[str]):
        """Clean up subprocesses that are no longer active."""
        inactive_processes = set(self.active_processes.keys()) - active_task_ids

        for process_id in inactive_processes:
            await self.unregister_subprocess(process_id)

    async def sync_with_subagent_manager(self, subagent_manager):
        """Synchronize display with current subagent manager state."""
        if not subagent_manager:
            return

        # Get currently active task IDs
        try:
            active_task_ids = set(subagent_manager.get_active_task_ids())
        except AttributeError:
            # Fallback if method doesn't exist
            active_task_ids = set()

        # Register any new subagents
        for task_id in active_task_ids:
            if not self.is_subprocess_registered(task_id):
                # Get subagent info if available
                process_name = task_id
                try:
                    if hasattr(subagent_manager, "subagents"):
                        subagent = subagent_manager.subagents.get(task_id)
                        if subagent and hasattr(subagent, "description"):
                            process_name = subagent.description[
                                :30
                            ]  # Truncate long descriptions
                except Exception:
                    pass

                await self.register_subprocess(task_id, process_name)

        # Clean up completed subagents
        await self.cleanup_inactive_subprocesses(active_task_ids)


# Global subprocess display coordinator instance
_subprocess_display_coordinator = None


def get_subprocess_display_coordinator(
    event_bus: Optional[EventBus] = None,
) -> SubprocessDisplayCoordinator:
    """Get the global subprocess display coordinator instance."""
    global _subprocess_display_coordinator
    if _subprocess_display_coordinator is None:
        _subprocess_display_coordinator = SubprocessDisplayCoordinator(event_bus)
    return _subprocess_display_coordinator
