"""
DisplayManager for routing events to appropriate display handlers.

This module provides centralized display management that converts events
into appropriate console output, replacing scattered print statements
throughout the codebase.
"""

import asyncio
import json
import logging
import sys
from typing import Dict, List, Optional, Set

from cli_agent.core.event_system import (
    ErrorEvent,
    Event,
    EventBus,
    EventType,
    InterruptEvent,
    StatusEvent,
    SystemEvent,
    SystemMessageEvent,
    TextEvent,
    ToolCallEvent,
    ToolExecutionStartEvent,
    ToolResultEvent,
    UIStatusEvent,
    UserInputEvent,
)
from cli_agent.core.terminal_manager import get_terminal_manager

logger = logging.getLogger(__name__)


class DisplayManager:
    """
    Central display coordinator that converts events to console output.

    Replaces scattered print statements with event-driven display logic.
    Supports both interactive and non-interactive modes.
    """

    def __init__(
        self, event_bus: EventBus, interactive: bool = True, json_handler=None
    ):
        self.event_bus = event_bus
        self.interactive = interactive
        self.enabled_events: Set[EventType] = set()
        self.quiet_mode = False
        self.last_status_line = ""
        self.terminal_manager = get_terminal_manager()
        self.json_handler = json_handler  # For JSON streaming mode

        # Default enabled events for interactive mode
        if interactive:
            self.enabled_events.update(
                [
                    EventType.TEXT,
                    EventType.TOOL_CALL,
                    EventType.TOOL_EXECUTION_START,
                    EventType.TOOL_RESULT,
                    EventType.STATUS,
                    EventType.ERROR,
                    EventType.SYSTEM,
                    EventType.SYSTEM_MESSAGE,
                    EventType.UI_STATUS,
                    EventType.USER_INPUT,
                    EventType.INTERRUPT,
                ]
            )

        # Subscribe to all enabled events
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        """Subscribe to all enabled event types."""
        for event_type in self.enabled_events:
            self.event_bus.subscribe(event_type, self._handle_event)

    def enable_event_type(self, event_type: EventType):
        """Enable display for a specific event type."""
        if event_type not in self.enabled_events:
            self.enabled_events.add(event_type)
            self.event_bus.subscribe(event_type, self._handle_event)
            logger.debug(f"Enabled display for {event_type.value}")

    def disable_event_type(self, event_type: EventType):
        """Disable display for a specific event type."""
        if event_type in self.enabled_events:
            self.enabled_events.remove(event_type)
            self.event_bus.unsubscribe(event_type, self._handle_event)
            logger.debug(f"Disabled display for {event_type.value}")

    def set_quiet_mode(self, quiet: bool):
        """Enable/disable quiet mode (suppress non-essential output)."""
        self.quiet_mode = quiet
        if quiet:
            logger.debug("Quiet mode enabled")
        else:
            logger.debug("Quiet mode disabled")

    async def _handle_event(self, event: Event):
        """Route events to appropriate display handlers."""
        if self.quiet_mode and event.event_type in [
            EventType.UI_STATUS,
            EventType.STATUS,
        ]:
            return

        # Let the global interrupt manager handle Ctrl+C - don't interfere here

        try:
            if event.event_type == EventType.TEXT:
                await self._display_text_event(event)
            elif event.event_type == EventType.TOOL_CALL:
                await self._display_tool_call_event(event)
            elif event.event_type == EventType.TOOL_EXECUTION_START:
                await self._display_tool_execution_start(event)
            elif event.event_type == EventType.TOOL_RESULT:
                await self._display_tool_result(event)
            elif event.event_type == EventType.STATUS:
                await self._display_status_event(event)
            elif event.event_type == EventType.ERROR:
                await self._display_error_event(event)
            elif event.event_type == EventType.SYSTEM:
                await self._display_system_event(event)
            elif event.event_type == EventType.SYSTEM_MESSAGE:
                await self._display_system_message(event)
            elif event.event_type == EventType.UI_STATUS:
                await self._display_ui_status(event)
            elif event.event_type == EventType.USER_INPUT:
                await self._display_user_input_event(event)
            elif event.event_type == EventType.INTERRUPT:
                await self._display_interrupt_event(event)
            else:
                logger.debug(f"Unhandled event type: {event.event_type}")

        except Exception as e:
            logger.error(f"Error displaying event {event.event_type}: {e}")

    async def _display_text_event(self, event: TextEvent):
        """Display text content from LLM responses with enhanced formatting."""
        logger.debug(f"DisplayManager received TextEvent: {repr(event.content[:50])}")

        # Let the global interrupt manager handle Ctrl+C - don't interfere here

        if self.json_handler and event.content:
            # JSON streaming mode - emit assistant text message
            # Only send final complete text (not streaming chunks)
            if not event.is_streaming:
                self.json_handler.send_assistant_text(event.content)
        elif self.interactive and event.content:
            # Handle streaming content immediately without delays
            if event.is_streaming:
                # Clear any status line before streaming text
                if self.last_status_line:
                    self.terminal_manager.write_above_prompt("\r\x1b[K")
                    self.last_status_line = ""

                # Display streaming content immediately (no character delays for responsiveness)
                self.terminal_manager.write_above_prompt(event.content)
            else:
                # Non-streaming content - display immediately
                content_to_display = event.content
                # Apply markdown formatting if requested
                if event.is_markdown and len(event.content.strip()) > 0:
                    try:
                        import re

                        from cli_agent.core.formatting import ResponseFormatter

                        # Extract reasoning and thinking content
                        reasoning_content = ""
                        thinking_content = ""
                        remaining_content = event.content

                        # Extract reasoning tags
                        reasoning_match = re.search(
                            r"<reasoning>(.*?)</reasoning>",
                            remaining_content,
                            re.DOTALL,
                        )
                        if reasoning_match:
                            reasoning_content = (
                                f"<reasoning>{reasoning_match.group(1)}</reasoning>"
                            )
                            remaining_content = remaining_content.replace(
                                reasoning_match.group(0), ""
                            ).strip()

                        # Extract thinking tags (both <thinking> and <think>)
                        thinking_match = re.search(
                            r"<think(?:ing)?>(.*?)</think(?:ing)?>", remaining_content, re.DOTALL
                        )
                        if thinking_match:
                            # Preserve the original tag format
                            full_match = thinking_match.group(0)
                            thinking_content = full_match
                            remaining_content = remaining_content.replace(
                                thinking_match.group(0), ""
                            ).strip()

                        # Format the remaining content with markdown
                        formatted_remaining = ""
                        if remaining_content:
                            formatter = ResponseFormatter()
                            formatted_remaining = formatter.format_markdown(
                                remaining_content
                            )

                        # Reconstruct: reasoning + thinking + formatted content
                        content_to_display = ""
                        if reasoning_content:
                            content_to_display += reasoning_content + "\n\n"
                        if thinking_content:
                            content_to_display += thinking_content + "\n\n"
                        if formatted_remaining:
                            content_to_display += formatted_remaining

                        # Ensure proper line spacing
                        if content_to_display and not content_to_display.startswith(
                            "\n"
                        ):
                            content_to_display = "\n" + content_to_display

                    except Exception as e:
                        logger.warning(
                            f"Failed to format markdown with custom tags: {e}"
                        )
                        # Fallback to original content with line spacing
                        content_to_display = event.content
                        if not content_to_display.startswith("\n"):
                            content_to_display = "\n" + content_to_display

                self.terminal_manager.write_above_prompt(content_to_display)
                
                # Ensure newline after LLM response for proper prompt positioning
                if not content_to_display.endswith('\n'):
                    self.terminal_manager.write_above_prompt('\n')

            logger.debug("Displayed text content to console")

    async def _display_tool_execution_start(self, event: ToolExecutionStartEvent):
        """Display tool execution start notification."""
        if self.json_handler:
            # In JSON streaming mode, suppress tool execution start messages
            return
        elif self.interactive:
            args_summary = self._format_tool_arguments(event.arguments)
            display_text = f"🔧 Executing {event.tool_name}({args_summary})\n"

            # Clear any previous status line and display
            if self.last_status_line:
                self.terminal_manager.write_above_prompt("\r\x1b[K")
                self.last_status_line = ""
            self.terminal_manager.write_above_prompt(display_text)

    async def _display_tool_result(self, event: ToolResultEvent):
        """Display tool execution results."""
        if self.json_handler:
            # JSON streaming mode - emit tool result message
            try:
                self.json_handler.send_tool_result(
                    tool_use_id=event.tool_id, content=event.result, is_error=event.is_error
                )
            except Exception as e:
                logger.error(f"Error sending tool_result JSON event: {e}")
        elif self.interactive:
            status_icon = "❌" if event.is_error else "✅"
            time_info = (
                f" ({event.execution_time:.2f}s)" if event.execution_time else ""
            )

            result_preview = (
                event.result[:100] + "..." if len(event.result) > 100 else event.result
            )
            display_text = f"{status_icon} {event.tool_name} completed{time_info}: {result_preview}\n"

            self.terminal_manager.write_above_prompt(display_text)

    async def _display_status_event(self, event: StatusEvent):
        """Display status updates."""
        if self.json_handler:
            # In JSON streaming mode, suppress status messages
            return
        elif self.interactive:
            level_icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
            icon = level_icons.get(event.level, "📊")

            display_text = f"{icon} {event.status}"
            if event.details:
                display_text += f": {event.details}"

            # All status messages now go above the prompt with newlines
            # This ensures the prompt stays at the bottom
            display_text += "\n"
            self.terminal_manager.write_above_prompt(display_text)

    async def _display_error_event(self, event: ErrorEvent):
        """Display error notifications."""
        if self.interactive:
            error_type = f" ({event.error_type})" if event.error_type else ""
            display_text = f"❌ Error{error_type}: {event.error_message}\n"

            self.terminal_manager.write_above_prompt(display_text)

            # Optionally show stack trace in debug mode
            if event.stack_trace and logger.isEnabledFor(logging.DEBUG):
                self.terminal_manager.write_above_prompt(
                    f"Stack trace: {event.stack_trace}\n"
                )

    async def _display_system_message(self, event: SystemMessageEvent):
        """Display system messages like welcome, goodbye, thinking."""
        if self.interactive:
            emoji = event.emoji or self._get_default_emoji(event.message_type)
            display_text = f"{emoji} {event.message}"

            # For permission requests, add extra spacing
            if event.message_type == "permission_request":
                display_text = f"\n{display_text}\n"
            else:
                # Regular system messages always get a new line
                display_text += "\n"

            self.terminal_manager.write_above_prompt(display_text)

    async def _display_ui_status(self, event: UIStatusEvent):
        """Display UI status updates."""
        if self.interactive and not self.quiet_mode:
            type_icons = {"info": "ℹ️", "progress": "⏳", "completion": "✅"}
            icon = type_icons.get(event.status_type, "📊")

            display_text = f"{icon} {event.status_text}\n"

            # UI status goes above prompt
            self.terminal_manager.write_above_prompt(display_text)

            if event.duration:
                # Note: With persistent prompt, we don't need to clear status lines
                # as they scroll up naturally
                pass

    async def _display_interrupt_event(self, event: InterruptEvent):
        """Display interrupt/cancellation notifications."""
        if self.interactive:
            interrupt_icons = {"user": "🛑", "system": "⚠️", "timeout": "⏰"}
            icon = interrupt_icons.get(event.interrupt_type, "🚫")

            reason_text = f": {event.reason}" if event.reason else ""
            display_text = f"\n{icon} Operation interrupted{reason_text}\n"

            self.terminal_manager.write_above_prompt(display_text)

    async def _display_tool_call_event(self, event: ToolCallEvent):
        """Display tool call request event."""
        if self.json_handler:
            # JSON streaming mode - emit tool use message
            try:
                self.json_handler.send_assistant_tool_use(
                    tool_name=event.tool_name,
                    tool_input=event.arguments,
                    tool_use_id=event.tool_id,
                )
            except Exception as e:
                logger.error(f"Error sending tool_use JSON event: {e}")
        elif self.interactive:
            args_summary = self._format_tool_arguments(event.arguments)
            description = f" - {event.description}" if event.description else ""
            display_text = (
                f"🔗 Tool Call: {event.tool_name}({args_summary}){description}\n"
            )

            self.terminal_manager.write_above_prompt(display_text)

    async def _display_system_event(self, event: SystemEvent):
        """Display system-level events."""
        if self.interactive:
            # Handle hook execution events with meaningful information
            if event.system_type == "hook_execution_start":
                hook_type = getattr(event, 'hook_type', 'Unknown')
                hook_name = event.data.get('hook_name', 'unnamed') if hasattr(event, 'data') and event.data else 'unnamed'
                
                # Show informative hook execution message with name
                display_text = f"🪝 {hook_type} hook: {hook_name}\n"
                self.terminal_manager.write_above_prompt(display_text)
                return
            elif event.system_type == "hook_execution_complete":
                # Don't show completion messages - just clutter
                return
                
            system_icons = {"init": "🚀", "shutdown": "🛑", "config_change": "⚙️"}
            icon = system_icons.get(event.system_type, "🔧")

            display_text = f"{icon} System: {event.system_type}"
            if event.data:
                display_text += f" - {event.data}"
            display_text += "\n"

            self.terminal_manager.write_above_prompt(display_text)

    async def _display_user_input_event(self, event: UserInputEvent):
        """Display user input events (mainly for logging/debugging)."""
        if self.interactive and not self.quiet_mode:
            input_icons = {"chat": "💬", "command": "⌨️", "interrupt": "🛑"}
            icon = input_icons.get(event.input_type, "📝")

            # Only show first 50 chars of input for privacy
            input_preview = (
                event.input_text[:50] + "..."
                if len(event.input_text) > 50
                else event.input_text
            )
            display_text = f"{icon} User Input ({event.input_type}): {input_preview}\n"

            self.terminal_manager.write_above_prompt(display_text)

    def _format_tool_arguments(self, arguments: Dict) -> str:
        """Format tool arguments for display."""
        if not arguments:
            return ""

        # Show first few characters of each argument
        formatted_args = []
        for key, value in arguments.items():
            value_str = str(value)
            if len(value_str) > 30:
                value_str = value_str[:27] + "..."
            formatted_args.append(f"{key}={value_str}")

        return ", ".join(formatted_args)

    def _get_default_emoji(self, message_type: str) -> str:
        """Get default emoji for system message types."""
        emojis = {
            "welcome": "🤖",
            "goodbye": "👋",
            "thinking": "💭",
            "status": "📊",
            "info": "ℹ️",
        }
        return emojis.get(message_type, "🔔")

    async def _clear_status_after_delay(self, delay: float):
        """Clear status line after specified delay."""
        await asyncio.sleep(delay)
        # With persistent prompt, status lines scroll up naturally
        # No need to clear them manually
        pass

    def shutdown(self):
        """Clean up display manager."""
        # Unsubscribe from all events
        for event_type in self.enabled_events:
            self.event_bus.unsubscribe(event_type, self._handle_event)
        self.enabled_events.clear()

        # Stop the persistent prompt
        self.terminal_manager.stop_persistent_prompt()

        logger.debug("DisplayManager shutdown complete")


class JSONDisplayManager(DisplayManager):
    """
    Specialized display manager that outputs JSON events only.

    Used for non-interactive mode where external tools consume events.
    """

    def __init__(self, event_bus: EventBus, output_file=None):
        super().__init__(event_bus, interactive=False)
        self.output_file = output_file or sys.stdout

        # JSON mode shows all events
        self.enabled_events.update(
            [
                EventType.TEXT,
                EventType.TOOL_CALL,
                EventType.TOOL_EXECUTION_START,
                EventType.TOOL_RESULT,
                EventType.STATUS,
                EventType.ERROR,
                EventType.SYSTEM,
                EventType.SYSTEM_MESSAGE,
                EventType.UI_STATUS,
                EventType.USER_INPUT,
                EventType.INTERRUPT,
            ]
        )

        # Re-subscribe with updated events
        self._subscribe_to_events()

    async def _handle_event(self, event: Event):
        """Output events as JSON instead of console display."""
        try:
            json_output = event.to_json()
            print(json_output, file=self.output_file, flush=True)
        except Exception as e:
            logger.error(f"Error outputting JSON event: {e}")
