"""Chat interface and interaction handling for MCP agents."""

import asyncio
import logging
import signal
import sys
from typing import Any, Dict, List, Optional

from cli_agent.core.event_system import (
    ErrorEvent,
    EventEmitter,
    InterruptEvent,
    StatusEvent,
    SystemMessageEvent,
    TextEvent,
)
from cli_agent.core.global_interrupt import get_global_interrupt_manager
from cli_agent.core.terminal_manager import get_terminal_manager

logger = logging.getLogger(__name__)


class ChatInterface:
    """Handles interactive chat sessions, input processing, and conversation management."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent
        self.conversation_active = False
        self.interrupt_received = False
        self.global_interrupt_manager = get_global_interrupt_manager()
        self.terminal_manager = get_terminal_manager()

        # Initialize event emitter if event bus is available
        self.event_emitter = None
        if hasattr(agent, "event_bus") and agent.event_bus:
            self.event_emitter = EventEmitter(agent.event_bus)

    async def interactive_chat(
        self, input_handler, existing_messages: Optional[List[Dict[str, Any]]] = None
    ):
        """Main interactive chat loop implementation."""
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        # Store input handler on agent for permission system
        self.agent._input_handler = input_handler

        # Initialize messages from existing if provided
        messages = existing_messages[:] if existing_messages else []

        # Display welcome message
        if not existing_messages:
            await self.display_welcome_message()

        # Set up signal handlers
        self.setup_signal_handlers()
        self.start_conversation()

        # Start event bus processing if available
        if hasattr(self.agent, "event_bus") and self.agent.event_bus:
            await self.agent.event_bus.start_processing()

        current_task = None

        while self.is_conversation_active():
            try:
                # Check for global interrupt first - this should ALWAYS return to prompt
                if self.global_interrupt_manager.is_interrupted():
                    if current_task and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
                    await self._emit_interruption(
                        "Operation cancelled, returning to prompt", "global"
                    )
                    self.global_interrupt_manager.clear_interrupt()
                    input_handler.interrupted = False
                    current_task = None
                    continue

                # Check if we were interrupted during a previous operation
                if input_handler.interrupted:
                    if current_task and not current_task.done():
                        current_task.cancel()
                        await self._emit_interruption(
                            "Operation cancelled by user", "user"
                        )
                    input_handler.interrupted = False
                    current_task = None
                    continue

                # Check if subagents are active - if so, don't prompt for input
                if (
                    hasattr(self.agent, "subagent_manager")
                    and self.agent.subagent_manager
                ):
                    active_count = self.agent.subagent_manager.get_active_count()
                    if active_count > 0:
                        # Subagents are running, wait instead of prompting for input
                        await asyncio.sleep(0.5)
                        continue

                # Start persistent prompt for user input
                self.terminal_manager.start_persistent_prompt("You: ")

                # Get user input with smart multiline detection (without prompt since it's persistent)
                user_input = input_handler.get_multiline_input("")

                # Stop persistent prompt once input is received
                self.terminal_manager.stop_persistent_prompt()

                if user_input is None:  # Interrupted or EOF
                    if current_task and not current_task.done():
                        current_task.cancel()
                        await self._emit_interruption(
                            "Operation cancelled by user", "user"
                        )
                    current_task = None
                    # If interrupted (EOF), exit the conversation
                    if input_handler.interrupted:
                        input_handler.interrupted = False
                        break
                    continue

                # Check for empty input (could indicate EOF without None return)
                if user_input == "":
                    # Empty input could indicate EOF in non-interactive mode
                    if not sys.stdin.isatty():
                        await self._emit_system_message(
                            "End of input detected, exiting...", "info", "🔚"
                        )
                        break
                    continue

                # Handle user input
                if user_input.startswith("/"):
                    # Handle slash command asynchronously
                    slash_result = await self.handle_slash_command(user_input)
                    if slash_result:
                        # Check if it's a quit command
                        if isinstance(slash_result, dict) and slash_result.get("quit"):
                            await self._emit_system_message(
                                slash_result.get("status", "Goodbye!"), "goodbye", "👋"
                            )
                            break
                        # Check if it's a reload_host command
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "reload_host"
                        ):
                            await self._emit_system_message(
                                slash_result.get("status", "Reloading..."),
                                "status",
                                "🔄",
                            )
                            # Return dict with reload_host key and current messages
                            return {
                                "reload_host": slash_result["reload_host"],
                                "messages": messages,
                            }
                        # Check if it's a clear_messages command
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "clear_messages"
                        ):
                            await self._emit_system_message(
                                slash_result.get("status", "Messages cleared."),
                                "status",
                                "🗑️",
                            )
                            messages.clear()  # Clear the messages list
                        # Check if it's a compacted_messages command
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "compacted_messages"
                        ):
                            await self._emit_system_message(
                                slash_result.get("status", "Messages compacted."),
                                "status",
                                "🗃",
                            )
                            messages[:] = slash_result[
                                "compacted_messages"
                            ]  # Replace messages with compacted ones
                        # Check if it's a send_to_llm command (like /init)
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "send_to_llm"
                        ):
                            await self._emit_system_message(
                                slash_result.get("status", "Sending to LLM..."),
                                "status",
                                "📤",
                            )
                            # Add the prompt as a user message and continue processing
                            messages.append(
                                {"role": "user", "content": slash_result["send_to_llm"]}
                            )
                            # Don't continue, let it process the LLM prompt
                        else:
                            await self._emit_system_message(str(slash_result), "info")
                            continue

                    # If we got here, it means we have a send_to_llm command to process
                    if isinstance(slash_result, dict) and slash_result.get(
                        "send_to_llm"
                    ):
                        # Continue to process the LLM request (don't skip to next iteration)
                        pass
                    else:
                        continue

                processed_input = self.handle_user_input(user_input)
                if processed_input is None:
                    continue  # Empty input or handled specially

                if not self.is_conversation_active():
                    break  # User quit

                # Process the user input
                if processed_input.strip():
                    messages.append({"role": "user", "content": processed_input})

                    # Check if we should auto-compact before making the API call
                    if self.should_compact_conversation(messages):
                        tokens_before = self.agent._estimate_token_count(messages)
                        await self._emit_status(
                            f"Auto-compacting conversation (was ~{tokens_before} tokens)...",
                            "info",
                        )
                        try:
                            messages = self.agent.compact_conversation(messages)
                            tokens_after = self.agent._estimate_token_count(messages)
                            await self._emit_status(
                                f"Compacted to ~{tokens_after} tokens", "info"
                            )
                        except Exception as e:
                            await self._emit_error(
                                f"Auto-compact failed: {e}", "compaction_error"
                            )

                    try:
                        # Make API call interruptible by running in a task
                        # Emit thinking message via event system
                        await self._emit_system_message(
                            "Thinking... (press ESC to interrupt)", "thinking", "💭"
                        )

                        logger.info("Creating task for generate_response")
                        current_task = asyncio.create_task(
                            self.agent.generate_response(messages, stream=True)
                        )
                        logger.info("Task created, waiting for completion")
                        logger.info(
                            f"Messages being sent to LLM: {len(messages)} messages"
                        )

                        # Create a background task to monitor for escape key
                        async def monitor_escape():
                            import select
                            import termios
                            import tty

                            old_settings = None
                            try:
                                while not current_task.done():
                                    try:
                                        if (
                                            sys.stdin.isatty()
                                            and select.select([sys.stdin], [], [], 0.1)[
                                                0
                                            ]
                                        ):
                                            # Set up raw mode for a single character read
                                            if old_settings is None:
                                                old_settings = termios.tcgetattr(
                                                    sys.stdin.fileno()
                                                )
                                                tty.setraw(sys.stdin.fileno())

                                            char = sys.stdin.read(1)
                                            if char == "\x1b":  # Escape key
                                                input_handler.interrupted = True
                                                return
                                        await asyncio.sleep(0.1)
                                    except Exception:
                                        await asyncio.sleep(0.1)
                            finally:
                                if old_settings is not None:
                                    termios.tcsetattr(
                                        sys.stdin.fileno(),
                                        termios.TCSADRAIN,
                                        old_settings,
                                    )

                        monitor_task = asyncio.create_task(monitor_escape())

                        # Wait for either completion or interruption
                        response = None
                        logger.info("Waiting for task completion or interruption...")
                        done, pending = await asyncio.wait(
                            [current_task, monitor_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        logger.info(
                            f"Task completed. Done: {len(done)}, Pending: {len(pending)}"
                        )

                        # Cancel any remaining tasks
                        for task in pending:
                            task.cancel()

                        # Check for any kind of interruption (global or local)
                        if (
                            input_handler.interrupted
                            or self.global_interrupt_manager.is_interrupted()
                        ):
                            # Try to emit as event if event bus is available
                            await self._emit_interruption(
                                "Request cancelled by user", "user"
                            )
                            input_handler.interrupted = False
                            self.global_interrupt_manager.clear_interrupt()
                            current_task = None
                            continue

                        if (
                            current_task
                            and current_task.done()
                            and not current_task.cancelled()
                        ):
                            logger.info("Task completed successfully")
                            response = current_task.result()
                            logger.info(
                                f"Response received: {type(response)} - {repr(response[:100] if isinstance(response, str) else response)}"
                            )
                            current_task = None
                        else:
                            continue  # Request was cancelled, go back to input

                        # Handle the response
                        if hasattr(response, "__aiter__"):
                            # This shouldn't happen - generate_response should return final content
                            # But handle it just in case
                            response_content = await self._collect_response_content(
                                response
                            )
                        elif isinstance(response, str):
                            # Direct string response - this is the normal case
                            response_content = response
                            logger.info(
                                f"Got string response: {repr(response_content[:100])}"
                            )

                            # Wait for event processing to complete before continuing
                            await asyncio.sleep(0.1)
                        else:
                            # Other response type - convert to string
                            response_content = str(response)
                            logger.info(
                                f"Got non-string response: {type(response)} - {repr(response_content[:100])}"
                            )

                        # Check if response handler has updated conversation history (with tool calls/results)
                        updated_messages = None
                        if hasattr(self.agent, "response_handler"):
                            updated_messages = (
                                self.agent.response_handler.get_updated_messages()
                            )

                        if updated_messages:
                            # Extract new messages (assistant message with tool calls + tool results)
                            # Skip the original messages that were passed to the response handler
                            original_length = len(messages)
                            new_messages = updated_messages[original_length:]

                            # Add the new messages (tool calls and results) to conversation
                            messages.extend(new_messages)
                        elif response_content:
                            # Fallback: just add the assistant response if no tool execution occurred
                            messages.append(
                                {"role": "assistant", "content": response_content}
                            )

                        # Reset interrupt count after successful operation
                        from cli_agent.core.global_interrupt import (
                            reset_interrupt_count,
                        )

                        reset_interrupt_count()

                    except ToolDeniedReturnToPrompt as e:
                        await self._emit_error(
                            f"Tool access denied: {e.reason}", "tool_denied"
                        )
                        continue  # Return to prompt without adding to conversation
                    except Exception as e:
                        logger.error(f"Error in chat: {e}")
                        await self._emit_error(f"Error: {str(e)}", "general")
                        continue

            except KeyboardInterrupt:
                await self._emit_system_message("Goodbye!", "goodbye", "👋")
                break
            except Exception as e:
                error_msg = self.handle_conversation_error(e)
                await self._emit_error(error_msg, "conversation_error")
                continue

        # Stop persistent prompt and display goodbye message
        self.terminal_manager.stop_persistent_prompt()
        if not existing_messages:
            await self.display_goodbye_message()

        return messages

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful interruption."""

        def signal_handler(signum, frame):
            logger.info("Interrupt signal received, stopping chat...")
            self.interrupt_received = True
            self.conversation_active = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start_conversation(self):
        """Mark conversation as active."""
        self.conversation_active = True
        self.interrupt_received = False

    def stop_conversation(self):
        """Mark conversation as inactive."""
        self.conversation_active = False

    def is_conversation_active(self) -> bool:
        """Check if conversation is currently active."""
        return self.conversation_active and not self.interrupt_received

    def handle_user_input(self, user_input: str) -> Optional[str]:
        """Process user input and handle special commands."""
        if not user_input or not user_input.strip():
            return None

        user_input = user_input.strip()

        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "bye"]:
            self.stop_conversation()
            return "Goodbye!"

        return user_input

    async def handle_slash_command(self, command: str) -> Optional[str]:
        """Handle slash commands - delegates to agent's slash command manager."""
        if hasattr(self.agent, "slash_commands"):
            return await self.agent.slash_commands.handle_slash_command(command)
        else:
            return f"Unknown command: {command}"

    def format_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation history for display."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                formatted.append(f"You: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"System: {content}")
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "unknown")
                formatted.append(f"Tool ({tool_id}): {content}")

        return "\n".join(formatted)

    def get_conversation_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        stats = {
            "total_messages": len(messages),
            "user_messages": 0,
            "assistant_messages": 0,
            "tool_messages": 0,
            "system_messages": 0,
            "total_characters": 0,
        }

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                stats["user_messages"] += 1
            elif role == "assistant":
                stats["assistant_messages"] += 1
            elif role == "tool":
                stats["tool_messages"] += 1
            elif role == "system":
                stats["system_messages"] += 1

            stats["total_characters"] += len(str(content))

        return stats

    def handle_conversation_error(self, error: Exception) -> str:
        """Handle errors during conversation."""
        logger.error(f"Conversation error: {error}")
        return f"An error occurred: {str(error)}"

    async def display_welcome_message(self):
        """Display welcome message at start of chat."""
        model_name = getattr(
            self.agent, "_get_current_runtime_model", lambda: "Unknown"
        )()

        await self._emit_system_message(
            f"Starting interactive chat with {model_name}", "welcome", "🤖"
        )
        await self._emit_system_message(
            "Type '/help' for available commands or '/quit' to exit.", "info"
        )
        await self._emit_system_message("-" * 50, "info")

    async def display_goodbye_message(self):
        """Display goodbye message at end of chat."""
        await self._emit_system_message("\n" + "-" * 50, "info")
        await self._emit_system_message("Thanks for chatting!", "goodbye", "👋")

    def clean_conversation_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Clean and validate conversation messages."""
        cleaned = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Ensure content is a string
                content = msg["content"]
                if not isinstance(content, str):
                    content = str(content)

                cleaned_msg = {"role": msg["role"], "content": content}

                # Preserve other important fields
                for field in ["tool_calls", "tool_call_id", "name"]:
                    if field in msg:
                        cleaned_msg[field] = msg[field]

                cleaned.append(cleaned_msg)

        return cleaned

    def should_compact_conversation(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if conversation should be compacted."""
        if hasattr(self.agent, "get_token_limit"):
            # Use agent's token management logic
            try:
                token_count = self.agent._estimate_token_count(messages)
                token_limit = self.agent.get_token_limit()
                return token_count > (token_limit * 0.8)  # 80% threshold
            except Exception:
                pass

        # Fallback: compact if too many messages
        return len(messages) > 50

    # Event emission helper methods
    async def _emit_interruption(self, reason: str, interrupt_type: str = "user"):
        """Emit interruption event instead of print statement."""
        if self.event_emitter:
            await self.event_emitter.emit_interrupt(interrupt_type, reason)

    async def _emit_system_message(
        self, message: str, message_type: str = "info", emoji: str = None
    ):
        """Emit system message event instead of print statement."""
        if self.event_emitter:
            await self.event_emitter.emit_system_message(message, message_type, emoji)

    async def _emit_status(self, message: str, level: str = "info"):
        """Emit status event instead of print statement."""
        if self.event_emitter:
            await self.event_emitter.emit_status(message, level=level)

    async def _emit_error(self, message: str, error_type: str = "general"):
        """Emit error event instead of print statement."""
        if self.event_emitter:
            await self.event_emitter.emit_error(message, error_type)

    async def _emit_text(self, content: str, is_markdown: bool = False):
        """Emit text event instead of print statement."""
        if self.event_emitter:
            await self.event_emitter.emit_text(content, is_markdown=is_markdown)

    async def _collect_response_content(self, response) -> str:
        """Collect response content for conversation history without display logic."""
        content = ""
        if hasattr(response, "__aiter__"):
            # Streaming response - content collection only
            async for chunk in response:
                # Check for global interrupt during content collection
                if self.global_interrupt_manager.is_interrupted():
                    await self._emit_interruption(
                        "Request cancelled during streaming", "user"
                    )
                    break

                chunk_text = str(chunk)
                content += chunk_text
                # Display is handled by events in _process_streaming_chunks_with_events
        else:
            # Non-streaming response
            content = str(response)
            if content:
                await self._emit_text(content, is_markdown=True)

        return content
