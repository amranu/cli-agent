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

        # Queue for handling input received during LLM processing
        self.input_queue = []

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
        
        # Store agent reference on input handler for accessing subagent coordinator
        input_handler._agent = self.agent

        # Initialize messages from existing if provided
        messages = existing_messages[:] if existing_messages else []

        # Start conversation (let global interrupt manager handle signals)
        self.start_conversation()

        # Start event bus processing if available and not already running
        if (
            hasattr(self.agent, "event_bus")
            and self.agent.event_bus
            and not self.agent.event_bus.is_running
        ):
            await self.agent.event_bus.start_processing()

        # Display welcome message (after event bus is running for immediate display)
        if not existing_messages:
            await self.display_welcome_message()

        # Initialize token display
        self._update_token_display(messages)

        current_task = None

        while self.is_conversation_active():
            try:
                # Check for global interrupt - clear and continue
                if self.global_interrupt_manager.is_interrupted():
                    if current_task and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
                    
                    # Clear interrupt state and continue conversation
                    self.global_interrupt_manager.clear_interrupt()
                    input_handler.interrupted = False
                    self.input_queue.clear()
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
                    # Clear input queue on interruption
                    self.input_queue.clear()
                    current_task = None
                    continue

                # Check if subagents are active - if so, don't prompt for input
                # Unless BACKGROUND_SUBAGENTS is enabled, then allow parallel input
                if (
                    hasattr(self.agent, "subagent_manager")
                    and self.agent.subagent_manager
                    and not self.agent.config.background_subagents
                ):
                    active_count = self.agent.subagent_manager.get_active_count()
                    if active_count > 0:
                        # Subagents are running, wait instead of prompting for input
                        await asyncio.sleep(0.5)
                        continue

                # Get user input with proper prompt - let prompt_toolkit handle the display
                user_input = input_handler.get_multiline_input("> ")

                # Keep the prompt as "> " instead of changing it
                # self.terminal_manager.update_prompt("Processing... ")

                if user_input is None:  # Interrupted or EOF
                    # Check if this is an immediate exit from empty prompt
                    if hasattr(input_handler, 'immediate_exit_requested') and input_handler.immediate_exit_requested:
                        # Immediate exit - don't show operation cancelled message
                        input_handler.immediate_exit_requested = False
                        self.interrupt_received = True
                        break
                    
                    # Regular interruption handling
                    if current_task and not current_task.done():
                        current_task.cancel()
                        await self._emit_interruption(
                            "Operation cancelled by user", "user"
                        )
                    current_task = None
                    # If interrupted (EOF), exit the conversation
                    if input_handler.interrupted:
                        input_handler.interrupted = False
                        self.interrupt_received = True  # Mark that we're exiting due to interruption
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
                            # Return special quit indicator to the session loop
                            return {"quit": True, "messages": messages}
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

                    # Update token display after adding user message
                    self._update_token_display(messages)

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
                            # Update token display after compaction
                            self._update_token_display(messages)
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

                        # Create a background task to handle input during LLM output
                        async def handle_concurrent_input():
                            import select
                            import termios
                            import tty

                            old_settings = None
                            input_buffer = ""

                            try:
                                while not current_task.done():
                                    try:
                                        if (
                                            sys.stdin.isatty()
                                            and select.select([sys.stdin], [], [], 0.1)[
                                                0
                                            ]
                                        ):
                                            # Set up raw mode for character-by-character input
                                            if old_settings is None:
                                                old_settings = termios.tcgetattr(
                                                    sys.stdin.fileno()
                                                )
                                                tty.setraw(sys.stdin.fileno())

                                            char = sys.stdin.read(1)

                                            if char == "\x1b":  # Escape key - interrupt
                                                input_handler.interrupted = True
                                                return
                                            elif (
                                                char == "\r" or char == "\n"
                                            ):  # Enter key
                                                if input_buffer.strip():
                                                    # Process the queued input
                                                    await self._handle_concurrent_input(
                                                        input_buffer.strip()
                                                    )
                                                    input_buffer = ""
                                                    # Keep the prompt as "> "
                                                    # self.terminal_manager.update_prompt("Processing... ")
                                            elif (
                                                char == "\x7f" or char == "\x08"
                                            ):  # Backspace
                                                if input_buffer:
                                                    input_buffer = input_buffer[:-1]
                                                    # Update prompt to show current input
                                                    display_prompt = (
                                                        f"Queue: {input_buffer}"
                                                        if input_buffer
                                                        else "> "
                                                    )
                                                    self.terminal_manager.update_prompt(
                                                        display_prompt
                                                    )
                                            elif (
                                                char.isprintable()
                                            ):  # Regular character
                                                input_buffer += char
                                                # Update prompt to show current input being typed
                                                self.terminal_manager.update_prompt(
                                                    f"Queue: {input_buffer}"
                                                )

                                        await asyncio.sleep(
                                            0.05
                                        )  # Shorter sleep for more responsive input
                                    except Exception:
                                        await asyncio.sleep(0.1)
                            finally:
                                if old_settings is not None:
                                    termios.tcsetattr(
                                        sys.stdin.fileno(),
                                        termios.TCSADRAIN,
                                        old_settings,
                                    )

                        monitor_task = asyncio.create_task(handle_concurrent_input())

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

                        # Check for any kind of interruption (global or local) first
                        if (
                            input_handler.interrupted
                            or self.global_interrupt_manager.is_interrupted()
                        ):
                            # Cancel task if it's still running
                            if current_task and not current_task.done():
                                current_task.cancel()
                                try:
                                    await current_task
                                except asyncio.CancelledError:
                                    pass
                                except Exception:
                                    pass
                            
                            # Try to emit as event if event bus is available
                            await self._emit_interruption(
                                "Request cancelled by user", "user"
                            )
                            input_handler.interrupted = False
                            # Don't clear interrupt state immediately - let user press Ctrl+C again to exit
                            # self.global_interrupt_manager.clear_interrupt()
                            # Clear input queue on interruption
                            self.input_queue.clear()
                            current_task = None
                            continue

                        # Only process successful, non-cancelled tasks
                        if (
                            current_task
                            and current_task.done()
                            and not current_task.cancelled()
                        ):
                            logger.info("Task completed successfully")
                            try:
                                response = current_task.result()
                                logger.info(
                                    f"Response received: {type(response)} - {repr(response[:100] if isinstance(response, str) else response)}"
                                )
                                current_task = None
                            except Exception as e:
                                # Check for different types of interruptions
                                from cli_agent.core.interrupt_aware_streaming import FirstInterruptException
                                
                                if isinstance(e, FirstInterruptException):
                                    # First interrupt: handle gracefully and continue
                                    logger.info(f"First interrupt detected, handling gracefully: {e}")
                                    current_task = None
                                    await self._emit_interruption(
                                        "Operation interrupted by user", "user"
                                    )
                                    # Let the main loop handle clearing the interrupt state
                                    continue
                                elif isinstance(e, KeyboardInterrupt):
                                    # KeyboardInterrupt (second+ interrupt): re-raise to exit application
                                    logger.info(f"KeyboardInterrupt detected, re-raising: {e}")
                                    current_task = None
                                    raise
                                else:
                                    # Other exception: re-raise
                                    logger.error(f"Other exception in task result: {type(e).__name__}: {e}")
                                    current_task = None
                                    raise
                        elif current_task and current_task.done() and current_task.cancelled():
                            # Task was cancelled, clean up and continue
                            logger.info("Task was cancelled, returning to prompt")
                            current_task = None
                            continue
                        else:
                            continue  # Task not done yet, keep waiting

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
                            # Update token display after tool execution
                            self._update_token_display(messages)
                        elif response_content:
                            # Fallback: just add the assistant response if no tool execution occurred
                            messages.append(
                                {"role": "assistant", "content": response_content}
                            )

                        # Update and show token display after assistant response
                        # Check for auto-compaction and update messages if needed
                        compaction_result = await self._update_token_display_async(messages, show_display=True)
                        
                        # EXECUTE STOP HOOKS - triggered when agent finishes responding
                        if hasattr(self.agent, 'hook_manager') and self.agent.hook_manager:
                            try:
                                from cli_agent.core.hooks.hook_config import HookType
                                from datetime import datetime
                                
                                stop_context = {
                                    "timestamp": datetime.now().isoformat(),
                                    "session_id": getattr(self.agent, 'session_id', 'unknown'),
                                    "conversation_length": len(messages),
                                    "response_length": len(response_content) if isinstance(response_content, str) else 0
                                }
                                
                                await self.agent.hook_manager.execute_hooks(HookType.STOP, stop_context)
                            except Exception as e:
                                logger.warning(f"Error executing stop hooks: {e}")
                        if compaction_result and compaction_result.get("auto_compacted"):
                            # Replace messages with compacted version
                            messages[:] = compaction_result["compacted_messages"]

                        # Reset interrupt count and clear interrupt state after successful operation
                        from cli_agent.core.global_interrupt import (
                            reset_interrupt_count,
                        )

                        reset_interrupt_count()
                        self.global_interrupt_manager.clear_interrupt()
                        
                        # Reset prompt to ready state for next input
                        self.terminal_manager.update_prompt("> ")

                        # Process any queued input that was received during LLM processing
                        await self._process_input_queue(messages)

                    except ToolDeniedReturnToPrompt as e:
                        await self._emit_error(
                            f"Tool access denied: {e.reason}", "tool_denied"
                        )
                        # Ensure newline before prompt reset
                        self.terminal_manager.write_above_prompt('\n')
                        # Reset prompt to ready state
                        self.terminal_manager.update_prompt("> ")
                        continue  # Return to prompt without adding to conversation
                    except Exception as e:
                        # Check if this is a 429 rate limit error
                        error_str = str(e).lower()
                        if "429" in error_str or "rate limit" in error_str:
                            logger.warning(f"Rate limit detected: {e}")
                            await self._emit_system_message(
                                "Rate limit reached. Retrying in 10 seconds...",
                                "rate_limit",
                                "⏳",
                            )

                            # Wait 10 seconds with visual countdown
                            for i in range(10, 0, -1):
                                await self._emit_status(
                                    f"Retrying in {i} second{'s' if i != 1 else ''}...",
                                    "info",
                                )
                                await asyncio.sleep(1)

                            await self._emit_status("Retrying now...", "info")

                            # Retry by recursively calling the same logic
                            # Remove the last user message to avoid duplication
                            if messages and messages[-1]["role"] == "user":
                                logger.info(
                                    "Retrying after rate limit with preserved conversation"
                                )

                                # Retry the entire request process recursively
                                return await self._retry_llm_request(
                                    messages, input_handler
                                )
                            else:
                                await self._emit_error(
                                    "Unable to retry - no user message found",
                                    "retry_error",
                                )
                                # Ensure newline before prompt reset
                                self.terminal_manager.write_above_prompt('\n')
                                self.terminal_manager.update_prompt("> ")
                                continue
                        else:
                            logger.error(f"Error in chat: {e}")
                            await self._emit_error(f"Error: {str(e)}", "general")
                            # Ensure newline before prompt reset
                            self.terminal_manager.write_above_prompt('\n')
                            # Reset prompt to ready state
                            self.terminal_manager.update_prompt("> ")
                            continue

            except KeyboardInterrupt:
                # Check if this is the second interrupt (should exit app)
                if self.global_interrupt_manager._interrupt_count >= 2:
                    # Let the global manager handle the exit
                    sys.exit(0)
                else:
                    # First interrupt - return special interrupt indicator to prevent session restart
                    await self._emit_system_message("Goodbye!", "goodbye", "👋")
                    return {"interrupted": True, "messages": messages}
            except Exception as e:
                error_msg = self.handle_conversation_error(e)
                await self._emit_error(error_msg, "conversation_error")
                # Ensure newline before prompt reset
                self.terminal_manager.write_above_prompt('\n')
                # Reset prompt to ready state
                self.terminal_manager.update_prompt("> ")
                continue

        # Stop persistent prompt and display goodbye message
        self.terminal_manager.stop_persistent_prompt()
        # Clear token display when conversation ends
        self.terminal_manager.clear_token_display()
        if not existing_messages:
            await self.display_goodbye_message()

        # Check if we're exiting due to interruption (EOF or Ctrl+C)
        if self.interrupt_received:
            return {"interrupted": True, "messages": messages}
            
        return messages

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful interruption."""
        # NOTE: Signal handling is now done by the global interrupt manager
        # to avoid conflicts. This method is kept for backward compatibility.
        pass

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

        # Process @ file references
        processed_input = self._process_file_references(user_input)

        return processed_input

    def _process_file_references(self, user_input: str) -> str:
        """Process @ file references in user input and include file contents."""
        import os
        import re

        # Find all @ file references using regex
        # Match @/path/to/file or @~/path/to/file
        file_pattern = r"@([~/][^\s]*)"
        matches = re.findall(file_pattern, user_input)

        if not matches:
            return user_input

        # Process each file reference
        file_contents = []
        processed_text = user_input

        for file_path in matches:
            # Expand user home directory if needed
            expanded_path = os.path.expanduser(file_path)

            try:
                # Check if file exists and is readable
                if os.path.isfile(expanded_path):
                    with open(
                        expanded_path, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        content = f.read()

                    # Add file content to our collection
                    file_contents.append(
                        f"\n\n--- File: {file_path} ---\n{content}\n--- End of {file_path} ---"
                    )

                    # Remove the @file_path from the original text since we're including it separately
                    processed_text = processed_text.replace(
                        f"@{file_path}", f"[File: {file_path}]"
                    )

                elif os.path.isdir(expanded_path):
                    # If it's a directory, list its contents
                    try:
                        dir_contents = os.listdir(expanded_path)
                        dir_listing = "\n".join(dir_contents)
                        file_contents.append(
                            f"\n\n--- Directory: {file_path} ---\n{dir_listing}\n--- End of {file_path} ---"
                        )
                        processed_text = processed_text.replace(
                            f"@{file_path}", f"[Directory: {file_path}]"
                        )
                    except PermissionError:
                        processed_text = processed_text.replace(
                            f"@{file_path}",
                            f"[Directory: {file_path} - Permission denied]",
                        )

                else:
                    # File doesn't exist
                    processed_text = processed_text.replace(
                        f"@{file_path}", f"[File: {file_path} - Not found]"
                    )

            except Exception as e:
                # Error reading file
                processed_text = processed_text.replace(
                    f"@{file_path}", f"[File: {file_path} - Error: {str(e)}]"
                )

        # Combine the processed text with file contents
        if file_contents:
            return processed_text + "\n" + "".join(file_contents)
        else:
            return processed_text

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
        """Determine if conversation should be compacted using accurate token counting."""
        try:
            # Use the enhanced token manager
            if hasattr(self.agent, "token_manager") and self.agent.token_manager:
                return self.agent.token_manager.should_compact(messages)
            
            # Fallback to simple token counting
            elif hasattr(self.agent, "get_token_limit"):
                token_count = self.agent._estimate_token_count(messages)
                token_limit = self.agent.get_token_limit()
                return token_count > (token_limit * 0.8)  # 80% threshold
        except Exception as e:
            logger.debug(f"Token-based compaction check failed: {e}")

        # Final fallback: compact if too many messages
        return len(messages) > 50

    def _update_token_display(self, messages: List[Dict[str, Any]], show_display: bool = False):
        """Update the token count display and check for automatic compaction.
        
        Args:
            messages: Current conversation messages
            show_display: Whether to actually print the display (True only before user input)
            
        Returns:
            Dict with compaction info if auto-compaction was triggered, None otherwise
        """
        try:
            if hasattr(self.agent, "token_manager") and self.agent.token_manager:
                # Update token manager with current model
                self.agent._update_token_manager_model()
                
                # Create a full message list including system prompt and tools
                full_messages = self._prepare_full_message_context(messages)
                
                # Get current token count including system prompt and tools
                current_tokens = self.agent.token_manager.count_conversation_tokens(full_messages)
                
                # Get token limit (try to get from model if possible)
                token_limit = self.agent.token_manager.get_token_limit()
                
                # Get model name
                model_name = ""
                try:
                    model_name = self.agent._get_current_runtime_model()
                except:
                    model_name = "Unknown"
                
                # Check if we have reliable token information
                has_reliable_info = self.agent.token_manager.has_reliable_token_info()
                
                # Show token display if we have reliable info  
                if has_reliable_info:
                    self.terminal_manager.update_token_display(
                        current_tokens=current_tokens,
                        token_limit=token_limit,
                        model_name=model_name,
                        show_display=show_display
                    )
                elif show_display and not has_reliable_info:
                    # Add spacing only when token display is hidden AND we need spacing
                    print()  # Add newline for spacing between LLM output and prompt
                
                return None
                
        except Exception as e:
            # Token display is non-critical, so we just log and continue
            logger.debug(f"Failed to update token display: {e}")
            return None
    
    async def _update_token_display_async(self, messages: List[Dict[str, Any]], show_display: bool = False):
        """Async version of _update_token_display that can perform auto-compaction.
        
        Args:
            messages: Current conversation messages
            show_display: Whether to actually print the display (True only before user input)
            
        Returns:
            Dict with compaction info if auto-compaction was triggered, None otherwise
        """
        try:
            if hasattr(self.agent, "token_manager") and self.agent.token_manager:
                # Update token manager with current model
                self.agent._update_token_manager_model()
                
                # Create a full message list including system prompt and tools
                full_messages = self._prepare_full_message_context(messages)
                
                # Get current token count including system prompt and tools
                current_tokens = self.agent.token_manager.count_conversation_tokens(full_messages)
                
                # Get token limit (try to get from model if possible)
                token_limit = self.agent.token_manager.get_token_limit()
                
                # Get model name
                model_name = ""
                try:
                    model_name = self.agent._get_current_runtime_model()
                except:
                    model_name = "Unknown"
                
                # Check if we have reliable token information
                has_reliable_info = self.agent.token_manager.has_reliable_token_info()
                
                # Only perform auto-compaction and show token display if we have reliable info
                if has_reliable_info:
                    # Check for automatic compaction (95% threshold)
                    percentage = (current_tokens / token_limit) * 100 if token_limit > 0 else 0
                    
                    # Auto-compact if above 95% and we have enough messages to compact
                    if percentage >= 95 and len(messages) > 5:
                        return await self._perform_auto_compaction(messages, current_tokens, token_limit, model_name)
                    
                    # Update the terminal display
                    self.terminal_manager.update_token_display(
                        current_tokens=current_tokens,
                        token_limit=token_limit,
                        model_name=model_name,
                        show_display=show_display
                    )
                elif show_display and not has_reliable_info:
                    # Add spacing only when token display is hidden AND we need spacing
                    print()  # Add newline for spacing between LLM output and prompt
                # If we don't have reliable info, don't show token display or auto-compact
                
                return None
                
        except Exception as e:
            # Token display is non-critical, so we just log and continue
            logger.debug(f"Failed to update token display: {e}")
            return None
    
    def _prepare_full_message_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare full message context including system prompt and tools for accurate token counting."""
        full_messages = []
        
        # Add system prompt if available
        try:
            if hasattr(self.agent, '_create_system_prompt'):
                system_prompt = self.agent._create_system_prompt()
                if system_prompt and system_prompt.strip():
                    full_messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
        except Exception as e:
            logger.debug(f"Failed to get system prompt for token counting: {e}")
        
        # Add the conversation messages
        full_messages.extend(messages)
        
        # Add tool information for models that include tools in token calculation
        try:
            if hasattr(self.agent, 'available_tools') and self.agent.available_tools:
                # Create a summary of available tools to include in token count
                tool_summary = self._create_tool_summary()
                if tool_summary:
                    # Add as a system message or append to existing system message
                    if full_messages and full_messages[0].get("role") == "system":
                        full_messages[0]["content"] += f"\n\nAvailable tools: {tool_summary}"
                    else:
                        full_messages.insert(0, {
                            "role": "system", 
                            "content": f"Available tools: {tool_summary}"
                        })
        except Exception as e:
            logger.debug(f"Failed to add tools for token counting: {e}")
        
        return full_messages
    
    def _create_tool_summary(self) -> str:
        """Create a brief summary of available tools for token counting."""
        try:
            if not hasattr(self.agent, 'available_tools') or not self.agent.available_tools:
                return ""
            
            tool_names = []
            for tool_key, tool_info in self.agent.available_tools.items():
                if isinstance(tool_info, dict) and 'name' in tool_info:
                    tool_names.append(tool_info['name'])
                else:
                    # Fallback to tool key
                    tool_names.append(tool_key.split(':')[-1] if ':' in tool_key else tool_key)
            
            return f"{len(tool_names)} tools ({', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''})"
        except Exception as e:
            logger.debug(f"Failed to create tool summary: {e}")
            return ""
    
    async def _perform_auto_compaction(self, messages: List[Dict[str, Any]], current_tokens: int, token_limit: int, model_name: str) -> Dict[str, Any]:
        """Perform automatic conversation compaction when token usage is too high.
        
        Args:
            messages: Current conversation messages
            current_tokens: Current token count
            token_limit: Token limit for the model
            model_name: Current model name
            
        Returns:
            Dict with compaction result and new messages
        """
        try:
            logger.info(f"Auto-compacting conversation: {current_tokens}/{token_limit} tokens (95%+ usage)")
            
            # Emit a notification about auto-compaction
            await self._emit_system_message(
                f"🗜️ Auto-compacting conversation ({current_tokens:,}/{token_limit:,} tokens, 95%+ usage)",
                "auto_compact",
                "🔄"
            )
            
            # Perform the compaction using the agent's method
            if hasattr(self.agent, "compact_conversation"):
                compacted_messages = await self.agent.compact_conversation(messages)
                
                # Calculate new token count
                full_compacted = self._prepare_full_message_context(compacted_messages)
                new_tokens = self.agent.token_manager.count_conversation_tokens(full_compacted)
                
                # Emit success notification
                await self._emit_system_message(
                    f"✅ Conversation auto-compacted: {len(messages)} → {len(compacted_messages)} messages, "
                    f"{current_tokens:,} → {new_tokens:,} tokens",
                    "auto_compact_success",
                    "📉"
                )
                
                # Update the token display with new values
                self.terminal_manager.update_token_display(
                    current_tokens=new_tokens,
                    token_limit=token_limit,
                    model_name=model_name,
                    show_display=False  # Don't show display during auto-compaction
                )
                
                return {
                    "auto_compacted": True,
                    "compacted_messages": compacted_messages,
                    "tokens_before": current_tokens,
                    "tokens_after": new_tokens,
                    "messages_before": len(messages),
                    "messages_after": len(compacted_messages)
                }
            else:
                await self._emit_system_message(
                    "❌ Auto-compaction failed: feature not available for this agent",
                    "auto_compact_error",
                    "⚠️"
                )
                return None
                
        except Exception as e:
            logger.error(f"Auto-compaction failed: {e}")
            await self._emit_system_message(
                f"❌ Auto-compaction failed: {str(e)}",
                "auto_compact_error",
                "⚠️"
            )
            return None

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

    async def _retry_llm_request(self, messages, input_handler):
        """Retry LLM request with the same messages after rate limit."""
        try:
            # Make API call interruptible by running in a task
            await self._emit_system_message(
                "Retrying... (press ESC to interrupt)", "thinking", "💭"
            )

            logger.info("Creating retry task for generate_response")
            current_task = asyncio.create_task(
                self.agent.generate_response(messages, stream=True)
            )
            logger.info("Retry task created, waiting for completion")

            # Create a background task to handle input during LLM output (simplified version)
            async def handle_concurrent_input():
                try:
                    while not current_task.done():
                        if input_handler.check_for_interrupt():
                            input_handler.interrupted = True
                            return
                        await asyncio.sleep(0.1)
                except Exception:
                    await asyncio.sleep(0.1)

            monitor_task = asyncio.create_task(handle_concurrent_input())

            # Wait for either completion or interruption
            response = None
            logger.info("Waiting for retry task completion or interruption...")
            done, pending = await asyncio.wait(
                [current_task, monitor_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            logger.info(
                f"Retry task completed. Done: {len(done)}, Pending: {len(pending)}"
            )

            # Cancel any remaining tasks
            for task in pending:
                task.cancel()

            # Check for interruption
            if (
                input_handler.interrupted
                or self.global_interrupt_manager.is_interrupted()
            ):
                await self._emit_interruption("Retry cancelled by user", "user")
                input_handler.interrupted = False
                # Don't clear interrupt state immediately - let user press Ctrl+C again to exit
                # self.global_interrupt_manager.clear_interrupt()
                return

            if current_task and current_task.done() and not current_task.cancelled():
                logger.info("Retry task completed successfully")
                response = current_task.result()
                logger.info(f"Retry response received: {type(response)}")

                # Handle the response (similar to main chat logic)
                if hasattr(response, "__aiter__"):
                    response_content = await self._collect_response_content(response)
                elif isinstance(response, str):
                    response_content = response
                    logger.info(
                        f"Got retry string response: {repr(response_content[:100])}"
                    )
                else:
                    response_content = str(response)
                    logger.info(f"Got retry other response: {type(response)}")

                # Check if response handler has updated messages (tool execution)
                if hasattr(self.agent, "response_handler") and hasattr(
                    self.agent.response_handler, "get_updated_messages"
                ):
                    updated_messages = (
                        self.agent.response_handler.get_updated_messages()
                    )
                    if updated_messages and len(updated_messages) > len(messages):
                        # Tool execution occurred - update conversation with tool results
                        original_length = len(messages)
                        new_messages = updated_messages[original_length:]
                        messages.extend(new_messages)
                        # Update token display after tool execution
                        self._update_token_display(messages)
                elif response_content:
                    # Just add the assistant response
                    messages.append({"role": "assistant", "content": response_content})

                # Update token display after assistant response
                self._update_token_display(messages)

                # Ensure newline before prompt reset  
                self.terminal_manager.write_above_prompt('\n')
                # Reset prompt to ready state for next input
                self.terminal_manager.update_prompt("> ")

                # Clear interrupt state after successful retry
                self.global_interrupt_manager.clear_interrupt()
                await self._emit_status("Retry successful!", "info")

        except Exception as e:
            # If retry also fails, just log and continue
            logger.error(f"Retry also failed: {e}")
            await self._emit_error(f"Retry failed: {str(e)}", "retry_error")
            # Ensure newline before prompt reset
            self.terminal_manager.write_above_prompt('\n')
            self.terminal_manager.update_prompt("> ")

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

    async def _handle_concurrent_input(self, input_text: str):
        """Handle input received while LLM is processing."""
        await self._emit_system_message(f"Queued input: {input_text}", "info", "📥")

        # Add to input queue for processing after LLM response
        self.input_queue.append(input_text)

    async def _process_input_queue(self, messages: List[Dict[str, Any]]):
        """Process any queued input received during LLM processing."""
        if not self.input_queue:
            return

        await self._emit_system_message(
            f"Processing {len(self.input_queue)} queued input(s)...", "info", "⚡"
        )

        # Process each queued input
        for queued_input in self.input_queue:
            # Handle slash commands
            if queued_input.startswith("/"):
                slash_result = await self.handle_slash_command(queued_input)
                if slash_result:
                    # Handle special command results
                    if isinstance(slash_result, dict) and slash_result.get("quit"):
                        await self._emit_system_message(
                            slash_result.get("status", "Goodbye!"), "goodbye", "👋"
                        )
                        self.stop_conversation()
                        return
                    elif isinstance(slash_result, dict) and slash_result.get(
                        "clear_messages"
                    ):
                        await self._emit_system_message(
                            slash_result.get("status", "Messages cleared."),
                            "status",
                            "🗑️",
                        )
                        messages.clear()
                    elif isinstance(slash_result, dict) and slash_result.get(
                        "compacted_messages"
                    ):
                        await self._emit_system_message(
                            slash_result.get("status", "Messages compacted."),
                            "status",
                            "🗃",
                        )
                        messages[:] = slash_result["compacted_messages"]
                    elif isinstance(slash_result, dict) and slash_result.get(
                        "send_to_llm"
                    ):
                        # Add the prompt as a user message
                        messages.append(
                            {"role": "user", "content": slash_result["send_to_llm"]}
                        )
                        await self._emit_system_message(
                            f"Added to conversation: {slash_result['send_to_llm'][:50]}...",
                            "info",
                            "📝",
                        )
                    else:
                        await self._emit_system_message(str(slash_result), "info")
            else:
                # Regular input - add to conversation
                processed_input = self.handle_user_input(queued_input)
                if processed_input and processed_input.strip():
                    messages.append({"role": "user", "content": processed_input})
                    await self._emit_system_message(
                        f"Added to conversation: {processed_input[:50]}...",
                        "info",
                        "📝",
                    )

        # Clear the queue
        self.input_queue.clear()

        # If we have new messages, let the user know they can continue
        if len([msg for msg in messages if msg.get("role") == "user"]) > 0:
            await self._emit_system_message(
                "Queued input processed. Continue the conversation or press Enter to send new messages to LLM.",
                "info",
                "✅",
            )
