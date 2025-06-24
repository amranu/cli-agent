"""Chat interface and interaction handling for MCP agents."""

import asyncio
import logging
import signal
import sys
from typing import Any, Dict, List, Optional

from cli_agent.core.global_interrupt import get_global_interrupt_manager

logger = logging.getLogger(__name__)


class ChatInterface:
    """Handles interactive chat sessions, input processing, and conversation management."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent
        self.conversation_active = False
        self.interrupt_received = False
        self.global_interrupt_manager = get_global_interrupt_manager()

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
            self.display_welcome_message()

        # Set up signal handlers
        self.setup_signal_handlers()
        self.start_conversation()

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
                    print("🛑 Operation cancelled, returning to prompt")
                    self.global_interrupt_manager.clear_interrupt()
                    input_handler.interrupted = False
                    current_task = None
                    continue

                # Check if we were interrupted during a previous operation
                if input_handler.interrupted:
                    if current_task and not current_task.done():
                        current_task.cancel()
                        print("🛑 Operation cancelled by user")
                    input_handler.interrupted = False
                    current_task = None
                    continue

                # Get user input with smart multiline detection
                user_input = input_handler.get_multiline_input("You: ")

                if user_input is None:  # Interrupted or EOF
                    if current_task and not current_task.done():
                        current_task.cancel()
                        print("🛑 Operation cancelled by user")
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
                        print("🔚 End of input detected, exiting...")
                        break
                    continue

                # Handle user input
                if user_input.startswith("/"):
                    # Handle slash command asynchronously
                    slash_result = await self.handle_slash_command(user_input)
                    if slash_result:
                        # Check if it's a quit command
                        if isinstance(slash_result, dict) and slash_result.get("quit"):
                            print(slash_result.get("status", "Goodbye!"))
                            break
                        # Check if it's a reload_host command
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "reload_host"
                        ):
                            print(slash_result.get("status", "Reloading..."))
                            # Return dict with reload_host key and current messages
                            return {
                                "reload_host": slash_result["reload_host"],
                                "messages": messages,
                            }
                        # Check if it's a clear_messages command
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "clear_messages"
                        ):
                            print(slash_result.get("status", "Messages cleared."))
                            messages.clear()  # Clear the messages list
                        # Check if it's a compacted_messages command
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "compacted_messages"
                        ):
                            print(slash_result.get("status", "Messages compacted."))
                            messages[:] = slash_result[
                                "compacted_messages"
                            ]  # Replace messages with compacted ones
                        # Check if it's a send_to_llm command (like /init)
                        elif isinstance(slash_result, dict) and slash_result.get(
                            "send_to_llm"
                        ):
                            print(slash_result.get("status", "Sending to LLM..."))
                            # Add the prompt as a user message and continue processing
                            messages.append(
                                {"role": "user", "content": slash_result["send_to_llm"]}
                            )
                            # Don't continue, let it process the LLM prompt
                        else:
                            print(slash_result)
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
                        print(
                            f"\n🗜️  Auto-compacting conversation (was ~{tokens_before} tokens)..."
                        )
                        try:
                            messages = self.agent.compact_conversation(messages)
                            tokens_after = self.agent._estimate_token_count(messages)
                            print(f"✅ Compacted to ~{tokens_after} tokens")
                        except Exception as e:
                            print(f"⚠️  Auto-compact failed: {e}")

                    try:
                        # Make API call interruptible by running in a task
                        print("\n💭 Thinking... (press ESC to interrupt)")
                        current_task = asyncio.create_task(
                            self.agent.generate_response(messages, stream=True)
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
                        done, pending = await asyncio.wait(
                            [current_task, monitor_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Cancel any remaining tasks
                        for task in pending:
                            task.cancel()

                        # Check for any kind of interruption (global or local)
                        if (
                            input_handler.interrupted
                            or self.global_interrupt_manager.is_interrupted()
                        ):
                            print("\n🛑 Request cancelled by user")
                            input_handler.interrupted = False
                            self.global_interrupt_manager.clear_interrupt()
                            current_task = None
                            continue

                        if (
                            current_task
                            and current_task.done()
                            and not current_task.cancelled()
                        ):
                            response = current_task.result()
                            current_task = None
                        else:
                            continue  # Request was cancelled, go back to input

                        # Handle the response
                        if hasattr(response, "__aiter__"):
                            # Streaming response
                            print("\nAssistant (press ESC to interrupt):")
                            sys.stdout.flush()

                            response_content = await self.handle_streaming_display(
                                response, interactive=True
                            )

                            # Add newline after streaming response completes
                            print()

                            # Add response to messages
                            if response_content:
                                messages.append(
                                    {"role": "assistant", "content": response_content}
                                )

                            # Reset interrupt count after successful operation
                            from cli_agent.core.global_interrupt import (
                                reset_interrupt_count,
                            )

                            reset_interrupt_count()

                        elif isinstance(response, str):
                            # Non-streaming text response
                            print(f"\nAssistant: {response}")
                            messages.append({"role": "assistant", "content": response})

                        else:
                            # Handle other response types
                            response_str = str(response)
                            print(f"\nAssistant: {response_str}")
                            messages.append(
                                {"role": "assistant", "content": response_str}
                            )

                    except ToolDeniedReturnToPrompt as e:
                        print(f"\n🚫 {e.reason}")
                        continue  # Return to prompt without adding to conversation
                    except Exception as e:
                        logger.error(f"Error in chat: {e}")
                        print(f"\n❌ Error: {str(e)}")
                        continue

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                error_msg = self.handle_conversation_error(e)
                print(f"\n{error_msg}")
                continue

        # Display goodbye message
        if not existing_messages:
            self.display_goodbye_message()

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

    def display_welcome_message(self):
        """Display welcome message at start of chat."""
        model_name = getattr(
            self.agent, "_get_current_runtime_model", lambda: "Unknown"
        )()
        print(f"🤖 Starting interactive chat with {model_name}")
        print("Type '/help' for available commands or '/quit' to exit.")
        print("-" * 50)

    def display_goodbye_message(self):
        """Display goodbye message at end of chat."""
        print("\n" + "-" * 50)
        print("Thanks for chatting! 👋")

    async def handle_streaming_display(
        self, response_generator, interactive: bool = True
    ):
        """Handle display of streaming responses."""
        # Simplified approach - process the response directly for now
        if hasattr(response_generator, "__aiter__"):
            content = ""
            async for chunk in response_generator:
                # Check for global interrupt during streaming
                if self.global_interrupt_manager.is_interrupted():
                    print("\n🛑 Streaming interrupted by user")
                    break

                chunk_text = str(chunk)
                content += chunk_text
                if interactive:
                    print(chunk_text, end="", flush=True)
            return content
        else:
            content = str(response_generator)
            if interactive:
                print(content)
            return content

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
