"""Subagent coordination and management for BaseMCPAgent."""

import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from cli_agent.core.global_interrupt import get_global_interrupt_manager
from cli_agent.core.permission_display import get_clean_permission_display, format_tool_description

logger = logging.getLogger(__name__)


class SubagentCoordinator:
    """Handles subagent coordination, messaging, and lifecycle management."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent
        self.global_interrupt_manager = get_global_interrupt_manager()

        # Initialize event emitter if event bus is available
        self.event_emitter = None
        if hasattr(agent, "event_bus") and agent.event_bus:
            from cli_agent.core.event_system import EventEmitter

            self.event_emitter = EventEmitter(agent.event_bus)
        
        # Get clean permission display manager
        self.clean_permission_display = get_clean_permission_display()
        
        # Initialize permission queue immediately to avoid race conditions
        try:
            self._permission_queue = asyncio.Queue()
            self._permission_processor_task = asyncio.create_task(self._process_permission_requests())
            logger.info("Permission queue and processor initialized")
        except RuntimeError:
            # No event loop yet - will be initialized when first permission request arrives
            self._permission_queue = None
            self._permission_processor_task = None
            logger.debug("Permission queue initialization deferred (no event loop)")

    def _start_permission_processor(self):
        """Start the background task that processes permission requests sequentially."""
        import asyncio
        
        # Only start if we have an event loop and haven't started yet
        try:
            if self._permission_queue is None:
                self._permission_queue = asyncio.Queue()
            
            if self._permission_processor_task is None or self._permission_processor_task.done():
                self._permission_processor_task = asyncio.create_task(self._process_permission_requests())
        except RuntimeError:
            # No event loop running yet - will be started later
            pass

    async def _process_permission_requests(self):
        """Background task that processes permission requests one at a time."""
        while True:
            try:
                # Wait for next permission request
                message, task_id = await self._permission_queue.get()
                
                # Process the request (this will block until user responds)
                await self._handle_subagent_permission_request_impl(message, task_id)
                
                # Mark task as done
                self._permission_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing permission request: {e}")

    async def handle_subagent_permission_request(self, message, task_id):
        """Handle permission requests from subagents by queueing them for sequential processing."""
        # Ensure permission processor is started with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._start_permission_processor()
                if self._permission_queue is not None:
                    break
            except Exception as e:
                logger.warning(f"Failed to start permission processor (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to start permission processor after {max_retries} retries")
                    # Create a default denial response file to prevent subagent from hanging
                    response_file = message.data.get("response_file")
                    if response_file:
                        try:
                            with open(response_file, "w") as f:
                                f.write("n")  # Default to deny
                            logger.info(f"Created fallback denial response for {task_id}")
                        except Exception as write_error:
                            logger.error(f"Failed to create fallback response: {write_error}")
                    return
                await asyncio.sleep(0.1)
        
        # Add to queue for sequential processing instead of concurrent execution
        try:
            await self._permission_queue.put((message, task_id))
            logger.info(f"Queued permission request from subagent {task_id}")
        except Exception as e:
            logger.error(f"Failed to queue permission request from {task_id}: {e}")
            # Create a fallback response to prevent hanging
            response_file = message.data.get("response_file")
            if response_file:
                try:
                    with open(response_file, "w") as f:
                        f.write("n")  # Default to deny
                    logger.info(f"Created fallback denial response for {task_id} due to queue error")
                except Exception as write_error:
                    logger.error(f"Failed to create fallback response: {write_error}")
    
    async def _handle_subagent_permission_request_impl(self, message, task_id):
        """Internal implementation of permission request handling."""
        try:
            # Extract permission request details
            request_id = message.data.get("request_id")
            response_file = message.data.get("response_file")
            tool_name = message.data.get("tool_name", "unknown")
            arguments = message.data.get("arguments", {})
            tool_description = message.data.get("description", f"Execute tool: {tool_name}")

            if not request_id or not response_file:
                logger.error("Invalid permission request format")
                return

            logger.info(
                f"Handling permission request from subagent {task_id}: {tool_name}"
            )

            # Use clean permission display
            try:
                # Calculate queue status
                queue_size = self._permission_queue.qsize() if self._permission_queue else 0
                total_requests = queue_size + 1  # Include current request
                current_position = 1  # We're processing this one now
                
                # Show clean permission display
                use_clean_display = self.clean_permission_display.show_permission_request(
                    tool_name=tool_name,
                    tool_description=tool_description,
                    arguments=arguments,
                    task_id=task_id,
                    queue_position=current_position,
                    total_requests=total_requests,
                )
                
                if use_clean_display:
                    logger.debug(f"Using clean display for permission request from {task_id}")
                else:
                    logger.warning(f"Clean display not available for {task_id}")
                        
            except Exception as e:
                logger.error(f"Clean permission display failed: {e}")
                use_clean_display = False

            # Get user input
            try:
                # Use the main agent's input handler to get user input
                if hasattr(self.agent, "_input_handler") and self.agent._input_handler:
                    input_handler = self.agent._input_handler
                else:
                    from cli_agent.core.input_handler import InterruptibleInput
                    input_handler = InterruptibleInput()

                # Get user choice (clean display has already shown the prompt)
                user_choice = input_handler.get_input("")

                # Handle None return (EOF/interrupt)
                if user_choice is None:
                    logger.info(f"No input received for {task_id}, defaulting to deny")
                    response = "n"
                else:
                    user_choice = user_choice.strip().lower()
                    # Validate the choice
                    if user_choice in ["y", "a", "A", "n", "d"]:
                        response = user_choice
                    else:
                        response = "n"  # Default to deny for invalid input

                logger.info(f"User choice for {task_id}: {response}")
            except Exception as e:
                logger.error(f"Error getting user input: {e}")
                response = "n"  # Default to deny on error

            # Write response to file
            try:
                with open(response_file, "w") as f:
                    f.write(response)
                logger.info(f"Wrote permission response to {response_file}")
            except Exception as e:
                logger.error(f"Error writing permission response: {e}")
                
            # Restore terminal state if using clean display
            if use_clean_display:
                try:
                    self.clean_permission_display.restore_terminal()
                except Exception as e:
                    logger.warning(f"Failed to restore terminal state: {e}")
                
            # Show remaining requests in queue
            if self._permission_queue:
                remaining = self._permission_queue.qsize()
                if remaining > 0 and self.event_emitter:
                    await self.event_emitter.emit_system_message(
                        f"✅ Permission handled. Processing next request ({remaining} remaining)...",
                        "queue_update",
                        "➡️",
                    )

        except Exception as e:
            logger.error(f"Error handling subagent permission request: {e}")

    def on_subagent_message(self, message):
        """Callback for when a subagent message is received - display during yield period."""
        try:
            # Update timeout tracking - reset timer whenever we receive any message
            import time

            self.agent.last_subagent_message_time = time.time()

            # Get task_id for identification (if available in message data)
            task_id = (
                message.data.get("task_id", "unknown")
                if hasattr(message, "data") and message.data
                else "unknown"
            )

            if message.type == "output":
                formatted = f"🤖 [SUBAGENT-{task_id}] {message.content}"
            elif message.type == "status":
                status = (
                    message.data.get("status", "unknown") if message.data else "unknown"
                )
                formatted = f"📊 [SUBAGENT-{task_id}] Status: {status}"
            elif message.type == "result":
                formatted = f"🤖 [SUBAGENT-{task_id}] 🤖 Response: {message.content}"
            elif message.type == "error":
                formatted = f"❌ [SUBAGENT-{task_id}] Error: {message.content}"
            elif message.type == "permission_request":
                # Queue permission request for sequential processing instead of concurrent execution
                import asyncio
                
                # This creates a task but it just adds to queue and returns immediately
                asyncio.create_task(
                    self.handle_subagent_permission_request(message, task_id)
                )
                return  # Don't display permission requests
            else:
                formatted = f"📨 [SUBAGENT-{task_id}] {message.type}: {message.content}"

            # Display the message immediately
            self.display_subagent_message_immediately(formatted, message.type)

            logger.debug(f"Displayed subagent message: {message.type}")
        except Exception as e:
            logger.error(f"Error displaying subagent message: {e}")

    def display_subagent_message_immediately(self, formatted: str, message_type: str):
        """Display subagent message immediately during streaming or collection periods."""
        try:
            # Emit as event - event system is always available
            import asyncio

            from cli_agent.core.event_system import ErrorEvent, StatusEvent

            # Add extra spacing for subagent messages to prevent overwriting
            formatted_with_spacing = f"\n{formatted}"

            # Create appropriate event based on message type
            if message_type == "error":
                event = ErrorEvent(
                    error_message=formatted_with_spacing, error_type="subagent_error"
                )
            else:
                # Determine status level
                level = "error" if message_type == "error" else "info"
                event = StatusEvent(status=formatted_with_spacing, level=level)

            # Emit event synchronously (since this method isn't async)
            if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                self.agent.event_bus.emit_sync(event)
        except Exception as e:
            logger.error(f"Error displaying message immediately: {e}")

    async def collect_subagent_results(self):
        """Wait for all subagents to complete and collect their results."""
        if not self.agent.subagent_manager:
            return []

        import time

        results = []
        max_wait_time = 300  # 5 minutes max wait
        start_time = time.time()

        # Initialize last message time if not set
        if self.agent.last_subagent_message_time is None:
            self.agent.last_subagent_message_time = start_time

        # Wait for all active subagents to complete
        while self.agent.subagent_manager.get_active_count() > 0:
            current_time = time.time()

            # Check for cancellation messages
            pending_messages = await self.agent.subagent_manager.get_pending_messages()
            for msg in pending_messages:
                if (
                    msg.type == "status"
                    and hasattr(msg, "data")
                    and msg.data.get("status") == "cancelled"
                ):
                    logger.info(
                        f"Subagent {msg.data.get('task_id', 'unknown')} was cancelled by user"
                    )
                    # Terminate all subagents and return to prompt
                    await self.agent.subagent_manager.terminate_all()
                    return []  # Return empty results to indicate cancellation
                elif msg.type == "error" and "Tool execution denied" in msg.content:
                    logger.info(f"Subagent tool denied, terminating all subagents")
                    await self.agent.subagent_manager.terminate_all()
                    return []  # Return empty results to indicate cancellation

            # Check timeout based on time since last message received
            time_since_last_message = (
                current_time - self.agent.last_subagent_message_time
            )

            if time_since_last_message > max_wait_time:
                logger.error(
                    f"Timeout waiting for subagents to complete ({time_since_last_message:.1f}s since last message)"
                )
                break

            await asyncio.sleep(0.5)

        # Collect results from completed subagents
        # ONLY collect results that were explicitly provided via emit_result
        logger.info(
            f"Checking {len(self.agent.subagent_manager.subagents)} subagents for explicit results"
        )
        for task_id, subagent in self.agent.subagent_manager.subagents.items():
            logger.info(
                f"Subagent {task_id}: completed={subagent.completed}, has_explicit_result={subagent.result is not None}"
            )
            # Only collect results that were explicitly set via emit_result (type="result" messages)
            if subagent.result:
                results.append(
                    {
                        "task_id": task_id,
                        "description": subagent.description,
                        "result": subagent.result,
                    }
                )
                logger.info(
                    f"Collected explicit result from {task_id}: {subagent.result[:100]}..."
                )

        # Clear processed subagent messages from the queue
        if (
            hasattr(self.agent, "subagent_message_queue")
            and self.agent.subagent_message_queue
        ):
            queue_size = self.agent.subagent_message_queue.qsize()
            if queue_size > 0:
                logger.info(
                    f"Clearing {queue_size} messages from subagent message queue"
                )
                # Clear the queue
                while not self.agent.subagent_message_queue.empty():
                    try:
                        self.agent.subagent_message_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                logger.info("Cleared subagent message queue after collecting results")

        return results

    def detect_task_tool_execution(self, tool_calls) -> bool:
        """Detect if any task tools were executed that spawn subagents."""
        for tool_call in tool_calls:
            # Handle different tool call formats
            tool_name = ""

            if isinstance(tool_call, dict):
                # Dict format (Gemini function calls)
                if "function" in tool_call:
                    # DeepSeek-style dict format
                    tool_name = tool_call["function"].get("name", "")
                elif "name" in tool_call:
                    # Simplified dict format
                    tool_name = tool_call.get("name", "")
            elif hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
                # Object format (DeepSeek streaming)
                tool_name = tool_call.function.name
            elif hasattr(tool_call, "name"):
                # Simple object format
                tool_name = tool_call.name

            # Check if this is a task spawning tool
            if tool_name in ["builtin_task", "task"]:
                logger.debug(f"Detected task tool execution: {tool_name}")
                return True

        return False

    def create_subagent_continuation_message(
        self, original_request: str, subagent_results: List[Dict]
    ) -> Dict:
        """Create a continuation message that includes subagent results."""
        # Format results for inclusion in message
        results_summary = []
        for result in subagent_results:
            results_summary.append(
                f"**Task: {result['description']} (ID: {result['task_id']})**\n{result['result']}"
            )

        results_text = "\n\n".join(results_summary)

        return {
            "role": "user",
            "content": f"""Subagent results:

{results_text}

Please continue with your task.""",
        }

    async def handle_subagent_coordination(
        self,
        tool_calls,
        original_messages: List[Dict],
        interactive: bool = True,
        streaming_mode: bool = False,
    ) -> Optional[Dict]:
        """
        Centralized subagent coordination logic.

        Returns None if no subagents were spawned, or a structured result dict if subagents completed.
        The result dict contains: continuation_message, interrupt_msg, completion_msg, restart_msg
        """
        if not self.agent.subagent_manager:
            return None

        # Check if any task tools were executed
        task_tools_executed = self.detect_task_tool_execution(tool_calls)

        if not (
            task_tools_executed and self.agent.subagent_manager.get_active_count() > 0
        ):
            return None

        # Skip auto-injection if BACKGROUND_SUBAGENTS is enabled
        if self.agent.config.background_subagents:
            return None

        # Prepare status messages for centralized handling
        interrupt_msg = "\r\n🔄 Subagents spawned - interrupting main stream to wait for completion...\r\n"

        # Wait for all subagents to complete and collect results
        subagent_results = await self.collect_subagent_results()

        # Special handling for cancellation - empty list with no active subagents means cancelled
        if (
            subagent_results == []
            and self.agent.subagent_manager.get_active_count() == 0
        ):
            # Subagents were cancelled, return special result to trigger return to prompt
            return {
                "cancelled": True,
                "interrupt_msg": interrupt_msg,
                "completion_msg": "\r\n🚫 Subagent cancelled due to tool denial. Returning to prompt.\r\n",
                "restart_msg": "",
            }

        if subagent_results:
            completion_msg = f"\r\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...\r\n"
            restart_msg = "\r\n🔄 Restarting conversation with subagent results...\r\n"

            # Create continuation message with subagent results
            original_request = (
                original_messages[-1]["content"]
                if original_messages
                else "your request"
            )
            continuation_message = self.create_subagent_continuation_message(
                original_request, subagent_results
            )

            return {
                "continuation_message": continuation_message,
                "interrupt_msg": interrupt_msg,
                "completion_msg": completion_msg,
                "restart_msg": restart_msg,
                "subagent_count": len(subagent_results),
            }
        else:
            logger.warning(
                "No explicit results collected from subagents - they must call emit_result"
            )
            return None

    async def cleanup(self):
        """Clean up subagent coordinator resources."""
        try:
            # Cancel permission processor task if running
            if self._permission_processor_task and not self._permission_processor_task.done():
                self._permission_processor_task.cancel()
                try:
                    await self._permission_processor_task
                except asyncio.CancelledError:
                    pass
                logger.info("Permission processor task cancelled")
        except Exception as e:
            logger.warning(f"Error during subagent coordinator cleanup: {e}")
