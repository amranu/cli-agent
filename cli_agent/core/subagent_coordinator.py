"""Subagent coordination and management for BaseMCPAgent."""

import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SubagentCoordinator:
    """Handles subagent coordination, messaging, and lifecycle management."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    def handle_subagent_permission_request(self, message, task_id):
        """Handle permission requests from subagents."""
        try:
            # Extract permission request details
            request_id = message.data.get("request_id")
            response_file = message.data.get("response_file")
            prompt_text = message.content

            if not request_id or not response_file:
                logger.error("Invalid permission request format")
                return

            logger.info(
                f"Handling permission request from subagent {task_id}: {prompt_text}"
            )

            # The subagent has displayed the permission prompt and is waiting for our response
            # We need to get the user's input and write it to the response file
            try:
                # Use the main agent's input handler to get user input
                if hasattr(self.agent, "_input_handler") and self.agent._input_handler:
                    input_handler = self.agent._input_handler
                else:
                    from cli_agent.core.input_handler import InterruptibleInput

                    input_handler = InterruptibleInput()

                # Get user choice using the proper input handler
                user_choice = input_handler.get_input("Choice: ").strip().lower()

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
                # Handle permission requests immediately
                self.handle_subagent_permission_request(message, task_id)
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
            # Use carriage return to overwrite any current line, then print message
            if message_type in ["output", "result", "error"]:
                # For important messages, ensure they're visible
                print(f"\r{formatted}", flush=True)
            else:
                # For status messages, display but allow overwriting
                print(f"\r{formatted}", end="", flush=True)
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
        logger.info(
            f"Checking {len(self.agent.subagent_manager.subagents)} subagents for results"
        )
        for task_id, subagent in self.agent.subagent_manager.subagents.items():
            logger.info(
                f"Subagent {task_id}: completed={subagent.completed}, has_result={subagent.result is not None}"
            )
            if subagent.result:
                results.append(
                    {
                        "task_id": task_id,
                        "description": subagent.description,
                        "result": subagent.result,
                    }
                )
                logger.info(
                    f"Collected result from {task_id}: {subagent.result[:100]}..."
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
            "content": f"""I requested: {original_request}

Subagent results:

{results_text}""",
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

        Returns None if no subagents were spawned, or a continuation message dict if subagents completed.
        """
        if not self.agent.subagent_manager:
            return None

        # Check if any task tools were executed
        task_tools_executed = self.detect_task_tool_execution(tool_calls)

        if not (
            task_tools_executed and self.agent.subagent_manager.get_active_count() > 0
        ):
            return None

        # Display interrupt message
        interrupt_msg = "\n🔄 Subagents spawned - interrupting main stream to wait for completion..."
        if streaming_mode:
            # For streaming mode, this should be yielded by the caller
            pass  # Caller will handle yielding
        elif interactive:
            print(interrupt_msg, flush=True)

        # Wait for all subagents to complete and collect results
        subagent_results = await self.collect_subagent_results()

        if subagent_results:
            completion_msg = f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results..."
            if streaming_mode:
                # For streaming mode, this should be yielded by the caller
                pass  # Caller will handle yielding
            elif interactive:
                print(completion_msg, flush=True)

            # Create continuation message with subagent results
            original_request = (
                original_messages[-1]["content"]
                if original_messages
                else "your request"
            )
            continuation_message = self.create_subagent_continuation_message(
                original_request, subagent_results
            )

            return continuation_message
        else:
            logger.warning("No results collected from subagents")
            return None
