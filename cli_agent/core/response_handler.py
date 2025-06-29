"""Response handling framework for MCP agents."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from cli_agent.core.event_system import (
    EventBus,
    StatusEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from cli_agent.core.global_interrupt import get_global_interrupt_manager
from cli_agent.utils.tool_name_utils import ToolNameUtils

logger = logging.getLogger(__name__)


class ResponseHandler:
    """Handles response processing, streaming, and tool execution coordination."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent
        self.global_interrupt_manager = get_global_interrupt_manager()

        # Initialize event emitter if event bus is available
        self.event_emitter = None
        if hasattr(agent, "event_bus") and agent.event_bus:
            from cli_agent.core.event_system import EventEmitter

            self.event_emitter = EventEmitter(agent.event_bus)

    def _ensure_arguments_are_json_string(self, arguments: Any) -> str:
        """Ensure tool call arguments are formatted as JSON string.

        Args:
            arguments: Arguments in any format (dict, str, etc.)

        Returns:
            JSON string representation of arguments
        """
        if isinstance(arguments, str):
            return arguments
        elif isinstance(arguments, dict):
            return json.dumps(arguments)
        else:
            return json.dumps({}) if arguments is None else json.dumps(arguments)

    async def handle_complete_response_generic(
        self,
        response: Any,
        original_messages: List[Dict[str, str]],
        interactive: bool = True,
    ) -> Union[str, Any]:
        """Generic handler for non-streaming responses from any LLM provider."""
        current_messages = original_messages.copy()

        # Extract content from response
        text_content, tool_calls, provider_data = self.agent._extract_response_content(
            response
        )

        # Handle provider-specific features (like reasoning content)
        self.agent._handle_provider_specific_features(provider_data)

        # Display text content if present and interactive
        if text_content and interactive:
            # Extract text that appears before tool calls
            text_before_tools = self.agent._extract_text_before_tool_calls(text_content)
            if text_before_tools:
                # Text formatting is handled by _process_streaming_chunks_with_events
                # No display logic needed here
                pass

        # Process tool calls if any
        if tool_calls:
            logger.info(f"Processing {len(tool_calls)} tool calls")
            # Process tool calls using centralized framework
            updated_messages, continuation_message, tools_executed = (
                await self.agent._process_tool_calls_centralized(
                    response,
                    current_messages,
                    original_messages,
                    interactive=interactive,
                    streaming_mode=False,
                    accumulated_content=text_content,
                )
            )
            logger.info(
                f"Tool processing result: tools_executed={tools_executed}, continuation_message={continuation_message is not None}"
            )

            # If tools were executed, generate follow-up response with tool results
            if tools_executed:
                # Generate response with tool results (both normal tools and subagent continuation)
                follow_up_response = await self.agent.generate_response(
                    updated_messages, stream=False
                )
                return follow_up_response

        # Include provider-specific content formatting
        final_response = self.agent._format_provider_specific_content(provider_data)
        if text_content:
            final_response += text_content

        return final_response or ""

    def handle_streaming_response_generic(
        self,
        response,
        original_messages: List[Dict[str, str]] = None,
        interactive: bool = True,
    ):
        """Generic handler for streaming responses from any LLM provider."""

        # Store updated messages that can be accessed after streaming completes
        self._updated_messages = None

        async def async_stream_generator():
            logger.info("Starting async_stream_generator")
            # Process streaming chunks to get full response
            try:
                accumulated_content, tool_calls, provider_data = (
                    await self.agent._process_streaming_chunks(response)
                )
                logger.info(
                    f"Processed streaming chunks: content={len(accumulated_content)} chars, tools={len(tool_calls) if tool_calls else 0}"
                )
            except Exception as e:
                logger.error(f"Error processing streaming chunks: {e}")
                accumulated_content = ""
                tool_calls = []
                provider_data = {}

            # Handle provider-specific features (like reasoning content)
            self.agent._handle_provider_specific_features(provider_data)

            # Check for subagent spawning BEFORE yielding any content
            task_tools_executed = False
            if tool_calls:
                # Check if any task tools will be executed (subagent spawning)
                for tc in tool_calls:
                    tool_name = ""
                    # Handle dictionary format (most common)
                    # Use centralized tool name extraction
                    tool_name = ToolNameUtils.extract_tool_name_from_call(tc)

                    # Check if this is a task spawning tool
                    if "task" in tool_name:
                        task_tools_executed = True
                        logger.info(
                            f"Detected task tool execution: {tool_name} - will not yield content until subagents complete"
                        )
                        break

            # If subagents will be spawned, DO NOT yield content yet - wait for subagent completion
            if not task_tools_executed and accumulated_content:
                # Only yield content if no subagents will be spawned
                logger.debug("No task tools detected, yielding accumulated content")
                yield accumulated_content
            elif task_tools_executed:
                logger.info(
                    "Task tools detected - suppressing content yield until subagent completion"
                )

            # Process tool calls
            if tool_calls:

                # Add assistant message with tool calls to conversation
                current_messages = original_messages.copy() if original_messages else []
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": accumulated_content or "",
                        "tool_calls": [
                            {
                                "id": (
                                    tc.get("id", f"call_{i}")
                                    if isinstance(tc, dict)
                                    else getattr(tc, "id", f"call_{i}")
                                ),
                                "type": "function",
                                "function": {
                                    "name": (
                                        tc["function"]["name"]
                                        if isinstance(tc, dict)
                                        and "function" in tc
                                        and isinstance(tc["function"], dict)
                                        else (
                                            tc.get(
                                                "name", f"<missing_name_dict_{id(tc)}>"
                                            )
                                            if isinstance(tc, dict)
                                            else getattr(
                                                tc,
                                                "name",
                                                (
                                                    getattr(
                                                        tc.function,
                                                        "name",
                                                        f"<missing_function_name_{id(tc)}>",
                                                    )
                                                    if hasattr(tc, "function")
                                                    else f"<missing_name_obj_{id(tc)}>"
                                                ),
                                            )
                                        )
                                    ),
                                    "arguments": self._ensure_arguments_are_json_string(
                                        tc["function"]["arguments"]
                                        if isinstance(tc, dict)
                                        and "function" in tc
                                        and isinstance(tc["function"], dict)
                                        else (
                                            tc.get("args", {})
                                            if isinstance(tc, dict)
                                            else getattr(
                                                tc,
                                                "args",
                                                (
                                                    getattr(
                                                        tc.function, "arguments", {}
                                                    )
                                                    if hasattr(tc, "function")
                                                    else {}
                                                ),
                                            )
                                        )
                                    ),
                                },
                            }
                            for i, tc in enumerate(tool_calls)
                        ],
                    }
                )

                # Store updated messages for access after streaming
                self._updated_messages = current_messages

                # Tool execution status is handled by _process_streaming_chunks_with_events
                # Focus only on tool execution logic here

                # Execute each tool and add results to conversation
                for i, tool_call in enumerate(tool_calls):
                    try:
                        # Check for global interrupt before each tool execution
                        if self.global_interrupt_manager.is_interrupted():
                            # Tool execution interrupted - no display logic needed here
                            pass
                            break

                        # Also check input handler for additional interrupt detection
                        if (
                            hasattr(self.agent, "_input_handler")
                            and self.agent._input_handler
                        ):
                            # Check for pending interrupts (ESC/Ctrl+C)
                            if self.agent._input_handler.check_for_interrupt():
                                # Tool execution interrupted - break without display
                                break

                        # Extract tool name with better error handling for both dict and object formats
                        tool_name = f"<no_tool_name_in_response_{id(tool_call)}>"

                        # Handle dictionary format (most common)
                        if isinstance(tool_call, dict):
                            if "function" in tool_call and isinstance(
                                tool_call["function"], dict
                            ):
                                tool_name = tool_call["function"].get(
                                    "name", f"<missing_function_name_{id(tool_call)}>"
                                )
                            elif "name" in tool_call:
                                tool_name = tool_call["name"]
                        # Handle object format (backup)
                        elif hasattr(tool_call, "name") and tool_call.name:
                            tool_name = tool_call.name
                        elif hasattr(tool_call, "function") and hasattr(
                            tool_call.function, "name"
                        ):
                            tool_name = tool_call.function.name

                        logger.debug(
                            f"Extracted tool_name: {tool_name} from tool_call: {type(tool_call)} - {tool_call}"
                        )
                        # Extract tool call ID handling both dict and object formats
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id", f"call_{i}")
                        else:
                            tool_call_id = getattr(tool_call, "id", f"call_{i}")

                        # Try to get args from different possible formats
                        tool_args = None
                        if isinstance(tool_call, dict):
                            if "function" in tool_call and isinstance(
                                tool_call["function"], dict
                            ):
                                tool_args = tool_call["function"].get("arguments", {})
                            elif "args" in tool_call:
                                tool_args = tool_call["args"]
                            elif "input" in tool_call:
                                tool_args = tool_call["input"]
                            else:
                                tool_args = {}
                        elif hasattr(tool_call, "args"):
                            tool_args = tool_call.args
                        elif hasattr(tool_call, "input"):
                            tool_args = tool_call.input
                        elif hasattr(tool_call, "function") and hasattr(
                            tool_call.function, "arguments"
                        ):
                            tool_args = tool_call.function.arguments
                        else:
                            tool_args = {}

                        # Convert string arguments to dict if needed
                        if isinstance(tool_args, str):
                            import json

                            # Handle empty string case (tools with no arguments)
                            if tool_args.strip() == "":
                                tool_args = {}
                            else:
                                tool_args = json.loads(tool_args)

                        # Debug log for replace_in_file spacing issues
                        if tool_name == "replace_in_file" and isinstance(
                            tool_args, dict
                        ):
                            old_text = tool_args.get("old_text", "")
                            new_text = tool_args.get("new_text", "")
                            if old_text or new_text:
                                logger.debug(
                                    f"replace_in_file args - old_text: {repr(old_text)}, new_text: {repr(new_text)}"
                                )

                        # Emit only tool execution start event - tool_call events create duplication
                        if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                            # Emit tool execution start event
                            await self.agent.event_emitter.emit_tool_execution_start(
                                tool_name=tool_name,
                                tool_id=tool_call_id,
                                arguments=tool_args,
                                streaming_mode=True,
                            )

                        # Execute the tool - map builtin tools correctly
                        builtin_tools = [
                            "todo_read",
                            "todo_write",
                            "bash_execute",
                            "read_file",
                            "write_file",
                            "list_directory",
                            "get_current_directory",
                            "replace_in_file",
                            "webfetch",
                            "task",
                            "task_status",
                            "task_results",
                        ]

                        if tool_name in builtin_tools:
                            tool_key = f"builtin:{tool_name}"
                        elif tool_name.startswith("builtin_"):
                            # Handle case where LLM uses builtin_ prefix
                            actual_tool_name = tool_name.replace("builtin_", "", 1)
                            if actual_tool_name in builtin_tools:
                                tool_key = f"builtin:{actual_tool_name}"
                            else:
                                tool_key = tool_name
                        elif tool_name.startswith("builtin:"):
                            # Already properly formatted
                            tool_key = tool_name
                        else:
                            tool_key = tool_name

                        logger.debug(
                            f"Final tool_key for execution: {tool_key} (original tool_name: {tool_name})"
                        )
                        result = (
                            await self.agent.tool_execution_engine.execute_mcp_tool(
                                tool_key, tool_args
                            )
                        )

                        # Emit tool result event if event bus is available
                        if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                            await self.agent.event_emitter.emit_tool_result(
                                tool_id=tool_call_id,
                                tool_name=tool_name,
                                result=result,
                                is_error=False,
                            )
                        elif interactive:
                            # Use formatting utils to handle truncation
                            if hasattr(self.agent, "formatting_utils"):
                                display_result = (
                                    self.agent.formatting_utils.truncate_tool_result(
                                        result
                                    )
                                )
                                # Tool results are handled by event system
                                pass
                            else:
                                # Tool results are handled by event system
                                pass

                        # Add tool result to conversation
                        current_messages.append(
                            {
                                "role": "tool",
                                "content": result,
                                "tool_call_id": tool_call_id,
                                "name": tool_name,  # Add tool name for provider compatibility
                            }
                        )

                    except Exception as e:
                        # Check if this is a tool permission denial that should return to prompt
                        from cli_agent.core.tool_permissions import (
                            ToolDeniedReturnToPrompt,
                        )

                        if isinstance(e, ToolDeniedReturnToPrompt):
                            # Re-raise to let chat interface handle it
                            raise

                        error_msg = f"Error executing {tool_name}: {str(e)}"

                        # Emit tool result error event if event bus is available
                        if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                            await self.agent.event_emitter.emit_tool_result(
                                tool_id=getattr(tool_call, "id", f"call_{i}"),
                                tool_name=tool_name,
                                result=error_msg,
                                is_error=True,
                            )
                        elif interactive:
                            # Use formatting utils to handle truncation for errors too
                            if hasattr(self.agent, "formatting_utils"):
                                display_error = (
                                    self.agent.formatting_utils.truncate_tool_result(
                                        error_msg
                                    )
                                )
                                # Tool errors are handled by event system
                                pass
                            else:
                                # Tool errors are handled by event system
                                pass

                        # Add error to conversation
                        current_messages.append(
                            {
                                "role": "tool",
                                "content": error_msg,
                                "tool_call_id": getattr(tool_call, "id", f"call_{i}"),
                            }
                        )

                # Check if subagents were spawned and handle interruption
                if (
                    task_tools_executed
                    and self.agent.subagent_manager
                    and self.agent.subagent_manager.get_active_count() > 0
                ):
                    # Emit status event for subagent spawning
                    if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                        await self.agent.event_emitter.emit_status(
                            status="Subagents spawned",
                            details="Interrupting main stream to wait for completion",
                            level="info",
                        )
                    # Subagent status messages handled by event system
                    pass

                    # Wait for all subagents to complete and collect results
                    subagent_results = (
                        await self.agent.subagent_coordinator.collect_subagent_results()
                    )

                    if subagent_results:
                        # Emit status event for subagent results collection
                        if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                            await self.agent.event_emitter.emit_status(
                                status=f"Collected {len(subagent_results)} subagent result(s)",
                                details="Restarting with results",
                                level="info",
                            )
                        # Subagent collection status handled by event system
                        pass

                        # Create continuation message with subagent results
                        original_request = (
                            original_messages[-1]["content"]
                            if original_messages
                            else "your request"
                        )
                        continuation_message = self.agent.subagent_coordinator.create_subagent_continuation_message(
                            original_request, subagent_results
                        )

                        # Emit status event for conversation restart
                        if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                            await self.agent.event_emitter.emit_status(
                                status="Restarting conversation with subagent results",
                                details="Processing continuation with collected results",
                                level="info",
                            )
                        # Subagent restart status handled by event system
                        pass

                        # Include the full conversation history plus the continuation message
                        # This ensures the LLM has context of all previous tool calls and results
                        continuation_messages = current_messages + [
                            continuation_message
                        ]
                        restart_response = await self.agent.generate_response(
                            continuation_messages, stream=True
                        )
                        if hasattr(restart_response, "__aiter__"):
                            yield f"\n\n"
                            async for chunk in restart_response:
                                yield chunk
                        else:
                            yield f"\n\n{str(restart_response)}"
                        return  # Exit - don't continue with original tool results
                    else:
                        # Emit warning event for no subagent results
                        if hasattr(self.agent, "event_bus") and self.agent.event_bus:
                            await self.agent.event_emitter.emit_status(
                                status="No explicit results collected from subagents",
                                details="Subagents completed without calling emit_result tool",
                                level="warning",
                            )
                        # Subagent no-results status handled by event system
                        pass
                        return

            # Generate follow-up response with tool results (only if no subagents were spawned)
                # Skip status message for simple single tool calls to reduce verbosity
                if hasattr(self.agent, "event_bus") and self.agent.event_bus and len(tool_calls) > 1:
                    await self.agent.event_emitter.emit_status(
                        status="Processing tool results",
                        details="Generating follow-up response based on tool outputs",
                        level="info",
                    )

                try:
                    # Add instruction to prevent repeating identical tool calls (same tool + same arguments)
                    final_messages = current_messages.copy()

                    # Build a list of recently executed tool calls with their arguments
                    recent_tool_calls = []
                    for msg in current_messages[-10:]:  # Check last 10 messages
                        if msg.get("role") == "assistant" and "tool_calls" in msg:
                            for tc in msg["tool_calls"]:
                                # Use centralized tool name extraction
                                tool_name = ToolNameUtils.extract_tool_name_from_call(
                                    tc
                                )

                                # Extract tool arguments
                                if isinstance(tc, dict):
                                    if "function" in tc and isinstance(
                                        tc["function"], dict
                                    ):
                                        tool_args = tc["function"].get(
                                            "arguments", "{}"
                                        )
                                    else:
                                        tool_args = tc.get("arguments", "{}")
                                else:
                                    tool_args = getattr(
                                        tc,
                                        "arguments",
                                        (
                                            getattr(tc.function, "arguments", "{}")
                                            if hasattr(tc, "function")
                                            else "{}"
                                        ),
                                    )
                                recent_tool_calls.append(
                                    f"{tool_name} with arguments {tool_args}"
                                )

                    if recent_tool_calls:
                        # Create specific instruction about not repeating identical calls
                        final_messages.append(
                            {
                                "role": "user",
                                "content": f"You have already executed these exact tool calls in this conversation: {', '.join(recent_tool_calls)}. Please continue your task with minimal prompting to the user or provide your response based on those results.",
                            }
                        )
                    else:
                        final_messages.append(
                            {
                                "role": "user",
                                "content": "Please provide your response based on the tool results above.",
                            }
                        )

                    follow_up_response = await self.agent.generate_response(
                        final_messages, stream=True
                    )
                    if hasattr(follow_up_response, "__aiter__"):
                        yield f"\n\n"
                        async for chunk in follow_up_response:
                            yield chunk
                    else:
                        yield f"\n\n{str(follow_up_response)}"
                except Exception as e:
                    # Follow-up errors handled by event system
                    pass

        return async_stream_generator()

    def get_updated_messages(self) -> Optional[List[Dict[str, Any]]]:
        """Get the updated conversation messages after tool execution."""
        return getattr(self, "_updated_messages", None)

    async def process_tool_calls_centralized(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        original_messages: List[Dict[str, Any]],
        interactive: bool = True,
        streaming_mode: bool = False,
        accumulated_content: str = "",
    ):
        """Centralized tool call processing logic."""
        # Delegate to agent's existing method
        if hasattr(self.agent, "_process_tool_calls_centralized"):
            return await self.agent._process_tool_calls_centralized(
                response,
                messages,
                original_messages,
                interactive,
                streaming_mode,
                accumulated_content,
            )
        else:
            # Fallback implementation
            return messages, None, False

    def extract_response_content(
        self, response: Any
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Extract content from provider response - delegates to agent."""
        return self.agent._extract_response_content(response)

    async def process_streaming_chunks(
        self, response
    ) -> tuple[str, List[Any], Dict[str, Any]]:
        """Process streaming chunks - delegates to agent."""
        return await self.agent._process_streaming_chunks(response)

    def create_mock_response(self, content: str, tool_calls: List[Any]) -> Any:
        """Create mock response for centralized processing - delegates to agent."""
        return self.agent._create_mock_response(content, tool_calls)

    def handle_provider_specific_features(self, provider_data: Dict[str, Any]) -> None:
        """Handle provider-specific features - delegates to agent."""
        self.agent._handle_provider_specific_features(provider_data)

    def format_provider_specific_content(self, provider_data: Dict[str, Any]) -> str:
        """Format provider-specific content - delegates to agent."""
        return self.agent._format_provider_specific_content(provider_data)
