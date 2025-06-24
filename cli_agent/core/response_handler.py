"""Response handling framework for MCP agents."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ResponseHandler:
    """Handles response processing, streaming, and tool execution coordination."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

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
                formatted_text = self.agent.formatter.format_markdown(text_before_tools)
                print(f"\n{formatted_text}", flush=True)

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

        async def async_stream_generator():
            # Process streaming chunks to get full response
            accumulated_content, tool_calls, provider_data = (
                await self.agent._process_streaming_chunks(response)
            )

            # Handle provider-specific features (like reasoning content)
            self.agent._handle_provider_specific_features(provider_data)

            # First, yield the accumulated content (the actual LLM response)
            if accumulated_content:
                yield accumulated_content

            # If there are tool calls, check for subagent spawning first
            if tool_calls:
                # Check if any task tools will be executed (subagent spawning)
                task_tools_executed = any(
                    "task"
                    in getattr(
                        tc,
                        "name",
                        tc.function.name if hasattr(tc, "function") else "unknown",
                    )
                    for tc in tool_calls
                )

                # Add assistant message with tool calls to conversation
                current_messages = original_messages.copy() if original_messages else []
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": accumulated_content or "",
                        "tool_calls": [
                            {
                                "id": getattr(tc, "id", f"call_{i}"),
                                "type": "function",
                                "function": {
                                    "name": getattr(
                                        tc,
                                        "name",
                                        (
                                            tc.function.name
                                            if hasattr(tc, "function")
                                            else "unknown"
                                        ),
                                    ),
                                    "arguments": self._ensure_arguments_are_json_string(
                                        getattr(
                                            tc,
                                            "args",
                                            (
                                                tc.function.arguments
                                                if hasattr(tc, "function")
                                                else {}
                                            ),
                                        )
                                    ),
                                },
                            }
                            for i, tc in enumerate(tool_calls)
                        ],
                    }
                )

                if interactive:
                    yield f"\n\n🔧 Executing {len(tool_calls)} tool(s)..."

                # Execute each tool and add results to conversation
                for i, tool_call in enumerate(tool_calls):
                    try:
                        tool_name = getattr(
                            tool_call,
                            "name",
                            (
                                tool_call.function.name
                                if hasattr(tool_call, "function")
                                else "unknown"
                            ),
                        )
                        tool_args = getattr(
                            tool_call,
                            "args",
                            (
                                tool_call.function.arguments
                                if hasattr(tool_call, "function")
                                else {}
                            ),
                        )

                        # Convert string arguments to dict if needed
                        if isinstance(tool_args, str):
                            import json

                            tool_args = json.loads(tool_args)

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
                            actual_tool_name = tool_name.replace("builtin_", "")
                            tool_key = f"builtin:{actual_tool_name}"
                        else:
                            tool_key = tool_name
                        result = (
                            await self.agent.tool_execution_engine.execute_mcp_tool(
                                tool_key, tool_args
                            )
                        )

                        if interactive:
                            yield f"\n✅ {tool_name}: {result[:100]}{'...' if len(result) > 100 else ''}"

                        # Add tool result to conversation
                        current_messages.append(
                            {
                                "role": "tool",
                                "content": result,
                                "tool_call_id": getattr(tool_call, "id", f"call_{i}"),
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
                        if interactive:
                            yield f"\n❌ {error_msg}"

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
                    if interactive:
                        yield f"\n\n🔄 Subagents spawned - interrupting main stream to wait for completion...\n"

                    # Wait for all subagents to complete and collect results
                    subagent_results = (
                        await self.agent.subagent_coordinator.collect_subagent_results()
                    )

                    if subagent_results:
                        if interactive:
                            yield f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...\n"

                        # Create continuation message with subagent results
                        original_request = (
                            original_messages[-1]["content"]
                            if original_messages
                            else "your request"
                        )
                        continuation_message = self.agent.subagent_coordinator.create_subagent_continuation_message(
                            original_request, subagent_results
                        )

                        # Restart conversation with subagent results instead of continuing
                        if interactive:
                            yield f"\n🔄 Restarting conversation with subagent results...\n"

                        restart_response = await self.agent.generate_response(
                            [continuation_message], stream=True
                        )
                        if hasattr(restart_response, "__aiter__"):
                            yield f"\n\n"
                            async for chunk in restart_response:
                                yield chunk
                        else:
                            yield f"\n\n{str(restart_response)}"
                        return  # Exit - don't continue with original tool results
                    else:
                        if interactive:
                            yield f"\n⚠️ No results collected from subagents.\n"
                        return

                # Generate follow-up response with tool results (only if no subagents were spawned)
                if interactive:
                    yield f"\n\n💭 Processing tool results..."

                try:
                    follow_up_response = await self.agent.generate_response(
                        current_messages, stream=True
                    )
                    if hasattr(follow_up_response, "__aiter__"):
                        yield f"\n\n"
                        async for chunk in follow_up_response:
                            yield chunk
                    else:
                        yield f"\n\n{str(follow_up_response)}"
                except Exception as e:
                    if interactive:
                        yield f"\n❌ Error generating follow-up: {str(e)}"

        return async_stream_generator()

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
