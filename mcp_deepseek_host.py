#!/usr/bin/env python3
"""MCP Host implementation using Deepseek as the language model backend."""
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.utils.tool_conversion import OpenAIStyleToolConverter
from cli_agent.utils.tool_parsing import DeepSeekToolCallParser
from config import HostConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to see subagent messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MCPDeepseekHost(BaseMCPAgent):
    """MCP Host that uses Deepseek as the language model backend."""

    def __init__(self, config: HostConfig, is_subagent: bool = False):
        super().__init__(config, is_subagent)
        self.deepseek_config = config.get_deepseek_config()

        # Store last reasoning content for deepseek-reasoner
        self.last_reasoning_content: Optional[str] = None

        # Set streaming preference for centralized generate_response method
        self.stream = self.deepseek_config.stream

        # Initialize Deepseek client with appropriate timeout for reasoner model
        timeout_seconds = (
            600 if self.deepseek_config.model == "deepseek-reasoner" 
            else 600
        )
        self.deepseek_client = OpenAI(
            api_key=self.deepseek_config.api_key,
            base_url=self.deepseek_config.base_url,
            timeout=timeout_seconds,
        )

        logger.info(
            f"Initialized MCP Deepseek Host with model: {self.deepseek_config.model}"
        )

    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract any text that appears before tool calls in the response."""

        # Pattern to find text before tool call markers
        before_tool_pattern = (
            r'^(.*?)(?=<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>'
            r'|```json\s*\{\s*"function"|```python\s*<｜tool▁calls▁begin｜>)'
        )
        match = re.search(before_tool_pattern, content, re.DOTALL)

        if match:
            text_before = match.group(1).strip()
            # Remove code block markers if present
            text_before = re.sub(r"^```\w*\s*", "", text_before)
            text_before = re.sub(r"\s*```$", "", text_before)
            return text_before

        return ""

    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to Deepseek format using shared utilities."""
        converter = OpenAIStyleToolConverter()
        return converter.convert_tools(self.available_tools)

    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from Deepseek response using shared utilities."""
        if isinstance(response, str):
            tool_calls = DeepSeekToolCallParser.parse_tool_calls(response)
            # Convert to dict format for compatibility
            return [
                {"name": tc.function.name, "args": tc.function.arguments}
                for tc in tool_calls
            ]
        return []

    def _create_system_prompt(self, for_first_message: bool = False) -> str:
        """Create a system prompt that includes tool information."""
        # Get the base system prompt
        system_prompt = super()._create_system_prompt(for_first_message)

        # Add Deepseek-specific instructions for reasoning
        deepseek_instructions = """

**Special Instructions for Deepseek Reasoner:**
1.  **Reason:** Use the <reasoning> section to outline your plan before taking action.
2.  **Act:** Execute your plan with tool calls.
3.  **Respond:** Provide the final answer to the user."""

        # Insert Deepseek-specific instructions before the final line
        final_line = "\n\nYou are the expert. Complete the task."
        if final_line in system_prompt:
            system_prompt = system_prompt.replace(
                final_line, deepseek_instructions + final_line
            )

        return system_prompt

    async def _generate_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        interactive: bool = True,
    ) -> Any:
        """Generate completion using DeepSeek API."""
        # Use the stream parameter passed in (centralized logic decides streaming behavior)

        # Handle deepseek-reasoner specific message formatting
        enhanced_messages = self._format_messages_for_reasoner(messages)

        try:
            response = self.deepseek_client.chat.completions.create(
                model=self.deepseek_config.model,
                messages=enhanced_messages,
                tools=tools,
                temperature=self.deepseek_config.temperature,
                max_tokens=self.deepseek_config.max_tokens,
                stream=stream,
            )

            if stream:
                return self._handle_streaming_response(
                    response, enhanced_messages, interactive=interactive
                )
            else:
                return await self._handle_complete_response(
                    response, enhanced_messages, interactive=interactive
                )

        except Exception as e:
            # Re-raise tool permission denials so they can be handled at the chat level
            from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

            if isinstance(e, ToolDeniedReturnToPrompt):
                raise  # Re-raise the exception to bubble up to interactive chat

            logger.error(f"Error in generate completion: {e}")
            return f"Error: {str(e)}"

    def _format_messages_for_reasoner(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages specifically for deepseek-reasoner model."""
        # Check if this is the first message in a chat with deepseek-reasoner
        is_first_message = len(messages) == 1 and messages[0].get("role") == "user"
        is_reasoner = self.deepseek_config.model == "deepseek-reasoner"

        enhanced_messages = messages.copy()

        # Handle system prompt differently for deepseek-reasoner
        if is_reasoner:
            # For deepseek-reasoner, only add system prompt to first message
            if is_first_message:
                system_prompt = self._create_system_prompt(for_first_message=True)
                if enhanced_messages and enhanced_messages[0].get("role") == "user":
                    # Prepend system prompt to user message
                    user_content = enhanced_messages[0]["content"]
                    enhanced_messages[0][
                        "content"
                    ] = f"{system_prompt}\n\n---\n\nUser Request: {user_content}"
            else:
                # For subsequent messages, prepend last reasoning content to user message
                if (
                    self.last_reasoning_content
                    and enhanced_messages
                    and enhanced_messages[-1].get("role") == "user"
                ):
                    user_content = enhanced_messages[-1]["content"]
                    enhanced_messages[-1][
                        "content"
                    ] = f"Previous reasoning: {self.last_reasoning_content}\n\n---\n\nUser Request: {user_content}"

        return enhanced_messages

    def _add_tool_results_to_conversation(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
        tool_results: List[str],
    ) -> List[Dict[str, Any]]:
        """Add tool results to conversation in DeepSeek format (OpenAI-style)."""
        updated_messages = messages.copy()

        # Add assistant message with tool calls
        if tool_calls:
            # Convert tool calls to OpenAI format for assistant message
            openai_tool_calls = []
            for i, tool_call in enumerate(tool_calls):
                if hasattr(tool_call, "get"):
                    # Already in dict format
                    openai_tool_calls.append(tool_call)
                else:
                    # Convert from SimpleNamespace to dict format
                    openai_tool_calls.append(
                        {
                            "id": getattr(tool_call, "id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": getattr(tool_call, "name", "unknown"),
                                "arguments": getattr(tool_call, "args", {}),
                            },
                        }
                    )

            updated_messages.append(
                {"role": "assistant", "content": "", "tool_calls": openai_tool_calls}
            )

        # Add tool result messages
        for i, (tool_call, result) in enumerate(zip(tool_calls, tool_results)):
            # Handle different tool call formats
            if hasattr(tool_call, "get"):
                tool_call_id = tool_call.get("id", f"call_{i}")
                tool_name = tool_call.get("function", {}).get("name", "unknown")
            else:
                tool_call_id = getattr(tool_call, "id", f"call_{i}")
                tool_name = getattr(tool_call, "name", "unknown")

            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
            )

        return updated_messages

    async def _handle_complete_response(
        self,
        response,
        original_messages: List[Dict[str, str]],
        interactive: bool = True,
    ) -> Union[str, Any]:
        """Handle non-streaming response from Deepseek."""
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        current_messages = original_messages.copy()

        # Debug log the raw response
        logger.debug(f"Raw LLM response: {response}")
        logger.debug(f"Raw LLM message content: {response.choices[0].message.content}")
        if (
            hasattr(response.choices[0].message, "reasoning_content")
            and response.choices[0].message.reasoning_content
        ):
            logger.debug(
                f"Raw LLM reasoning content: {response.choices[0].message.reasoning_content}"
            )

        round_num = 0
        while True:
            round_num += 1
            choice = response.choices[0]
            message = choice.message

            # Debug: log the actual response
            logger.debug(f"Processing response round {round_num}")
            logger.debug(f"Message content: {repr(message.content)}")
            logger.debug(f"Message has tool_calls: {bool(message.tool_calls)}")
            if message.tool_calls:
                logger.debug(
                    f"Tool calls: {[tc.function.name for tc in message.tool_calls]}"
                )

            # Handle reasoning content if present (deepseek-reasoner)
            reasoning_content = ""
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content
                logger.debug(f"Found reasoning content: {len(reasoning_content)} chars")
                # Store reasoning content for next message if using reasoner
                if self.deepseek_config.model == "deepseek-reasoner":
                    self.last_reasoning_content = reasoning_content
                if interactive:
                    print(f"\n<reasoning>{reasoning_content}</reasoning>", flush=True)

            # Display regular message content if present and not tool-only
            if message.content and interactive:
                # Check if this is just tool calls or has actual text content
                has_only_tool_calls = (
                    message.tool_calls
                    or ("<｜tool▁calls▁begin｜>" in message.content)
                    or ("<｜tool▁call▁begin｜>" in message.content)
                    or (
                        '{"function":' in message.content
                        and message.content.strip().startswith('{"function"')
                    )
                )

                if not has_only_tool_calls:
                    # Extract text that appears before any tool calls
                    text_before_tools = self._extract_text_before_tool_calls(
                        message.content
                    )
                    if text_before_tools:
                        formatted_text = self.format_markdown(text_before_tools)
                        print(f"\n{formatted_text}", flush=True)

            # Check if the model wants to call tools
            # Handle both OpenAI-style tool calls and Deepseek's custom format
            tool_calls = message.tool_calls

            # If no standard tool calls, check for Deepseek's custom format
            if (
                not tool_calls
                and message.content
                and (
                    "<｜tool▁calls▁begin｜>" in message.content
                    or "<｜tool▁call▁begin｜>" in message.content
                    or '"function":' in message.content
                    or '{"function"' in message.content
                )
            ):
                logger.warning("Detected Deepseek custom tool calling format")
                tool_calls = DeepSeekToolCallParser.parse_tool_calls(message.content)

            if tool_calls:
                if interactive:
                    print(f"\n🔧 Executing {len(tool_calls)} tool(s):", flush=True)
                    for i, tc in enumerate(tool_calls, 1):
                        tool_name = tc.function.name.replace("_", ":", 1)
                        try:
                            args = json.loads(tc.function.arguments)
                            args_preview = (
                                str(args)[:100] + "..."
                                if len(str(args)) > 100
                                else str(args)
                            )
                            print(f"   {i}. {tool_name} - {args_preview}", flush=True)
                        except Exception:
                            print(f"   {i}. {tool_name}", flush=True)

                # Add assistant message with tool calls
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                # Execute tool calls in parallel
                tool_coroutines = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name.replace("_", ":", 1)
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        tool_coroutines.append(
                            self._execute_mcp_tool_with_keepalive(
                                tool_name,
                                arguments,
                                keepalive_interval=self.deepseek_config.keepalive_interval,
                            )
                        )
                    except json.JSONDecodeError as e:
                        # Handle JSON parsing errors immediately
                        error_content = (
                            f"Error parsing arguments for {tool_name}: {e}\n"
                            "⚠️  Command failed - take this into account for your next action."
                        )
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_content,
                            }
                        )

                # Run all tool calls concurrently
                tool_results = await asyncio.gather(
                    *tool_coroutines, return_exceptions=True
                )

                # Check if we just spawned subagents and should interrupt immediately
                task_tools_executed = any(
                    "task" in tool_call.function.name for tool_call in tool_calls
                )
                if (
                    task_tools_executed
                    and self.subagent_manager
                    and self.subagent_manager.get_active_count() > 0
                ):
                    if interactive:
                        print(
                            "\n🔄 Subagents spawned - interrupting to wait for completion...",
                            flush=True,
                        )

                    # Wait for all subagents to complete and collect results
                    subagent_results = await self._collect_subagent_results()

                    if subagent_results:
                        if interactive:
                            print(
                                f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...",
                                flush=True,
                            )

                        # Create new message with subagent results
                        results_summary = "\n".join(
                            [
                                f"**Subagent Task: {result['description']}**\n{result['content']}"
                                for result in subagent_results
                            ]
                        )

                        continuation_message = {
                            "role": "user",
                            "content": f"""I requested: {original_messages[-1]['content']}

You spawned subagents and they have completed their tasks. Here are the results:

{results_summary}

Please provide your final analysis based on these subagent results. Do not spawn any new subagents - just analyze the provided data.""",
                        }

                        # Restart with subagent results (non-streaming for subagents)
                        new_messages = [continuation_message]
                        response = self.deepseek_client.chat.completions.create(
                            model=self.deepseek_config.model,
                            messages=new_messages,
                            temperature=self.deepseek_config.temperature,
                            max_tokens=self.deepseek_config.max_tokens,
                            stream=False,
                            tools=self.convert_tools_to_llm_format(),
                        )

                        # Replace the old response and messages with the new context
                        # and continue the loop to process this new response.
                        current_messages = new_messages
                        continue

                # Process results - check for permission denials FIRST before adding to conversation
                for i, result in enumerate(tool_results):
                    tool_call = tool_calls[i]
                    # Handle tuple return from keep-alive version first
                    if isinstance(result, tuple) and len(result) == 2:
                        actual_result, keepalive_messages = result
                        if isinstance(actual_result, Exception):
                            result = actual_result  # Extract the exception from tuple
                        else:
                            tool_result_content, keepalive_messages = result
                            # In non-interactive mode, we can log keep-alive messages
                            for msg in keepalive_messages:
                                logger.info(f"Keep-alive: {msg}")
                            # Continue to add tool result below

                    # CHECK FOR PERMISSION DENIAL FIRST - before adding anything to conversation
                    if isinstance(result, Exception):
                        if isinstance(result, ToolDeniedReturnToPrompt):
                            raise result  # Exit immediately - don't add ANYTHING to conversation

                    # If we get here, permission was granted, process the result normally
                    if isinstance(result, Exception):
                        tool_result_content = f"Error executing tool: {result}"
                    else:
                        # Result is not an exception, use the extracted value from tuple
                        tool_result_content = result if not isinstance(result, tuple) else tool_result_content

                    # Check if tool failed and add notice to the content sent to the model
                    tool_failed = (
                        isinstance(result, Exception)
                        or tool_result_content.startswith("Error:")
                        or "error" in tool_result_content.lower()[:100]
                    )
                    if tool_failed:
                        tool_result_content += "\n⚠️  Command failed - take this into account for your next action."

                    # Display tool result to user (matching streaming behavior)
                    if interactive:
                        result_msg = f"Result: {tool_result_content}"
                        print(f"\nTool {i+1} {result_msg}\n", flush=True)

                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_content,
                        }
                    )

                # Make another request with tool results
                response = self.deepseek_client.chat.completions.create(
                    model=self.deepseek_config.model,
                    messages=current_messages,
                    temperature=self.deepseek_config.temperature,
                    max_tokens=self.deepseek_config.max_tokens,
                    stream=interactive,  # Stream if in interactive mode
                    tools=self.convert_tools_to_llm_format(),
                )

                # If interactive, we need to handle the new streaming response
                if interactive:
                    return self._handle_streaming_response(
                        response, current_messages, interactive=True
                    )

                # Continue the loop to check if the new response has more tool calls
                continue
            elif message.content:
                logger.debug(f"Final response content: {message.content}")
                # Include reasoning content if present
                final_response = ""
                if reasoning_content:
                    final_response += f"<reasoning>{reasoning_content}</reasoning>\n\n"
                final_response += message.content
                return final_response
            else:
                # No more tool calls, return the final content
                final_response = ""
                if reasoning_content:
                    final_response += f"<reasoning>{reasoning_content}</reasoning>\n\n"
                if message.content:
                    final_response += message.content
                return final_response or ""

        # If we hit the max rounds, return what we have
        return response.choices[0].message.content or ""

    def _handle_streaming_response(
        self,
        response,
        original_messages: List[Dict[str, str]] = None,
        interactive: bool = True,
    ):
        """Handle streaming response from Deepseek with tool call support."""
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        class StreamingContext:
            """Context to store exceptions that need to be raised after streaming."""
            def __init__(self):
                self.tool_denial_exception = None

        context = StreamingContext()

        async def async_stream_generator():
            current_messages = original_messages.copy() if original_messages else []
            current_response = response  # Store the initial response

            round_num = 0
            while True:
                round_num += 1
                accumulated_content = ""
                accumulated_reasoning_content = ""
                accumulated_tool_calls = []
                current_tool_call = None

                # Process the current streaming response
                for chunk in current_response:

                    if chunk.choices:
                        delta = chunk.choices[0].delta

                        # Handle reasoning content (deepseek-reasoner)
                        if (
                            hasattr(delta, "reasoning_content")
                            and delta.reasoning_content
                        ):
                            accumulated_reasoning_content += delta.reasoning_content
                            yield delta.reasoning_content

                        # Handle regular content
                        if delta.content:
                            accumulated_content += delta.content
                            yield delta.content

                        # Handle tool calls in streaming
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                # Handle new tool call
                                if tool_call_delta.index is not None:
                                    # Ensure we have enough space in our list
                                    while (
                                        len(accumulated_tool_calls)
                                        <= tool_call_delta.index
                                    ):
                                        accumulated_tool_calls.append(
                                            {
                                                "id": None,
                                                "type": "function",
                                                "function": {
                                                    "name": None,
                                                    "arguments": "",
                                                },
                                            }
                                        )

                                    current_tool_call = accumulated_tool_calls[
                                        tool_call_delta.index
                                    ]

                                    # Update tool call data
                                    if tool_call_delta.id:
                                        current_tool_call["id"] = tool_call_delta.id
                                    if tool_call_delta.type:
                                        current_tool_call["type"] = tool_call_delta.type
                                    if tool_call_delta.function:
                                        if tool_call_delta.function.name:
                                            current_tool_call["function"][
                                                "name"
                                            ] = tool_call_delta.function.name
                                        if tool_call_delta.function.arguments:
                                            current_tool_call["function"][
                                                "arguments"
                                            ] += tool_call_delta.function.arguments

                # Check for tool calls (both standard and custom format)
                has_tool_calls = False
                tool_calls = []

                # Check standard tool calls
                if accumulated_tool_calls and any(
                    tc["function"]["name"] for tc in accumulated_tool_calls
                ):
                    has_tool_calls = True
                    yield "\n\n🔧 Executing tools...\n"

                    # Convert to tool call objects
                    from types import SimpleNamespace

                    for tc in accumulated_tool_calls:
                        if tc["function"]["name"]:
                            tool_call = SimpleNamespace()
                            tool_call.id = tc["id"] or f"stream_call_{len(tool_calls)}"
                            tool_call.type = tc["type"]
                            tool_call.function = SimpleNamespace()
                            tool_call.function.name = tc["function"]["name"]
                            tool_call.function.arguments = tc["function"]["arguments"]
                            tool_calls.append(tool_call)

                # Check for Deepseek custom format if no standard tool calls
                elif accumulated_content and (
                    "<｜tool▁calls▁begin｜>" in accumulated_content
                    or "<｜tool▁call▁begin｜>" in accumulated_content
                    or '"function":' in accumulated_content
                ):
                    has_tool_calls = True
                    yield "\n\n🔧 Parsing custom tool calls...\n"
                    tool_calls = DeepSeekToolCallParser.parse_tool_calls(
                        accumulated_content
                    )

                # Execute tools if found
                if has_tool_calls and tool_calls:
                    # Add assistant message with tool calls to conversation
                    current_messages.append(
                        {
                            "role": "assistant",
                            "content": accumulated_content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )

                    # Execute tools and add results to conversation
                    # Execute all tools in parallel like the non-streaming version
                    tool_coroutines = []
                    tool_call_mapping = {}

                    # Prepare all tool executions
                    for i, tool_call in enumerate(tool_calls, 1):
                        tool_name = tool_call.function.name.replace("_", ":", 1)
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                            # Show tool name and abbreviated parameters
                            args_preview = (
                                str(arguments)[:100] + "..."
                                if len(str(arguments)) > 100
                                else str(arguments)
                            )
                            yield f"\n🔧 Tool {i}: {tool_name}\n📝 Parameters: {args_preview}\n"

                            tool_coroutines.append(
                                self._execute_mcp_tool_with_keepalive(
                                    tool_name,
                                    arguments,
                                    keepalive_interval=self.deepseek_config.keepalive_interval,
                                )
                            )
                            tool_call_mapping[len(tool_coroutines) - 1] = tool_call

                        except json.JSONDecodeError as e:
                            error_msg = f"Error parsing arguments for {tool_name}: {e} ⚠️  Command failed - take this into account for your next action."
                            yield f"{error_msg}\n"
                            # Add error to conversation
                            current_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": error_msg,
                                }
                            )

                    # Execute all tools with real-time streaming
                    if tool_coroutines:
                        yield f"\n⚡ Executing {len(tool_coroutines)} tool(s) in parallel...\n"

                        # Create tasks with identifiers for real-time completion tracking
                        tasks_to_tools = {}
                        for i, (coro, tool_call) in enumerate(
                            zip(tool_coroutines, tool_call_mapping.values())
                        ):
                            task = asyncio.create_task(coro)
                            tasks_to_tools[task] = (tool_call, i + 1)

                        pending_tasks = set(tasks_to_tools.keys())
                        completed_count = 0

                        # Process tools as they complete for real-time output
                        while pending_tasks:
                            done, pending_tasks = await asyncio.wait(
                                pending_tasks, return_when=asyncio.FIRST_COMPLETED
                            )

                            for task in done:
                                completed_count += 1
                                tool_call, tool_num = tasks_to_tools[task]

                                try:
                                    result = task.result()

                                    if isinstance(result, Exception):
                                        if isinstance(result, ToolDeniedReturnToPrompt):
                                            # Store the exception to raise after generator completes
                                            context.tool_denial_exception = result
                                            yield "\nTool execution denied - returning to prompt.\n"
                                            return  # Exit the generator

                                        error_msg = f"Error executing tool: {str(result)} ⚠️  Command failed - take this into account for your next action."
                                        yield f"\nTool {completed_count} Result: {error_msg}\n"
                                        current_messages.append(
                                            {
                                                "role": "tool",
                                                "tool_call_id": tool_call.id,
                                                "content": error_msg,
                                            }
                                        )
                                    else:
                                        tool_result, keepalive_messages = result

                                        # Yield any keep-alive messages immediately
                                        for msg in keepalive_messages:
                                            yield f"{msg}\n"

                                        # Subagent messages displayed immediately via callback - no duplication needed

                                        # Check if tool failed and add appropriate notice
                                        tool_failed = (
                                            tool_result.startswith("Error:")
                                            or "error" in tool_result.lower()[:100]
                                        )
                                        result_msg = f"Result: {tool_result}"
                                        if tool_failed:
                                            result_msg += " ⚠️  Command failed - take this into account for your next action."
                                        yield f"\nTool {completed_count} {result_msg}\n"

                                        # Add tool result to conversation
                                        current_messages.append(
                                            {
                                                "role": "tool",
                                                "tool_call_id": tool_call.id,
                                                "content": tool_result,
                                            }
                                        )

                                except Exception as e:
                                    # Handle tool permission denials specially - don't add to conversation
                                    if isinstance(e, ToolDeniedReturnToPrompt):
                                        context.tool_denial_exception = e
                                        yield "\nTool execution denied - returning to prompt.\n"
                                        return  # Exit the generator

                                    error_msg = f"Error executing tool: {str(e)} ⚠️  Command failed - take this into account for your next action."
                                    yield f"\nTool {completed_count} Result: {error_msg}\n"
                                    current_messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "content": error_msg,
                                        }
                                    )

                    # Check if we just spawned subagents and should interrupt immediately
                    task_tools_executed = any(
                        "task" in tool_call.function.name for tool_call in tool_calls
                    )
                    if (
                        task_tools_executed
                        and self.subagent_manager
                        and self.subagent_manager.get_active_count() > 0
                    ):
                        yield "\n🔄 Subagents spawned - interrupting main stream to wait for completion...\n"

                        # Wait for all subagents to complete and collect results
                        subagent_results = await self._collect_subagent_results()

                        if subagent_results:
                            yield f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...\n"

                            # Create new message with subagent results
                            results_summary = "\n".join(
                                [
                                    f"**Subagent Task: {result['description']}**\n{result['content']}"
                                    for result in subagent_results
                                ]
                            )

                            continuation_message = {
                                "role": "user",
                                "content": f"""I requested: {original_messages[-1]['content']}

You spawned subagents and they have completed their tasks. Here are the results:

{results_summary}

Please provide your final analysis based on these subagent results. Do not spawn any new subagents - just analyze the provided data.""",
                            }

                            # Restart with subagent results
                            new_messages = [continuation_message]
                            yield "\n🔄 Restarting conversation with subagent results...\n"

                            new_response = await self._generate_completion(
                                new_messages,
                                tools=self.convert_tools_to_llm_format(),
                                stream=True,
                                interactive=interactive,
                            )
                            if hasattr(new_response, "__aiter__"):
                                async for new_chunk in new_response:
                                    yield new_chunk
                            else:
                                yield str(new_response)
                            return
                        else:
                            yield "\n⚠️ No results collected from subagents.\n"
                            return

                    yield "\n✅ Tool execution complete. Continuing...\n"

                    # Make a new streaming request with tool results
                    try:
                        tools = (
                            self.convert_tools_to_llm_format()
                            if self.available_tools
                            else None
                        )
                        current_response = self.deepseek_client.chat.completions.create(
                            model=self.deepseek_config.model,
                            messages=current_messages,
                            temperature=self.deepseek_config.temperature,
                            max_tokens=self.deepseek_config.max_tokens,
                            stream=self.deepseek_config.stream,
                            tools=tools,
                        )
                        # Handle both streaming and non-streaming responses
                        if self.deepseek_config.stream:
                            # Continue to next round with new streaming response
                            continue
                        else:
                            # Handle non-streaming response and break out of streaming generator
                            if (
                                hasattr(current_response, "choices")
                                and current_response.choices
                            ):
                                choice = current_response.choices[0]
                                message = choice.message

                                # Handle reasoning content if present
                                if (
                                    hasattr(message, "reasoning_content")
                                    and message.reasoning_content
                                ):
                                    yield f"<reasoning>{message.reasoning_content}</reasoning>\n\n"

                                # Yield the content and exit
                                if message.content:
                                    yield message.content
                                break
                            else:
                                yield "Error: Invalid response format\n"
                                break

                    except Exception as e:
                        yield f"Error continuing conversation after tool execution: {e}\n"
                        break

                else:
                    # No tool calls, we're done
                    # Store reasoning content for next message if using reasoner
                    if (
                        accumulated_reasoning_content
                        and self.deepseek_config.model == "deepseek-reasoner"
                    ):
                        self.last_reasoning_content = accumulated_reasoning_content
                    break

        # Create a wrapper to handle tool denial exceptions
        async def wrapper():
            generator = async_stream_generator()

            try:
                async for chunk in generator:
                    yield chunk
            finally:
                # If we collected a tool denial exception, raise it now
                if context.tool_denial_exception:
                    raise context.tool_denial_exception

        return wrapper()
