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
    level=logging.DEBUG,  # Changed to INFO to see subagent messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MCPDeepseekHost(BaseMCPAgent):
    """MCP Host that uses Deepseek as the language model backend."""

    def __init__(self, config: HostConfig, is_subagent: bool = False):
        # Store last reasoning content for deepseek-reasoner (must be before super().__init__)
        self.last_reasoning_content: Optional[str] = None

        # Call parent initialization (which will call our abstract methods)
        super().__init__(config, is_subagent)

    # Centralized Client Initialization Implementation
    # ===============================================

    def _get_provider_config(self):
        """Get DeepSeek-specific configuration."""
        return self.config.get_deepseek_config()

    def _get_streaming_preference(self, provider_config) -> bool:
        """Get streaming preference for DeepSeek."""
        return provider_config.stream

    def _calculate_timeout(self, provider_config) -> float:
        """Calculate timeout based on DeepSeek model."""
        # Use same timeout for all DeepSeek models (reasoner needs longer but 600s works for all)
        return 600.0

    def _create_llm_client(self, provider_config, timeout_seconds):
        """Create the DeepSeek OpenAI client."""
        self.deepseek_config = provider_config  # Store for later use
        client = OpenAI(
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
            timeout=timeout_seconds,
        )
        # Store as both _client (from base class) and deepseek_client (for compatibility)
        self.deepseek_client = client
        return client

    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to Deepseek format using shared utilities."""
        converter = OpenAIStyleToolConverter()
        return converter.convert_tools(self.available_tools)

    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from Deepseek response using shared utilities."""
        logger.debug(f"parse_tool_calls received response type: {type(response)}")
        logger.debug(f"parse_tool_calls response content: {response}")

        # Handle standard OpenAI-style response objects with tool_calls attribute
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                logger.debug(
                    f"Found {len(message.tool_calls)} tool calls in response object"
                )
                result = []
                for tc in message.tool_calls:
                    try:
                        # Handle both dict and object arguments
                        if hasattr(tc.function, "arguments"):
                            args = tc.function.arguments
                            if isinstance(args, str):
                                import json

                                args = json.loads(args)
                        else:
                            args = {}

                        result.append({"name": tc.function.name, "args": args})
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse tool call {tc.function.name}: {e}"
                        )

                logger.debug(f"Parsed tool calls from response object: {result}")
                return result

        # Handle string responses (custom DeepSeek format)
        elif isinstance(response, str):
            logger.debug("Parsing string response for tool calls")
            tool_calls = DeepSeekToolCallParser.parse_tool_calls(response)
            logger.debug(f"DeepSeekToolCallParser found {len(tool_calls)} tool calls")
            # Convert to dict format for compatibility
            result = [
                {"name": tc.function.name, "args": tc.function.arguments}
                for tc in tool_calls
            ]
            logger.debug(f"Converted tool calls: {result}")
            return result
        else:
            logger.debug(f"Response type not handled, returning empty list")
        return []

    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract text that appears before DeepSeek tool calls."""
        import re

        # DeepSeek-specific patterns
        patterns = [
            # DeepSeek tool call markers
            r"^(.*?)(?=<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>)",
            r"^(.*?)(?=```json\s*\{\s*\"function\")",
            r"^(.*?)(?=```python\s*<｜tool▁calls▁begin｜>)",
            # JSON function calls
            r"^(.*?)(?=\{\s*\"function\":)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                text_before = match.group(1).strip()
                if text_before:  # Only return if there's actual content
                    # Remove code block markers if present
                    text_before = re.sub(r"^```\w*\s*", "", text_before)
                    text_before = re.sub(r"\s*```$", "", text_before)
                    return text_before

        return ""

    def _get_llm_specific_instructions(self) -> str:
        """Provide DeepSeek-specific instructions."""
        if self.is_subagent:
            return """**Special Instructions for DeepSeek Subagents:**
1.  **Reason:** Use the <reasoning> section to outline your plan before taking action.
2.  **Act:** Execute your plan with tool calls. You can execute tool calls during reasoning blocks.
3.  **MANDATORY:** You MUST end your response by calling `emit_result` with a comprehensive summary of your findings.
4.  **Focus:** You are a subagent - you cannot spawn other subagents, only execute tools and provide results.

**Required Pattern:** After completing your investigation/task with tools, always call `emit_result` with detailed findings."""
        else:
            return """**Special Instructions for Deepseek Reasoner:**
1.  **Reason:** Use the <reasoning> section to outline your plan before taking action.
2.  **Act:** Execute your plan with tool calls. You can execute tool calls during reasoning blocks
3.  **Respond:** Provide the final answer to the user."""

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: Optional[bool] = None,
        modify_messages_in_place: bool = False,
    ) -> Union[str, Any]:
        """DeepSeek-specific response generation with message cleaning."""
        # Clean messages first to prevent JSON deserialization errors
        cleaned_messages = self._clean_messages_for_deepseek(messages)

        # Call parent's generate_response with cleaned messages and all parameters
        return await super().generate_response(
            cleaned_messages, tools, stream, modify_messages_in_place
        )

    async def _generate_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        interactive: bool = True,
    ) -> Any:
        """Generate completion using DeepSeek API."""
        # Check if we should use buffering for streaming JSON
        use_buffering = (
            hasattr(self, "streaming_json_callback")
            and self.streaming_json_callback is not None
        )

        # Use the stream parameter passed in (centralized logic decides streaming behavior)

        # Clean messages again as a safety net in case they bypassed generate_response
        cleaned_messages = self._clean_messages_for_deepseek(messages)

        # Handle deepseek-reasoner specific message formatting
        enhanced_messages = self._format_messages_for_reasoner(cleaned_messages)

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

        enhanced_messages = messages

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

    def _clean_messages_for_deepseek(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Clean messages to remove non-standard fields that cause DeepSeek JSON deserialization errors."""
        logger.debug(f"Cleaning {len(messages)} messages for DeepSeek")
        cleaned_messages = []

        for i, message in enumerate(messages):
            logger.debug(
                f"Processing message {i}: {type(message)} with keys: {list(message.keys()) if isinstance(message, dict) else 'not_dict'}"
            )

            # Ensure content is always a string
            content = message.get("content", "")
            if not isinstance(content, str):
                logger.debug(
                    f"Converting non-string content from {type(content)} to string"
                )
                content = str(content)

            cleaned_msg = {"role": message.get("role", "user"), "content": content}

            # Only add tool_calls if DeepSeek supports them in the expected format
            if message.get("tool_calls") and isinstance(
                message.get("tool_calls"), list
            ):
                # Convert to DeepSeek format if needed
                tool_calls = []
                for tc in message["tool_calls"]:
                    if isinstance(tc, dict) and "function" in tc:
                        # Ensure function.arguments is a JSON string, not a dict
                        cleaned_tc = tc.copy()
                        # Ensure type field is present for DeepSeek compatibility
                        if "type" not in cleaned_tc:
                            cleaned_tc["type"] = "function"
                        if (
                            "function" in cleaned_tc
                            and "arguments" in cleaned_tc["function"]
                        ):
                            args = cleaned_tc["function"]["arguments"]
                            if isinstance(args, dict):
                                cleaned_tc["function"]["arguments"] = json.dumps(args)
                            elif not isinstance(args, str):
                                cleaned_tc["function"]["arguments"] = str(args)
                        tool_calls.append(cleaned_tc)
                    elif isinstance(tc, dict) and "name" in tc:
                        # Convert from simplified format
                        tool_calls.append(
                            {
                                "id": tc.get("id", f"call_{len(tool_calls)}"),
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": (
                                        json.dumps(tc.get("args", {}))
                                        if isinstance(tc.get("args"), dict)
                                        else tc.get("args", "{}")
                                    ),
                                },
                            }
                        )

                if tool_calls:
                    cleaned_msg["tool_calls"] = tool_calls

            # Handle tool results/responses
            if message.get("tool_call_id"):
                cleaned_msg["tool_call_id"] = message["tool_call_id"]

            cleaned_messages.append(cleaned_msg)

        logger.debug(f"Cleaned messages result: {len(cleaned_messages)} messages")
        for i, msg in enumerate(cleaned_messages):
            logger.debug(
                f"Message {i}: role={msg.get('role')}, content_len={len(str(msg.get('content', '')))}, has_tool_calls={bool(msg.get('tool_calls'))}"
            )

        return cleaned_messages

    def _normalize_tool_calls_to_standard_format(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert DeepSeek tool calls to standardized format."""
        normalized_calls = []

        for i, tool_call in enumerate(tool_calls):
            if hasattr(tool_call, "get"):
                # Dict format (already structured)
                if "function" in tool_call:
                    # DeepSeek OpenAI-style format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call["function"].get("name", "unknown"),
                            "arguments": tool_call["function"].get("arguments", {}),
                        }
                    )
                else:
                    # Simple dict format
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": tool_call.get("name", "unknown"),
                            "arguments": tool_call.get("arguments", {}),
                        }
                    )
            else:
                # SimpleNamespace format
                normalized_calls.append(
                    {
                        "id": getattr(tool_call, "id", f"call_{i}"),
                        "name": getattr(tool_call, "name", "unknown"),
                        "arguments": getattr(tool_call, "args", {}),
                    }
                )

        return normalized_calls

    def _get_current_runtime_model(self) -> str:
        """Get the actual DeepSeek model being used at runtime."""
        return self.deepseek_config.model

    async def _handle_complete_response(
        self,
        response,
        original_messages: List[Dict[str, str]],
        interactive: bool = True,
    ) -> Union[str, Any]:
        """Handle non-streaming response from Deepseek."""
        from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

        # Use original messages list if modify_messages_in_place is enabled for session persistence
        if getattr(self, "_modify_messages_in_place", False):
            current_messages = original_messages
        else:
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

            # Use centralized tool call processing
            updated_messages, continuation_message, has_tool_calls = (
                await self._process_tool_calls_centralized(
                    response,
                    current_messages,
                    original_messages,
                    interactive=interactive,
                    streaming_mode=False,
                )
            )

            if has_tool_calls:
                current_messages = updated_messages

                if continuation_message:
                    # Restart with subagent results (non-streaming for subagents)
                    new_messages = [continuation_message]
                    # Clean messages before API call to prevent JSON deserialization errors
                    cleaned_new_messages = self._clean_messages_for_deepseek(
                        new_messages
                    )
                    response = self.deepseek_client.chat.completions.create(
                        model=self.deepseek_config.model,
                        messages=cleaned_new_messages,
                        temperature=self.deepseek_config.temperature,
                        max_tokens=self.deepseek_config.max_tokens,
                        stream=False,
                        tools=self.convert_tools_to_llm_format(),
                    )

                    # Replace the old response and messages with the new context
                    # and continue the loop to process this new response.
                    current_messages = new_messages
                    continue

                # Make another request with tool results
                # Clean messages before API call to prevent JSON deserialization errors
                cleaned_current_messages = self._clean_messages_for_deepseek(
                    current_messages
                )
                response = self.deepseek_client.chat.completions.create(
                    model=self.deepseek_config.model,
                    messages=cleaned_current_messages,
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
            # Use original messages list if modify_messages_in_place is enabled for session persistence
            if getattr(self, "_modify_messages_in_place", False):
                current_messages = original_messages if original_messages else []
            else:
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
                    logger.debug(f"Streaming chunk: {chunk}")

                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        logger.debug(f"Delta content: {delta.content}")
                        logger.debug(
                            f"Delta has tool_calls: {hasattr(delta, 'tool_calls') and delta.tool_calls}"
                        )

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

                # Create a mock response for centralized processing
                # Build response object that includes both standard and custom tool calls
                logger.debug(
                    f"Accumulated content before tool parsing: {repr(accumulated_content)}"
                )
                logger.debug(f"Accumulated tool calls: {accumulated_tool_calls}")

                mock_response = type("MockResponse", (), {})()
                mock_response.choices = [type("MockChoice", (), {})()]
                mock_response.choices[0].message = type("MockMessage", (), {})()
                mock_response.choices[0].message.content = accumulated_content

                # Set tool_calls for standard format detection
                if accumulated_tool_calls and any(
                    tc["function"]["name"] for tc in accumulated_tool_calls
                ):
                    # Convert accumulated tool calls to SimpleNamespace format for parse_tool_calls
                    from types import SimpleNamespace

                    tool_calls_objs = []
                    for tc in accumulated_tool_calls:
                        if tc["function"]["name"]:
                            tool_call = SimpleNamespace()
                            tool_call.id = (
                                tc["id"] or f"stream_call_{len(tool_calls_objs)}"
                            )
                            tool_call.type = tc["type"]
                            tool_call.function = SimpleNamespace()
                            tool_call.function.name = tc["function"]["name"]
                            tool_call.function.arguments = tc["function"]["arguments"]
                            tool_calls_objs.append(tool_call)
                    mock_response.choices[0].message.tool_calls = tool_calls_objs
                else:
                    mock_response.choices[0].message.tool_calls = None

                # Use centralized tool call processing for streaming
                try:
                    updated_messages, continuation_message, has_tool_calls = (
                        await self._process_tool_calls_centralized(
                            mock_response,
                            current_messages,
                            original_messages,
                            interactive=interactive,
                            streaming_mode=True,
                            accumulated_content=accumulated_content,
                        )
                    )

                    if has_tool_calls:
                        current_messages = updated_messages

                        if continuation_message:
                            # Yield the interrupt and completion messages for streaming
                            yield "\n🔄 Subagents spawned - interrupting main stream to wait for completion...\n"

                            # The centralized method already collected results, so yield completion message
                            yield f"\n📋 Collected subagent result(s). Restarting with results...\n"

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

                        yield "\n✅ Tool execution complete. Continuing...\n"

                except ToolDeniedReturnToPrompt:
                    # Store the exception to raise after generator completes
                    context.tool_denial_exception = ToolDeniedReturnToPrompt()
                    # yield "\nTool execution denied - returning to prompt.\n"
                    return  # Exit the generator

                # Make a new streaming request with tool results if we had tool calls
                if has_tool_calls:
                    try:
                        tools = (
                            self.convert_tools_to_llm_format()
                            if self.available_tools
                            else None
                        )
                        # Clean messages before API call to prevent JSON deserialization errors
                        cleaned_current_messages = self._clean_messages_for_deepseek(
                            current_messages
                        )
                        current_response = self.deepseek_client.chat.completions.create(
                            model=self.deepseek_config.model,
                            messages=cleaned_current_messages,
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

    async def _handle_buffered_streaming_response(
        self,
        response,
        original_messages: List[Dict[str, str]] = None,
        interactive: bool = True,
    ) -> str:
        """Handle streaming response by collecting all chunks and emitting through buffer system."""
        # Collect all chunks first
        full_content = ""

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta

                # Handle regular content
                if delta.content:
                    full_content += delta.content

        # Now emit the complete content through the buffer callback system
        if self.streaming_json_callback and full_content.strip():
            self.streaming_json_callback(full_content)

        return full_content
