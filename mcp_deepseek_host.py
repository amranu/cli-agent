#!/usr/bin/env python3
"""This is the MCP host implementation that integrates with Deepseek's API."""
"""MCP Host implementation using Deepseek as the language model backend."""
import asyncio
import json
import logging
import re
import sys
import termios
import tty
import select
from typing import Any, Dict, List, Optional, Union

import click
from openai import OpenAI

from config import HostConfig, load_config, create_sample_env
from agent import BaseMCPAgent, InterruptibleInput

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from DEBUG to WARNING to suppress DEBUG and INFO messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPDeepseekHost(BaseMCPAgent):
    """MCP Host that uses Deepseek as the language model backend."""
    
    def __init__(self, config: HostConfig, is_subagent: bool = False):
        super().__init__(config, is_subagent)
        self.deepseek_config = config.get_deepseek_config()
        
        # Store last reasoning content for deepseek-reasoner
        self.last_reasoning_content: Optional[str] = None
        
        # Initialize Deepseek client with appropriate timeout for reasoner model
        timeout_seconds = 600 if self.deepseek_config.model == "deepseek-reasoner" else 600
        self.deepseek_client = OpenAI(
            api_key=self.deepseek_config.api_key,
            base_url=self.deepseek_config.base_url,
            timeout=timeout_seconds
        )
        
        logger.info(f"Initialized MCP Deepseek Host with model: {self.deepseek_config.model}")
    
    def get_token_limit(self) -> int:
        """Get the context token limit for DeepSeek models."""
        if self.deepseek_config.model == "deepseek-reasoner":
            return 128000  # DeepSeek-R1 has 128k context
        else:
            return 64000   # DeepSeek-Chat has 64k context
    
    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract any text that appears before tool calls in the response."""
        import re
        
        # Pattern to find text before tool call markers
        before_tool_pattern = r'^(.*?)(?=<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>|```json\s*\{\s*"function"|```python\s*<｜tool▁calls▁begin｜>)'
        match = re.search(before_tool_pattern, content, re.DOTALL)
        
        if match:
            text_before = match.group(1).strip()
            # Remove code block markers if present
            text_before = re.sub(r'^```\w*\s*', '', text_before)
            text_before = re.sub(r'\s*```$', '', text_before)
            return text_before
        
        return ""
    
    async def generate_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> Union[str, Any]:
        """Generate a response using Deepseek API."""
        # For subagents, use interactive=False to avoid terminal formatting issues
        # when output is forwarded to parent chat
        interactive = not self.is_subagent
        return await self.chat_completion(messages, stream=True, interactive=interactive)
    
    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to Deepseek format."""
        return self._convert_tools_to_deepseek_format()
    
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from Deepseek response."""
        if isinstance(response, str):
            return self._parse_deepseek_tool_calls(response)
        return []

    def _parse_deepseek_tool_calls(self, content: str) -> List:
        """Parse Deepseek's custom tool calling format."""
        import re
        import json
        from types import SimpleNamespace
        
        tool_calls = []
        
        # Multiple patterns to handle various Deepseek tool call formats
        patterns = [
            # Full format with begin/end markers
            r'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\s*```json\s*(.*?)\s*```<｜tool▁call▁end｜>',
            # Standard format with markers
            r'<｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\s*```json\s*(.*?)\s*```<｜tool▁call▁end｜>',
            # JSON object with function and parameters in code block
            r'```json\s*\{\s*"function":\s*"(\w+)"\s*,\s*"parameters":\s*(\{.*?\})\s*\}\s*```',
            # Direct JSON with function field
            r'\{\s*"function":\s*"(\w+)"\s*,\s*"parameters":\s*(\{.*?\})\s*\}',
            # Python-style function call format (with text before tool calls)
            r'```python\s*.*?<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\s*```json\s*(.*?)\s*```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            # Python-style function call format (simple)
            r'```python\s*<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\s*```json\s*(.*?)\s*```<｜tool▁call▁end｜>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            
            for i, match in enumerate(matches):
                try:
                    if isinstance(match, tuple) and len(match) >= 2:
                        func_name = match[0]
                        args_json = match[1]
                    else:
                        continue
                    
                    # Skip if we already have this function call
                    if any(tc.function.name == func_name for tc in tool_calls):
                        continue
                    
                    # Only accept valid builtin tool names
                    if not func_name.startswith('builtin_'):
                        continue
                    
                    # Validate JSON
                    try:
                        json.loads(args_json)
                    except json.JSONDecodeError:
                        continue
                    
                    # Create a mock tool call object similar to OpenAI's format
                    tool_call = SimpleNamespace()
                    tool_call.id = f"deepseek_call_{len(tool_calls)}"
                    tool_call.type = "function"
                    tool_call.function = SimpleNamespace()
                    tool_call.function.name = func_name
                    tool_call.function.arguments = args_json.strip()
                    
                    tool_calls.append(tool_call)
                    logger.warning(f"Parsed Deepseek tool call: {func_name}")
                    
                except Exception as e:
                    logger.error(f"Error parsing Deepseek tool call: {e}")
                    continue
        
        return tool_calls
    
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
            system_prompt = system_prompt.replace(final_line, deepseek_instructions + final_line)
        
        return system_prompt
    
    
    def _convert_tools_to_deepseek_format(self) -> List[Dict]:
        """Convert MCP tools to Deepseek function calling format."""
        deepseek_tools = []
        
        for tool_key, tool_info in self.available_tools.items():
            deepseek_tool = {
                "type": "function",
                "function": {
                    "name": tool_key.replace(":", "_"),  # Replace colon with underscore for Deepseek
                    "description": tool_info["description"] or f"Execute {tool_info['name']} tool",
                    "parameters": tool_info["schema"] or {"type": "object", "properties": {}}
                }
            }
            deepseek_tools.append(deepseek_tool)
        
        return deepseek_tools
    
    
    async def chat_completion(self, messages: List[Dict[str, str]], stream: bool = None, interactive: bool = False) -> Union[str, Any]:
        """Handle chat completion using Deepseek with MCP tool support."""
        if stream is None:
            stream = self.deepseek_config.stream
        
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
                    enhanced_messages[0]["content"] = f"{system_prompt}\n\n---\n\nUser Request: {user_content}"
            else:
                # For subsequent messages, prepend last reasoning content to user message
                if self.last_reasoning_content and enhanced_messages and enhanced_messages[-1].get("role") == "user":
                    user_content = enhanced_messages[-1]["content"]
                    enhanced_messages[-1]["content"] = f"Previous reasoning: {self.last_reasoning_content}\n\n---\n\nUser Request: {user_content}"
            # For subsequent messages with reasoner, don't add any system prompt
        else:
            # Standard behavior: add system prompt if not present
            if not enhanced_messages or enhanced_messages[0].get("role") != "system":
                system_prompt = self._create_system_prompt()
                enhanced_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Prepare tools for Deepseek
        tools = self._convert_tools_to_deepseek_format() if self.available_tools else None
        
        # For interactive mode, use streaming to handle tool calls properly
        if interactive:
            stream = True
        
        try:
            # Make request to Deepseek
            response = self.deepseek_client.chat.completions.create(
                model=self.deepseek_config.model,
                messages=enhanced_messages,
                temperature=self.deepseek_config.temperature,
                max_tokens=self.deepseek_config.max_tokens,
                stream=stream,
                tools=tools
            )
            
            if stream:
                return self._handle_streaming_response(response, enhanced_messages)
            else:
                return await self._handle_complete_response(response, enhanced_messages, interactive)
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return f"Error: {str(e)}"
    
    async def _handle_complete_response(self, response, original_messages: List[Dict[str, str]], interactive: bool = False) -> Union[str, Any]:
        """Handle non-streaming response from Deepseek."""
        current_messages = original_messages.copy()
        
        # Debug log the raw response
        logger.debug(f"Raw LLM response: {response}")
        logger.debug(f"Raw LLM message content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            logger.debug(f"Raw LLM reasoning content: {response.choices[0].message.reasoning_content}")
        
        
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
                logger.debug(f"Tool calls: {[tc.function.name for tc in message.tool_calls]}")
            
            # Handle reasoning content if present (deepseek-reasoner)
            reasoning_content = ""
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
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
                    message.tool_calls or 
                    ("<｜tool▁calls▁begin｜>" in message.content) or 
                    ("<｜tool▁call▁begin｜>" in message.content) or
                    ('{"function":' in message.content and message.content.strip().startswith('{"function"'))
                )
                
                if not has_only_tool_calls:
                    # Extract text that appears before any tool calls
                    text_before_tools = self._extract_text_before_tool_calls(message.content)
                    if text_before_tools:
                        formatted_text = self.format_markdown(text_before_tools)
                        print(f"\n{formatted_text}", flush=True)
            
            # Check if the model wants to call tools
            # Handle both OpenAI-style tool calls and Deepseek's custom format
            tool_calls = message.tool_calls
            
            # If no standard tool calls, check for Deepseek's custom format
            if not tool_calls and message.content and (
                "<｜tool▁calls▁begin｜>" in message.content or 
                "<｜tool▁call▁begin｜>" in message.content or 
                '"function":' in message.content or
                '{"function"' in message.content
            ):
                logger.warning("Detected Deepseek custom tool calling format")
                tool_calls = self._parse_deepseek_tool_calls(message.content)
                
            if tool_calls:
                if interactive:
                    print(f"\n🔧 Using {len(tool_calls)} tool(s)...", flush=True)
                
                # Add assistant message with tool calls
                current_messages.append({
                    "role": "assistant", 
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                })
                
                # Execute tool calls in parallel
                tool_coroutines = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name.replace("_", ":", 1)
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        tool_coroutines.append(
                            self._execute_mcp_tool_with_keepalive(tool_name, arguments, keepalive_interval=self.deepseek_config.keepalive_interval)
                        )
                    except json.JSONDecodeError as e:
                        # Handle JSON parsing errors immediately
                        error_content = f"Error parsing arguments for {tool_name}: {e}\n⚠️  Command failed - take this into account for your next action."
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_content
                        })

                # Run all tool calls concurrently
                tool_results = await asyncio.gather(*tool_coroutines, return_exceptions=True)

                # Process results
                for i, result in enumerate(tool_results):
                    tool_call = tool_calls[i]
                    if isinstance(result, Exception):
                        tool_result_content = f"Error executing tool: {result}"
                    else:
                        # Handle tuple return from keep-alive version
                        if isinstance(result, tuple) and len(result) == 2:
                            tool_result_content, keepalive_messages = result
                            # In non-interactive mode, we can log keep-alive messages
                            for msg in keepalive_messages:
                                logger.info(f"Keep-alive: {msg}")
                        else:
                            tool_result_content = result

                    # Check if tool failed and add notice to the content sent to the model
                    tool_failed = (isinstance(result, Exception) or 
                                 tool_result_content.startswith("Error:") or 
                                 "error" in tool_result_content.lower()[:100])
                    if tool_failed:
                        tool_result_content += "\n⚠️  Command failed - take this into account for your next action."

                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_content
                    })

                # Make another request with tool results
                response = self.deepseek_client.chat.completions.create(
                    model=self.deepseek_config.model,
                    messages=current_messages,
                    temperature=self.deepseek_config.temperature,
                    max_tokens=self.deepseek_config.max_tokens,
                    stream=interactive,  # Stream if in interactive mode
                    tools=self._convert_tools_to_deepseek_format()
                )
                
                # If interactive, we need to handle the new streaming response
                if interactive:
                    return self._handle_streaming_response(response, current_messages)

                
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
    
    def _handle_streaming_response(self, response, original_messages: List[Dict[str, str]] = None):
        """Handle streaming response from Deepseek with tool call support."""
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
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            accumulated_reasoning_content += delta.reasoning_content
                            # Stream reasoning content immediately as it arrives
                            yield delta.reasoning_content
                        
                        # Handle regular content
                        if delta.content:
                            accumulated_content += delta.content
                            yield delta.content
                        
                        # Handle tool calls in streaming
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                # Handle new tool call
                                if tool_call_delta.index is not None:
                                    # Ensure we have enough space in our list
                                    while len(accumulated_tool_calls) <= tool_call_delta.index:
                                        accumulated_tool_calls.append({
                                            'id': None,
                                            'type': 'function',
                                            'function': {'name': None, 'arguments': ''}
                                        })
                                    
                                    current_tool_call = accumulated_tool_calls[tool_call_delta.index]
                                    
                                    # Update tool call data
                                    if tool_call_delta.id:
                                        current_tool_call['id'] = tool_call_delta.id
                                    if tool_call_delta.type:
                                        current_tool_call['type'] = tool_call_delta.type
                                    if tool_call_delta.function:
                                        if tool_call_delta.function.name:
                                            current_tool_call['function']['name'] = tool_call_delta.function.name
                                        if tool_call_delta.function.arguments:
                                            current_tool_call['function']['arguments'] += tool_call_delta.function.arguments
                
                # Check for tool calls (both standard and custom format)
                has_tool_calls = False
                tool_calls = []
                
                # Check standard tool calls
                if accumulated_tool_calls and any(tc['function']['name'] for tc in accumulated_tool_calls):
                    has_tool_calls = True
                    yield "\n\n🔧 Executing tools...\n"
                    
                    # Convert to tool call objects
                    from types import SimpleNamespace
                    for tc in accumulated_tool_calls:
                        if tc['function']['name']:
                            tool_call = SimpleNamespace()
                            tool_call.id = tc['id'] or f"stream_call_{len(tool_calls)}"
                            tool_call.type = tc['type']
                            tool_call.function = SimpleNamespace()
                            tool_call.function.name = tc['function']['name']
                            tool_call.function.arguments = tc['function']['arguments']
                            tool_calls.append(tool_call)
                
                # Check for Deepseek custom format if no standard tool calls
                elif accumulated_content and (
                    "<｜tool▁calls▁begin｜>" in accumulated_content or 
                    "<｜tool▁call▁begin｜>" in accumulated_content or 
                    '"function":' in accumulated_content
                ):
                    has_tool_calls = True
                    yield "\n\n🔧 Parsing custom tool calls...\n"
                    tool_calls = self._parse_deepseek_tool_calls(accumulated_content)
                
                # Execute tools if found
                if has_tool_calls and tool_calls:
                    # Add assistant message with tool calls to conversation
                    current_messages.append({
                        "role": "assistant",
                        "content": accumulated_content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in tool_calls
                        ]
                    })
                    
                    # Execute tools and add results to conversation
                    # Execute all tools in parallel like the non-streaming version
                    tool_coroutines = []
                    tool_call_mapping = {}
                    
                    # Prepare all tool executions
                    for i, tool_call in enumerate(tool_calls, 1):
                        tool_name = tool_call.function.name.replace("_", ":", 1)
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                            yield f"\n{i}. Executing {tool_name}...\n"
                            
                            tool_coroutines.append(
                                self._execute_mcp_tool_with_keepalive(tool_name, arguments, keepalive_interval=self.deepseek_config.keepalive_interval)
                            )
                            tool_call_mapping[len(tool_coroutines) - 1] = tool_call
                            
                        except json.JSONDecodeError as e:
                            error_msg = f"Error parsing arguments for {tool_name}: {e} ⚠️  Command failed - take this into account for your next action."
                            yield f"{error_msg}\n"
                            # Add error to conversation
                            current_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg
                            })
                    
                    # Execute all tools with real-time streaming
                    if tool_coroutines:
                        yield f"\n🔧 Executing {len(tool_coroutines)} tool(s) in parallel...\n"
                        
                        # Create tasks with identifiers for real-time completion tracking
                        tasks_to_tools = {}
                        for i, (coro, tool_call) in enumerate(zip(tool_coroutines, tool_call_mapping.values())):
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
                                        error_msg = f"Error executing tool: {str(result)} ⚠️  Command failed - take this into account for your next action."
                                        yield f"\nTool {completed_count} Result: {error_msg}\n"
                                        current_messages.append({
                                            "role": "tool", 
                                            "tool_call_id": tool_call.id,
                                            "content": error_msg
                                        })
                                    else:
                                        tool_result, keepalive_messages = result
                                        
                                        # Yield any keep-alive messages immediately
                                        for msg in keepalive_messages:
                                            yield f"{msg}\n"
                                        
                                        # Check if tool failed and add appropriate notice
                                        tool_failed = tool_result.startswith("Error:") or "error" in tool_result.lower()[:100]
                                        result_msg = f"Result: {tool_result}"
                                        if tool_failed:
                                            result_msg += " ⚠️  Command failed - take this into account for your next action."
                                        yield f"\nTool {completed_count} {result_msg}\n"
                                        
                                        # Add tool result to conversation
                                        current_messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "content": tool_result
                                        })
                                        
                                except Exception as e:
                                    error_msg = f"Error executing tool: {str(e)} ⚠️  Command failed - take this into account for your next action."
                                    yield f"\nTool {completed_count} Result: {error_msg}\n"
                                    current_messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id, 
                                        "content": error_msg
                                    })
                    
                    yield "\n✅ Tool execution complete. Continuing...\n"
                    
                    # Make a new streaming request with tool results
                    try:
                        tools = self._convert_tools_to_deepseek_format() if self.available_tools else None
                        current_response = self.deepseek_client.chat.completions.create(
                            model=self.deepseek_config.model,
                            messages=current_messages,
                            temperature=self.deepseek_config.temperature,
                            max_tokens=self.deepseek_config.max_tokens,
                            stream=True,
                            tools=tools
                        )
                        # Continue to next round with new response
                        continue
                        
                    except Exception as e:
                        yield f"Error continuing conversation after tool execution: {e}\n"
                        break
                
                else:
                    # No tool calls, we're done
                    # Store reasoning content for next message if using reasoner
                    if accumulated_reasoning_content and self.deepseek_config.model == "deepseek-reasoner":
                        self.last_reasoning_content = accumulated_reasoning_content
                    break
        
        return async_stream_generator()
    


async def interactive_chat(host: MCPDeepseekHost):
    """Run an interactive chat session with streaming tool execution."""
    print(f"MCP Deepseek Host - Interactive Chat")
    print(f"Model: {host.deepseek_config.model}")
    print(f"Available tools: {len(host.available_tools)}")
    print("Commands: 'quit' to exit, 'tools' to list tools, 'ESC' to interrupt")
    print("Model switching: '/switch-chat', '/switch-reason', '/switch-gemini', '/switch-gemini-pro'")
    print("Utility: '/compact' to manually compact conversation, '/tokens' to show token count")
    print("Input: Single Enter to send, paste multiline content automatically detected")
    print("Navigation: Arrow keys for cursor movement, Backspace to delete")
    print("-" * 50)
    
    messages = []
    input_handler = InterruptibleInput()
    current_task = None
    
    while True:
        try:
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
            
            if user_input is None:  # Interrupted
                if current_task and not current_task.done():
                    current_task.cancel()
                    print("🛑 Operation cancelled by user")
                input_handler.interrupted = False
                current_task = None
                continue
            
            if user_input.lower().strip() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower().strip() == 'tools':
                if host.available_tools:
                    print("\nAvailable tools:")
                    for tool_key, tool_info in host.available_tools.items():
                        print(f"  - {tool_key}: {tool_info['description']}")
                else:
                    print("No tools available")
                continue
            elif user_input.lower().strip() == '/switch-chat':
                # Switch to deepseek-chat model
                config = load_config()
                config.deepseek_model = "deepseek-chat"
                config.save()
                # Update the host's config and recreate the deepseek client
                host.config = config
                host.deepseek_config = config.get_deepseek_config()
                timeout_seconds = 600 if host.deepseek_config.model == "deepseek-reasoner" else 600
                host.deepseek_client = host.deepseek_client.__class__(
                    api_key=host.deepseek_config.api_key,
                    base_url=host.deepseek_config.base_url,
                    timeout=timeout_seconds
                )
                print(f"✅ Model switched to: {config.deepseek_model}")
                continue
            elif user_input.lower().strip() == '/switch-reason':
                # Switch to deepseek-reasoner model
                config = load_config()
                config.deepseek_model = "deepseek-reasoner"
                config.save()
                # Update the host's config and recreate the deepseek client
                host.config = config
                host.deepseek_config = config.get_deepseek_config()
                timeout_seconds = 600 if host.deepseek_config.model == "deepseek-reasoner" else 600
                host.deepseek_client = host.deepseek_client.__class__(
                    api_key=host.deepseek_config.api_key,
                    base_url=host.deepseek_config.base_url,
                    timeout=timeout_seconds
                )
                print(f"✅ Model switched to: {config.deepseek_model}")
                continue
            elif user_input.lower().strip() == '/switch-gemini':
                # Switch to Gemini Flash backend
                config = load_config()
                config.deepseek_model = "gemini"  # Use this as a marker
                config.gemini_model = "gemini-2.5-flash"
                config.save()
                print(f"✅ Backend switched to: Gemini Flash 2.5 ({config.gemini_model})")
                print("⚠️  Note: Restart the chat session to use Gemini backend")
                continue
            elif user_input.lower().strip() == '/switch-gemini-pro':
                # Switch to Gemini Pro backend
                config = load_config()
                config.deepseek_model = "gemini"  # Use this as a marker
                config.gemini_model = "gemini-2.5-pro"
                config.save()
                print(f"✅ Backend switched to: Gemini Pro 2.5 ({config.gemini_model})")
                print("⚠️  Note: Restart the chat session to use Gemini backend")
                continue
            elif user_input.lower().strip() == '/compact':
                # Manually compact conversation
                if len(messages) > 3:
                    print(f"\n🗜️  Compacting conversation... ({len(messages)} messages)")
                    try:
                        messages = await host.compact_conversation(messages)
                        tokens = host.count_conversation_tokens(messages)
                        print(f"✅ Conversation compacted. Current tokens: ~{tokens}")
                    except Exception as e:
                        print(f"❌ Failed to compact: {e}")
                else:
                    print("📝 Conversation is already short, no need to compact")
                continue
            elif user_input.lower().strip() == '/tokens':
                # Show token count
                tokens = host.count_conversation_tokens(messages)
                limit = host.get_token_limit()
                percentage = (tokens / limit) * 100
                print(f"\n📊 Token usage: ~{tokens}/{limit} ({percentage:.1f}%)")
                if percentage > 80:
                    print("⚠️  Consider using '/compact' to reduce token usage")
                continue
            
            # Process the user input (no longer need buffer logic)
            if user_input.strip():  # Only process non-empty input
                messages.append({"role": "user", "content": user_input})
                
                # Check if we should auto-compact before making the API call
                if host.should_compact(messages):
                    tokens_before = host.count_conversation_tokens(messages)
                    print(f"\n🗜️  Auto-compacting conversation (was ~{tokens_before} tokens)...")
                    try:
                        messages = await host.compact_conversation(messages)
                        tokens_after = host.count_conversation_tokens(messages)
                        print(f"✅ Compacted to ~{tokens_after} tokens")
                    except Exception as e:
                        print(f"⚠️  Auto-compact failed: {e}")
                
                try:
                    # Make API call interruptible by running in a task
                    print("\n💭 Thinking... (press ESC to interrupt)")
                    current_task = asyncio.create_task(
                        host.chat_completion(messages, stream=True, interactive=True)
                    )
                    
                    # Create a background task to monitor for escape key
                    async def monitor_escape():
                        old_settings = None
                        try:
                            while not current_task.done():
                                try:
                                    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.1)[0]:
                                        # Set up raw mode for a single character read
                                        if old_settings is None:
                                            old_settings = termios.tcgetattr(sys.stdin.fileno())
                                            tty.setraw(sys.stdin.fileno())
                                        
                                        char = sys.stdin.read(1)
                                        if char == '\x1b':  # Escape key
                                            input_handler.interrupted = True
                                            return
                                    await asyncio.sleep(0.1)
                                except Exception as e:
                                    await asyncio.sleep(0.1)
                        finally:
                            if old_settings is not None:
                                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
                    
                    monitor_task = asyncio.create_task(monitor_escape())
                    
                    # Wait for either completion or interruption
                    response = None
                    done, pending = await asyncio.wait(
                        [current_task, monitor_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel any remaining tasks
                    for task in pending:
                        task.cancel()
                    
                    if input_handler.interrupted:
                        print("\n🛑 Request cancelled by user")
                        input_handler.interrupted = False
                        current_task = None
                        continue
                    
                    if current_task and current_task.done() and not current_task.cancelled():
                        response = current_task.result()
                        current_task = None
                    else:
                        continue  # Request was cancelled, go back to input
                    
                    if hasattr(response, '__aiter__'):
                        # Streaming response with potential tool execution
                        print("\nAssistant (press ESC to interrupt):")
                        sys.stdout.flush()
                        full_response = ""
                        
                        # Set up non-blocking input monitoring
                        stdin_fd = sys.stdin.fileno()
                        old_settings = termios.tcgetattr(stdin_fd)
                        tty.setraw(stdin_fd)
                        
                        interrupted = False
                        try:
                            async for chunk in response:
                                # Check for escape key on each chunk
                                if select.select([sys.stdin], [], [], 0)[0]:  # Non-blocking check
                                    char = sys.stdin.read(1)
                                    if char == '\x1b':  # Escape key
                                        interrupted = True
                                        break
                                
                                # Check for interruption flag
                                if input_handler.interrupted:
                                    interrupted = True
                                    input_handler.interrupted = False
                                    break
                                    
                                if isinstance(chunk, str):
                                    # Process chunk to ensure proper line positioning
                                    if '\n' in chunk:
                                        # Replace newlines to ensure next line starts at column 0
                                        processed_chunk = chunk.replace('\n', '\n\r')
                                        sys.stdout.write(processed_chunk)
                                        sys.stdout.flush()
                                    else:
                                        print(chunk, end="", flush=True)
                                    full_response += chunk
                                else:
                                    # Handle any non-string chunks if needed
                                    chunk_str = str(chunk)
                                    if '\n' in chunk_str:
                                        # Replace newlines to ensure next line starts at column 0
                                        processed_chunk = chunk_str.replace('\n', '\n\r')
                                        sys.stdout.write(processed_chunk)
                                        sys.stdout.flush()
                                    else:
                                        print(chunk_str, end="", flush=True)
                                    full_response += chunk_str
                        finally:
                            # Always restore terminal settings first
                            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
                            
                            # Clean up display if interrupted
                            if interrupted:
                                print("\n🛑 Streaming interrupted by user")
                                sys.stdout.flush()
                            else:
                                print()  # Normal newline after streaming
                                # Apply markdown formatting to the complete response
                                if full_response.strip():
                                    # Clear the previous unformatted output and show formatted version
                                    lines_to_clear = full_response.count('\n') + 2  # +2 for prompt and extra line
                                    for _ in range(lines_to_clear):
                                        sys.stdout.write('\x1b[A\x1b[K')  # Move up and clear line
                                    
                                    formatted_response = host.format_markdown(full_response)
                                    print(f"Assistant: {formatted_response}")
                        
                        # Add assistant response to messages
                        if full_response:  # Only add if not interrupted
                            messages.append({"role": "assistant", "content": full_response})
                    else:
                        # Non-streaming response (happens when tools are used)
                        formatted_response = host.format_markdown(str(response))
                        print(f"\nAssistant: {formatted_response}")
                        messages.append({"role": "assistant", "content": response})
                        
                except asyncio.CancelledError:
                    print("\n🛑 Request cancelled")
                    current_task = None
                except Exception as e:
                    print(f"\nError: {e}")
                    current_task = None
            else:
                # Empty input, just continue
                continue
            
        except KeyboardInterrupt:
            # Move to beginning of line and clear, then print exit message
            sys.stdout.write('\r\x1b[KExiting...\n')
            sys.stdout.flush()
            break
        except Exception as e:
            print(f"\nError: {e}")


@click.group()
@click.option('--config-file', default=None, help='Path to the configuration file (default: ~/.mcp/config.json)')
@click.pass_context
def cli(ctx, config_file):
    """MCP Deepseek Host - Run AI models with MCP tool integration."""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config_file


@cli.command()
def init():
    """Initialize configuration file."""
    create_sample_env()


@cli.command('switch-chat')
@click.pass_context
def switch_chat(ctx):
    """Switch the model to deepseek-chat."""
    config = load_config()
    config.deepseek_model = "deepseek-chat"
    click.echo(f"Model switched to: {config.deepseek_model}")
    # Save the updated config
    config.save()


@cli.command('switch-reason')
@click.pass_context
def switch_reason(ctx):
    """Switch the model to deepseek-reasoner."""
    config = load_config()
    config.deepseek_model = "deepseek-reasoner"
    click.echo(f"Model switched to: {config.deepseek_model}")
    # Save the updated config
    config.save()


@cli.command('switch-gemini')
@click.pass_context
def switch_gemini(ctx):
    """Switch to use Gemini Flash 2.5 as the backend model."""
    config = load_config()
    # Set Gemini Flash as the model and switch backend
    config.deepseek_model = "gemini"  # Use this as a marker
    config.gemini_model = "gemini-2.5-flash"
    click.echo(f"Backend switched to: Gemini Flash 2.5 ({config.gemini_model})")
    # Save the updated config
    config.save()


@cli.command('switch-gemini-pro')
@click.pass_context
def switch_gemini_pro(ctx):
    """Switch to use Gemini Pro 2.5 as the backend model."""
    config = load_config()
    # Set Gemini Pro as the model and switch backend
    config.deepseek_model = "gemini"  # Use this as a marker
    config.gemini_model = "gemini-2.5-pro"
    click.echo(f"Backend switched to: Gemini Pro 2.5 ({config.gemini_model})")
    # Save the updated config
    config.save()


@cli.command()
@click.option('--server', multiple=True, help='MCP server to connect to (format: name:command:arg1:arg2)')
@click.pass_context
async def chat(ctx, server):
    """Start interactive chat session."""
    try:
        # Load configuration
        config = load_config()
        
        # Check if Gemini backend should be used
        if config.deepseek_model == "gemini":
            if not config.gemini_api_key:
                click.echo("Error: GEMINI_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            # Import and create Gemini host
            from mcp_gemini_host import MCPGeminiHost, interactive_chat_gemini
            host = MCPGeminiHost(config)
            chat_function = interactive_chat_gemini
        else:
            if not config.deepseek_api_key:
                click.echo("Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            # Create Deepseek host
            host = MCPDeepseekHost(config)
            chat_function = interactive_chat
        
        # Connect to specified MCP servers
        for server_spec in server:
            parts = server_spec.split(':')
            if len(parts) < 2:
                click.echo(f"Invalid server spec: {server_spec}")
                continue
            
            server_name = parts[0]
            command = parts[1:]
            
            config.add_mcp_server(server_name, command)
            
            success = await host.start_mcp_server(server_name, config.mcp_servers[server_name])
            if not success:
                click.echo(f"Failed to start server: {server_name}")
        
        # Start interactive chat
        await chat_function(host)
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'host' in locals():
            await host.shutdown()


@cli.command()
@click.argument('message')
@click.option('--server', multiple=True, help='MCP server to connect to')
@click.pass_context
async def ask(ctx, message, server):
    """Ask a single question."""
    try:
        config = load_config()
        
        # Check if Gemini backend should be used
        if config.deepseek_model == "gemini":
            if not config.gemini_api_key:
                click.echo("Error: GEMINI_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            # Import and create Gemini host
            from mcp_gemini_host import MCPGeminiHost
            host = MCPGeminiHost(config)
        else:
            if not config.deepseek_api_key:
                click.echo("Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            host = MCPDeepseekHost(config)
        
        # Connect to servers
        for server_spec in server:
            parts = server_spec.split(':')
            if len(parts) < 2:
                continue
            
            server_name = parts[0]
            command = parts[1:]
            config.add_mcp_server(server_name, command)
            await host.start_mcp_server(server_name, config.mcp_servers[server_name])
        
        # Get response
        messages = [{"role": "user", "content": message}]
        response = await host.chat_completion(messages, stream=False)
        
        click.echo(response)
        
    finally:
        if 'host' in locals():
            await host.shutdown()


def main():
    """Main entry point."""
    # Store original async callbacks
    original_chat = chat.callback
    original_ask = ask.callback
    
    # Convert async commands to sync
    def sync_chat(**kwargs):
        asyncio.run(original_chat(**kwargs))
    
    def sync_ask(**kwargs):
        asyncio.run(original_ask(**kwargs))
    
    # Replace command callbacks
    chat.callback = sync_chat
    ask.callback = sync_ask
    
    cli()


if __name__ == "__main__":
    main()
