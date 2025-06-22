#!/usr/bin/env python3
"""This is the MCP host implementation that integrates with Google Gemini's API."""

import asyncio
import json
import logging
import sys
import select
import termios
import time
import tty
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import HostConfig, load_config, create_sample_env, GeminiConfig
from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from cli_agent.utils.tool_conversion import GeminiToolConverter
from cli_agent.utils.tool_parsing import GeminiToolCallParser


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for comprehensive logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPGeminiHost(BaseMCPAgent):
    """MCP Host that uses Google Gemini as the language model backend."""
    
    def __init__(self, config: HostConfig, is_subagent: bool = False):
        super().__init__(config, is_subagent)
        self.gemini_config = config.get_gemini_config()
        
        # Initialize Gemini client with timeout configuration
        try:
            import httpx
            
            # Configure timeout for Gemini requests (longer for tool-heavy conversations)
            timeout_seconds = 120.0  # 2 minutes for Gemini requests
            
            # Create HTTP client with custom timeout
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
            
            self.gemini_client = genai.Client(
                api_key=self.gemini_config.api_key,
                http_client=http_client
            )
            self.http_client = http_client  # Store reference for cleanup
            logger.debug(f"Gemini client initialized with {timeout_seconds}s timeout")
            
        except Exception as e:
            import traceback
            logger.warning(f"Failed to create custom HTTP client: {e}")
            logger.debug(f"HTTP client creation traceback: {traceback.format_exc()}")
            # Fallback to default client
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_config.api_key)
                self.http_client = None
            except Exception as fallback_error:
                logger.error(f"Failed to create even default Gemini client: {fallback_error}")
                raise
        
        # Set streaming preference for centralized generate_response method
        self.stream = True  # Gemini always uses streaming
        
        logger.info(f"Initialized MCP Gemini Host with model: {self.gemini_config.model}")
    
    

    
    def convert_tools_to_llm_format(self) -> List[types.Tool]:
        """Convert tools to Gemini format using shared utilities."""
        converter = GeminiToolConverter()
        function_declarations = converter.convert_tools(self.available_tools)
        return [types.Tool(function_declarations=function_declarations)]
    
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from Gemini response using shared utilities."""
        tool_calls = GeminiToolCallParser.parse_all_formats(response, "")
        # Convert to dict format for compatibility
        return [{'name': tc.function.name, 'args': tc.function.arguments} for tc in tool_calls]
    
    
    async def _make_gemini_request_with_retry(self, request_func, max_retries: int = 3, base_delay: float = 1.0):
        """Make a Gemini API request with exponential backoff retry logic."""
        import asyncio
        import random
        
        for attempt in range(max_retries + 1):
            try:
                # Execute the request function
                if asyncio.iscoroutinefunction(request_func):
                    return await request_func()
                else:
                    return request_func()
                    
            except Exception as e:
                error_str = str(e)
                logger.error(f"Gemini API request failed (attempt {attempt+1}/{max_retries+1}): {error_str}")
                
                # Try to extract more details from the exception
                if hasattr(e, 'response'):
                    try:
                        response_text = await e.response.aread() if hasattr(e.response, 'aread') else e.response.text
                        logger.error(f"Response body: {response_text}")
                    except:
                        logger.error(f"Could not read response body from exception")
                
                if hasattr(e, '__cause__') and e.__cause__:
                    logger.error(f"Root cause: {e.__cause__}")
                
                # Check for 500 Internal Server Error specifically
                if "500" in error_str or "Internal Server Error" in error_str:
                    logger.error("Gemini API returned 500 Internal Server Error - likely prompt/content issue")
                
                is_retryable = (
                    "RetryError" in error_str or
                    "timeout" in error_str.lower() or
                    "network" in error_str.lower() or
                    "connection" in error_str.lower() or
                    "rate limit" in error_str.lower() or
                    "429" in error_str or
                    "502" in error_str or
                    "503" in error_str or
                    "504" in error_str or
                    "500" in error_str  # Add 500 as retryable for now
                )
                
                if attempt == max_retries or not is_retryable:
                    # Last attempt or non-retryable error
                    raise e
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Gemini API request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                logger.warning(f"Retrying in {delay:.2f} seconds...")
                
                await asyncio.sleep(delay)
    
    def _parse_python_style_function_calls(self, text: str) -> List:
        """Parse Python-style function calls that Gemini sometimes generates."""
        import re
        from types import SimpleNamespace
        
        function_calls = []
        
        # Look for patterns like: function_name('arg1', 'arg2', key='value')
        # This is a simplified parser - may need enhancement for complex cases
        python_call_pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        
        matches = re.finditer(python_call_pattern, text)
        
        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)
            
            # Check if this looks like one of our tools
            tool_key = f"builtin:{func_name}"
            if tool_key not in self.available_tools:
                continue
                
            # Try to parse the arguments
            try:
                # This is a very basic parser - for production you'd want something more robust
                arguments = {}
                
                # Simple parsing for common cases
                if args_str.strip():
                    # Remove quotes and split by commas (very basic)
                    args_parts = [part.strip() for part in args_str.split(',')]
                    
                    for i, part in enumerate(args_parts):
                        if '=' in part:
                            # Keyword argument
                            key, value = part.split('=', 1)
                            key = key.strip().strip('\'"')
                            value = value.strip().strip('\'"')
                            # Try to convert to appropriate type
                            try:
                                if value.isdigit():
                                    value = int(value)
                                elif value.lower() in ['true', 'false']:
                                    value = value.lower() == 'true'
                            except:
                                pass
                            arguments[key] = value
                        else:
                            # Positional argument - map to expected parameter
                            value = part.strip().strip('\'"')
                            # Try to convert to appropriate type
                            try:
                                # Try integer first
                                if value.isdigit():
                                    value = int(value)
                            except:
                                pass
                            
                            # Map to common parameter names based on function
                            if func_name == 'edit_file' and i == 0:
                                arguments['file_path'] = value
                            elif func_name == 'read_file' and i == 0:
                                arguments['file_path'] = value
                            elif func_name == 'read_file' and i == 1:
                                arguments['limit'] = value
                            elif func_name == 'write_file' and i == 0:
                                arguments['file_path'] = value
                            elif func_name == 'list_directory' and i == 0:
                                arguments['path'] = value
                            elif func_name == 'bash_execute' and i == 0:
                                arguments['command'] = value
                
                # Create a mock function call object
                function_call = SimpleNamespace()
                function_call.name = f"builtin_{func_name}"
                function_call.args = arguments
                function_calls.append(function_call)
                
                logger.info(f"Parsed Python-style function call: {func_name} with args: {arguments}")
                
            except Exception as e:
                logger.warning(f"Failed to parse Python-style function call {func_name}: {e}")
                continue
        
        return function_calls

    def _parse_xml_style_tool_calls(self, content: str) -> List:
        """Parse XML-style tool calls from content supporting multiple formats."""
        import re
        import json
        from types import SimpleNamespace
        
        function_calls = []
        
        # Pattern 1: Complex format with tool_name and parameters
        # <execute_tool>{"tool_name": "builtin:bash_execute", "parameters": {...}}</execute_tool>
        complex_pattern = r'<execute_tool>\s*\{\s*"tool_name":\s*"([^"]+)"\s*,\s*"parameters":\s*(\{.*?\})\s*\}\s*</execute_tool>'
        
        # Pattern 2: Simple format with tool name and direct args
        # <execute_tool>builtin:bash_execute{"command": "..."}</execute_tool>
        simple_pattern = r'<execute_tool>\s*(\w+:\w+)\s*(\{[^}]*\})\s*</execute_tool>'
        
        # Pattern 3: Inline tool format
        # Tool: builtin:replace_in_file
        # Tool Input:
        # ```json
        # {"arg": "value"}
        # ```
        inline_pattern = r'Tool:\s*(\w+:\w+)\s*\n\s*Tool Input:\s*\n\s*```json\s*\n(.*?)\n\s*```'
        
        # Try complex pattern first
        complex_matches = re.findall(complex_pattern, content, re.DOTALL)
        for match in complex_matches:
            try:
                tool_name = match[0]  # e.g., "builtin:bash_execute"
                parameters_json = match[1].strip()
                
                # Convert tool name format (builtin:bash_execute -> builtin_bash_execute)
                gemini_tool_name = tool_name.replace(":", "_")
                
                # Validate JSON
                try:
                    json.loads(parameters_json)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in complex XML tool call: {parameters_json}")
                    continue
                
                # Create function call object
                function_call = SimpleNamespace()
                function_call.name = gemini_tool_name
                function_call.args = parameters_json
                
                function_calls.append(function_call)
                logger.info(f"Parsed complex XML-style tool call: {gemini_tool_name} with args: {function_call.args}")
                
            except Exception as e:
                logger.warning(f"Failed to parse complex XML-style tool call: {e}")
                continue
        
        # Try simple pattern if no complex matches found
        if not function_calls:
            simple_matches = re.findall(simple_pattern, content, re.DOTALL)
            for match in simple_matches:
                try:
                    tool_name = match[0]  # e.g., "builtin:bash_execute"
                    args_json = match[1].strip()
                    
                    # Convert tool name format (builtin:bash_execute -> builtin_bash_execute)
                    gemini_tool_name = tool_name.replace(":", "_")
                    
                    # Validate JSON
                    try:
                        json.loads(args_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in simple XML tool call: {args_json}")
                        continue
                    
                    # Create function call object
                    function_call = SimpleNamespace()
                    function_call.name = gemini_tool_name
                    function_call.args = args_json
                    
                    function_calls.append(function_call)
                    logger.info(f"Parsed simple XML-style tool call: {gemini_tool_name} with args: {function_call.args}")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse simple XML-style tool call: {e}")
                    continue
        
        # Try inline pattern if no other matches found
        if not function_calls:
            inline_matches = re.findall(inline_pattern, content, re.DOTALL)
            for match in inline_matches:
                try:
                    tool_name = match[0]  # e.g., "builtin:replace_in_file"
                    args_json = match[1].strip()
                    
                    # Convert tool name format (builtin:replace_in_file -> builtin_replace_in_file)
                    gemini_tool_name = tool_name.replace(":", "_")
                    
                    # Validate JSON
                    try:
                        json.loads(args_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in inline tool call: {args_json}")
                        continue
                    
                    # Create function call object
                    function_call = SimpleNamespace()
                    function_call.name = gemini_tool_name
                    function_call.args = args_json
                    
                    function_calls.append(function_call)
                    logger.info(f"Parsed inline tool call: {gemini_tool_name} with args: {function_call.args}")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse inline tool call: {e}")
                    continue
        
        return function_calls

    def _extract_text_before_tool_calls(self, content: str) -> str:
        """Extract any text that appears before tool calls in the response."""
        import re
        
        # Pattern to find text before various tool call formats
        patterns = [
            r'^(.*?)(?=<execute_tool>)',  # XML-style tool calls
            r'^(.*?)(?=\w+\s*\()',        # Python-style function calls
            r'^(.*?)(?=Tool:\s*\w+:\w+)', # Inline tool calls
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                text_before = match.group(1).strip()
                if text_before:  # Only return if there's actual content
                    return text_before
        
        return ""

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Gemini format."""
        # Gemini expects a single string prompt, so we combine all messages
        gemini_prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                gemini_prompt_parts.append(f"System: {content}")
            elif role == "user":
                gemini_prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                gemini_prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                # Handle tool results
                gemini_prompt_parts.append(f"Tool Result: {content}")
        
        return "\n\n".join(gemini_prompt_parts)
    
    
    async def chat_completion(self, messages: List[Dict[str, str]], stream: bool = None, interactive: bool = False, input_handler=None) -> Union[str, Any]:
        """Handle chat completion using Gemini with MCP tool support."""
        if stream is None:
            stream = self.gemini_config.stream
        
        # Add system prompt only for the first message in a conversation
        enhanced_messages = messages.copy()
        is_first_message = len(messages) == 1 and messages[0].get("role") == "user"
        
        if is_first_message and (not enhanced_messages or enhanced_messages[0].get("role") != "system"):
            system_prompt = self._create_system_prompt()
            enhanced_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Convert messages to Gemini format
        gemini_prompt = self._convert_messages_to_gemini_format(enhanced_messages)
        
        # Prepare tools and config
        tools = self.convert_tools_to_llm_format()
        
        # Configure tool calling behavior
        tool_config = None
        if tools:
            try:
                # Configure function calling mode for compositional function calling
                mode_map = {
                    "AUTO": types.FunctionCallingConfigMode.AUTO,
                    "ANY": types.FunctionCallingConfigMode.ANY,
                    "NONE": types.FunctionCallingConfigMode.NONE
                }
                
                # Use configured mode, or fall back to legacy force_function_calling setting
                if self.gemini_config.function_calling_mode in mode_map:
                    mode = mode_map[self.gemini_config.function_calling_mode]
                elif self.gemini_config.force_function_calling:
                    mode = types.FunctionCallingConfigMode.ANY
                else:
                    mode = types.FunctionCallingConfigMode.AUTO
                
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=mode
                    )
                )
                logger.debug(f"Configured function calling mode: {mode}")
                
            except Exception as e:
                logger.warning(f"Could not configure function calling: {e}")
        
        config = types.GenerateContentConfig(
            temperature=self.gemini_config.temperature,
            max_output_tokens=self.gemini_config.max_output_tokens,
            top_p=self.gemini_config.top_p,
            top_k=self.gemini_config.top_k,
            tools=tools if tools else None,
            tool_config=tool_config
        )
        
        try:
            if stream:
                return await self._handle_streaming_response(gemini_prompt, config, interactive, enhanced_messages, input_handler)
            else:
                return await self._handle_complete_response(gemini_prompt, config, interactive, enhanced_messages)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            logger.info("Chat completion interrupted by user")
            raise
        except Exception as e:
            import traceback
            logger.error(f"Error in Gemini chat completion: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"
    

    async def _execute_function_calls(self, function_calls: List, interactive: bool, input_handler=None, streaming_mode=False) -> tuple:
        """Execute a list of function calls and return results and output."""
        function_results = []
        all_tool_output = []  # Collect all tool execution output for non-interactive mode
        
        # Prepare tool info for parallel execution
        tool_info_list = []
        tool_coroutines = []
        
        # Check for interruption before starting any tool execution
        if input_handler and input_handler.interrupted:
            all_tool_output.append("🛑 Tool execution interrupted by user")
            return function_results, all_tool_output
        
        for i, function_call in enumerate(function_calls, 1):
            tool_name = function_call.name.replace("_", ":", 1)  # Convert back to MCP format
            
            # Parse arguments from function call
            arguments = {}
            if hasattr(function_call, 'args') and function_call.args:
                try:
                    # First try to access as dict directly
                    if hasattr(function_call.args, 'items'):
                        arguments = dict(function_call.args)
                    elif hasattr(function_call.args, '__iter__'):
                        arguments = dict(function_call.args)
                    else:
                        # If args is a string, try to parse as JSON
                        if isinstance(function_call.args, str):
                            arguments = json.loads(function_call.args)
                        else:
                            arguments = {}
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in function call args: {e}")
                    logger.warning(f"Raw args: {function_call.args}")
                    arguments = {}
                except Exception as e:
                    logger.warning(f"Error parsing function call args: {e}")
                    logger.warning(f"Raw args: {function_call.args}")
                    arguments = {}
            
            # Store tool info for processing
            tool_info_list.append((i, tool_name, arguments))
            
            # Display tool execution step
            tool_execution_msg = self.display_tool_execution_step(i, tool_name, arguments, self.is_subagent, interactive=not self.is_subagent)
            if interactive and not streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            elif interactive and streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            else:
                all_tool_output.append(tool_execution_msg)
            
            # Create coroutine for parallel execution
            tool_coroutines.append(self._execute_mcp_tool(tool_name, arguments))
        
        # Execute all tools in parallel
        if tool_coroutines:
            try:
                # Execute all tool calls concurrently like DeepSeek
                tool_results = await asyncio.gather(*tool_coroutines, return_exceptions=True)
                
                # Process results in order
                for (i, tool_name, arguments), tool_result in zip(tool_info_list, tool_results):
                    tool_success = True
                    
                    # Handle exceptions
                    if isinstance(tool_result, Exception):
                        tool_success = False
                        tool_result = f"Exception during execution: {str(tool_result)}"
                    elif isinstance(tool_result, str):
                        # Check if tool result indicates an error
                        if tool_result.startswith("Error:") or "error" in tool_result.lower()[:100]:
                            tool_success = False
                    else:
                        # Convert non-string results to string
                        tool_result = str(tool_result)
                    
                    # Format result with success/failure status
                    status = "SUCCESS" if tool_success else "FAILED"
                    result_content = f"Tool {tool_name} {status}: {tool_result}"
                    if not tool_success:
                        result_content += "\n⚠️  Command failed - take this into account for your next action."
                    function_results.append(result_content)
                    
                    # Use unified tool result display
                    tool_result_msg = self.display_tool_execution_result(tool_result, not tool_success, self.is_subagent, interactive=not self.is_subagent)
                    
                    # Fix newlines in tool result messages to have proper cursor positioning
                    if interactive and (not streaming_mode or streaming_mode):
                        # Replace any bare newlines with \n\r to ensure proper cursor positioning
                        formatted_result_msg = tool_result_msg.replace('\n', '\n\r')
                        print(f"\r\x1b[K{formatted_result_msg}", flush=True)
                    else:
                        # Only add to tool output for non-interactive mode
                        all_tool_output.append(tool_result_msg)
                        
            except Exception as e:
                # Handle any unexpected errors during parallel execution
                error_msg = f"Error during parallel tool execution: {str(e)}"
                all_tool_output.append(error_msg)
                function_results.append(f"PARALLEL EXECUTION FAILED: {error_msg}")
        
        return function_results, all_tool_output

    async def _handle_complete_response(self, prompt: str, config: types.GenerateContentConfig, interactive: bool, original_messages: List[Dict[str, str]]) -> str:
        """Handle complete response from Gemini with iterative tool calling."""
        current_prompt = prompt
        all_accumulated_output = []
        
        try:
            while True:
                
                # Make API call to Gemini
                response = await self._make_gemini_request_with_retry(
                    lambda: self.gemini_client.models.generate_content(
                        model=self.gemini_config.model,
                        contents=current_prompt,
                        config=config
                    )
                )
                
                # Parse response content
                function_calls = self.parse_tool_calls(response)
                # Extract text response from response
                text_response = ""
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_response += part.text
                elif hasattr(response, 'text') and response.text:
                    text_response = response.text
                elif response.candidates and hasattr(response.candidates[0], 'text') and response.candidates[0].text:
                    text_response = response.candidates[0].text
                
                # Debug logging
                logger.debug(f"Parsed {len(function_calls)} function calls and {len(text_response)} chars of text")
                logger.debug(f"Full text response: {repr(text_response)}")
                if function_calls:
                    logger.debug(f"Function calls: {[fc.name for fc in function_calls]}")
                
                # Accumulate text response for non-interactive mode only
                # Interactive mode printing is handled by the chat loop to avoid duplication
                if text_response and not interactive:
                    if function_calls:
                        # Text with function calls - add to accumulated output
                        all_accumulated_output.append(f"Assistant: {text_response}")
                    else:
                        # Text without function calls - this is the final response
                        all_accumulated_output.append(f"Assistant: {text_response}")
                
                if function_calls:
                    # Handle function calls
                    if interactive:
                        print(f"\r\x1b[K\n\r{self.display_tool_execution_start(len(function_calls), self.is_subagent, interactive=not self.is_subagent)}", flush=True)
                    
                    # Execute function calls
                    function_results, tool_output = await self._execute_function_calls(function_calls, interactive)
                    
                    # Note: We don't add tool execution status messages to accumulated output
                    # as they are only for user feedback and cause LLM hallucinations
                    
                    # Create follow-up prompt with tool results and clear instruction
                    tool_results_text = "\n".join(function_results)
                    current_prompt = f"{current_prompt}\n\nTool Results:\n{tool_results_text}\n\nThe tool execution is complete. Please continue with either another tool use if needed or your response based on these results." 
                    
                    # Continue the loop - let Gemini decide if more tools are needed
                    continue
                else:
                    # No function calls - this is the final response
                    if not interactive and all_accumulated_output:
                        # Include all accumulated output (final response already included in loop above)
                        return "\n".join(all_accumulated_output)
                    else:
                        return text_response if text_response else ""
                
        except Exception as e:
            import traceback
            
            # Log detailed error information
            logger.error(f"Error in Gemini complete response: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Extract more specific error details if available
            error_details = str(e)
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"Caused by: {e.__cause__}")
                error_details += f" (caused by: {e.__cause__})"
            
            if hasattr(e, '__context__') and e.__context__:
                logger.error(f"Context: {e.__context__}")
            
            # Check for common Gemini API error patterns
            if "RetryError" in str(e):
                return f"Error: Gemini API retry failed - likely a network or rate limit issue. Details: {error_details}"
            elif "ClientError" in str(e):
                return f"Error: Gemini API client error - check API key and configuration. Details: {error_details}"
            elif "timeout" in str(e).lower():
                return f"Error: Request timed out. Details: {error_details}"
            else:
                return f"Error: {error_details}"
    

    async def _handle_streaming_response(self, prompt: str, config: types.GenerateContentConfig, interactive: bool, original_messages: List[Dict[str, str]], input_handler=None):
        """Handle streaming response from Gemini with iterative tool calling."""
        async def async_stream_generator():
            current_prompt = prompt
            
            try:
                while True:
                    
                    # Stream a single response
                    accumulated_content = ""
                    function_calls = []
                    
                    logger.debug(f"Starting Gemini streaming with prompt length: {len(current_prompt)}")
                    
                    # Make streaming API call
                    stream_response = None
                    try:
                        logger.debug(f"Making Gemini streaming request with prompt length: {len(current_prompt)}")
                        logger.debug(f"Model: {self.gemini_config.model}")
                        
                        # Log the last 1000 chars of the prompt to see what's being sent
                        if len(current_prompt) > 1000:
                            logger.debug(f"Prompt excerpt (last 1000 chars): ...{current_prompt[-1000:]}")
                        else:
                            logger.debug(f"Full prompt: {current_prompt}")
                        
                        # Log the config being used
                        logger.debug(f"Request config: {config}")
                        
                        stream_response = await self._make_gemini_request_with_retry(
                            lambda: self.gemini_client.models.generate_content_stream(
                                model=self.gemini_config.model,
                                contents=current_prompt,
                                config=config
                            )
                        )
                        
                        if stream_response is None:
                            logger.error("Gemini stream response is None")
                            yield "Error: No response from Gemini (stream is None)"
                            return
                        
                        logger.debug("Successfully created Gemini stream")
                        
                    except Exception as e:
                        logger.error(f"Failed to create Gemini stream: {e}")
                        logger.error(f"Exception type: {type(e)}")
                        logger.error(f"Exception args: {e.args}")
                        
                        # Check for specific error types
                        if "timeout" in str(e).lower():
                            yield f"⏱️ Request timed out. Gemini may be overloaded. Error: {str(e)}"
                        elif "429" in str(e) or "rate limit" in str(e).lower():
                            yield f"🚫 Rate limited by Gemini API. Please wait and try again. Error: {str(e)}"
                        elif "500" in str(e) or "Internal Server Error" in str(e):
                            yield f"🔥 Gemini API Internal Server Error (500). This usually means the prompt is too long or contains problematic content.\n"
                            yield f"💡 Try using '/compact' to reduce conversation length, or start a new conversation.\n"
                            yield f"Error details: {str(e)}"
                        else:
                            yield f"❌ Error creating stream: {str(e)}"
                        return
                    
                    # Process streaming chunks
                    chunk_count = 0
                    has_any_content = False
                    stream_started = False
                    try:
                        for chunk in stream_response:
                            try:
                                chunk_count += 1
                                stream_started = True
                                logger.debug(f"Processing chunk {chunk_count}")
                                
                                if chunk is None:
                                    logger.warning(f"Chunk {chunk_count} is None, skipping")
                                    continue
                                
                                if hasattr(chunk, 'text') and chunk.text:
                                    accumulated_content += chunk.text
                                    has_any_content = True
                                    
                                    # Yield content normally first
                                    yield chunk.text
                            
                                # Check for function calls in chunk
                                if hasattr(chunk, 'candidates') and chunk.candidates:
                                    try:
                                        if chunk.candidates[0] and hasattr(chunk.candidates[0], 'content') and chunk.candidates[0].content:
                                            if hasattr(chunk.candidates[0].content, 'parts') and chunk.candidates[0].content.parts:
                                                for part in chunk.candidates[0].content.parts:
                                                    if hasattr(part, 'function_call') and part.function_call:
                                                        function_calls.append(part.function_call)
                                    except (IndexError, AttributeError) as e:
                                        logger.warning(f"Error processing chunk {chunk_count} candidates: {e}")
                                        continue
                                
                            except Exception as e:
                                logger.error(f"Error processing chunk {chunk_count}: {e}")
                                # Don't yield error to user, just log and continue
                                continue
                    except Exception as stream_error:
                        logger.error(f"Error iterating stream after {chunk_count} chunks: {stream_error}")
                        if not stream_started:
                            logger.error("Stream never started - this suggests Gemini API is not responding")
                            yield f"\n⚠️ Gemini API is not responding. Stream never started. Error: {stream_error}\n"
                            return
                        elif not has_any_content:
                            logger.error("Stream started but produced no content")
                            yield f"\n⚠️ Gemini stream started but failed after {chunk_count} chunks. Error: {stream_error}\n"
                            return
                    
                    # Check if we got any content at all from the stream
                    if not has_any_content and not accumulated_content:
                        if not stream_started:
                            logger.warning("Gemini stream never started - API may be unresponsive")
                            yield "\n⚠️ Gemini API is unresponsive. Stream never started.\n"
                        else:
                            logger.warning(f"No content received from Gemini stream after {chunk_count} chunks - stream ended unexpectedly")
                            yield f"\n⚠️ No response from Gemini after {chunk_count} chunks. Ending conversation.\n"
                        return
                    
                    # Parse additional function calls from text using shared method
                    if accumulated_content:
                        python_calls = self._parse_python_style_function_calls(accumulated_content)
                        if python_calls:
                            function_calls.extend(python_calls)
                        
                        xml_calls = self._parse_xml_style_tool_calls(accumulated_content)
                        if xml_calls:
                            function_calls.extend(xml_calls)
                    
                    # Check if we have function calls to execute
                    if function_calls:
                        # Show tool execution indicator to user via print (not yielded to avoid LLM contamination)
                        if accumulated_content.strip():
                            print(f"\r\x1b[K\n\r\n\r{self.display_tool_execution_start(len(function_calls), self.is_subagent, interactive=not self.is_subagent)}", flush=True)
                        else:
                            print(f"\r\x1b[K{self.display_tool_execution_start(len(function_calls), self.is_subagent, interactive=not self.is_subagent)}", flush=True)
                        
                        # Execute function calls using shared method
                        function_results, tool_output = await self._execute_function_calls(function_calls, True, input_handler, streaming_mode=True)  # Always interactive for streaming

                        # Check if we just spawned subagents and should interrupt immediately
                        if self.subagent_manager and self.subagent_manager.get_active_count() > 0:
                            # Check if any of the function calls were "task" tools (subagent spawning)
                            task_tools_executed = any("task" in fc.name for fc in function_calls)
                            if task_tools_executed:
                                # Interrupt immediately after spawning subagents
                                yield f"\n🔄 Subagents spawned - interrupting main stream to wait for completion...\n"
                                
                                # Wait for all subagents to complete and collect results
                                subagent_results = await self._collect_subagent_results()
                                
                                if subagent_results:
                                    # Add subagent results to the conversation and restart
                                    yield f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...\n"
                                    
                                    # Create new message with subagent results
                                    results_summary = "\n".join([
                                        f"**Subagent Task: {result['description']}**\n{result['content']}"
                                        for result in subagent_results
                                    ])
                                    
                                    # Create a new conversation context that includes the original request and subagent results
                                    # but frames it as analysis rather than a new spawning request
                                    continuation_message = {
                                        "role": "user", 
                                        "content": f"""I requested: {original_messages[-1]['content']}

You spawned subagents and they have completed their tasks. Here are the results:

{results_summary}

Please provide your final analysis based on these subagent results. Do not spawn any new subagents - just analyze the provided data."""
                                    }
                                    
                                    # Replace conversation with just the continuation context
                                    new_messages = [continuation_message]
                                    
                                    # Restart the conversation with subagent results
                                    yield f"\n🔄 Restarting conversation with subagent results...\n"
                                    new_response = await self.chat_completion(new_messages, stream=True, interactive=interactive)
                                    
                                    # Yield the new response (check if it's a generator or string)
                                    if hasattr(new_response, '__aiter__'):
                                        async for new_chunk in new_response:
                                            yield new_chunk
                                    else:
                                        # If it's a string, yield it directly
                                        yield str(new_response)
                                    
                                    # Exit since we've restarted
                                    return
                                else:
                                    yield f"\n⚠️ No results collected from subagents.\n"
                                    return

                        # Don't yield tool execution details to avoid LLM hallucinations
                        # The tool results will be included in the next iteration's prompt context
                        
                        # Indicate we're getting the follow-up response (via print to avoid LLM contamination)
                        print(f"\n\r\n\r{self.display_tool_processing(self.is_subagent, interactive=not self.is_subagent)}\n\r", flush=True)
                        
                        # Create follow-up prompt for next iteration
                        tool_results_text = "\n".join(function_results)
                        
                        # Check if prompt is getting too long and limit growth
                        if len(current_prompt) > 50000:  # 50k characters - prevent exponential growth
                            logger.warning(f"Prompt is getting very long ({len(current_prompt)} chars), truncating context")
                            # Keep only the original user request and recent tool results
                            current_prompt = f"Original request: {original_messages[-1]['content']}\n\nTool execution complete. Results:\n{tool_results_text}\n\nPlease continue with your response based on these results."
                        else:
                            # Keep original context but add tool results for continuation
                            current_prompt = f"{current_prompt}\n\nTool execution complete. Results:\n{tool_results_text}\n\nPlease continue with your response based on these results."
                        
                        logger.debug(f"Updated prompt length after tool execution: {len(current_prompt)}")
                        logger.debug(f"Tool results length: {len(tool_results_text)}")
                        
                        # Continue the loop - let Gemini decide if more tools are needed
                        continue
                    else:
                        # No function calls - this is the final response
                        # Yield any accumulated content before exiting
                        if accumulated_content.strip():
                            yield accumulated_content
                        
                        # Check for subagent interrupts before ending
                        if self.subagent_manager and self.subagent_manager.get_active_count() > 0:
                            # INTERRUPT STREAMING - collect subagent results and restart
                            yield f"\n🔄 Subagents active - interrupting main stream to collect results...\n"
                            
                            # Wait for all subagents to complete and collect results
                            subagent_results = await self._collect_subagent_results()
                            
                            if subagent_results:
                                # Add subagent results to the conversation and restart
                                yield f"\n📋 Collected {len(subagent_results)} subagent result(s). Restarting with results...\n"
                                
                                # Create new message with subagent results
                                results_summary = "\n".join([
                                    f"**Subagent Task: {result['description']}**\n{result['content']}"
                                    for result in subagent_results
                                ])
                                
                                # Create a new conversation context that includes the original request and subagent results
                                # but frames it as analysis rather than a new spawning request
                                continuation_message = {
                                    "role": "user", 
                                    "content": f"""I requested: {original_messages[-1]['content']}

You spawned subagents and they have completed their tasks. Here are the results:

{results_summary}

Please provide your final analysis based on these subagent results. Do not spawn any new subagents - just analyze the provided data."""
                                }
                                
                                # Replace conversation with just the continuation context
                                new_messages = [continuation_message]
                                
                                # Restart the conversation with subagent results
                                yield f"\n🔄 Restarting conversation with subagent results...\n"
                                new_response = await self.chat_completion(new_messages, stream=True, interactive=interactive)
                                
                                # Yield the new response (check if it's a generator or string)
                                if hasattr(new_response, '__aiter__'):
                                    async for new_chunk in new_response:
                                        yield new_chunk
                                else:
                                    # If it's a string, yield it directly
                                    yield str(new_response)
                                
                                # Exit since we've restarted
                                return
                        
                        return
                        
            except GeneratorExit:
                logger.debug("Stream generator closed by client")
                return
            except Exception as e:
                error_msg = f"Error in streaming: {str(e)}"
                logger.error(error_msg)
                yield error_msg
        
        return async_stream_generator()
    
    async def shutdown(self):
        """Shutdown all MCP connections and HTTP client."""
        # Close HTTP client first
        if hasattr(self, 'http_client') and self.http_client:
            try:
                await self.http_client.aclose()
                logger.info("Closed Gemini HTTP client")
            except Exception as e:
                logger.error(f"Error closing Gemini HTTP client: {e}")
        
        # Call parent shutdown for MCP connections
        await super().shutdown()
    




