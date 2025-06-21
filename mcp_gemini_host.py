#!/usr/bin/env python3
"""This is the MCP host implementation that integrates with Google Gemini's API."""

import asyncio
import json
import logging
import sys
import select
import termios
import tty
from typing import Any, Dict, List, Optional, Union

import click
from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import HostConfig, load_config, create_sample_env, GeminiConfig
from agent import BaseMCPAgent, InterruptibleInput


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for comprehensive logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPGeminiHost(BaseMCPAgent):
    """MCP Host that uses Google Gemini as the language model backend."""
    
    def __init__(self, config: HostConfig):
        super().__init__(config)
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
        
        logger.info(f"Initialized MCP Gemini Host with model: {self.gemini_config.model}")
    
    def get_token_limit(self) -> int:
        """Get the context token limit for Gemini models."""
        if "pro" in self.gemini_config.model.lower():
            return 128000  # Gemini Pro has higher context limit
        else:
            return 64000   # Gemini Flash
    
    async def generate_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> Union[str, Any]:
        """Generate a response using Gemini API."""
        return await self.chat_completion(messages, stream=True, interactive=True)
    
    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to Gemini format."""
        return self._convert_tools_to_gemini_format()
    
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from Gemini response."""
        if hasattr(response, 'candidates'):
            function_calls, _ = self._parse_response_content(response)
            return [{'name': fc.name, 'args': fc.args} for fc in function_calls]
        return []
    
    def _convert_tools_to_gemini_format(self) -> List[types.Tool]:
        """Convert MCP tools to Gemini function calling format."""
        function_declarations = []
        
        for tool_key, tool_info in self.available_tools.items():
            function_declaration = {
                "name": tool_key.replace(":", "_"),  # Replace colon with underscore for Gemini
                "description": tool_info["description"] or f"Execute {tool_info['name']} tool",
                "parameters": tool_info["schema"] or {"type": "object", "properties": {}}
            }
            function_declarations.append(function_declaration)
        
        return [types.Tool(function_declarations=function_declarations)] if function_declarations else []
    
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
        tools = self._convert_tools_to_gemini_format()
        
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
    
    def _parse_response_content(self, response) -> tuple:
        """Parse Gemini response content into function calls and text."""
        function_calls = []
        text_response = ""
        
        # Debug response structure
        logger.debug(f"Response type: {type(response)}")
        logger.debug(f"Has candidates: {hasattr(response, 'candidates') and bool(response.candidates)}")
        if hasattr(response, 'candidates') and response.candidates:
            logger.debug(f"Candidate 0 has content: {hasattr(response.candidates[0], 'content') and bool(response.candidates[0].content)}")
            if hasattr(response.candidates[0], 'content') and response.candidates[0].content:
                logger.debug(f"Content has parts: {hasattr(response.candidates[0].content, 'parts') and bool(response.candidates[0].content.parts)}")
        
        # First try to get content from parts structure
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
                elif hasattr(part, 'text') and part.text:
                    text_response += part.text
        
        # If no content from parts, try alternative locations
        if not text_response and not function_calls:
            # Try direct text property on response
            if hasattr(response, 'text') and response.text:
                text_response = response.text
                logger.debug("Found text on response.text")
            # Try text on candidate
            elif response.candidates and hasattr(response.candidates[0], 'text') and response.candidates[0].text:
                text_response = response.candidates[0].text
                logger.debug("Found text on candidate.text")
            else:
                # Log the full response structure for debugging
                logger.warning("No text content found in response")
                logger.warning(f"Response structure: {repr(response)}")
                if response.candidates:
                    logger.warning(f"Candidate 0 structure: {repr(response.candidates[0])}")
                    if response.candidates[0].content:
                        logger.warning(f"Content structure: {repr(response.candidates[0].content)}")
                
                # Return empty response rather than failing
                return [], ""
                    
        # Also check for Python-style function calls in text that Gemini sometimes generates
        if text_response:
            python_calls = self._parse_python_style_function_calls(text_response)
            if python_calls:
                function_calls.extend(python_calls)
                logger.debug(f"Found {len(python_calls)} Python-style function calls in addition to {len(function_calls) - len(python_calls)} structured calls")
        
        # Also check for XML-style tool calls like <execute_tool>tool_name{args}</execute_tool>
        if text_response:
            xml_calls = self._parse_xml_style_tool_calls(text_response)
            if xml_calls:
                function_calls.extend(xml_calls)
                logger.debug(f"Found {len(xml_calls)} XML-style tool calls in addition to {len(function_calls) - len(xml_calls)} other calls")
        
        return function_calls, text_response

    async def _execute_function_calls(self, function_calls: List, interactive: bool, input_handler=None, streaming_mode=False) -> tuple:
        """Execute a list of function calls and return results and output."""
        function_results = []
        all_tool_output = []  # Collect all tool execution output for non-interactive mode
        
        for i, function_call in enumerate(function_calls, 1):
            # Check for interruption before each tool execution
            if input_handler and input_handler.interrupted:
                all_tool_output.append("🛑 Tool execution interrupted by user")
                break
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
            
            tool_execution_msg = f"  {i}. Executing {tool_name} with args: {arguments}"
            if interactive and not streaming_mode:
                print(f"\r\x1b[K{tool_execution_msg}", flush=True)
            else:
                all_tool_output.append(tool_execution_msg)
            
            # Execute the tool with interruption support
            tool_success = True
            tool_result = ""
            
            try:
                if interactive and input_handler:
                    # Create interruptible tool execution
                    tool_task = asyncio.create_task(self._execute_mcp_tool(tool_name, arguments))
                    
                    # Monitor for interruption while tool executes
                    while not tool_task.done():
                        if input_handler.interrupted:
                            tool_task.cancel()
                            all_tool_output.append(f"🛑 Tool {tool_name} execution cancelled by user")
                            function_results.append(f"Tool {tool_name} CANCELLED: User interrupted execution")
                            return function_results, all_tool_output
                        
                        # Check for escape key input
                        try:
                            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.1)[0]:
                                old_settings = termios.tcgetattr(sys.stdin.fileno())
                                tty.setraw(sys.stdin.fileno())
                                char = sys.stdin.read(1)
                                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
                                if char == '\x1b':  # Escape key
                                    input_handler.interrupted = True
                                    tool_task.cancel()
                                    all_tool_output.append(f"🛑 Tool {tool_name} execution cancelled by user")
                                    function_results.append(f"Tool {tool_name} CANCELLED: User interrupted execution")
                                    return function_results, all_tool_output
                        except:
                            pass  # Ignore errors in interrupt checking
                        
                        await asyncio.sleep(0.1)
                    
                    try:
                        tool_result = tool_task.result()
                    except asyncio.CancelledError:
                        all_tool_output.append(f"🛑 Tool {tool_name} execution cancelled")
                        function_results.append(f"Tool {tool_name} CANCELLED: Task was cancelled")
                        return function_results, all_tool_output
                else:
                    # Non-interactive or no input handler - execute normally
                    tool_result = await self._execute_mcp_tool(tool_name, arguments)
                
                # Check if tool result indicates an error
                if tool_result.startswith("Error:") or "error" in tool_result.lower()[:100]:
                    tool_success = False
                    
            except Exception as e:
                tool_success = False
                tool_result = f"Exception during execution: {str(e)}"
            
            # Format result with success/failure status
            status = "SUCCESS" if tool_success else "FAILED"
            result_content = f"Tool {tool_name} {status}: {tool_result}"
            if not tool_success:
                result_content += "\n⚠️  Command failed - take this into account for your next action."
            function_results.append(result_content)
            
            tool_result_msg = f"     Result: {tool_result[:200]}..." if len(tool_result) > 200 else f"     Result: {tool_result}"
            if not tool_success:
                tool_result_msg += " ⚠️  Command failed - take this into account for your next action."
            if interactive and not streaming_mode:
                print(f"\r\x1b[K{tool_result_msg}", flush=True)
            else:
                all_tool_output.append(tool_result_msg)
        
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
                function_calls, text_response = self._parse_response_content(response)
                
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
                        print(f"\r\x1b[K\n🔧 Using {len(function_calls)} tool(s)...", flush=True)
                    
                    # Execute function calls
                    function_results, tool_output = await self._execute_function_calls(function_calls, interactive)
                    
                    # Accumulate output for non-interactive mode
                    if not interactive:
                        all_accumulated_output.append(f"🔧 Using {len(function_calls)} tool(s)...")
                        all_accumulated_output.extend(tool_output)
                        all_accumulated_output.append("")  # Empty line for spacing
                    
                    # Create follow-up prompt with tool results and clear instruction
                    tool_results_text = "\n".join(function_results)
                    current_prompt = f"{current_prompt}\n\nTool Results:\n{tool_results_text}\n\nThe tool execution is complete. Please continue with either another tool use if needed or your response based on these results." 
                    
                    # Continue the loop - let Gemini decide if more tools are needed
                    continue
                else:
                    # No function calls - this is the final response
                    if not interactive and all_accumulated_output:
                        # Include all accumulated tool output plus final response
                        final_output = "\n".join(all_accumulated_output) + (text_response if text_response else "")
                        return final_output
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
                    for chunk in stream_response:
                        try:
                            chunk_count += 1
                            logger.debug(f"Processing chunk {chunk_count}")
                            
                            if chunk is None:
                                logger.warning(f"Chunk {chunk_count} is None, skipping")
                                continue
                            
                            if hasattr(chunk, 'text') and chunk.text:
                                accumulated_content += chunk.text
                                has_any_content = True
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
                    
                    # Check if we got any content at all from the stream
                    if not has_any_content and not accumulated_content:
                        logger.warning("No content received from Gemini stream - this may indicate the stream ended unexpectedly")
                        yield "\n⚠️ No response from Gemini. Ending conversation.\n"
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
                        # Show tool execution indicator
                        if accumulated_content.strip():
                            yield "\r\x1b[K\n\n🔧 Executing function calls...\n"
                        else:
                            yield "\r\x1b[K🔧 Executing function calls...\n"
                        
                        # Execute function calls using shared method
                        function_results, tool_output = await self._execute_function_calls(function_calls, True, input_handler, streaming_mode=True)  # Always interactive for streaming

                        # Yield each line of the tool output so it's displayed in streaming mode
                        for line in tool_output:
                            yield f"\r\x1b[K{line}\n"
                        
                        # Indicate we're getting the follow-up response
                        yield "\n\n💭 Processing results and generating response...\n"
                        
                        # Create follow-up prompt for next iteration
                        tool_results_text = "\n".join(function_results)
                        current_prompt = f"{current_prompt}\n\nTool Results:\n{tool_results_text}\n\nThe tool execution is complete. Please continue with your response based on these results."
                        
                        logger.debug(f"Updated prompt length after tool execution: {len(current_prompt)}")
                        logger.debug(f"Tool results length: {len(tool_results_text)}")
                        
                        # Check if prompt is getting too long
                        if len(current_prompt) > 100000:  # 100k characters
                            logger.warning(f"Prompt is very long ({len(current_prompt)} chars), this might cause issues")
                            yield f"\n⚠️ Conversation is getting very long. Consider using '/compact' to reduce length.\n"
                        
                        # Continue the loop - let Gemini decide if more tools are needed
                        continue
                    else:
                        # No function calls - this is the final response
                        # Yield any accumulated content before exiting
                        if accumulated_content.strip():
                            yield accumulated_content
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
    


async def interactive_chat_gemini(host: MCPGeminiHost):
    """Run an interactive chat session with Gemini and streaming tool execution."""
    import signal
    import atexit
    
    # Set up cleanup handlers
    def cleanup_handler():
        """Cleanup handler for unexpected exits."""
        try:
            if hasattr(host, 'http_client') and host.http_client:
                # Note: We can't use async here, so just log
                logger.warning("Process terminating - HTTP client may not be properly closed")
        except:
            pass
    
    atexit.register(cleanup_handler)
    
    print(f"MCP Gemini Host - Interactive Chat")
    print(f"Model: {host.gemini_config.model}")
    print(f"Available tools: {len(host.available_tools)}")
    print("Commands: 'quit' to exit, 'tools' to list tools, 'ESC' to interrupt")
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
            
            # Process the user input (no longer need buffer logic)
            if user_input.strip():  # Only process non-empty input
                # Ensure no other request is running
                if current_task and not current_task.done():
                    print("⚠️  Another request is already in progress. Please wait or press ESC to cancel it.")
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                # Get response from Gemini (interactive mode with tools)
                try:
                    # Make API call interruptible by running in a task
                    print("\n💭 Thinking... (press ESC to interrupt)")
                    current_task = asyncio.create_task(
                        host.chat_completion(messages, stream=None, interactive=True, input_handler=input_handler)
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
                    
                    full_response = ""  # Initialize outside the streaming block
                    
                    if hasattr(response, '__aiter__'):
                        # Streaming response
                        print("\nAssistant (press ESC to interrupt):")
                        sys.stdout.flush()
                        
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
                                    
                                try:
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
                                except Exception as chunk_error:
                                    logger.error(f"Error processing streaming chunk: {chunk_error}")
                                    error_msg = f"\n[Error processing response chunk: {chunk_error}]"
                                    print(error_msg, end="", flush=True)
                                    full_response += error_msg
                        except Exception as stream_error:
                            logger.error(f"Error in streaming response: {stream_error}")
                            error_msg = f"\n[Streaming error: {stream_error}]"
                            print(error_msg, end="", flush=True)
                            full_response += error_msg
                        finally:
                            # Always restore terminal settings first
                            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
                            
                            # Clean up display if interrupted
                            if interrupted:
                                print("\n🛑 Streaming interrupted by user")
                                sys.stdout.flush()
                            else:
                                print()  # Normal newline after streaming
                        
                        # Add assistant response to messages
                        if full_response:  # Only add if not interrupted
                            messages.append({"role": "assistant", "content": full_response})
                    else:
                        # Non-streaming response
                        if response:
                            formatted_response = host.format_markdown(str(response))
                            print(f"\nAssistant: {formatted_response}")
                            messages.append({"role": "assistant", "content": response})
                        
                except Exception as response_error:
                    error_msg = f"Error getting response from Gemini: {response_error}"
                    logger.error(error_msg)
                    print(f"\n{error_msg}")
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
            import traceback
            logger.error(f"Unexpected error in interactive chat: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"\n💥 Unexpected error: {e}")
            print("The chat session will continue, but please report this error.")
            
            # Reset any interrupted state
            if input_handler:
                input_handler.interrupted = False
            if current_task and not current_task.done():
                current_task.cancel()
            current_task = None


# CLI commands would be added to the main mcp_deepseek_host.py file
if __name__ == "__main__":
    async def test_gemini():
        """Test Gemini integration."""
        config = load_config()
        
        if not config.gemini_api_key:
            print("Error: GEMINI_API_KEY not set. Please update .env file.")
            return
        
        host = MCPGeminiHost(config)
        
        # Test a simple conversation
        messages = [{"role": "user", "content": "Hello! Can you tell me what 2+2 is?"}]
        response = await host.chat_completion(messages, stream=False)
        print(f"Gemini response: {response}")
        
        await host.shutdown()
    
    asyncio.run(test_gemini())
