#!/usr/bin/env python3
"""Base MCP Agent implementation with shared functionality."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import signal
import termios
import tty
import select
import time
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

from fastmcp.client import Client as FastMCPClient, StdioTransport

from config import HostConfig
from cli_agent.core.slash_commands import SlashCommandManager
from cli_agent.tools.builtin_tools import get_all_builtin_tools

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Suppress WARNING messages during interactive chat
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseMCPAgent(ABC):
    """Base class for MCP agents with shared functionality."""
    
    def __init__(self, config: HostConfig, is_subagent: bool = False):
        self.config = config
        self.is_subagent = is_subagent
        self.mcp_clients: Dict[str, FastMCPClient] = {}
        self.available_tools: Dict[str, Dict] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Task management for subprocess-based subagents
        self.running_tasks: Dict[str, Dict] = {}  # task_id -> task_info
        self.task_counter = 0
        self.current_batch_id = None  # Track batches of parallel tasks
        self.batch_counter = 0
        
        # Communication socket for subagent forwarding (set by parent process)
        self.comm_socket = None
        
        # Centralized subagent management system
        if not is_subagent:
            try:
                import sys
                import os
                # Add current directory to path for subagent import
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                from subagent import SubagentManager
                self.subagent_manager = SubagentManager(config)
                
                # Event-driven message handling
                self.subagent_message_queue = asyncio.Queue()
                self.subagent_manager.add_message_callback(self._on_subagent_message)
                logger.info("Initialized centralized subagent management system")
            except ImportError as e:
                logger.warning(f"Failed to import subagent manager: {e}")
                self.subagent_manager = None
                self.subagent_message_queue = None
        else:
            self.subagent_manager = None
            self.subagent_message_queue = None
        
        # Add built-in tools
        self._add_builtin_tools()
        
        # Initialize slash command manager
        self.slash_commands = SlashCommandManager(self)
        
        logger.info(f"Initialized Base MCP Agent with {len(self.available_tools)} built-in tools")
    
    def _add_builtin_tools(self):
        """Add built-in tools to the available tools."""
        builtin_tools = get_all_builtin_tools()
        self.available_tools.update(builtin_tools)
    
    async def _execute_builtin_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a built-in tool."""
        if tool_name == "bash_execute":
            return self._bash_execute(args)
        elif tool_name == "read_file":
            return self._read_file(args)
        elif tool_name == "write_file":
            return self._write_file(args)
        elif tool_name == "list_directory":
            return self._list_directory(args)
        elif tool_name == "get_current_directory":
            return self._get_current_directory(args)
        elif tool_name == "todo_read":
            return self._todo_read(args)
        elif tool_name == "todo_write":
            return self._todo_write(args)
        elif tool_name == "replace_in_file":
            return self._replace_in_file(args)
        elif tool_name == "webfetch":
            return self._webfetch(args)
        elif tool_name == "task":
            return await self._task(args)
        elif tool_name == "task_status":
            return self._task_status(args)
        elif tool_name == "task_results":
            return self._task_results(args)
        else:
            return f"Unknown built-in tool: {tool_name}"
    
    def _bash_execute(self, args: Dict[str, Any]) -> str:
        """Execute a bash command and return the output."""
        command = args.get("command", "")
        timeout = args.get("timeout", 120)
        
        if not command:
            return "Error: No command provided"
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}"
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nReturn code: {result.returncode}"
            
            return output if output else "Command executed successfully (no output)"
            
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _read_file(self, args: Dict[str, Any]) -> str:
        """Read contents of a file with line numbers."""
        file_path = args.get("file_path", "")
        offset = args.get("offset", 1)
        limit = args.get("limit", None)
        
        if not file_path:
            return "Error: No file path provided"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_idx = max(0, offset - 1)  # Convert to 0-based index
            end_idx = len(lines) if limit is None else min(len(lines), start_idx + limit)
            
            result = []
            for i in range(start_idx, end_idx):
                result.append(f"{i + 1:6d}→{lines[i].rstrip()}")
            
            return "\n".join(result)
            
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _write_file(self, args: Dict[str, Any]) -> str:
        """Write content to a file."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")
        
        if not file_path:
            return "Error: No file path provided"
        
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create directory if it's not empty (i.e., file is not in current dir)
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {file_path}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _list_directory(self, args: Dict[str, Any]) -> str:
        """List contents of a directory."""
        directory_path = args.get("directory_path", ".")
        
        try:
            path = Path(directory_path)
            if not path.exists():
                return f"Error: Directory does not exist: {directory_path}"
            
            if not path.is_dir():
                return f"Error: Path is not a directory: {directory_path}"
            
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
            
            return "\n".join(items) if items else "Directory is empty"
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def _get_current_directory(self, args: Dict[str, Any]) -> str:
        """Get the current working directory."""
        try:
            return os.getcwd()
        except Exception as e:
            return f"Error getting current directory: {str(e)}"
    
    def _todo_read(self, args: Dict[str, Any]) -> str:
        """Read the current todo list."""
        todo_file = "todo.json"
        
        try:
            if not os.path.exists(todo_file):
                return "[]"  # Empty todo list
            
            with open(todo_file, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            return f"Error reading todo list: {str(e)}"
    
    def _todo_write(self, args: Dict[str, Any]) -> str:
        """Write/update the todo list."""
        todos = args.get("todos", [])
        todo_file = "todo.json"
        
        try:
            with open(todo_file, 'w', encoding='utf-8') as f:
                json.dump(todos, f, indent=2)
            
            return f"Successfully updated todo list with {len(todos)} items"
            
        except Exception as e:
            return f"Error writing todo list: {str(e)}"
    
    def _replace_in_file(self, args: Dict[str, Any]) -> str:
        """Replace text in a file."""
        file_path = args.get("file_path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        
        if not file_path:
            return "Error: No file path provided"
        if not old_text:
            return "Error: No old text provided"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_text not in content:
                return f"Error: Text not found in file: {old_text}"
            
            new_content = content.replace(old_text, new_text)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            count = content.count(old_text)
            return f"Successfully replaced {count} occurrence(s) of text in {file_path}"
            
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error replacing text in file: {str(e)}"
    
    def _webfetch(self, args: Dict[str, Any]) -> str:
        """Fetch a webpage using curl and return its content."""
        url = args.get("url", "")
        limit = args.get("limit", 1000)  # Default to 1000 lines

        if not url:
            return "Error: No URL provided"

        # Use curl to fetch the webpage with a timeout, capturing raw output
        result = subprocess.run(
            ["curl", "-L", "--max-time", "30", url],
            capture_output=True,
            timeout=35  # Slightly longer than curl timeout
        )

        # Try to decode with utf-8, then fall back to latin-1 (which rarely fails)
        try:
            content = result.stdout.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {url}. Falling back to latin-1.")
            content = result.stdout.decode('latin-1', errors='replace')

        if result.returncode != 0:
            # Try to decode stderr for a better error message
            try:
                stderr = result.stderr.decode('utf-8', errors='replace')
            except:
                stderr = repr(result.stderr)
            
            error_msg = f"Error fetching URL (curl return code {result.returncode}): {stderr}"
            
            # If we have content despite the error, include it
            if content.strip():
                return f"{error_msg}\n\nContent retrieved:\n{content}"
            else:
                return error_msg

        # Truncate the content by lines if limit is specified
        if limit is not None and isinstance(limit, int) and limit > 0:
            lines = content.split('\n')
            if len(lines) > limit:
                content = '\n'.join(lines[:limit])
                content += f"\n\n[Content truncated at {limit} lines. Original had {len(lines)} lines.]"

        # Return the content (truncated if limit was provided)
        return content
    
    async def _task(self, args: Dict[str, Any]) -> str:
        """Spawn a subagent subprocess to investigate a task and return immediately."""
        description = args.get("description", "")
        prompt = args.get("prompt", "")
        context = args.get("context", "")
        
        if not description:
            return "Error: No task description provided"
        if not prompt:
            return "Error: No task prompt provided"
        
        try:
            # Generate unique task ID and batch tracking
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
            
            # Create a new batch if this is the first task in a while
            # (assumes tasks called within 5 seconds are part of the same batch)
            current_time = time.time()
            if (self.current_batch_id is None or 
                not self.running_tasks or 
                current_time - max(task['start_time'] for task in self.running_tasks.values()) > 5):
                self.batch_counter += 1
                self.current_batch_id = f"batch_{self.batch_counter}"
            
            # Create task prompt
            task_prompt = f"""You are a specialized subagent tasked with investigating and completing the following task:

TASK: {description}

INSTRUCTIONS:
{prompt}

IMPORTANT: After completing your investigation/task, provide a clear and concise summary of your findings. Your summary should include:
1. What you investigated or accomplished
2. Key findings or results
3. Any important observations or insights
4. Conclusions or recommendations if applicable

Please structure your response so that the main findings are easily extractable for analysis."""
            
            if context:
                task_prompt += f"""

ADDITIONAL CONTEXT:
{context}"""
            
            task_prompt += """

Your goal is to thoroughly investigate this task using all available tools and provide a comprehensive summary of your findings. Be systematic, thorough, and provide actionable insights.

IMPORTANT: 
- Use tools extensively to gather information
- Provide a clear, well-structured summary
- Include specific details and findings
- If you encounter any issues, document them clearly
- Your response will be returned to the parent agent, so make it complete and self-contained"""
            
            # Write task prompt to a temporary file
            import tempfile
            import json
            import sys
            import os
            
            # Create a communication socket for tool execution forwarding
            import socket
            comm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            comm_socket.bind(('localhost', 0))  # Bind to any available port
            comm_port = comm_socket.getsockname()[1]
            comm_socket.listen(1)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as task_file:
                task_data = {
                    "task_id": task_id,
                    "description": description,
                    "prompt": task_prompt,
                    "timestamp": time.time(),
                    "comm_port": comm_port  # Add communication port
                }
                json.dump(task_data, task_file)
                task_file_path = task_file.name
            
            # Start subprocess for subagent
            python_executable = sys.executable
            # Get the path to the main agent.py script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, '..', '..', 'agent.py')
            script_path = os.path.abspath(script_path)
            
            # Start subprocess with task execution
            process = await asyncio.create_subprocess_exec(
                python_executable, script_path, "execute-task", task_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Store task info
            self.running_tasks[task_id] = {
                "description": description,
                "process": process,
                "task_file": task_file_path,
                "result_file": task_file_path.replace('.json', '_result.json'),
                "start_time": current_time,
                "output_buffer": "",
                "completed": False,
                "result": None,
                "batch_id": self.current_batch_id,
                "comm_socket": comm_socket,
                "comm_port": comm_port
            }
            
            # Start monitoring task in background
            asyncio.create_task(self._monitor_task(task_id))
            
            # Start communication handler for tool execution forwarding
            asyncio.create_task(self._handle_subagent_communication(task_id))
            
            # Return immediately to allow main LLM to continue
            print(f"\n\r🤖 [SUBAGENT] Starting task {task_id}: {description}\r")
            print(f"🎯 [SUBAGENT] Task running in subprocess...\r")
            
            return f"""[SUBAGENT TASK STARTED]
Task ID: {task_id}
Description: {description}
Status: Running in subprocess

The subagent is now running independently and will report progress. You can continue with other tasks while this completes."""
            
        except Exception as e:
            logger.error(f"Error starting subagent task: {e}")
            return f"Error starting subagent task '{description}': {str(e)}"
    
    async def _monitor_task(self, task_id: str):
        """Monitor a running task subprocess and display its output."""
        if task_id not in self.running_tasks:
            return
        
        task_info = self.running_tasks[task_id]
        process = task_info["process"]
        
        try:
            import os
            
            # Monitor stdout and stderr
            while True:
                # Check if process is still running
                if process.returncode is not None:
                    break
                
                # Read available output with shorter timeout for real-time display
                try:
                    stdout_data = await asyncio.wait_for(process.stdout.read(1024), timeout=0.01)
                    if stdout_data:
                        output = stdout_data.decode('utf-8', errors='ignore')
                        # Store output for buffer
                        task_info["output_buffer"] += output
                        
                        # Store output for streaming integration
                        if "display_messages" not in task_info:
                            task_info["display_messages"] = []
                        task_info["display_messages"].append(f"🤖 [SUBAGENT {task_id}] {output}")
                        
                        # Print immediately for non-streaming contexts (fallback)
                        formatted_output = output.replace('\n', '\n\r')
                        print(f"🤖 [SUBAGENT {task_id}] {formatted_output}", end='', flush=True)
                except asyncio.TimeoutError:
                    pass
                
                # Shorter sleep for more responsive output
                await asyncio.sleep(0.01)
            
            # Process completed, get final output
            try:
                stdout, stderr = await process.communicate()
                if stdout:
                    output = stdout.decode('utf-8', errors='ignore')
                    formatted_output = output.replace('\n', '\n\r')
                    print(f"{formatted_output}", end='', flush=True)
                    task_info["output_buffer"] += output
                
                if stderr and stderr.strip():
                    error_output = stderr.decode('utf-8', errors='ignore')
                    print(f"\n\r🤖 [SUBAGENT ERROR {task_id}]: {error_output}\r")
            except Exception as e:
                print(f"\n\r🤖 [SUBAGENT ERROR {task_id}]: Failed to read final output: {e}\r")
            
            # Mark task as completed
            task_info["completed"] = True
            task_info["end_time"] = time.time()
            
            # Try to collect the result
            try:
                import json
                result_file = task_info["result_file"]
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        task_info["result"] = result_data.get("result", "No result captured")
                    # Clean up result file
                    os.unlink(result_file)
                else:
                    task_info["result"] = "Result file not found"
            except Exception as e:
                task_info["result"] = f"Error collecting result: {e}"
            
            print(f"\n\r🤖 [SUBAGENT] Task {task_id} completed: {task_info['description']}\r")
            
            # Check if all tasks in the current batch are completed
            self._check_all_tasks_completed()
            
            # Clean up task file
            try:
                os.unlink(task_info["task_file"])
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Error monitoring task {task_id}: {e}")
            print(f"\n\r🤖 [SUBAGENT ERROR {task_id}]: Monitoring failed: {e}\r")
    
    def get_pending_subagent_messages(self):
        """Get all pending display messages from running subagents."""
        messages = []
        for task_id, task_info in self.running_tasks.items():
            if "display_messages" in task_info and task_info["display_messages"]:
                messages.extend(task_info["display_messages"])
                task_info["display_messages"].clear()  # Clear after retrieving
        return messages
    
    async def _handle_subagent_communication(self, task_id: str):
        """Handle communication from subagent for tool execution forwarding."""
        if task_id not in self.running_tasks:
            return
        
        import socket
        import json
        
        task_info = self.running_tasks[task_id]
        comm_socket = task_info["comm_socket"]
        
        try:
            # Wait for subagent to connect with timeout
            comm_socket.settimeout(10.0)  # 10 second timeout for connection
            try:
                # Use asyncio to make the blocking accept() non-blocking for the event loop
                loop = asyncio.get_event_loop()
                client_socket, addr = await loop.run_in_executor(None, comm_socket.accept)
                logger.debug(f"Subagent {task_id} connected for communication")
            except socket.timeout:
                logger.warning(f"Subagent {task_id} did not connect within timeout")
                return
            
            # Handle messages from subagent
            with client_socket:
                client_socket.settimeout(1.0)  # 1 second timeout for messages
                buffer = ""
                
                # Store client socket reference for sending responses
                task_info["client_socket"] = client_socket
                
                while task_id in self.running_tasks and not self.running_tasks[task_id].get("completed", False):
                    try:
                        # Use executor to make recv non-blocking for event loop
                        data = await loop.run_in_executor(None, client_socket.recv, 4096)
                        if not data:
                            break
                        data = data.decode('utf-8')
                        
                        buffer += data
                        
                        # Process complete messages (newline-delimited JSON)
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                try:
                                    message = json.loads(line.strip())
                                    await self._process_subagent_message(task_id, message)
                                except json.JSONDecodeError as e:
                                    logger.error(f"Invalid JSON from subagent {task_id}: {e}")
                    
                    except socket.timeout:
                        # Continue waiting for more messages
                        continue
                    except Exception as e:
                        logger.error(f"Error receiving from subagent {task_id}: {e}")
                        break
                        
        except Exception as e:
            logger.error(f"Error in subagent communication for {task_id}: {e}")
        finally:
            try:
                comm_socket.close()
            except:
                pass
    
    async def _process_subagent_message(self, task_id: str, message: dict):
        """Process a message from a subagent."""
        try:
            msg_type = message.get("type")
            
            if msg_type == "tool_execution_request":
                # Handle tool execution request from subagent (don't await - run concurrently)
                asyncio.create_task(self._handle_subagent_tool_request(task_id, message))
                
            elif msg_type == "display_message":
                # Handle display messages from subagent - store for streaming integration
                display_msg = message.get("message", "")
                # Store in task info for streaming integration
                if task_id in self.running_tasks:
                    task_info = self.running_tasks[task_id]
                    if "display_messages" not in task_info:
                        task_info["display_messages"] = []
                    task_info["display_messages"].append(display_msg)
                # Also print immediately for non-streaming contexts
                print(display_msg, flush=True)
                
            elif msg_type == "tool_execution":
                # Forward tool execution to main chat
                tool_name = message.get("tool_name", "unknown")
                tool_args = message.get("tool_args", {})
                tool_result = message.get("tool_result", "")
                
                # Display tool execution in main chat
                print(f"\n\r🔧 [SUBAGENT {task_id}] Executing tool: {tool_name}\r")
                if tool_args:
                    print(f"📝 [SUBAGENT {task_id}] Arguments: {str(tool_args)[:100]}{'...' if len(str(tool_args)) > 100 else ''}\r")
                if tool_result:
                    # Truncate long results for display
                    result_preview = str(tool_result)[:300] + "..." if len(str(tool_result)) > 300 else str(tool_result)
                    print(f"✅ [SUBAGENT {task_id}] Result: {result_preview}\r")
                    
            elif msg_type == "status":
                # Handle status updates
                status = message.get("status", "")
                print(f"\n\r📊 [SUBAGENT {task_id}] {status}\r")
                
            elif msg_type == "error":
                # Handle error messages
                error = message.get("error", "")
                print(f"\n\r❌ [SUBAGENT {task_id}] Error: {error}\r")
                
        except Exception as e:
            logger.error(f"Error processing subagent message from {task_id}: {e}")
    
    async def _handle_subagent_tool_request(self, task_id: str, message: dict):
        """Handle a tool execution request from a subagent."""
        try:
            import json
            import asyncio
            
            request_id = message.get("request_id")
            tool_key = message.get("tool_key")
            tool_name = message.get("tool_name")
            tool_args = message.get("tool_args", {})
            
            if not request_id or not tool_key or not tool_name:
                logger.error(f"Invalid tool request from subagent {task_id}: missing required fields")
                return
            
            # Get the subagent's communication socket
            if task_id not in self.running_tasks:
                logger.error(f"Task {task_id} not found for tool request")
                return
                
            task_info = self.running_tasks[task_id]
            comm_socket = task_info.get("comm_socket")
            if not comm_socket:
                logger.error(f"No communication socket for task {task_id}")
                return
            
            # Display tool execution in main chat
            print(f"\n🔧 [SUBAGENT {task_id}] Executing tool: {tool_name}", flush=True)
            if tool_args:
                args_preview = str(tool_args)[:100] + "..." if len(str(tool_args)) > 100 else str(tool_args)
                print(f"📝 [SUBAGENT {task_id}] Arguments: {args_preview}", flush=True)
            
            # Execute the tool on behalf of the subagent
            try:
                # Temporarily disable subagent mode to execute locally
                original_is_subagent = self.is_subagent
                self.is_subagent = False
                
                result = await self._execute_mcp_tool(tool_key, tool_args)
                
                # Restore subagent mode
                self.is_subagent = original_is_subagent
                
                # Display result in main chat
                result_preview = str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
                print(f"✅ [SUBAGENT {task_id}] Result: {result_preview}", flush=True)
                
                # Send successful response back to subagent (with full result so it can continue)
                response = {
                    "type": "tool_execution_response", 
                    "request_id": request_id,
                    "success": True,
                    "result": result  # Full result for subagent to continue its work
                }
                
            except SystemExit as e:
                # Handle Click's sys.exit() gracefully
                if e.code == 0:
                    # Successful exit, treat as success
                    result = "Command completed successfully"
                    result_preview = str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
                    print(f"✅ [SUBAGENT {task_id}] Result: {result_preview}", flush=True)
                    
                    response = {
                        "type": "tool_execution_response", 
                        "request_id": request_id,
                        "success": True,
                        "result": result
                    }
                else:
                    # Non-zero exit, treat as error
                    error_msg = f"Tool exited with code {e.code}"
                    logger.error(f"Tool execution error for subagent {task_id}: {error_msg}")
                    print(f"❌ [SUBAGENT {task_id}] Tool error: {error_msg}", flush=True)
                    
                    response = {
                        "type": "tool_execution_response",
                        "request_id": request_id,
                        "success": False,
                        "error": error_msg
                    }
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                logger.error(f"Tool execution error for subagent {task_id}: {e}")
                
                # Display error in main chat
                print(f"❌ [SUBAGENT {task_id}] Tool error: {error_msg}", flush=True)
                
                # Send error response back to subagent
                response = {
                    "type": "tool_execution_response",
                    "request_id": request_id,
                    "success": False,
                    "error": error_msg
                }
            
            # Send response back to subagent through client socket
            try:
                client_socket = task_info.get("client_socket")
                if client_socket:
                    response_json = json.dumps(response) + "\n"
                    client_socket.send(response_json.encode('utf-8'))
                else:
                    logger.error(f"No client socket available for subagent {task_id}")
                
            except Exception as e:
                logger.error(f"Error sending response to subagent {task_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error handling tool request from subagent {task_id}: {e}")
    
    def _task_status(self, args: Dict[str, Any]) -> str:
        """Check the status of running subagent tasks."""
        task_id = args.get("task_id", None)
        
        if not self.running_tasks:
            return "No tasks are currently running."
        
        if task_id:
            # Check specific task
            if task_id not in self.running_tasks:
                return f"Task {task_id} not found."
            
            task_info = self.running_tasks[task_id]
            status = "Completed" if task_info["completed"] else "Running"
            runtime = time.time() - task_info["start_time"]
            
            result = f"""Task Status: {task_id}
Description: {task_info["description"]}
Status: {status}
Runtime: {runtime:.2f} seconds
Output Buffer Size: {len(task_info["output_buffer"])} characters"""
            
            if task_info["completed"] and "end_time" in task_info:
                total_time = task_info["end_time"] - task_info["start_time"]
                result += f"\nTotal Time: {total_time:.2f} seconds"
            
            return result
        else:
            # Check all tasks
            result = f"Task Status Summary ({len(self.running_tasks)} tasks):\n"
            for tid, task_info in self.running_tasks.items():
                status = "Completed" if task_info["completed"] else "Running"
                runtime = time.time() - task_info["start_time"]
                result += f"\n{tid}: {task_info['description']} - {status} ({runtime:.1f}s)"
            
            return result
    
    def _task_results(self, args: Dict[str, Any]) -> str:
        """Retrieve the results and summaries from completed subagent tasks."""
        try:
            # Validate arguments - the error suggests args might not be a dict
            if not isinstance(args, dict):
                logger.error(f"_task_results received invalid args type: {type(args)}")
                return f"Error: Invalid arguments type: {type(args)}. Expected dictionary."
            
            include_running = args.get("include_running", False)
            clear_after_retrieval = args.get("clear_after_retrieval", True)
            
            # Check if running_tasks exists and is valid
            if not hasattr(self, 'running_tasks') or not isinstance(self.running_tasks, dict):
                return "No task manager found or tasks corrupted."
                
            if not self.running_tasks:
                return "No tasks found."
            
            # Simple approach - just return basic info to avoid complex iteration bugs
            task_count = len(self.running_tasks)
            completed_count = sum(1 for task in self.running_tasks.values() 
                                if isinstance(task, dict) and task.get("completed", False))
            running_count = task_count - completed_count
            
            result_parts = [
                f"=== TASK RESULTS SUMMARY ===",
                f"Total tasks: {task_count}",
                f"Completed: {completed_count}",
                f"Running: {running_count}",
                ""
            ]
            
            # Show completed tasks
            if completed_count > 0:
                result_parts.append("=== COMPLETED TASKS ===")
                for task_id, task_info in self.running_tasks.items():
                    if not isinstance(task_info, dict) or not task_info.get("completed", False):
                        continue
                    
                    runtime = task_info.get('end_time', time.time()) - task_info.get('start_time', 0)
                    result_parts.append(f"\n{task_id}: {task_info.get('description', 'Unknown')} - ✅ Completed ({runtime:.2f}s)")
                    
                    result = task_info.get("result", "No result")
                    if result and len(str(result)) > 10:
                        # Include full results (no truncation)
                        result_parts.append(f"Result: {str(result)}")
            
            # Show running tasks if requested
            if include_running and running_count > 0:
                result_parts.append("\n=== RUNNING TASKS ===")
                for task_id, task_info in self.running_tasks.items():
                    if not isinstance(task_info, dict) or task_info.get("completed", False):
                        continue
                    
                    runtime = time.time() - task_info.get('start_time', 0)
                    result_parts.append(f"\n{task_id}: {task_info.get('description', 'Unknown')} - ⏳ Running ({runtime:.2f}s)")
            
            # Clear completed tasks if requested
            if clear_after_retrieval and completed_count > 0:
                to_remove = [tid for tid, task in self.running_tasks.items() 
                           if isinstance(task, dict) and task.get("completed", False)]
                for task_id in to_remove:
                    del self.running_tasks[task_id]
                result_parts.append(f"\n--- {len(to_remove)} completed tasks cleared from memory ---")
            
            return "\n".join(result_parts)
        
        except Exception as e:
            logger.error(f"Error in _task_results: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error retrieving task results: {str(e)}"
    
    def _check_all_tasks_completed(self):
        """Check if all tasks in the current batch are completed and automatically report results."""
        if not self.running_tasks:
            return
        
        # Group tasks by batch
        batches = {}
        for task_id, task_info in self.running_tasks.items():
            batch_id = task_info.get("batch_id", "unknown")
            if batch_id not in batches:
                batches[batch_id] = []
            batches[batch_id].append((task_id, task_info))
        
        # Check each batch for completion
        for batch_id, batch_tasks in batches.items():
            completed_in_batch = [task for task_id, task in batch_tasks if task["completed"]]
            
            if (len(completed_in_batch) == len(batch_tasks) and 
                len(completed_in_batch) > 1 and 
                batch_id == self.current_batch_id):
                # All tasks in current batch are completed - generate consolidated summary
                print(f"\n\r🎉 [TASK MANAGER] All {len(completed_in_batch)} subagent tasks in {batch_id} have completed!\r")
                print(f"📋 [TASK MANAGER] Generating consolidated summary...\r")
                
                # Create consolidated summary for this batch
                summary = self._generate_consolidated_summary(batch_id)
                
                # Display the summary to the user
                print(f"\n\r📊 [SUBAGENT SUMMARY] Consolidated Results from {batch_id} ({len(completed_in_batch)} tasks):\r")
                print(f"{summary}\r")
                
                # Remove completed batch tasks
                for task_id, _ in batch_tasks:
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]
    
    def _generate_consolidated_summary(self, batch_id: str = None) -> str:
        """Generate a consolidated summary of completed subagent tasks for a specific batch."""
        if not self.running_tasks:
            return "No completed tasks to summarize."
        
        # Filter tasks by batch if specified
        if batch_id:
            batch_tasks = {task_id: task_info for task_id, task_info in self.running_tasks.items() 
                          if task_info.get("batch_id") == batch_id and task_info["completed"]}
        else:
            batch_tasks = {task_id: task_info for task_id, task_info in self.running_tasks.items() 
                          if task_info["completed"]}
        
        if not batch_tasks:
            return "No completed tasks found for the specified batch."
        
        completed_tasks = list(batch_tasks.values())
        
        # Create structured summary
        summary_parts = [
            f"=== SUBAGENT TASK SUMMARY ({batch_id or 'All Tasks'}) ===",
            f"Completed Tasks: {len(completed_tasks)}",
        ]
        
        # Calculate total runtime with error handling
        try:
            # Extract valid timestamps and calculate runtime
            start_times = []
            end_times = []
            for task in completed_tasks:
                start_time = task.get('start_time')
                end_time = task.get('end_time', time.time())
                
                # Validate that times are numeric
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    start_times.append(start_time)
                    end_times.append(end_time)
            
            if start_times and end_times:
                total_runtime = max(end_times) - min(start_times)
                summary_parts.append(f"Total Runtime: {total_runtime:.2f} seconds")
            else:
                summary_parts.append("Total Runtime: Unable to calculate (invalid time data)")
        except Exception as e:
            logger.error(f"Error calculating total runtime: {e}")
            summary_parts.append("Total Runtime: Unable to calculate (calculation error)")
        
        summary_parts.extend([
            "",
            "=== INDIVIDUAL TASK RESULTS ==="
        ])
        
        for i, (task_id, task_info) in enumerate(sorted(batch_tasks.items()), 1):
            if not task_info["completed"]:
                continue
                
            # Calculate individual task runtime with error handling
            try:
                start_time = task_info.get('start_time')
                end_time = task_info.get('end_time', time.time())
                
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    runtime = end_time - start_time
                    runtime_str = f"{runtime:.2f} seconds"
                else:
                    runtime_str = "Unable to calculate (invalid time data)"
            except Exception as e:
                logger.error(f"Error calculating runtime for task {task_id}: {e}")
                runtime_str = "Unable to calculate (calculation error)"
            
            summary_parts.extend([
                f"\n--- Task {i}: {task_id} ---",
                f"Description: {task_info['description']}",
                f"Runtime: {runtime_str}",
                f"Status: {'✅ Completed' if task_info['completed'] else '⏳ Running'}",
                ""
            ])
            
            # Add the actual result from the subagent
            result = task_info.get("result", "No result available")
            if result and result != "No result available":
                # Include full results (no truncation)
                summary_parts.extend([
                    "Result:",
                    result,
                    ""
                ])
        
        summary_parts.extend([
            "=== SUMMARY COMPLETE ===",
            "",
            "All parallel subagent tasks have finished. The above results are now available",
            "for your analysis and next steps."
        ])
        
        return "\n".join(summary_parts)
    
    async def start_mcp_server(self, server_name: str, server_config) -> bool:
        """Start and connect to an MCP server using FastMCP."""
        try:
            logger.info(f"Starting MCP server: {server_name}")
            
            # Construct command and args for FastMCP client
            command = server_config.command[0]
            args = server_config.command[1:] + server_config.args
            
            # Create FastMCP client with stdio transport
            transport = StdioTransport(command=command, args=args, env=server_config.env)
            client = FastMCPClient(transport=transport)
            
            # Enter the context manager and store it for cleanup
            context_manager = client.__aenter__()
            await context_manager
            
            # Store the client and context manager
            self.mcp_clients[server_name] = client
            self._mcp_contexts = getattr(self, '_mcp_contexts', {})
            self._mcp_contexts[server_name] = client
            
            # Get available tools from this server
            tools_result = await client.list_tools()
            if tools_result and hasattr(tools_result, 'tools'):
                for tool in tools_result.tools:
                    tool_key = f"{server_name}:{tool.name}"
                    self.available_tools[tool_key] = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        "client": client
                    }
                    logger.info(f"Registered tool: {tool_key}")
            elif hasattr(tools_result, '__len__'):
                # Handle list format
                for tool in tools_result:
                    tool_key = f"{server_name}:{tool.name}"
                    self.available_tools[tool_key] = {
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        "client": client
                    }
                    logger.info(f"Registered tool: {tool_key}")
            
            logger.info(f"Successfully connected to MCP server: {server_name}")
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to start MCP server {server_name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    async def shutdown(self):
        """Shutdown all MCP connections."""
        logger.info("Shutting down MCP connections...")
        
        # Close FastMCP client sessions
        for server_name, client in self.mcp_clients.items():
            try:
                # Exit the context manager properly
                await client.__aexit__(None, None, None)
                logger.info(f"Closed client session for {server_name}")
            except Exception as e:
                logger.error(f"Error closing client session for {server_name}: {e}")
        
        self.mcp_clients.clear()
        self.available_tools.clear()
        if hasattr(self, '_mcp_contexts'):
            self._mcp_contexts.clear()
        
        # Shutdown subagent manager if present
        if hasattr(self, 'subagent_manager') and self.subagent_manager:
            await self.subagent_manager.terminate_all()
    
    # Centralized Subagent Management Methods
    def _on_subagent_message(self, message):
        """Callback for when a subagent message is received - display during yield period."""
        try:
            # Get task_id for identification (if available in message data)
            task_id = message.data.get('task_id', 'unknown') if hasattr(message, 'data') and message.data else 'unknown'
            
            if message.type == 'output':
                formatted = f"🤖 [SUBAGENT-{task_id}] {message.content}"
            elif message.type == 'status':
                formatted = f"📋 [SUBAGENT-{task_id}] {message.content}"
            elif message.type == 'error':
                formatted = f"❌ [SUBAGENT-{task_id}] {message.content}"
            elif message.type == 'result':
                formatted = f"✅ [SUBAGENT-{task_id}] Result: {message.content}"
            else:
                formatted = f"[SUBAGENT-{task_id} {message.type}] {message.content}"
            
            # Only display immediately if we're in yielding mode (subagents active)
            # This ensures clean separation between main agent and subagent output
            if self.subagent_manager and self.subagent_manager.get_active_count() > 0:
                # Subagents are active - display immediately during yield period
                self._display_subagent_message_immediately(formatted, message.type)
            else:
                # No active subagents - just log for now (main agent controls display)
                logger.info(f"Subagent message logged: {message.type} - {message.content[:50]}")
                
        except Exception as e:
            logger.error(f"Error handling subagent message: {e}")
    
    def _display_subagent_message_immediately(self, formatted: str, message_type: str):
        """Display subagent message immediately with proper terminal handling."""
        try:
            # Handle raw terminal mode properly
            import sys
            import termios
            import tty
            
            try:
                # Check if terminal is in raw mode
                stdin_fd = sys.stdin.fileno()
                current_attrs = termios.tcgetattr(stdin_fd)
                
                # Check if we're in raw mode (no echo, no canonical processing)
                in_raw_mode = not (current_attrs[3] & termios.ECHO) and not (current_attrs[3] & termios.ICANON)
                
                if in_raw_mode:
                    # Terminal is in raw mode - convert all \n to \r\n for proper display
                    formatted_with_crlf = formatted.replace('\n', '\r\n')
                    output = f"\r\n{formatted_with_crlf}\r\n"
                    sys.stdout.write(output)
                    sys.stdout.flush()
                    logger.info(f"Displayed subagent message in raw mode: {message_type}")
                else:
                    # Normal mode - use prompt_toolkit
                    try:
                        from prompt_toolkit.patch_stdout import patch_stdout
                        from prompt_toolkit import print_formatted_text
                        
                        with patch_stdout():
                            print_formatted_text(formatted)
                        logger.info(f"Displayed subagent message via patch_stdout: {message_type}")
                    except ImportError:
                        # Fallback to stderr
                        print(formatted, file=sys.stderr, flush=True)
                        logger.info(f"Displayed subagent message via stderr: {message_type}")
                        
            except (OSError, termios.error):
                # Terminal control not available - use stderr fallback
                print(formatted, file=sys.stderr, flush=True)
                logger.info(f"Displayed subagent message via stderr fallback: {message_type}")
                
        except Exception as e:
            logger.error(f"Error displaying subagent message immediately: {e}")
    
    async def _collect_subagent_results(self):
        """Wait for all subagents to complete and collect their results."""
        if not self.subagent_manager:
            return []
        
        import time
        results = []
        max_wait_time = 300  # 5 minutes max wait
        start_time = time.time()
        
        # Wait for all active subagents to complete
        while self.subagent_manager.get_active_count() > 0:
            if time.time() - start_time > max_wait_time:
                logger.error("Timeout waiting for subagents to complete")
                break
            await asyncio.sleep(0.5)
        
        # Collect results from completed subagents
        logger.info(f"Checking {len(self.subagent_manager.subagents)} subagents for results")
        for task_id, subagent in self.subagent_manager.subagents.items():
            logger.info(f"Subagent {task_id}: completed={subagent.completed}, has_result={subagent.result is not None}")
            if subagent.completed and subagent.result:
                results.append({
                    'task_id': task_id,
                    'description': subagent.description,
                    'content': subagent.result,
                    'runtime': time.time() - subagent.start_time
                })
                logger.info(f"Collected result from {task_id}: {len(subagent.result)} chars")
            elif subagent.completed:
                logger.warning(f"Subagent {task_id} completed but has no result stored")
            else:
                logger.info(f"Subagent {task_id} not yet completed")
        
        logger.info(f"Collected {len(results)} results from {len(self.subagent_manager.subagents)} subagents")
        return results

    async def _execute_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool (built-in or external) and return the result."""
        try:
            if tool_key not in self.available_tools:
                # Debug: show available tools when tool not found
                available_list = list(self.available_tools.keys())[:10]  # First 10 tools
                return f"Error: Tool {tool_key} not found. Available tools: {available_list}"
            
            tool_info = self.available_tools[tool_key]
            tool_name = tool_info["name"]
            
            # Forward to parent if this is a subagent (except for subagent management tools)
            import sys
            if self.is_subagent and self.comm_socket:
                excluded_tools = ['task', 'task_status', 'task_results']
                if tool_name not in excluded_tools:
                    # Tool forwarding happens silently
                    return await self._forward_tool_to_parent(tool_key, tool_name, arguments)
            elif self.is_subagent:
                sys.stderr.write(f"🤖 [SUBAGENT] WARNING: is_subagent=True but no comm_socket for tool {tool_name}\n")
                sys.stderr.flush()
            
            # Check if it's a built-in tool
            if tool_info["server"] == "builtin":
                logger.info(f"Executing built-in tool: {tool_name}")
                return await self._execute_builtin_tool(tool_name, arguments)
            
            # Handle external MCP tools with FastMCP
            client = tool_info["client"]
            if client is None:
                return f"Error: No client session for tool {tool_key}"
            
            logger.info(f"Executing MCP tool: {tool_name}")
            result = await client.call_tool(tool_name, arguments)
            
            # Format the result for FastMCP
            if hasattr(result, 'content') and result.content:
                content_parts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        content_parts.append(content.text)
                    elif hasattr(content, 'data'):
                        content_parts.append(str(content.data))
                    else:
                        content_parts.append(str(content))
                return "\n".join(content_parts)
            elif isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return f"Tool executed successfully. Result type: {type(result)}, Content: {result}"
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_key}: {e}")
            return f"Error executing tool {tool_key}: {str(e)}"

    async def _forward_tool_to_parent(self, tool_key: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Forward tool execution to parent agent via communication socket."""
        try:
            import json
            import uuid
            
            # Create unique request ID for tracking
            request_id = str(uuid.uuid4())
            
            # Prepare tool execution message
            message = {
                "type": "tool_execution_request",
                "request_id": request_id,
                "tool_key": tool_key,
                "tool_name": tool_name,
                "tool_args": arguments,
                "timestamp": time.time()
            }
            
            # Send request to parent (synchronous)
            message_json = json.dumps(message) + "\n"
            self.comm_socket.send(message_json.encode('utf-8'))
            
            # Wait for response with timeout
            response_timeout = 300.0  # 5 minutes timeout for tool execution
            self.comm_socket.settimeout(response_timeout)
            
            # Read response (synchronous)
            buffer = ""
            while True:
                try:
                    data = self.comm_socket.recv(4096).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    
                    # Process complete messages (newline-delimited JSON)
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                response = json.loads(line.strip())
                                if (response.get("type") == "tool_execution_response" and 
                                    response.get("request_id") == request_id):
                                    
                                    # Return tool result
                                    if response.get("success", False):
                                        return response.get("result", "Tool executed successfully")
                                    else:
                                        error = response.get("error", "Unknown error")
                                        return f"Error from parent: {error}"
                                        
                            except json.JSONDecodeError:
                                continue
                                
                except Exception as e:
                    logger.error(f"Error receiving response from parent: {e}")
                    break
            
            return f"Error: No response received from parent for tool {tool_name}"
            
        except Exception as e:
            logger.error(f"Error forwarding tool {tool_name} to parent: {e}")
            return f"Error forwarding tool to parent: {str(e)}"

    async def _execute_mcp_tool_with_keepalive(self, tool_key: str, arguments: Dict[str, Any], input_handler=None, keepalive_interval: float = 5.0) -> tuple:
        """Execute an MCP tool with keep-alive messages, returning (result, keepalive_messages)."""
        import asyncio
        
        # Create the tool execution task
        tool_task = asyncio.create_task(self._execute_mcp_tool(tool_key, arguments))
        
        # Keep-alive configuration
        keepalive_messages = []
        start_time = asyncio.get_event_loop().time()
        
        # Monitor the task and collect keep-alive messages
        while not tool_task.done():
            try:
                # Check for interruption before waiting
                if input_handler and input_handler.interrupted:
                    tool_task.cancel()
                    keepalive_messages.append("🛑 Tool execution cancelled by user")
                    try:
                        await tool_task
                    except asyncio.CancelledError:
                        pass
                    return "Tool execution cancelled", keepalive_messages
                
                # Wait for either task completion or timeout
                await asyncio.wait_for(asyncio.shield(tool_task), timeout=keepalive_interval)
                break  # Task completed
            except asyncio.TimeoutError:
                # Task is still running, send keep-alive message
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                # Create a keep-alive message
                keepalive_msg = f"⏳ Tool {tool_key} still running... ({elapsed:.1f}s elapsed)"
                if input_handler:
                    keepalive_msg += ", press ESC to cancel"
                keepalive_messages.append(keepalive_msg)
                logger.debug(f"Keep-alive: {keepalive_msg}")
                continue
        
        # Get the final result
        result = await tool_task
        return result, keepalive_messages
    
    def _create_system_prompt(self, for_first_message: bool = False) -> str:
        """Create a basic system prompt that includes tool information."""
        tool_descriptions = []
        
        for tool_key, tool_info in self.available_tools.items():
            # Use the converted name format (with underscores)
            converted_tool_name = tool_key.replace(":", "_")
            description = tool_info["description"]
            tool_descriptions.append(f"- **{converted_tool_name}**: {description}")
        
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        
        # Basic system prompt - subclasses can override this
        system_prompt = f"""You are a top-tier autonomous software development agent. You are in control and responsible for completing the user's request.

**Mission:** Use the available tools to solve the user's request.

**Guiding Principles:**
- **Ponder, then proceed:** Briefly outline your plan before you act. State your assumptions.
- **Bias for action:** You are empowered to take initiative. Do not ask for permission, just do the work.
- **Problem-solve:** If a tool fails, analyze the error and try a different approach.
- **Break large changes into smaller chunks:** For large code changes, divide the work into smaller, manageable tasks to ensure clarity and reduce errors.

**File Reading Strategy:**
- **Be surgical:** Do not read entire files at once. It is a waste of your context window.
- **Locate, then read:** Use tools like `grep` or `find` to locate the specific line numbers or functions you need to inspect.
- **Read in chunks:** Read files in smaller, targeted chunks of 50-100 lines using the `offset` and `limit` parameters in the `read_file` tool.
- **Full reads as a last resort:** Only read a full file if you have no other way to find what you are looking for.

**File Editing Workflow:**
1.  **Read first:** Always read a file before you try to edit it, following the file reading strategy above.
2.  **Greedy Grepping:** Always `grep` or look for a small section around where you want to do an edit. This is faster and more reliable than reading the whole file.
3.  **Use `replace_in_file`:** For all file changes, use `builtin_replace_in_file` to replace text in files.
4.  **Chunk changes:** Break large edits into smaller, incremental changes to maintain control and clarity.

**Todo List Workflow:**
- **Use the Todo list:** Use `builtin_todo_read` and `builtin_todo_write` to manage your tasks.
- **Start with a plan:** At the beginning of your session, create a todo list to outline your steps.
- **Update as you go:** As you complete tasks, update the todo list to reflect your progress.

**Subagent Workflow:**
- **Parallel execution:** For complex investigations requiring multiple independent tasks, spawn multiple subagents simultaneously by making multiple `builtin_task` calls in the same response.
- **Automatic coordination:** After spawning subagents, the main agent automatically pauses, waits for all subagents to complete, then restarts with their combined results.
- **Do not poll status:** Avoid calling `builtin_task_status` repeatedly - the system handles coordination automatically.
- **Single response spawning:** To spawn multiple subagents, include all `builtin_task` calls in one response, not across multiple responses.

**Workflow:**
1.  **Reason:** Outline your plan.
2.  **Act:** Use one or more tool calls to execute your plan. Use parallel tool calls when it makes sense.
3.  **Respond:** When you have completed the request, provide the final answer to the user.

**Available Tools:**
{tools_text}

You are the expert. Complete the task."""

        return system_prompt
    

    def format_markdown(self, text: str) -> str:
        """Format markdown text for terminal display."""
        if not text:
            return text
            
        # Simple terminal-friendly markdown formatting
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Headers
            if line.startswith('# '):
                formatted_lines.append(f"\n\033[1m\033[4m{line[2:]}\033[0m")  # Bold + underline
            elif line.startswith('## '):
                formatted_lines.append(f"\n\033[1m{line[3:]}\033[0m")  # Bold
            elif line.startswith('### '):
                formatted_lines.append(f"\n\033[1m{line[4:]}\033[0m")  # Bold
            
            # Code blocks
            elif line.strip().startswith('```'):
                if line.strip() == '```':
                    formatted_lines.append("\033[2m" + line + "\033[0m")  # Dim
                else:
                    formatted_lines.append("\033[2m" + line + "\033[0m")  # Dim
            
            # Lists
            elif re.match(r'^\s*[-*+]\s', line):
                formatted_lines.append(f"\033[36m•\033[0m{line[line.index(' ', line.index('-') if '-' in line else line.index('*') if '*' in line else line.index('+')):]}")
            elif re.match(r'^\s*\d+\.\s', line):
                formatted_lines.append(f"\033[36m{line.split('.')[0]}.\033[0m{line[line.index('.') + 1:]}")
            
            # Regular line - process inline formatting
            else:
                # Bold
                line = re.sub(r'\*\*(.*?)\*\*', r'\033[1m\1\033[0m', line)
                # Italic (using dim since true italic isn't widely supported)
                line = re.sub(r'\*(.*?)\*', r'\033[3m\1\033[0m', line)
                # Inline code
                line = re.sub(r'`(.*?)`', r'\033[47m\033[30m\1\033[0m', line)
                
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def display_tool_execution_start(self, tool_count: int, is_subagent: bool = False, interactive: bool = True) -> str:
        """Display tool execution start message."""
        if is_subagent:
            return f"🤖 [SUBAGENT] Executing {tool_count} tool(s)..."
        else:
            return f"🔧 Using {tool_count} tool(s)..."
    
    def display_tool_execution_step(self, step_num: int, tool_name: str, arguments: dict, is_subagent: bool = False, interactive: bool = True) -> str:
        """Display individual tool execution step."""
        if is_subagent:
            return f"🤖 [SUBAGENT] Step {step_num}: Executing {tool_name}..."
        else:
            return f"{step_num}. Executing {tool_name}..."
    
    def display_tool_execution_result(self, result: str, is_error: bool = False, is_subagent: bool = False, interactive: bool = True) -> str:
        """Display tool execution result."""
        if is_error:
            prefix = "❌ [SUBAGENT] Error:" if is_subagent else "❌ Error:"
        else:
            prefix = "✅ [SUBAGENT] Result:" if is_subagent else "✅ Result:"
        
        # Truncate long results for display
        if len(result) > 200:
            result_preview = result[:200] + "..."
        else:
            result_preview = result
            
        return f"{prefix} {result_preview}"
    
    def display_tool_processing(self, is_subagent: bool = False, interactive: bool = True) -> str:
        """Display tool processing message."""
        if is_subagent:
            return "🤖 [SUBAGENT] Processing tool results..."
        else:
            return "⚙️ Processing tool results..."
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters for most models)."""
        return len(text) // 4
    
    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count estimated tokens in the conversation."""
        total_tokens = 0
        for message in messages:
            if isinstance(message.get('content'), str):
                total_tokens += self.estimate_tokens(message['content'])
            # Add small overhead for role and structure
            total_tokens += 10
        return total_tokens
    
    def get_token_limit(self) -> int:
        """Get the context token limit for the current model."""
        # Enhanced centralized token limit management with model configuration support
        model_limits = self._get_model_token_limits()
        
        # Try to get model name from config
        model_name = self._get_current_model_name()
        
        if model_name and model_name in model_limits:
            return model_limits[model_name]
        
        # Fallback: check for model patterns
        if model_name:
            for pattern, limit in model_limits.items():
                if pattern in model_name.lower():
                    return limit
        
        # Conservative default - subclasses can override
        return 32000
    
    def _get_model_token_limits(self) -> Dict[str, int]:
        """Define token limits for known models. Subclasses can extend this."""
        return {
            # DeepSeek models
            "deepseek-reasoner": 128000,
            "deepseek-chat": 64000,
            
            # Gemini models  
            "gemini-pro": 128000,
            "pro": 128000,  # Pattern matching for any "pro" model
            "gemini-flash": 64000,
            "flash": 64000,  # Pattern matching for any "flash" model
            
            # Common defaults
            "gpt-4": 128000,
            "gpt-3.5": 16000,
            "claude": 200000,
        }
    
    def _get_current_model_name(self) -> Optional[str]:
        """Get the current model name. Subclasses should override to provide specific model."""
        # Try common config patterns
        for attr_name in ['deepseek_config', 'gemini_config', 'openai_config']:
            if hasattr(self, attr_name):
                config = getattr(self, attr_name)
                if hasattr(config, 'model'):
                    return config.model
        
        return None
    
    def should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if conversation should be compacted."""
        current_tokens = self.count_conversation_tokens(messages)
        limit = self.get_token_limit()
        # Compact when we're at 80% of the limit
        return current_tokens > (limit * 0.8)
    
    async def compact_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a compact summary of the conversation to preserve context while reducing tokens."""
        if len(messages) <= 3:  # Keep conversations that are already short
            return messages
        
        # Always keep the first message (system prompt) and last 2 messages
        system_message = messages[0] if messages[0].get('role') == 'system' else None
        recent_messages = messages[-2:]
        
        # Messages to summarize (everything except system and last 2)
        start_idx = 1 if system_message else 0
        messages_to_summarize = messages[start_idx:-2]
        
        if not messages_to_summarize:
            return messages
        
        # Create summary prompt
        conversation_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages_to_summarize
        ])
        
        summary_prompt = f"""Please create a concise summary of this conversation that preserves:
1. Key decisions and actions taken
2. Important file changes or tool usage
3. Current project state and context
4. Any pending tasks or next steps

Conversation to summarize:
{conversation_text}

Provide a brief but comprehensive summary that maintains continuity for ongoing work."""

        try:
            # Use the current model to create summary
            summary_messages = [{"role": "user", "content": summary_prompt}]
            summary_response = await self.generate_response(summary_messages, tools=None)
            
            # Create condensed conversation
            condensed = []
            if system_message:
                condensed.append(system_message)
            
            # Add summary as a system message
            condensed.append({
                "role": "system", 
                "content": f"[CONVERSATION SUMMARY] {summary_response}"
            })
            
            # Add recent messages
            condensed.extend(recent_messages)
            
            print(f"\n🗜️  Conversation compacted: {len(messages)} → {len(condensed)} messages")
            return condensed
            
        except Exception as e:
            print(f"⚠️  Failed to compact conversation: {e}")
            # Fallback: just keep system + last 5 messages
            fallback = []
            if system_message:
                fallback.append(system_message)
            fallback.extend(messages[-5:])
            return fallback
    
    async def generate_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> Union[str, Any]:
        """Generate a response using the specific LLM. Centralized implementation."""
        # For subagents, use interactive=False to avoid terminal formatting issues
        interactive = not self.is_subagent
        
        # Default to streaming behavior, but allow subclasses to override
        stream = getattr(self, 'stream', True)
        
        # Call the concrete implementation's chat_completion method
        return await self.chat_completion(messages, stream=stream, interactive=interactive)
    
    # Tool conversion and parsing helper methods
    def normalize_tool_name(self, tool_key: str) -> str:
        """Normalize tool name by replacing colons with underscores."""
        return tool_key.replace(":", "_")
    
    def generate_default_description(self, tool_info: dict) -> str:
        """Generate a default description for a tool if none exists."""
        return tool_info.get("description") or f"Execute {tool_info['name']} tool"
    
    def get_tool_schema(self, tool_info: dict) -> dict:
        """Get tool schema with fallback to basic object schema."""
        return tool_info.get("schema") or {"type": "object", "properties": {}}
    
    def validate_json_arguments(self, args_json: str) -> bool:
        """Validate that a string contains valid JSON."""
        try:
            json.loads(args_json)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def validate_tool_name(self, tool_name: str) -> bool:
        """Validate tool name format."""
        return tool_name and (tool_name.startswith("builtin_") or "_" in tool_name)
    
    def create_tool_call_object(self, name: str, args: str, call_id: str = None):
        """Create a standardized tool call object."""
        import types
        
        # Create a SimpleNamespace object similar to OpenAI's format
        tool_call = types.SimpleNamespace()
        tool_call.function = types.SimpleNamespace()
        tool_call.function.name = name
        tool_call.function.arguments = args
        tool_call.id = call_id or f"call_{name}_{int(time.time())}"
        tool_call.type = "function"
        
        return tool_call
    
    @abstractmethod
    def convert_tools_to_llm_format(self) -> List[Dict]:
        """Convert tools to the specific LLM's format. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from the LLM response. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, Any]], stream: bool = True, interactive: bool = True) -> Union[str, Any]:
        """Generate chat completion. Must be implemented by subclasses."""
        pass
    
    async def interactive_chat(self, input_handler, existing_messages: List[Dict[str, Any]] = None):
        """Interactive chat session with shared functionality."""
        from cli_agent.core.input_handler import InterruptibleInput
        
        messages = existing_messages or []
        current_task = None
        
        print("Starting interactive chat. Type /quit or /exit to end, /tools to list available tools.")
        print("Use /help for slash commands. Press ESC at any time to interrupt operations.\n")
        
        while True:
            try:
                # Cancel any pending task if interrupted
                if input_handler.interrupted and current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
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
                
                # Handle slash commands
                if user_input.strip().startswith('/'):
                    try:
                        slash_response = await self.slash_commands.handle_slash_command(user_input.strip(), messages)
                        if slash_response:
                            # Handle special command responses
                            if isinstance(slash_response, dict):
                                if "compacted_messages" in slash_response:
                                    print(f"\n{slash_response['status']}\n")
                                    messages[:] = slash_response["compacted_messages"]  # Update messages in place
                                elif "clear_messages" in slash_response:
                                    print(f"\n{slash_response['status']}\n")
                                    messages.clear()  # Clear the local messages list
                                elif "quit" in slash_response:
                                    print(f"\n{slash_response['status']}")
                                    break  # Exit the chat loop
                                elif "reload_host" in slash_response:
                                    print(f"\n{slash_response['status']}")
                                    return {"reload_host": slash_response["reload_host"], "messages": messages}
                                else:
                                    print(f"\n{slash_response.get('status', str(slash_response))}\n")
                            else:
                                print(f"\n{slash_response}\n")
                            continue
                    except Exception as e:
                        print(f"\nError handling slash command: {e}\n")
                        continue
                
                if not user_input.strip():
                    # Empty input, just continue
                    continue
                
                # Add user message
                messages.append({"role": "user", "content": user_input})
                
                # Show thinking message
                print("\nThinking...")
                
                # Create response task
                tools_list = self.convert_tools_to_llm_format()
                current_task = asyncio.create_task(
                    self.generate_response(messages, tools_list)
                )
                
                # Wait for response with simple interruption handling
                try:
                    await current_task
                except asyncio.CancelledError:
                    print("\n🛑 Request cancelled")
                    input_handler.interrupted = False
                    current_task = None
                    continue
                except Exception as e:
                    print(f"\nError generating response: {e}")
                    current_task = None
                    continue
                
                # Get the response
                response = current_task.result()
                current_task = None
                
                if hasattr(response, '__aiter__'):
                    # Streaming response
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
                                # Convert \n to \r\n for proper terminal display in raw mode
                                display_chunk = chunk.replace('\n', '\r\n')
                                print(display_chunk, end="", flush=True)
                                full_response += chunk
                            else:
                                # Handle any non-string chunks if needed
                                display_chunk = str(chunk).replace('\n', '\r\n')
                                print(display_chunk, end="", flush=True)
                                full_response += str(chunk)
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
                    print(f"\nAssistant: {response}")
                    messages.append({"role": "assistant", "content": str(response)})
                
            except KeyboardInterrupt:
                # Move to beginning of line and clear, then print exit message
                sys.stdout.write('\r\x1b[KExiting...\n')
                sys.stdout.flush()
                break
            except Exception as e:
                print(f"\nError: {e}")