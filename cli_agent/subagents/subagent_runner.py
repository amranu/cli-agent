#!/usr/bin/env python3
"""
Subagent Runner - executes tasks for the new subagent system
"""

import asyncio
import json
import os
import sys
import tempfile
import time

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cli_agent.utils.tool_name_utils import ToolNameUtils
from config import load_config
from subagent import emit_error, emit_message, emit_output, emit_result, emit_status, emit_tool_request, emit_tool_result

# Global task_id for use in emit functions
current_task_id = None


def emit_output_with_id(text: str):
    """Emit output with task_id."""
    emit_message("output", text, task_id=current_task_id)


def emit_status_with_id(status: str, details: str = ""):
    """Emit status with task_id."""
    emit_message(
        "status",
        f"Status: {status}",
        status=status,
        details=details,
        task_id=current_task_id,
    )


def emit_result_with_id(result: str):
    """Emit result with task_id."""
    emit_message("result", result, task_id=current_task_id)


def _get_default_provider_for_model(model_name: str) -> str:
    """Map model name to its default provider:model format."""
    model_lower = model_name.lower()

    # Gemini models -> Google provider
    if any(keyword in model_lower for keyword in ["gemini", "flash", "pro"]):
        return f"google:{model_name}"

    # Claude models -> Anthropic provider
    elif any(
        keyword in model_lower for keyword in ["claude", "sonnet", "haiku", "opus"]
    ):
        return f"anthropic:{model_name}"

    # GPT/OpenAI models -> OpenAI provider
    elif any(keyword in model_lower for keyword in ["gpt", "o1", "turbo"]):
        return f"openai:{model_name}"

    # DeepSeek models -> DeepSeek provider
    elif any(keyword in model_lower for keyword in ["deepseek", "chat", "reasoner"]):
        return f"deepseek:{model_name}"

    # Default to DeepSeek provider for unknown models
    else:
        return f"deepseek:{model_name}"


def emit_error_with_id(error: str, details: str = ""):
    """Emit error with task_id."""
    emit_message("error", error, details=details, task_id=current_task_id)


async def run_subagent_task(task_file_path: str):
    """Run a subagent task from a task file."""
    global current_task_id

    # Set up signal handlers for subagent process
    try:
        from cli_agent.core.global_interrupt import get_global_interrupt_manager

        interrupt_manager = get_global_interrupt_manager()

        # Add subagent-specific interrupt callback
        def subagent_interrupt_callback():
            if current_task_id:
                emit_status_with_id("interrupted", "Task interrupted by user")
                emit_error_with_id(
                    "Task execution interrupted", "User requested interrupt"
                )

        interrupt_manager.add_callback(subagent_interrupt_callback)
    except Exception as e:
        # Continue without interrupt handling if setup fails
        print(f"Warning: Could not set up subagent interrupt handling: {e}")

    try:
        # Load task data
        with open(task_file_path, "r") as f:
            task_data = json.load(f)

        task_id = task_data["task_id"]

        # Set global task_id for emit functions IMMEDIATELY after getting task_id
        current_task_id = task_id
        description = task_data["description"]
        prompt = task_data["prompt"]

        emit_status_with_id("started", f"Task {task_id} started")
        emit_output_with_id(f"Starting task: {description}")

        # Load config and create host
        config = load_config()
        emit_output_with_id("Configuration loaded successfully")

        # Use new provider-model architecture for subagents
        # Check if task specifies a specific model to use
        task_model = task_data.get("model", None)

        if task_model:
            # Use task-specific provider-model format
            if ":" in task_model:
                # Already in provider:model format
                provider_model = task_model
            else:
                # Map model name to its default provider
                provider_model = _get_default_provider_for_model(task_model)

            # Create host using provider-model architecture
            host = config.create_host_from_provider_model(
                provider_model, is_subagent=True
            )
            emit_output_with_id(f"Created {provider_model} subagent")
        else:
            # Use current default provider-model
            host = config.create_host_from_provider_model(is_subagent=True)
            emit_output_with_id(f"Created {config.default_provider_model} subagent")

        # Disable permission manager entirely for subagents in stream-json mode
        if os.environ.get("STREAM_JSON_MODE") == "true":
            host.permission_manager = None
            emit_output_with_id("Disabled permission manager for stream-json mode")

        # Set up JSON handler if in stream-json mode (detected via environment)
        if os.environ.get("STREAM_JSON_MODE") == "true":
            try:
                from streaming_json import StreamingJSONHandler
                import uuid
                
                # Create JSON handler for subagent
                json_handler = StreamingJSONHandler(session_id=str(uuid.uuid4()))
                
                # Set JSON handler on display manager to enable tool JSON emission
                if hasattr(host, 'display_manager'):
                    host.display_manager.json_handler = json_handler
                    emit_output_with_id("Subagent configured for stream-json mode")
                    
                    # Send system init message for subagent
                    json_handler.send_system_init(
                        cwd=os.getcwd(),
                        tools=list(host.available_tools.keys()),
                        mcp_servers=[],
                        model=host._get_current_runtime_model() if hasattr(host, '_get_current_runtime_model') else "subagent"
                    )
                else:
                    emit_output_with_id("Warning: Host has no display_manager for JSON setup")
            except ImportError as e:
                emit_output_with_id(f"Warning: Could not set up JSON handler: {e}")

        # Create custom input handler for subagent that connects to main terminal
        from cli_agent.core.input_handler import InterruptibleInput
        
        class SubagentInputHandler(InterruptibleInput):
            def __init__(self, task_id):
                super().__init__()
                self.subagent_context = task_id
                # Store tool info for permission requests
                self.current_tool_name = None
                self.current_tool_arguments = None

            def set_current_tool_info(self, tool_name: str, arguments: dict):
                """Set the current tool information for permission requests."""
                self.current_tool_name = tool_name
                self.current_tool_arguments = arguments

            def get_input(
                self,
                prompt_text: str,
                multiline_mode: bool = False,
                allow_escape_interrupt: bool = False,
            ):
                # Auto-approve all tools when in stream-json mode (detected via environment)
                if os.environ.get("STREAM_JSON_MODE") == "true":
                    return "y"
                
                # For subagents, emit a permission request and wait for response via a temp file
                try:
                    import os
                    import tempfile
                    import time
                    import uuid

                    # Create unique request ID
                    request_id = str(uuid.uuid4())

                    # Create temp file for response
                    temp_dir = tempfile.gettempdir()
                    response_file = os.path.join(
                        temp_dir, f"subagent_response_{request_id}.txt"
                    )

                    # Emit permission request to main process
                    # Use current tool info if available, otherwise fall back to basic values
                    tool_name = self.current_tool_name or 'unknown'
                    tool_arguments = self.current_tool_arguments or {}
                    
                    # Create simple description from tool name and arguments  
                    if tool_name != 'unknown':
                        if tool_name == 'bash_execute' and 'command' in tool_arguments:
                            tool_description = f"Execute bash command: {tool_arguments['command']}"
                        elif tool_name == 'read_file' and 'file_path' in tool_arguments:
                            tool_description = f"Read file: {tool_arguments['file_path']}"
                        elif tool_name == 'write_file' and 'file_path' in tool_arguments:
                            tool_description = f"Write to file: {tool_arguments['file_path']}"
                        else:
                            tool_description = f"Execute tool: {tool_name}"
                    else:
                        tool_description = 'Unknown tool'
                    
                    full_prompt = f"Allow {tool_name}? {tool_description}"
                    
                    emit_message(
                        "permission_request",
                        full_prompt,  # Send the full formatted prompt for display
                        task_id=self.subagent_context,
                        request_id=request_id,
                        response_file=response_file,
                        tool_name=tool_name,
                        arguments=tool_arguments,
                        description=tool_description,
                    )

                    # Wait for response file to be created by main process
                    timeout = 60  # 60 seconds timeout
                    start_time = time.time()

                    while not os.path.exists(response_file):
                        if time.time() - start_time > timeout:
                            emit_output_with_id(
                                "Permission request timeout, defaulting to allow"
                            )
                            return "y"
                        time.sleep(0.1)

                    # Read response from file
                    with open(response_file, "r") as f:
                        response = f.read().strip()

                    # Clean up temp file
                    try:
                        os.remove(response_file)
                    except:
                        pass

                    return response

                except Exception as e:
                    emit_output_with_id(
                        f"Permission request error, defaulting to allow: {e}"
                    )
                    return "y"

        # Set up input handler for subagent with task context
        host._input_handler = SubagentInputHandler(task_id)

        emit_output_with_id("Tool permission manager and input handler configured")

        # Connect to MCP servers (inherit from parent config)
        for server_name, server_config in config.mcp_servers.items():
            emit_output_with_id(f"Connecting to MCP server: {server_name}")
            success = await host.start_mcp_server(server_name, server_config)
            if success:
                emit_output_with_id(f"✅ Connected to MCP server: {server_name}")
            else:
                emit_output_with_id(f"⚠️ Failed to connect to MCP server: {server_name}")

        emit_output_with_id(
            f"Executing task with {len(host.available_tools)} tools available..."
        )

        # Execute the task with custom tool execution monitoring
        # Add explicit tool usage instructions for subagents
        enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS FOR SUBAGENT:
- You are a subagent that MUST use tools to complete tasks.
- Start immediately by using the appropriate tools (like bash_execute, read_file, etc.).
- DO NOT provide explanations or analysis without first executing the requested tools.
- Execute tools step by step to gather the required information.
- After ALL tool execution is complete, analyze the results and call emit_result.
- The emit_result tool takes 'result' (required - your findings) and 'summary' (optional - brief description).
- IMPORTANT: You MUST call emit_result as your final action to complete the task.
- The conversation will continue until you call emit_result.
- Begin with tool usage immediately - no preliminary explanations needed.
"""

        messages = [{"role": "user", "content": enhanced_prompt}]

        # Override tool execution methods to emit messages
        original_execute_mcp_tool = host._execute_mcp_tool

        async def emit_tool_execution(tool_key, arguments):
            emit_output_with_id(f"🔧 Executing tool: {tool_key}")
            if arguments:
                # Show important parameters (limit size)
                args_str = (
                    str(arguments)[:200] + "..."
                    if len(str(arguments)) > 200
                    else str(arguments)
                )
                emit_output_with_id(f"📝 Parameters: {args_str}")

            try:
                result = await original_execute_mcp_tool(tool_key, arguments)
                # Use proper formatting for tool results (same as main agent)
                from cli_agent.core.formatting import ResponseFormatter

                formatter = ResponseFormatter()
                tool_result_msg = formatter.display_tool_execution_result(
                    result,
                    is_error=False,
                    is_subagent=True,
                    interactive=True,
                )
                emit_output_with_id(tool_result_msg)
                return result
            except Exception as e:
                # Use proper formatting for tool errors (same as main agent)
                from cli_agent.core.formatting import ResponseFormatter

                formatter = ResponseFormatter()
                tool_error_msg = formatter.display_tool_execution_result(
                    str(e),
                    is_error=True,
                    is_subagent=True,
                    interactive=True,
                )
                emit_output_with_id(tool_error_msg)
                raise

        # NOTE: Don't override _execute_mcp_tool - let normal tool call handling work
        # host._execute_mcp_tool = emit_tool_execution

        try:
            # Conversation loop - continue until emit_result is called or max iterations reached
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            emit_result_called = False

            # Track if emit_result was called
            original_execute_mcp_tool = host.tool_execution_engine.execute_mcp_tool

            async def track_emit_result_tool(tool_key, arguments):
                nonlocal emit_result_called
                
                # Set tool information on input handler before execution for permission display
                if hasattr(host, '_input_handler') and host._input_handler and hasattr(host._input_handler, 'set_current_tool_info'):
                    # Resolve tool name from tool_key
                    tool_name = tool_key.split(":")[-1] if ":" in tool_key else tool_key
                    host._input_handler.set_current_tool_info(tool_name, arguments)
                
                # Generate request_id and emit tool request
                tool_name = tool_key.split(":")[-1] if ":" in tool_key else tool_key
                request_id = f"subagent_{current_task_id}_{tool_name}_{int(time.time())}"
                emit_tool_request(tool_name, arguments, request_id)
                
                try:
                    result = await original_execute_mcp_tool(tool_key, arguments)
                except Exception as e:
                    # Emit tool error result
                    emit_tool_result(str(e), request_id, is_error=True)
                    raise

                # Check if tool was denied by user
                if isinstance(result, str) and "Tool execution denied" in result:
                    emit_error_with_id(
                        "Tool execution denied by user",
                        "User denied tool permission, terminating subagent",
                    )
                    emit_status_with_id(
                        "cancelled", "Task cancelled due to tool denial"
                    )
                    # Exit the subagent cleanly
                    import sys

                    sys.exit(0)

                # Check if emit_result was called
                if tool_key == "builtin:emit_result":
                    emit_result_called = True
                    emit_output_with_id("✅ Task completed - emit_result called")

                # Emit tool result event  
                emit_tool_result(str(result), request_id, is_error=False)

                return result

            host.tool_execution_engine.execute_mcp_tool = track_emit_result_tool

            try:
                # Conversation loop for multi-step tasks
                while iteration < max_iterations and not emit_result_called:
                    iteration += 1
                    emit_output_with_id(f"🔄 Conversation iteration {iteration}")

                    # Generate response using the main agent's system
                    emit_output_with_id("🚀 Calling host.generate_response...")

                    # Create normalized tools mapping for subagent use
                    # This ensures both normalized and original tool names are available
                    original_available_tools = host.available_tools

                    # Use centralized tool name utilities
                    normalized_tools = ToolNameUtils.create_normalized_tools_mapping(
                        original_available_tools
                    )
                    host.available_tools = normalized_tools

                    try:
                        response = await host.generate_response(messages, stream=False)
                    finally:
                        # Restore original tools
                        host.available_tools = original_available_tools

                    emit_output_with_id("✅ host.generate_response completed")

                    # Add the response to conversation
                    if isinstance(response, str) and response.strip():
                        messages.append({"role": "assistant", "content": response})
                        emit_output_with_id(f"📝 Added response: {response}")

                    # If emit_result was called during this iteration, the loop will exit
                    # Otherwise, continue to next iteration

                if iteration >= max_iterations:
                    emit_error_with_id(
                        "Max conversation iterations reached",
                        f"Subagent reached {max_iterations} iterations without calling emit_result",
                    )
                elif not emit_result_called:
                    emit_error_with_id(
                        "Task completed without explicit result",
                        "Subagent finished without calling emit_result tool",
                    )

            finally:
                # Restore original method
                host.tool_execution_engine.execute_mcp_tool = original_execute_mcp_tool

        except SystemExit:
            # This is expected when emit_result calls sys.exit(0)
            return
        except Exception as e:
            # Check if this is a tool denial that should terminate the subagent
            if "ToolDeniedReturnToPrompt" in str(
                type(e)
            ) or "Tool execution denied" in str(e):
                emit_error_with_id(
                    "Tool execution denied by user",
                    "User denied tool permission, terminating subagent",
                )
                emit_status_with_id("cancelled", "Task cancelled due to tool denial")
                return  # Terminate subagent cleanly
            emit_error_with_id(f"Task execution error: {str(e)}", str(e))
            return  # Exit on error instead of raising

    except Exception as e:
        emit_error_with_id(f"Task failed: {str(e)}", str(e))
        emit_status_with_id("failed", f"Task failed with error: {str(e)}")

    finally:
        # Clean up task file
        try:
            os.unlink(task_file_path)
        except:
            pass


if __name__ == "__main__":
    # Parse arguments - simplified since stream-json mode is detected via environment
    if len(sys.argv) != 2:
        print("Usage: python subagent_runner.py <task_file>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    asyncio.run(run_subagent_task(task_file))
