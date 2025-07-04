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

# For subagent runner, we need to ensure we can import from both
# the cli_agent package and the top-level modules (config, subagent)
# Add parent directories to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

# Add grandparent (project root) for config import
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Add current dir for subagent import 
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cli_agent.utils.tool_name_utils import ToolNameUtils
from config import load_config
from subagent import emit_error, emit_message, emit_output, emit_result, emit_status, emit_tool_request, emit_tool_result

# Global task_id for use in emit functions
current_task_id = None


def emit_output_with_id(text: str):
    """Emit output with task_id."""
    from cli_agent.subagents.subagent import emit_message as local_emit_message
    local_emit_message("output", text, task_id=current_task_id)


def emit_status_with_id(status: str, details: str = ""):
    """Emit status with task_id."""
    from cli_agent.subagents.subagent import emit_message as local_emit_message
    local_emit_message(
        "status",
        f"Status: {status}",
        status=status,
        details=details,
        task_id=current_task_id,
    )


def emit_result_with_id(result: str):
    """Emit result with task_id."""
    # Use the local subagent module's emit_message directly to ensure correct import
    from cli_agent.subagents.subagent import emit_message as local_emit_message
    local_emit_message("result", result, task_id=current_task_id)


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
    from cli_agent.subagents.subagent import emit_message as local_emit_message
    local_emit_message("error", error, details=details, task_id=current_task_id)


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

        # Load config and create host
        config = load_config()

        # Use new provider-model architecture for subagents
        # Check if task specifies a specific model to use
        task_model = task_data.get("model", None)
        task_role = task_data.get("role", None)

        if task_model:
            # Use task-specific provider-model format
            if ":" in task_model:
                # Already in provider:model format
                provider_model = task_model
            else:
                # Map model name to its default provider
                provider_model = _get_default_provider_for_model(task_model)

            # Create host using provider-model architecture
            try:
                with open("/tmp/subagent_host_creation.txt", "a") as f:
                    f.write(f"=== Attempting to create subagent host ===\n")
                    f.write(f"provider_model: {provider_model}\n")
                    f.write(f"task_id: {task_id}\n")
                    f.write("==========================================\n")
            except:
                pass
            
            host = config.create_host_from_provider_model(
                provider_model, is_subagent=True
            )
            
            try:
                with open("/tmp/subagent_host_creation.txt", "a") as f:
                    f.write(f"=== Subagent host created successfully ===\n")
                    f.write(f"host type: {type(host)}\n")
                    f.write(f"host.is_subagent: {getattr(host, 'is_subagent', 'MISSING')}\n")
                    f.write(f"has tool_execution_engine: {hasattr(host, 'tool_execution_engine')}\n")
                    f.write("========================================\n")
            except:
                pass
            
            emit_output_with_id(f"Created {provider_model} subagent")

            
        else:
            # Use current default provider-model
            host = config.create_host_from_provider_model(is_subagent=True)
            emit_output_with_id(f"Created {config.default_provider_model} subagent")

        

        # Set role on host if specified, otherwise default to subagent role
        if task_role:
            host.set_role(task_role)
            emit_output_with_id(f"Using role: {task_role}")
        else:
            # Default to subagent role for subagents
            host.set_role("subagent")
            emit_output_with_id("Using default subagent role")

        # Set up tool permission manager for subagent (inherits main agent settings)
        from cli_agent.core.tool_permissions import (
            ToolPermissionConfig,
            ToolPermissionManager,
        )

        permission_config = ToolPermissionConfig(
            allowed_tools=list(config.allowed_tools),
            disallowed_tools=list(config.disallowed_tools),
            auto_approve_session=config.auto_approve_tools,
        )
        permission_manager = ToolPermissionManager(permission_config)
        host.permission_manager = permission_manager

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
                    
                    # Send system init message for subagent
                    json_handler.send_system_init(
                        cwd=os.getcwd(),
                        tools=list(host.available_tools.keys()),
                        mcp_servers=[],
                        model=host._get_current_runtime_model() if hasattr(host, '_get_current_runtime_model') else "subagent"
                    )
            except ImportError:
                pass  # Silently handle missing JSON handler

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
                    # Get permission details if available - try current tool info first, then fallback to attributes
                    tool_name = self.current_tool_name or getattr(self, '_permission_tool_name', 'unknown')
                    tool_arguments = self.current_tool_arguments or getattr(self, '_permission_arguments', {})
                    tool_description = getattr(self, '_permission_description', 'Unknown tool')
                    full_prompt = getattr(self, '_permission_full_prompt', prompt_text)
                    
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
        
        # Also store task_id directly on the agent for emit_result tool access
        host._subagent_task_id = task_id

        # Connect to MCP servers (inherit from parent config)
        for server_name, server_config in config.mcp_servers.items():
            success = await host.start_mcp_server(server_name, server_config)
            # Only emit messages for failures
            if not success:
                emit_output_with_id(f"⚠️ Failed to connect to MCP server: {server_name}")

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
            # Conversation loop - continue until emit_result is called
            iteration = 0
            emit_result_called = False

            # Track if emit_result was called
            original_execute_mcp_tool = host.tool_execution_engine.execute_mcp_tool
            
            # Debug: Check what host this method is bound to
            try:
                with open("/tmp/original_execute_debug.txt", "a") as f:
                    f.write(f"=== original_execute_mcp_tool binding ===\n")
                    f.write(f"original_execute_mcp_tool: {original_execute_mcp_tool}\n")
                    f.write(f"bound to engine: {original_execute_mcp_tool.__self__}\n")
                    f.write(f"engine.agent: {original_execute_mcp_tool.__self__.agent}\n")
                    f.write(f"engine.agent.is_subagent: {getattr(original_execute_mcp_tool.__self__.agent, 'is_subagent', 'MISSING')}\n")
                    f.write(f"engine.agent == host: {original_execute_mcp_tool.__self__.agent == host}\n")
                    f.write(f"host id: {id(host)}\n")
                    f.write(f"engine.agent id: {id(original_execute_mcp_tool.__self__.agent)}\n")
                    f.write("========================================\n")
            except:
                pass

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
                emit_tool_request(tool_name, arguments, request_id, task_id=current_task_id)
                
                try:
                    # Special handling for emit_result - bypass the builtin tool executor
                    # and use our emit_result_with_id function directly
                    if tool_key == "builtin:emit_result":
                        result_text = arguments.get("result", "")
                        summary = arguments.get("summary", "")
                        
                        if not result_text:
                            result = "Error: result parameter is required"
                        else:
                            # Use our emit_result_with_id that includes the task_id
                            emit_result_with_id(result_text)
                            
                            # If summary is provided, also emit it with task_id
                            if summary:
                                emit_message("result", f"Summary: {summary}", task_id=current_task_id)
                            
                            # Terminate the subagent process
                            import sys
                            sys.exit(0)
                    else:
                        # Use the original tool execution method to ensure permission checks
                        result = await original_execute_mcp_tool(tool_key, arguments)
                except Exception as e:
                    # Emit tool error result
                    emit_tool_result(str(e), request_id, is_error=True, task_id=current_task_id)
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

                # Emit tool result event  
                emit_tool_result(str(result), request_id, is_error=False, task_id=current_task_id)

                return result

            host.tool_execution_engine.execute_mcp_tool = track_emit_result_tool

            try:
                # Conversation loop for multi-step tasks
                while not emit_result_called:
                    iteration += 1

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

                    # Add the response to conversation
                    if isinstance(response, str) and response.strip():
                        messages.append({"role": "assistant", "content": response})

                    # If emit_result was called during this iteration, the loop will exit
                    # Otherwise, continue to next iteration

                if not emit_result_called:
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
