#!/usr/bin/env python3
"""
Subagent Runner - executes tasks for the new subagent system
"""

import asyncio
import json
import os
import sys
import tempfile

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import load_config
from subagent import emit_error, emit_message, emit_output, emit_result, emit_status

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

        # Use new provider-model architecture for subagents
        try:
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
                host = config.create_host_from_provider_model(provider_model)
                if hasattr(host, "is_subagent"):
                    host.is_subagent = True
                emit_output_with_id(f"Created {provider_model} subagent")
            else:
                # Use current default provider-model
                host = config.create_host_from_provider_model()
                if hasattr(host, "is_subagent"):
                    host.is_subagent = True
                emit_output_with_id(f"Created {config.default_provider_model} subagent")

        except Exception as e:
            # Fallback to legacy for compatibility
            emit_output_with_id(
                f"Provider-model creation failed: {e}, falling back to legacy"
            )
            if config.deepseek_model == "gemini":
                from mcp_gemini_host import MCPGeminiHost

                host = MCPGeminiHost(config, is_subagent=True)
                emit_output_with_id("Created legacy Gemini subagent")
            else:
                from mcp_deepseek_host import MCPDeepseekHost

                host = MCPDeepseekHost(config, is_subagent=True)
                emit_output_with_id("Created legacy DeepSeek subagent")

        # Set up tool permission manager for subagent (inherits main agent settings)
        from cli_agent.core.input_handler import InterruptibleInput
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

        # Create custom input handler for subagent that connects to main terminal
        class SubagentInputHandler(InterruptibleInput):
            def __init__(self, task_id):
                super().__init__()
                self.subagent_context = task_id

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
                    emit_message(
                        "permission_request",
                        prompt_text,
                        task_id=current_task_id,
                        request_id=request_id,
                        response_file=response_file,
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

CRITICAL INSTRUCTIONS:
- You are a subagent focused on executing tasks.
- You MUST use the available tools to complete your task.
- You can provide reasoning and explanations as you work.
- When the task is complete, you MUST use the 'emit_result' tool to provide a summary of your results.
- The emit_result tool takes two parameters: 'result' (required - the main findings/output) and 'summary' (optional - brief description of what was accomplished).
- Always call emit_result as your final action to terminate the subagent and return results to the main agent.
- The subagent will continue running until you call emit_result.
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

        # Approach: Override tool execution engine to capture results
        try:
            # Capture tool execution results
            captured_tool_results = []

            # Override the tool execution engine's execute_mcp_tool method directly
            original_execute_mcp_tool = host.tool_execution_engine.execute_mcp_tool

            async def capture_tool_execution(tool_key, arguments):
                result = await original_execute_mcp_tool(tool_key, arguments)

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

                captured_tool_results.append({"tool": tool_key, "result": result})

                # Emit tool result immediately for real-time feedback
                tool_name = tool_key.split(":")[-1]
                emit_output_with_id(f"🔧 {tool_name}: {str(result).strip()}")

                return result

            host.tool_execution_engine.execute_mcp_tool = capture_tool_execution

            try:
                # Generate response with tool capture
                response = await host.generate_response(messages, stream=False)

                # Build result from captured tool outputs
                if captured_tool_results:
                    result_parts = []
                    for tool_result in captured_tool_results:
                        tool_name = tool_result["tool"].split(":")[
                            -1
                        ]  # Get tool name without prefix
                        result_content = str(tool_result["result"]).strip()

                        # For bash commands, extract just the output
                        if tool_name == "bash_execute" and result_content:
                            # Look for actual command output
                            lines = result_content.split("\n")
                            for line in lines:
                                if line.strip() and not line.startswith(
                                    ("Executing", "Command completed", "Exit code")
                                ):
                                    result_parts.append(line.strip())
                        else:
                            result_parts.append(result_content)

                    if result_parts:
                        result_text = "\n".join(result_parts)
                    else:
                        result_text = f"Task '{description}' executed successfully"
                else:
                    result_text = (
                        str(response).strip()
                        if response
                        else f"Task '{description}' completed successfully"
                    )

            finally:
                # Restore original method
                host.tool_execution_engine.execute_mcp_tool = original_execute_mcp_tool

            # Use emit_result to return the captured output
            from subagent import emit_result

            emit_result(result_text)
            return  # Exit successfully

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
            raise

        # Fallback: If the main approach fails, this shouldn't be reached
        emit_error_with_id(
            "Unexpected: Reached fallback section", "Main execution path failed"
        )

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
    if len(sys.argv) != 2:
        print("Usage: python subagent_runner.py <task_file>")
        sys.exit(1)

    task_file = sys.argv[1]
    asyncio.run(run_subagent_task(task_file))
