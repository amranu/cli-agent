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

        # Set global task_id for emit functions
        global current_task_id
        current_task_id = task_id
        description = task_data["description"]
        prompt = task_data["prompt"]

        emit_status_with_id("started", f"Task {task_id} started")
        emit_output_with_id(f"Starting task: {description}")

        # Load config and create host
        config = load_config()

        # Check if task specifies a specific model to use
        task_model = task_data.get("model", None)

        if task_model:
            # Use task-specific model
            if task_model.startswith("gemini"):
                # Create config copy with task-specific model
                import copy

                from mcp_gemini_host import MCPGeminiHost

                task_config = copy.deepcopy(config)
                task_config.gemini_model = task_model
                task_config.deepseek_model = "gemini"  # Signal to use Gemini backend
                host = MCPGeminiHost(task_config, is_subagent=True)
                emit_output_with_id(f"Created {task_model} subagent")
            else:  # deepseek models
                # Create config copy with task-specific model
                import copy

                from mcp_deepseek_host import MCPDeepseekHost

                task_config = copy.deepcopy(config)
                task_config.deepseek_model = task_model
                host = MCPDeepseekHost(task_config, is_subagent=True)
                emit_output_with_id(f"Created {task_model} subagent")
        else:
            # Inherit from main agent (existing logic)
            if config.deepseek_model == "gemini":
                from mcp_gemini_host import MCPGeminiHost

                host = MCPGeminiHost(config, is_subagent=True)
                emit_output_with_id(
                    "Created Gemini subagent (inherited from main agent)"
                )
            else:
                from mcp_deepseek_host import MCPDeepseekHost

                host = MCPDeepseekHost(config, is_subagent=True)
                emit_output_with_id(
                    "Created DeepSeek subagent (inherited from main agent)"
                )

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

        host._execute_mcp_tool = emit_tool_execution

        # Run the subagent in a conversation loop until it calls emit_result
        conversation_round = 0
        max_rounds = 20  # Prevent infinite loops

        try:
            while conversation_round < max_rounds:
                conversation_round += 1
                emit_output_with_id(f"💭 Conversation round {conversation_round}")

                response = await host.generate_response(messages)

                # Handle the response
                if isinstance(response, str):
                    response_content = response
                else:
                    response_content = str(response)

                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": response_content})

                # Emit the assistant's response
                if response_content.strip():
                    emit_output_with_id(f"🤖 Response: {response_content}")

                # The emit_result tool will call sys.exit(0) when called, so if we reach here,
                # the subagent hasn't finished yet. We continue the conversation by asking
                # it to continue or complete the task.

                # Add a follow-up message to encourage completion if we've had multiple rounds
                # and the last response didn't contain tool calls
                if (
                    conversation_round > 2
                    and "🔧 Executing tool:" not in response_content
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue working on the task. Use the available tools if needed. When you have completed the task, use the emit_result tool to provide your final results and terminate.",
                        }
                    )

        except SystemExit:
            # This is expected when emit_result calls sys.exit(0)
            # The subagent has completed successfully
            return
        except Exception as e:
            emit_error_with_id(f"Task execution error: {str(e)}", str(e))
            raise
        finally:
            # Restore original method
            host._execute_mcp_tool = original_execute_mcp_tool

        # If we reach here without SystemExit, the subagent didn't call emit_result
        if conversation_round >= max_rounds:
            emit_error_with_id(
                f"Task did not complete after {max_rounds} rounds. Subagent should call emit_result to terminate.",
                "Max rounds exceeded",
            )
            emit_status_with_id(
                "failed",
                f"Task exceeded maximum {max_rounds} conversation rounds without calling emit_result",
            )
        else:
            # This shouldn't happen if emit_result was called (it should cause SystemExit)
            emit_error_with_id(
                "Task completed without calling emit_result", "Missing emit_result call"
            )
            emit_status_with_id("failed", "Task completed without calling emit_result")

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
