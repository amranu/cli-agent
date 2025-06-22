#!/usr/bin/env python3
"""
Subagent Runner - executes tasks for the new subagent system
"""

import asyncio
import json
import sys
import tempfile
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import load_config
from subagent import emit_message, emit_output, emit_status, emit_result, emit_error

# Global task_id for use in emit functions
current_task_id = None

def emit_output_with_id(text: str):
    """Emit output with task_id."""
    emit_message('output', text, task_id=current_task_id)

def emit_status_with_id(status: str, details: str = ""):
    """Emit status with task_id."""
    emit_message('status', f"Status: {status}", status=status, details=details, task_id=current_task_id)

def emit_result_with_id(result: str):
    """Emit result with task_id."""
    emit_message('result', result, task_id=current_task_id)

def emit_error_with_id(error: str, details: str = ""):
    """Emit error with task_id."""
    emit_message('error', error, details=details, task_id=current_task_id)


async def run_subagent_task(task_file_path: str):
    """Run a subagent task from a task file."""
    global current_task_id
    try:
        # Load task data
        with open(task_file_path, 'r') as f:
            task_data = json.load(f)
        
        task_id = task_data['task_id']
        current_task_id = task_id  # Set global for emit functions
        description = task_data['description']
        prompt = task_data['prompt']
        
        emit_status_with_id("started", f"Task {task_id} started")
        emit_output_with_id(f"Starting task: {description}")
        
        # Load config and create host
        config = load_config()
        
        if config.deepseek_model == "gemini":
            from mcp_gemini_host import MCPGeminiHost
            host = MCPGeminiHost(config, is_subagent=True)
            emit_output_with_id("Created Gemini subagent")
        else:
            from mcp_deepseek_host import MCPDeepseekHost
            host = MCPDeepseekHost(config, is_subagent=True)
            emit_output_with_id("Created DeepSeek subagent")
        
        # Connect to MCP servers (inherit from parent config)
        for server_name, server_config in config.mcp_servers.items():
            emit_output_with_id(f"Connecting to MCP server: {server_name}")
            success = await host.start_mcp_server(server_name, server_config)
            if success:
                emit_output_with_id(f"✅ Connected to MCP server: {server_name}")
            else:
                emit_output_with_id(f"⚠️ Failed to connect to MCP server: {server_name}")
        
        emit_output_with_id(f"Executing task with {len(host.available_tools)} tools available...")
        
        # Execute the task with custom tool execution monitoring
        # Add summary instructions to ensure subagent provides comprehensive results
        enhanced_prompt = f"""{prompt}

IMPORTANT: After completing your investigation/task, provide a clear and concise summary of your findings. Your summary should include:
1. What you investigated or accomplished
2. Key findings or results
3. Any important observations or insights
4. Conclusions or recommendations if applicable

Please structure your response so that the main findings are easily extractable for analysis."""
        
        messages = [{"role": "user", "content": enhanced_prompt}]
        
        # Override tool execution methods to emit messages
        original_execute_mcp_tool = host._execute_mcp_tool
        async def emit_tool_execution(tool_key, arguments):
            emit_output_with_id(f"🔧 Executing tool: {tool_key}")
            if arguments:
                # Show important parameters (limit size)
                args_str = str(arguments)[:200] + "..." if len(str(arguments)) > 200 else str(arguments)
                emit_output_with_id(f"📝 Parameters: {args_str}")
            
            try:
                result = await original_execute_mcp_tool(tool_key, arguments)
                # Show result preview
                result_str = str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
                emit_output_with_id(f"✅ Tool result: {result_str}")
                return result
            except Exception as e:
                emit_output_with_id(f"❌ Tool error: {str(e)}")
                raise
        
        host._execute_mcp_tool = emit_tool_execution
        
        try:
            response = await host.chat_completion(messages, stream=False, interactive=False)
        finally:
            # Restore original method
            host._execute_mcp_tool = original_execute_mcp_tool
        
        # Emit the result
        if isinstance(response, str):
            emit_result_with_id(response)
        else:
            emit_result_with_id(str(response))
        
        emit_status_with_id("completed", f"Task {task_id} completed successfully")
        
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