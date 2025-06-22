#!/usr/bin/env python3
# This script implements the main command-line interface for the MCP Agent.
"""Main CLI interface for the MCP Agent with modular imports."""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import click

from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from config import HostConfig, load_config

# Import local modules with error handling for different environments
try:
    from streaming_json import StreamingJSONHandler
except ImportError:
    # Try importing from current directory if package import fails
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from streaming_json import StreamingJSONHandler

try:
    from session_manager import SessionManager
except ImportError:
    # Try importing from current directory if package import fails
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Suppress WARNING messages during interactive chat
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# CLI functionality
@click.group()
@click.option(
    "--config-file",
    default=None,
    help="Path to the configuration file (default: ~/.mcp/config.json)",
)
@click.pass_context
def cli(ctx, config_file):
    """MCP Agent - Run AI models with MCP tool integration."""
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file


@cli.command()
def init():
    """Initialize configuration file."""
    from config import create_sample_env

    create_sample_env()


@cli.command("switch-chat")
@click.pass_context
def switch_chat(ctx):
    """Switch the model to deepseek-chat."""
    config = load_config()
    config.deepseek_model = "deepseek-chat"
    click.echo(f"Model switched to: {config.deepseek_model}")
    # Save the updated config
    config.save()


@cli.command("switch-reason")
@click.pass_context
def switch_reason(ctx):
    """Switch the model to deepseek-reasoner."""
    config = load_config()
    config.deepseek_model = "deepseek-reasoner"
    click.echo(f"Model switched to: {config.deepseek_model}")
    # Save the updated config
    config.save()


@cli.command("switch-gemini")
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


@cli.command("switch-gemini-pro")
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
@click.option(
    "--server",
    multiple=True,
    help="MCP server to connect to (format: name:command:arg1:arg2)",
)
@click.option(
    "--input-format",
    type=click.Choice(["text", "stream-json"]),
    default="text",
    help="Input format (text or stream-json)",
)
@click.option(
    "--output-format",
    type=click.Choice(["text", "stream-json"]),
    default="text",
    help="Output format (text or stream-json)",
)
@click.option(
    "-c",
    "--continue",
    "continue_session",
    is_flag=True,
    help="Continue the last conversation",
)
@click.option("--resume", "resume_session_id", help="Resume a specific session by ID")
@click.option(
    "--allowed-tools",
    multiple=True,
    help='Comma or space-separated list of tool names to allow (e.g. "Bash(git:*) Edit")',
)
@click.option(
    "--disallowed-tools",
    multiple=True,
    help='Comma or space-separated list of tool names to deny (e.g. "Bash(git:*) Edit")',
)
@click.option(
    "--auto-approve-tools",
    is_flag=True,
    help="Auto-approve all tool executions for this session",
)
@click.pass_context
async def chat(
    ctx,
    server,
    input_format,
    output_format,
    continue_session,
    resume_session_id,
    allowed_tools,
    disallowed_tools,
    auto_approve_tools,
):
    """Start interactive chat session."""
    try:
        # Parse tool permissions from CLI arguments
        def parse_tool_list(tool_args):
            """Parse comma or space-separated tool lists."""
            tools = []
            for arg in tool_args:
                # Split by comma or space and clean up
                parts = [t.strip() for t in arg.replace(",", " ").split() if t.strip()]
                tools.extend(parts)
            return tools

        parsed_allowed_tools = parse_tool_list(allowed_tools)
        parsed_disallowed_tools = parse_tool_list(disallowed_tools)

        # Initialize session manager
        session_manager = SessionManager()

        # Handle session resumption
        if continue_session:
            session_id = session_manager.continue_last_session()
            if session_id:
                click.echo(f"Continuing session: {session_id[:8]}...")
                messages = session_manager.get_messages()
                click.echo(f"Restored {len(messages)} messages")
            else:
                click.echo("No previous session found, starting new conversation")
                session_id = session_manager.create_new_session()
                messages = []
        elif resume_session_id:
            session_id = session_manager.resume_session(resume_session_id)
            if session_id:
                click.echo(f"Resumed session: {session_id[:8]}...")
                messages = session_manager.get_messages()
                click.echo(f"Restored {len(messages)} messages")
            else:
                click.echo(
                    f"Session {resume_session_id} not found, starting new conversation"
                )
                session_id = session_manager.create_new_session()
                messages = []
        else:
            # Start new session
            session_id = session_manager.create_new_session()
            messages = []
            click.echo(f"Started new session: {session_id[:8]}...")

        # Handle mixed modes or invalid combinations FIRST
        if input_format != output_format:
            click.echo(
                "Error: Mixed input/output formats not supported. Both must be 'text' or both must be 'stream-json'."
            )
            return

        # Handle streaming JSON mode
        if input_format == "stream-json" or output_format == "stream-json":
            return await handle_streaming_json_chat(
                server, input_format, output_format, session_manager
            )

        # Handle text mode (default behavior)
        if input_format == "text" and output_format == "text":
            return await handle_text_chat(
                server,
                session_manager,
                messages,
                parsed_allowed_tools,
                parsed_disallowed_tools,
                auto_approve_tools,
            )

    except KeyboardInterrupt:
        pass
    finally:
        if "host" in locals():
            if hasattr(host.shutdown, "__call__") and asyncio.iscoroutinefunction(
                host.shutdown
            ):
                await host.shutdown()
            else:
                host.shutdown()


@cli.command()
@click.argument("message")
@click.option("--server", multiple=True, help="MCP server to connect to")
@click.option(
    "--input-format",
    type=click.Choice(["text", "stream-json"]),
    default="text",
    help="Input format (text or stream-json)",
)
@click.option(
    "--output-format",
    type=click.Choice(["text", "stream-json"]),
    default="text",
    help="Output format (text or stream-json)",
)
@click.option(
    "--allowed-tools",
    multiple=True,
    help='Comma or space-separated list of tool names to allow (e.g. "Bash(git:*) Edit")',
)
@click.option(
    "--disallowed-tools",
    multiple=True,
    help='Comma or space-separated list of tool names to deny (e.g. "Bash(git:*) Edit")',
)
@click.option(
    "--auto-approve-tools",
    is_flag=True,
    help="Auto-approve all tool executions for this session",
)
@click.pass_context
async def ask(
    ctx,
    message,
    server,
    input_format,
    output_format,
    allowed_tools,
    disallowed_tools,
    auto_approve_tools,
):
    """Ask a single question."""
    try:
        # Validate format combinations
        if input_format != output_format:
            click.echo(
                "Error: Mixed input/output formats not supported. Both must be 'text' or both must be 'stream-json'."
            )
            return

        # Handle streaming JSON mode for ask command
        if input_format == "stream-json" or output_format == "stream-json":
            click.echo(
                "Error: Streaming JSON mode not supported for single question mode. Use 'chat' command instead."
            )
            return

        config = load_config()

        # Check if Gemini backend should be used
        if config.deepseek_model == "gemini":
            if not config.gemini_api_key:
                click.echo(
                    "Error: GEMINI_API_KEY not set. Run 'init' command first and update .env file."
                )
                return

            # Import and create Gemini host
            from mcp_gemini_host import MCPGeminiHost

            host = MCPGeminiHost(config)
        else:
            if not config.deepseek_api_key:
                click.echo(
                    "Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file."
                )
                return

            from mcp_deepseek_host import MCPDeepseekHost

            host = MCPDeepseekHost(config)

        # Parse tool permissions from CLI arguments
        def parse_tool_list(tool_args):
            """Parse comma or space-separated tool lists."""
            tools = []
            for arg in tool_args:
                # Split by comma or space and clean up
                parts = [t.strip() for t in arg.replace(",", " ").split() if t.strip()]
                tools.extend(parts)
            return tools

        parsed_allowed_tools = parse_tool_list(allowed_tools)
        parsed_disallowed_tools = parse_tool_list(disallowed_tools)

        # Initialize tool permission manager
        from cli_agent.core.tool_permissions import (
            ToolPermissionConfig,
            ToolPermissionManager,
        )

        # Merge CLI args with config
        final_allowed_tools = list(config.allowed_tools) + parsed_allowed_tools
        final_disallowed_tools = list(config.disallowed_tools) + parsed_disallowed_tools
        final_auto_approve = config.auto_approve_tools or auto_approve_tools

        permission_config = ToolPermissionConfig(
            allowed_tools=final_allowed_tools,
            disallowed_tools=final_disallowed_tools,
            auto_approve_session=final_auto_approve,
        )
        permission_manager = ToolPermissionManager(permission_config)

        # Set the permission manager on the host (no input handler for ask command)
        host.permission_manager = permission_manager

        # Connect to servers
        for server_spec in server:
            parts = server_spec.split(":")
            if len(parts) < 2:
                continue

            server_name = parts[0]
            command = parts[1:]
            config.add_mcp_server(server_name, command)
            success = await host.start_mcp_server(
                server_name, config.mcp_servers[server_name]
            )
            if not success:
                click.echo(
                    f"Warning: Failed to connect to MCP server '{server_name}', continuing without it..."
                )

        # Get response with AGENT.md prepended to first message
        messages = [{"role": "user", "content": message}]
        enhanced_messages = host._prepend_agent_md_to_first_message(
            messages, is_first_message=True
        )
        response = await host.chat_completion(enhanced_messages, stream=False)

        click.echo(response)

    finally:
        if "host" in locals():
            if hasattr(host.shutdown, "__call__") and asyncio.iscoroutinefunction(
                host.shutdown
            ):
                await host.shutdown()
            else:
                host.shutdown()


@cli.command()
@click.pass_context
async def compact(ctx):
    """Show conversation token usage and compacting options."""
    click.echo("Compact functionality is only available in interactive chat mode.")
    click.echo("Use 'python agent.py chat' and then '/tokens' or '/compact' commands.")


@cli.command("execute-task")
@click.argument("task_file_path")
def execute_task_command(task_file_path):
    """Execute a task from a task file (used for subprocess execution)."""
    import asyncio

    asyncio.run(execute_task_subprocess(task_file_path))


async def execute_task_subprocess(task_file_path: str):
    """Execute a task from a JSON file in subprocess mode."""
    try:
        import json
        import os
        import time

        # Load task data from file
        if not os.path.exists(task_file_path):
            print(f"Error: Task file not found: {task_file_path}")
            return

        with open(task_file_path, "r") as f:
            task_data = json.load(f)

        task_id = task_data.get("task_id", "unknown")
        description = task_data.get("description", "")
        task_prompt = task_data.get("prompt", "")
        comm_port = task_data.get("comm_port")

        print(f"🤖 [SUBAGENT {task_id}] Starting task: {description}")

        # Connect to parent for tool execution forwarding
        comm_socket = None
        if comm_port:
            try:
                import socket

                comm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                comm_socket.connect(("localhost", comm_port))
                print(
                    f"🤖 [SUBAGENT {task_id}] Connected to parent for tool forwarding"
                )
            except Exception as e:
                print(
                    f"🤖 [SUBAGENT {task_id}] Warning: Could not connect to parent: {e}"
                )
                comm_socket = None

        # Load configuration
        config = load_config()

        # Create appropriate host instance with subagent flag and communication socket
        if hasattr(config, "deepseek_model") and config.deepseek_model == "gemini":
            from mcp_gemini_host import MCPGeminiHost

            subagent = MCPGeminiHost(config, is_subagent=True)
            print(
                f"🤖 [SUBAGENT {task_id}] Created Gemini subagent with is_subagent=True"
            )
        else:
            from mcp_deepseek_host import MCPDeepseekHost

            subagent = MCPDeepseekHost(config, is_subagent=True)
            print(
                f"🤖 [SUBAGENT {task_id}] Created DeepSeek subagent with is_subagent=True"
            )

        # Set communication socket for tool forwarding
        if comm_socket:
            subagent.comm_socket = comm_socket
            print(
                f"🤖 [SUBAGENT {task_id}] Communication socket configured for tool forwarding"
            )
        else:
            print(
                f"🤖 [SUBAGENT {task_id}] WARNING: No communication socket - tools will execute locally"
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
        subagent.permission_manager = permission_manager

        # Set up input handler for subagent to enable permission prompts
        subagent._input_handler = InterruptibleInput()

        print(
            f"🤖 [SUBAGENT {task_id}] Tool permission manager and input handler configured"
        )

        # Connect to MCP servers
        for server_name, server_config in config.mcp_servers.items():
            try:
                await subagent.start_mcp_server(server_name, server_config)
            except Exception as e:
                print(
                    f"🤖 [SUBAGENT {task_id}] Warning: Failed to connect to MCP server {server_name}: {e}"
                )

        # Execute the task
        messages = [{"role": "user", "content": task_prompt}]
        tools_list = subagent.convert_tools_to_llm_format()

        print(
            f"🤖 [SUBAGENT {task_id}] Executing task with {len(tools_list)} tools available..."
        )

        # Get response from subagent
        response = await subagent.generate_response(messages, tools_list)

        # Handle streaming response
        if hasattr(response, "__aiter__"):
            full_response = ""
            async for chunk in response:
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_response += chunk
            response = full_response
        else:
            print(response)

        # Clean up connections
        await subagent.shutdown()

        # Extract the final response for summary
        final_response = response if isinstance(response, str) else str(response)

        # Write result to a result file for the parent to collect
        result_file_path = task_file_path.replace(".json", "_result.json")
        result_data = {
            "task_id": task_id,
            "description": description,
            "status": "completed",
            "result": final_response,
            "timestamp": time.time(),
        }

        with open(result_file_path, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"\n🤖 [SUBAGENT {task_id}] Task completed successfully")

    except Exception as e:
        print(f"🤖 [SUBAGENT ERROR] Failed to execute task: {e}")
        import traceback

        traceback.print_exc()


@cli.group()
def mcp():
    """Manage MCP servers."""
    pass


@cli.group()
def sessions():
    """Manage conversation sessions."""
    pass


@sessions.command("list")
def list_sessions():
    """List recent conversation sessions."""
    session_manager = SessionManager()
    sessions_list = session_manager.list_sessions()

    if not sessions_list:
        click.echo("No sessions found.")
        return

    click.echo("Recent sessions:")
    for i, session in enumerate(sessions_list, 1):
        created = session["created_at"][:19].replace("T", " ")
        session_id = session["session_id"]
        message_count = session["message_count"]
        first_message = session.get("first_message", "")

        click.echo(f"{i}. {session_id[:8]}... ({message_count} messages) - {created}")
        if first_message:
            click.echo(f'   "{first_message}"')


@sessions.command()
@click.argument("session_id")
def show(session_id):
    """Show details of a specific session."""
    session_manager = SessionManager()
    summary = session_manager.get_session_summary(session_id)

    if not summary:
        click.echo(f"Session {session_id} not found.")
        return

    click.echo(f"Session: {summary['session_id']}")
    click.echo(f"Created: {summary['created_at']}")
    click.echo(f"Last Updated: {summary['last_updated']}")
    click.echo(f"Messages: {summary['message_count']}")
    if summary["first_message"]:
        click.echo(f"First Message: \"{summary['first_message']}\"")


@sessions.command()
@click.argument("session_id")
@click.confirmation_option(prompt="Are you sure you want to delete this session?")
def delete(session_id):
    """Delete a conversation session."""
    session_manager = SessionManager()

    if session_manager.delete_session(session_id):
        click.echo(f"Deleted session: {session_id}")
    else:
        click.echo(f"Session {session_id} not found.")


@sessions.command()
@click.confirmation_option(prompt="Are you sure you want to delete all sessions?")
def clear():
    """Clear all conversation sessions."""
    session_manager = SessionManager()
    session_manager.clear_all_sessions()
    click.echo("All sessions cleared.")


@mcp.command()
@click.argument("server_spec")
@click.option("--env", multiple=True, help="Environment variable (format: KEY=VALUE)")
def add(server_spec, env):
    """Add a persistent MCP server configuration.

    Format: name:command:arg1:arg2:...

    Examples:
        python agent.py mcp add digitalocean:node:/path/to/digitalocean-mcp/dist/index.js
        python agent.py mcp add filesystem:python:-m:mcp.server.stdio:filesystem:--root:.
    """
    try:
        config = load_config()

        # Parse server specification
        parts = server_spec.split(":")
        if len(parts) < 2:
            click.echo(
                "❌ Invalid server specification. Format: name:command:arg1:arg2:..."
            )
            return

        name = parts[0]
        command = parts[1]
        args = parts[2:] if len(parts) > 2 else []

        # Parse environment variables
        env_dict = {}
        for env_var in env:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                env_dict[key] = value
            else:
                click.echo(f"Warning: Invalid environment variable format: {env_var}")

        # Add the server
        config.add_mcp_server(name, [command], args, env_dict)
        config.save_mcp_servers()

        click.echo(f"✅ Added MCP server '{name}'")
        click.echo(f"   Command: {command} {' '.join(args)}")
        if env_dict:
            click.echo(f"   Environment: {env_dict}")

    except Exception as e:
        click.echo(f"❌ Error adding MCP server: {e}")


@mcp.command("list")
def list_mcp_servers():
    """List all configured MCP servers."""
    try:
        config = load_config()

        if not config.mcp_servers:
            click.echo("No MCP servers configured.")
            click.echo(
                "Add a server with: python agent.py mcp add <name:command:args...>"
            )
            return

        click.echo("Configured MCP servers:")
        click.echo()

        for name, server_config in config.mcp_servers.items():
            click.echo(f"📡 {name}")
            click.echo(
                f"   Command: {' '.join(server_config.command + server_config.args)}"
            )
            if server_config.env:
                click.echo(f"   Environment: {server_config.env}")
            click.echo()

    except Exception as e:
        click.echo(f"❌ Error listing MCP servers: {e}")


@mcp.command()
@click.argument("name")
def remove(name):
    """Remove a persistent MCP server configuration."""
    try:
        config = load_config()

        if config.remove_mcp_server(name):
            config.save_mcp_servers()
            click.echo(f"✅ Removed MCP server '{name}'")
        else:
            click.echo(f"❌ MCP server '{name}' not found")

    except Exception as e:
        click.echo(f"❌ Error removing MCP server: {e}")


async def handle_streaming_json_chat(
    server, input_format, output_format, session_manager=None
):
    """Handle streaming JSON chat mode compatible with Claude Code."""
    import os

    # Initialize streaming JSON handler
    handler = StreamingJSONHandler()

    # Load configuration
    config = load_config()

    # Create host
    if config.deepseek_model == "gemini":
        if not config.gemini_api_key:
            click.echo("Error: GEMINI_API_KEY not set.")
            return
        from mcp_gemini_host import MCPGeminiHost

        host = MCPGeminiHost(config)
        model_name = config.gemini_model
    else:
        if not config.deepseek_api_key:
            click.echo("Error: DEEPSEEK_API_KEY not set.")
            return
        from mcp_deepseek_host import MCPDeepseekHost

        host = MCPDeepseekHost(config)
        model_name = config.deepseek_model

    # Connect to servers
    for server_spec in server:
        parts = server_spec.split(":")
        if len(parts) < 2:
            continue
        server_name = parts[0]
        command = parts[1:]
        config.add_mcp_server(server_name, command)
        success = await host.start_mcp_server(
            server_name, config.mcp_servers[server_name]
        )
        if not success:
            click.echo(f"Warning: Failed to connect to MCP server '{server_name}'")

    # Get available tools
    tool_names = list(host.available_tools.keys())

    # Send system init
    if output_format == "stream-json":
        handler.send_system_init(
            cwd=os.getcwd(),
            tools=tool_names,
            mcp_servers=list(config.mcp_servers.keys()),
            model=model_name,
        )

    # Main chat loop
    try:
        if input_format == "stream-json":
            # Read JSON input from stdin
            while True:
                input_data = handler.read_input_json()
                if input_data is None:
                    break

                if input_data.get("type") == "user":
                    message_content = input_data.get("message", {}).get("content", "")
                    if isinstance(message_content, list) and len(message_content) > 0:
                        # Extract text from message content
                        text_content = ""
                        for content in message_content:
                            if content.get("type") == "text":
                                text_content = content.get("text", "")
                                break

                        if text_content:
                            # Process message
                            messages = [{"role": "user", "content": text_content}]

                            if output_format == "stream-json":
                                # Stream response in JSON format
                                await stream_json_response(host, handler, messages)
                            else:
                                # Regular text response
                                response = await host.chat_completion(
                                    messages, stream=False
                                )
                                click.echo(response)
        else:
            # Regular interactive mode but with JSON output
            input_handler = InterruptibleInput()
            messages = []

            while True:
                user_input = await input_handler.get_input("You: ")
                if not user_input or user_input.lower() in ["quit", "exit"]:
                    break

                messages.append({"role": "user", "content": user_input})

                if output_format == "stream-json":
                    await stream_json_response(host, handler, messages)
                else:
                    response = await host.chat_completion(messages, stream=False)
                    click.echo(f"Assistant: {response}")
                    messages.append({"role": "assistant", "content": response})

    finally:
        await host.shutdown()


async def stream_json_response(host, handler, messages):
    """Stream response in JSON format."""
    # Generate response and stream it
    response = await host.chat_completion(messages, stream=True)

    if hasattr(response, "__aiter__"):
        # Handle streaming response
        full_response = ""
        async for chunk in response:
            if isinstance(chunk, str):
                full_response += chunk
                # Send partial text updates
                handler.send_assistant_text(chunk)

        # Send final response
        if full_response:
            handler.send_assistant_text(full_response)
    else:
        # Handle non-streaming response
        handler.send_assistant_text(str(response))


async def handle_text_chat(
    server,
    session_manager,
    messages,
    allowed_tools=None,
    disallowed_tools=None,
    auto_approve_tools=False,
):
    """Handle standard text-based chat mode."""
    # Load configuration
    config = load_config()

    # Check if Gemini backend should be used
    if config.deepseek_model == "gemini":
        if not config.gemini_api_key:
            click.echo(
                "Error: GEMINI_API_KEY not set. Run 'init' command first and update .env file."
            )
            return

        # Import and create Gemini host
        from mcp_gemini_host import MCPGeminiHost

        host = MCPGeminiHost(config)
        click.echo(f"Using model: {config.gemini_model}")
        click.echo(f"Temperature: {config.gemini_temperature}")
    else:
        if not config.deepseek_api_key:
            click.echo(
                "Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file."
            )
            return

        # Create Deepseek host with new subagent system
        from mcp_deepseek_host import MCPDeepseekHost

        host = MCPDeepseekHost(config)
        click.echo(f"Using model: {config.deepseek_model}")
        click.echo(f"Temperature: {config.deepseek_temperature}")

    # Initialize tool permission manager
    from cli_agent.core.tool_permissions import (
        ToolPermissionConfig,
        ToolPermissionManager,
    )

    # Merge CLI args with config
    final_allowed_tools = list(config.allowed_tools) + (allowed_tools or [])
    final_disallowed_tools = list(config.disallowed_tools) + (disallowed_tools or [])
    final_auto_approve = config.auto_approve_tools or auto_approve_tools

    permission_config = ToolPermissionConfig(
        allowed_tools=final_allowed_tools,
        disallowed_tools=final_disallowed_tools,
        auto_approve_session=final_auto_approve,
    )
    permission_manager = ToolPermissionManager(permission_config)

    # Set the permission manager on the host
    host.permission_manager = permission_manager

    # Connect to additional MCP servers specified via --server option
    for server_spec in server:
        parts = server_spec.split(":")
        if len(parts) < 2:
            click.echo(f"Invalid server spec: {server_spec}")
            continue

        server_name = parts[0]
        command = parts[1:]

        config.add_mcp_server(server_name, command)

    # Connect to all configured MCP servers (persistent + command-line)
    for server_name, server_config in config.mcp_servers.items():
        click.echo(f"Starting MCP server: {server_name}")
        success = await host.start_mcp_server(server_name, server_config)
        if not success:
            click.echo(f"Failed to start server: {server_name}")
        else:
            click.echo(f"✅ Connected to MCP server: {server_name}")

    # Start interactive chat with host reloading support
    input_handler = InterruptibleInput()

    try:
        while True:
            chat_result = await host.interactive_chat(input_handler, messages)

            # Save session after each interaction
            if messages:
                for msg in messages:
                    if msg not in [m for m in session_manager.get_messages()]:
                        session_manager.add_message(msg)

            # Check if we need to reload the host
            if isinstance(chat_result, dict) and "reload_host" in chat_result:
                messages = chat_result.get("messages", [])
                # Save updated messages to session
                session_manager.current_messages = messages
                session_manager._save_current_session()
                reload_type = chat_result["reload_host"]

                # Shutdown current host
                await host.shutdown()

                # Reload config and create new host
                config = load_config()

                if reload_type == "gemini":
                    if not config.gemini_api_key:
                        click.echo(
                            "Error: GEMINI_API_KEY not set. Cannot switch to Gemini."
                        )
                        break
                    from mcp_gemini_host import MCPGeminiHost

                    host = MCPGeminiHost(config)
                    click.echo(f"Switched to model: {config.gemini_model}")
                    click.echo(f"Temperature: {config.gemini_temperature}")
                else:  # deepseek
                    if not config.deepseek_api_key:
                        click.echo(
                            "Error: DEEPSEEK_API_KEY not set. Cannot switch to DeepSeek."
                        )
                        break
                    from mcp_deepseek_host import MCPDeepseekHost

                    host = MCPDeepseekHost(config)
                    click.echo(f"Switched to model: {config.deepseek_model}")
                    click.echo(f"Temperature: {config.deepseek_temperature}")

                # Reconnect to MCP servers
                for server_name, server_config in config.mcp_servers.items():
                    click.echo(f"Reconnecting to MCP server: {server_name}")
                    success = await host.start_mcp_server(server_name, server_config)
                    if success:
                        click.echo(f"✅ Reconnected to MCP server: {server_name}")
                    else:
                        click.echo(
                            f"⚠️  Failed to reconnect to MCP server: {server_name}"
                        )

                # Continue with the same input handler and preserved messages
                print(
                    f"\n🔄 Continuing chat with {len(messages)} preserved messages...\n"
                )
                continue
            else:
                # Normal exit from chat
                break

    finally:
        if "host" in locals():
            if hasattr(host.shutdown, "__call__") and asyncio.iscoroutinefunction(
                host.shutdown
            ):
                await host.shutdown()
            else:
                host.shutdown()


def main():
    """Main entry point."""
    # Store original async callbacks
    original_chat = chat.callback
    original_ask = ask.callback
    original_compact = compact.callback

    # Convert async commands to sync
    def sync_chat(**kwargs):
        asyncio.run(original_chat(**kwargs))

    def sync_ask(**kwargs):
        asyncio.run(original_ask(**kwargs))

    def sync_compact(**kwargs):
        asyncio.run(original_compact(**kwargs))

    # Replace command callbacks
    chat.callback = sync_chat
    ask.callback = sync_ask
    compact.callback = sync_compact

    cli()


if __name__ == "__main__":
    main()
