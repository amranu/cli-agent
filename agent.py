#!/usr/bin/env python3
# This script implements the main command-line interface for the CLI Agent.
"""Main CLI interface for the CLI Agent with modular imports."""

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
from config import HostConfig, load_config, get_config_dir

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

# Configure logging - will be updated by CLI options
logging.basicConfig(
    level=logging.INFO,  # Default level, can be overridden by --debug
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Global instances to avoid repeated loading
_global_config = None
_global_session_manager = None


def _get_global_config():
    """Get or create global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def _get_global_session_manager():
    """Get or create global session manager instance."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager(config=_get_global_config())
    return _global_session_manager


def _display_chat_history(messages, max_messages=5):
    """Display recent chat history when continuing a session."""
    if not messages:
        return

    click.echo("\n" + "=" * 60)
    click.echo("📜 Recent Chat History")
    click.echo("=" * 60)

    # Show last few messages
    recent_messages = (
        messages[-max_messages:] if len(messages) > max_messages else messages
    )

    if len(messages) > max_messages:
        click.echo(f"... (showing last {max_messages} of {len(messages)} messages)")
        click.echo()

    for i, msg in enumerate(recent_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            click.echo(f"👤 User:")
            # Truncate very long messages
            if len(content) > 200:
                content = content[:200] + "..."
            click.echo(f"   {content}")
        elif role == "assistant":
            click.echo(f"🤖 Assistant:")
            # Truncate very long messages
            if len(content) > 300:
                content = content[:300] + "..."
            click.echo(f"   {content}")

        # Add spacing between messages
        if i < len(recent_messages) - 1:
            click.echo()

    click.echo("=" * 60)
    click.echo("Continuing conversation...\n")


# Suppress noisy third-party library logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("FastMCP.fastmcp.server.server").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def create_host(
    config: HostConfig, is_subagent: bool = False, provider_model: Optional[str] = None
) -> BaseMCPAgent:
    """Create host using provider-model architecture.

    Args:
        config: Host configuration
        is_subagent: Whether this is a subagent instance
        provider_model: Optional provider:model string override

    Returns:
        BaseMCPAgent instance using provider-model architecture
    """
    # Use new provider-model architecture
    if provider_model:
        host = config.create_host_from_provider_model(provider_model)
    else:
        host = config.create_host_from_provider_model()

    # Set subagent flag if needed
    if is_subagent:
        host.is_subagent = True

    logger.info(
        f"Created provider-model host: {getattr(config, 'default_provider_model', 'default')}"
    )
    return host


# CLI functionality
@click.group()
@click.option(
    "--config-file",
    default=None,
    help="Path to the configuration file (default: ~/.mcp/config.json)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging (shows detailed tool execution info)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (shows more detailed info)",
)
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=lambda ctx, param, value: ctx.exit(click.echo("mcp-cli-agent 1.0.0")) if value else None,
    help="Show version and exit",
)
@click.pass_context
def cli(ctx, config_file, debug, verbose):
    """CLI Agent - Run AI models with MCP tool integration."""
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file

    # Update logging level based on options
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Verbose logging enabled")
    else:
        # Default: suppress most logging except warnings and errors
        logging.getLogger().setLevel(logging.WARNING)


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
    # Save to persistent config instead of .env
    config.save_persistent_config()


@cli.command("switch-reason")
@click.pass_context
def switch_reason(ctx):
    """Switch the model to deepseek-reasoner."""
    config = load_config()
    config.deepseek_model = "deepseek-reasoner"
    click.echo(f"Model switched to: {config.deepseek_model}")
    # Save to persistent config instead of .env
    config.save_persistent_config()


@cli.command("switch-gemini")
@click.pass_context
def switch_gemini(ctx):
    """Switch to use Gemini Flash 2.5 as the backend model."""
    config = load_config()
    # Set Gemini Flash as the model and switch backend
    config.deepseek_model = "gemini"  # Use this as a marker
    config.gemini_model = "gemini-2.5-flash"
    click.echo(f"Backend switched to: Gemini Flash 2.5 ({config.gemini_model})")
    # Save to persistent config instead of .env
    config.save_persistent_config()


@cli.command("switch-gemini-pro")
@click.pass_context
def switch_gemini_pro(ctx):
    """Switch to use Gemini Pro 2.5 as the backend model."""
    config = load_config()
    # Set Gemini Pro as the model and switch backend
    config.deepseek_model = "gemini"  # Use this as a marker
    config.gemini_model = "gemini-2.5-pro"
    click.echo(f"Backend switched to: Gemini Pro 2.5 ({config.gemini_model})")
    # Save to persistent config instead of .env
    config.save_persistent_config()


@cli.command()
@click.argument("provider_model")
def switch(provider_model):
    """Switch to a specific provider:model combination.

    Examples:
        agent switch anthropic:claude-3.5-sonnet
        agent switch openai:gpt-4-turbo-preview
        agent switch deepseek:deepseek-chat
        agent switch gemini:gemini-2.5-flash
    """
    config = load_config()

    try:
        # Validate the provider:model format
        if ":" not in provider_model:
            click.echo(
                f"Error: Invalid format. Use 'provider:model' format (e.g., 'anthropic:claude-3.5-sonnet')"
            )
            return

        # Test that the provider-model combination is valid by trying to create a host
        test_host = config.create_host_from_provider_model(provider_model)

        # If successful, update the config
        config.default_provider_model = provider_model
        config.save_persistent_config()

        provider, model = provider_model.split(":", 1)
        click.echo(f"Model switched to: {model} via {provider} provider")
        click.echo(
            f"Configuration saved. Use 'agent chat' to start chatting with the new model."
        )

    except Exception as e:
        click.echo(f"Error switching to {provider_model}: {e}")
        click.echo("Available provider:model combinations:")
        available = config.get_available_provider_models()
        for provider, models in available.items():
            for model in models:
                click.echo(f"  {provider}:{model}")


@cli.command()
@click.option(
    "--model",
    help="Provider:model combination to use (e.g., 'anthropic:claude-3.5-sonnet')",
)
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
@click.option(
    "--event-driven",
    is_flag=True,
    help="Use event-driven display system with JSON output",
)
@click.pass_context
async def chat(
    ctx,
    model,
    server,
    input_format,
    output_format,
    continue_session,
    resume_session_id,
    allowed_tools,
    disallowed_tools,
    auto_approve_tools,
    event_driven,
):
    """Start interactive chat session."""
    try:
        # Set stream-json mode flag IMMEDIATELY at the start
        if output_format == "stream-json":
            import os
            import logging
            os.environ["STREAM_JSON_MODE"] = "true"
            # Suppress all non-critical logging for clean JSON output (unless debug is enabled)
            current_level = logging.getLogger().getEffectiveLevel()
            if current_level != logging.DEBUG:  # Don't suppress if debug mode is on
                logging.getLogger().setLevel(logging.WARNING)
                logging.getLogger("cli_agent").setLevel(logging.WARNING)
                logging.getLogger("asyncio").setLevel(logging.WARNING)
            
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

        # Use global session manager instance
        session_manager = _get_global_session_manager()

        # Handle session resumption
        if continue_session:
            session_id = session_manager.continue_last_session()
            if session_id:
                if output_format != "stream-json":
                    click.echo(f"Continuing session: {session_id[:8]}...")
                messages = session_manager.get_messages()
                if output_format != "stream-json":
                    click.echo(f"Restored {len(messages)} messages")
                    # Display recent chat history for context
                    _display_chat_history(messages)
            else:
                if output_format != "stream-json":
                    click.echo("No previous session found, starting new conversation")
                session_id = session_manager.create_new_session()
                messages = []
        elif resume_session_id:
            session_id = session_manager.resume_session(resume_session_id)
            if session_id:
                if output_format != "stream-json":
                    click.echo(f"Resumed session: {session_id[:8]}...")
                messages = session_manager.get_messages()
                if output_format != "stream-json":
                    click.echo(f"Restored {len(messages)} messages")
                    # Display recent chat history for context
                    _display_chat_history(messages)
            else:
                if output_format != "stream-json":
                    click.echo(
                        f"Session {resume_session_id} not found, starting new conversation"
                    )
                session_id = session_manager.create_new_session()
                messages = []
        else:
            # Start new session
            session_id = session_manager.create_new_session()
            messages = []
            if output_format != "stream-json":
                click.echo(f"Started new session: {session_id[:8]}...")

        # Initialize todo file for this session
        import json
        import os

        config_dir = get_config_dir()
        todos_dir = config_dir / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        todo_file = todos_dir / f"{session_id}.json"
        with open(todo_file, "w") as f:
            json.dump([], f)

        # Handle mixed modes or invalid combinations FIRST
        # Allow text input with stream-json output (but not the reverse)
        if input_format != output_format and not (
            input_format == "text" and output_format == "stream-json"
        ):
            click.echo(
                "Error: Invalid format combination. Supported: text->text, stream-json->stream-json, or text->stream-json."
            )
            return


        # Handle streaming JSON mode
        if input_format == "stream-json" or output_format == "stream-json":
            return await handle_streaming_json_chat(
                model, server, input_format, output_format, session_manager
            )

        # Handle text mode (default behavior)
        if input_format == "text" and output_format == "text":
            return await handle_text_chat(
                model,
                server,
                session_manager,
                messages,
                session_id,
                parsed_allowed_tools,
                parsed_disallowed_tools,
                auto_approve_tools,
                event_driven,
            )

    except KeyboardInterrupt:
        # Check if this is the second interrupt (user wants to exit)
        from cli_agent.core.global_interrupt import get_global_interrupt_manager

        interrupt_manager = get_global_interrupt_manager()

        if interrupt_manager._interrupt_count >= 2:
            # Second interrupt - user wants to exit, re-raise to exit the CLI
            click.echo("\n👋 Exiting...")
            raise
        else:
            # First interrupt - just return to exit this chat session
            click.echo("\n👋 Chat interrupted by user")
            return


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
    "--model",
    help="Provider-model to use (e.g. 'deepseek:deepseek-chat', 'anthropic:claude-3.5-sonnet')",
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
    model,
    allowed_tools,
    disallowed_tools,
    auto_approve_tools,
):
    """Ask a single question."""
    try:
        # Validate format combinations
        # Allow text input with stream-json output (but not the reverse)
        if input_format != output_format and not (
            input_format == "text" and output_format == "stream-json"
        ):
            click.echo(
                "Error: Invalid format combination. Supported: text->text, stream-json->stream-json, or text->stream-json."
            )
            return

        # Handle streaming JSON mode for ask command
        if input_format == "stream-json" or output_format == "stream-json":
            click.echo(
                "Error: Streaming JSON mode not supported for single question mode. Use 'chat' command instead."
            )
            return

        config = load_config()

        # Create host using helper function
        try:
            host = create_host(config, provider_model=model)
        except Exception as e:
            click.echo(f"Error creating host: {e}")
            return

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
        enhanced_messages = (
            host.system_prompt_builder.enhance_first_message_with_agent_md(messages)
        )
        response = await host.generate_response(enhanced_messages, stream=False)

        click.echo(response)

    finally:
        if "host" in locals():
            host_instance = locals()["host"]
            if hasattr(
                host_instance.shutdown, "__call__"
            ) and asyncio.iscoroutinefunction(host_instance.shutdown):
                await host_instance.shutdown()
            else:
                host_instance.shutdown()


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
        subagent = create_host(config, is_subagent=True)
        print(
            f"🤖 [SUBAGENT {task_id}] Created subagent with provider-model: {config.default_provider_model}"
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

        # Set up custom input handler for subagent that connects to main terminal
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
                # For subagents in agent.py, permission requests should be auto-approved
                # since this is the interactive chat interface where user has already
                # initiated the subagent. Return default approval.
                logger.info(
                    f"Subagent {self.subagent_context} permission request: {prompt_text}"
                )
                logger.info("Auto-approving for interactive subagent")
                return "a"  # Auto-approve for session

            def get_multiline_input(self, prompt: str) -> Optional[str]:
                # For multiline input, use the same auto-approval logic
                return "a"

            def check_for_interrupt(self) -> bool:
                return False

        subagent._input_handler = SubagentInputHandler(task_id)

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

        print(f"🤖 [SUBAGENT {task_id}] Executing task...")

        # Use the interactive chat interface to handle tool execution properly
        # This ensures proper conversation flow and tool execution
        from cli_agent.core.input_handler import InterruptibleInput

        # Create a mock input handler that provides the task prompt
        class MockInputHandler:
            def __init__(self, task_prompt):
                self.task_prompt = task_prompt
                self.used = False
                self.interrupted = False

            def get_multiline_input(self, prompt):
                if not self.used:
                    self.used = True
                    return self.task_prompt
                return None  # EOF after first input

            def check_for_interrupt(self):
                return False

        mock_input = MockInputHandler(task_prompt)

        # Use interactive chat which handles tool execution properly
        final_messages = await subagent.interactive_chat(
            mock_input, existing_messages=[]
        )

        # Extract the final response from the conversation
        if final_messages and len(final_messages) > 0:
            # Get the last assistant message
            for msg in reversed(final_messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    response = msg["content"]
                    break
            else:
                response = "Task completed successfully"
        else:
            response = "Task completed successfully"

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


@mcp.command()
@click.option("--port", default=3000, help="Port to serve on")
@click.option("--host", default="localhost", help="Host to serve on")
@click.option("--stdio", is_flag=True, help="Use stdio transport instead of TCP")
def serve(port, host, stdio):
    """Serve all available models over MCP protocol."""
    try:
        from cli_agent.mcp.model_server import create_model_server

        server = create_model_server()

        if stdio:
            # Use stdio transport for MCP clients that expect it
            click.echo("Starting MCP model server on stdio transport", err=True)
            asyncio.run(server.run_stdio_async())
        else:
            # Use TCP transport for web-based or network clients
            click.echo(f"Starting MCP model server on {host}:{port}", err=True)
            asyncio.run(server.run_async(host=host, port=port))

    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Please install FastMCP with: pip install fastmcp", err=True)
    except Exception as e:
        click.echo(f"Error starting MCP server: {e}", err=True)


@mcp.command("list-models")
def list_models():
    """List all available models that would be exposed via MCP."""
    try:
        from cli_agent.mcp.model_server import normalize_model_name

        config = load_config()
        available = config.get_available_provider_models()

        if not available:
            click.echo("No models available. Check your API key configuration.")
            return

        click.echo("Available models for MCP server:")
        total_models = 0
        for provider, models in available.items():
            click.echo(f"\n{provider}:")
            for model in models:
                tool_name = f"{provider}_{normalize_model_name(model)}"
                click.echo(f"  - {tool_name} ({provider}:{model})")
                total_models += 1

        click.echo(f"\nTotal: {total_models} models available")

    except Exception as e:
        click.echo(f"Error listing models: {e}", err=True)


@cli.group()
def sessions():
    """Manage conversation sessions."""
    pass


@sessions.command("list")
def list_sessions():
    """List recent conversation sessions."""
    session_manager = _get_global_session_manager()
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
    session_manager = _get_global_session_manager()
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
    session_manager = _get_global_session_manager()

    if session_manager.delete_session(session_id):
        click.echo(f"Deleted session: {session_id}")
    else:
        click.echo(f"Session {session_id} not found.")


@sessions.command()
@click.confirmation_option(prompt="Are you sure you want to delete all sessions?")
def clear():
    """Clear all conversation sessions."""
    session_manager = _get_global_session_manager()
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
    model, server, input_format, output_format, session_manager=None
):
    """Handle streaming JSON chat mode compatible with Claude Code."""
    import os

    # Initialize streaming JSON handler
    handler = StreamingJSONHandler()

    # Load configuration
    config = load_config()

    # Create host using helper function
    try:
        host = create_host(config, provider_model=model)
        # Extract model name from provider-model string
        _, model_name = config.parse_provider_model_string(
            config.default_provider_model
        )
        if not model_name:
            model_name = config.default_provider_model  # Fallback for legacy format
    except Exception as e:
        click.echo(f"Error creating host: {e}")
        return

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

    # Set up DisplayManager for JSON output globally (not just during response streaming)
    if output_format == "stream-json" and hasattr(host, "display_manager"):
        host.display_manager.json_handler = handler

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
            processed_message = False
            while True:
                input_data = handler.read_input_json()
                if input_data is None:
                    # If we've processed at least one message, don't exit immediately
                    # Allow time for streaming response to complete
                    if processed_message:
                        break
                    else:
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
                            processed_message = True

                            if output_format == "stream-json":
                                # Stream response in JSON format with proper tool call/result messages
                                await stream_json_response(
                                    host, handler, messages, model_name
                                )
                            else:
                                # Regular text response
                                response = await host.generate_response(
                                    messages, stream=False
                                )
                                click.echo(response)
        else:
            # Regular interactive mode but with JSON output
            # Check if input is from a pipe or terminal
            if not sys.stdin.isatty():
                # Non-interactive: read from pipe/stdin
                user_input = sys.stdin.read().strip()
                if user_input:
                    messages = [{"role": "user", "content": user_input}]
                    if output_format == "stream-json":
                        await stream_json_response(host, handler, messages, model_name)
                    else:
                        response = await host.generate_response(messages, stream=False)
                        click.echo(f"Assistant: {response}")
            else:
                # Interactive mode
                input_handler = InterruptibleInput(host)
                messages = []

                while True:
                    user_input = input_handler.get_input("You: ")
                    if not user_input or user_input.lower() in ["quit", "exit"]:
                        break

                    messages.append({"role": "user", "content": user_input})

                    if output_format == "stream-json":
                        await stream_json_response(host, handler, messages, model_name)
                    else:
                        response = await host.generate_response(messages, stream=False)
                        click.echo(f"Assistant: {response}")
                        messages.append({"role": "assistant", "content": response})

    finally:
        # Clean up JSON handler
        if output_format == "stream-json" and hasattr(host, "display_manager"):
            host.display_manager.json_handler = None
            
        # Clean up HTTP clients to prevent "Event loop is closed" errors
        try:
            from cli_agent.utils.http_client import http_client_manager

            await http_client_manager.cleanup_all()
        except Exception as e:
            # Don't fail cleanup if HTTP client cleanup fails
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"HTTP client cleanup failed: {e}")

        await host.shutdown()


async def stream_json_response(host, handler, messages, model_name):
    """Stream response in JSON format using unified event system."""

    # JSON handler is now set globally in handle_streaming_json_chat, no need to set it here
    
    # Start event bus processing if available and not already running
    if (
        hasattr(host, "event_bus")
        and host.event_bus
        and not host.event_bus.is_running
    ):
        await host.event_bus.start_processing()
    
    # Use streaming mode to ensure proper tool result integration and conversation state maintenance
    # This ensures tool calls and tool results are properly added to the conversation
    response = await host.generate_response(messages, stream=True)

    # Note: Text responses are handled by the event system in stream-json mode
    # No need to send directly to handler as events handle all display


async def handle_text_chat(
    model,
    server,
    session_manager,
    messages,
    session_id,
    allowed_tools=None,
    disallowed_tools=None,
    auto_approve_tools=False,
    event_driven=False,
):
    """Handle standard text-based chat mode."""
    # Load configuration
    config = load_config()

    # Create host using helper function
    try:
        host = create_host(config, provider_model=model)
        # Set session ID on host for session-specific todo files
        host._session_id = session_id
        # Display the actual model being used (either from CLI or default)
        actual_model = model or config.default_provider_model or config.get_intelligent_default_provider_model()
        provider_name, model_name = config.parse_provider_model_string(actual_model)
        click.echo(f"Using provider-model: {actual_model}")
        click.echo(f"Provider: {provider_name}, Model: {model_name}")
    except Exception as e:
        click.echo(f"Error creating host: {e}")
        return

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

    # Set up display manager for event-driven mode
    display_manager = None
    if event_driven:
        from cli_agent.core.display_manager import JSONDisplayManager

        # Disable the interactive display manager to avoid dual output
        if hasattr(host, "display_manager"):
            host.display_manager.shutdown()
        # Use JSON-only display manager
        display_manager = JSONDisplayManager(host.event_bus)
    else:
        # Use the standard interactive DisplayManager initialized in host
        display_manager = host.display_manager

    # Start interactive chat with host reloading support
    input_handler = InterruptibleInput(host)

    # Start continuous background interrupt monitoring for the entire chat session
    from cli_agent.core.interrupt_aware_streaming import run_with_interrupt_monitoring

    async def chat_session_loop():
        """Main chat session loop with continuous interrupt monitoring."""
        nonlocal host, messages, display_manager, config

        from cli_agent.core.global_interrupt import get_global_interrupt_manager

        interrupt_manager = get_global_interrupt_manager()

        while True:
            # Check if user interrupted and wants to exit
            if interrupt_manager.is_interrupted():
                if interrupt_manager._interrupt_count >= 2:
                    # Second interrupt - exit entirely
                    import sys

                    sys.exit(0)
                # First interrupt - break out of session loop to prevent restart
                break

            chat_result = await host.interactive_chat(input_handler, messages)

            # Check if user quit the chat or interrupted
            if isinstance(chat_result, dict) and (
                chat_result.get("quit") or chat_result.get("interrupted")
            ):
                # Save final messages before quitting/breaking
                if "messages" in chat_result:
                    messages = chat_result["messages"]
                
                # Save session before exiting
                if messages:
                    session_manager.current_messages = messages
                    session_manager._save_current_session()
                
                break

            # Update messages from the interactive chat result
            if chat_result is not None:
                if isinstance(chat_result, dict) and "reload_host" in chat_result:
                    messages = chat_result.get("messages", [])
                    # Save updated messages to session
                    session_manager.current_messages = messages
                elif isinstance(chat_result, list):
                    # Normal case: chat_result is the updated messages list
                    messages = chat_result

            # Save session after each interaction
            if messages:
                # Replace all messages in session (simple and reliable)
                session_manager.current_messages = messages
                session_manager._save_current_session()

            # Check if we need to reload the host
            if isinstance(chat_result, dict) and "reload_host" in chat_result:
                reload_type = chat_result["reload_host"]

                # Shutdown current host
                await host.shutdown()

                # Reload config and create new host
                config = load_config()

                # Check if reload_type is a provider:model combination or provider-model reload
                if ":" in reload_type or reload_type == "provider-model":
                    # New provider-model format
                    try:
                        if reload_type == "provider-model":
                            # Use the current default_provider_model from config
                            provider_model = config.default_provider_model
                        else:
                            provider_model = reload_type

                        host = create_host(config, provider_model=provider_model)
                        # Preserve session ID for session-specific todo files
                        host._session_id = session_id
                        provider_name, model_name = config.parse_provider_model_string(
                            provider_model
                        )
                        click.echo(f"Switched to: {provider_model}")
                        click.echo(f"Provider: {provider_name}, Model: {model_name}")
                    except Exception as e:
                        click.echo(f"Error switching to {reload_type}: {e}")
                        break
                else:
                    # Legacy model switching
                    if reload_type == "gemini":
                        try:
                            host = create_host(
                                config, provider_model="google:gemini-2.5-flash"
                            )
                            # Preserve session ID for session-specific todo files
                            host._session_id = session_id
                            click.echo("Switched to Google Gemini")
                        except Exception as e:
                            click.echo(f"Error switching to Gemini: {e}")
                            break
                    elif reload_type == "gemini-pro":
                        try:
                            host = create_host(
                                config, provider_model="google:gemini-1.5-pro"
                            )
                            # Preserve session ID for session-specific todo files
                            host._session_id = session_id
                            click.echo("Switched to Google Gemini Pro")
                        except Exception as e:
                            click.echo(f"Error switching to Gemini Pro: {e}")
                            break
                    elif reload_type == "chat":
                        try:
                            host = create_host(
                                config, provider_model="deepseek:deepseek-chat"
                            )
                            # Preserve session ID for session-specific todo files
                            host._session_id = session_id
                            click.echo("Switched to DeepSeek Chat")
                        except Exception as e:
                            click.echo(f"Error switching to DeepSeek Chat: {e}")
                            break
                    elif reload_type == "reason":
                        try:
                            host = create_host(
                                config, provider_model="deepseek:deepseek-reasoner"
                            )
                            # Preserve session ID for session-specific todo files
                            host._session_id = session_id
                            click.echo("Switched to DeepSeek Reasoner")
                        except Exception as e:
                            click.echo(f"Error switching to DeepSeek Reasoner: {e}")
                            break
                    else:
                        click.echo(f"Unknown model type: {reload_type}")
                        break

                # Re-initialize permission manager for new host
                host.permission_manager = permission_manager

                # Reconnect to MCP servers with new host
                for server_name, server_config in config.mcp_servers.items():
                    success = await host.start_mcp_server(server_name, server_config)
                    if not success:
                        click.echo(f"Failed to reconnect to server: {server_name}")

                # Update display manager if needed
                if event_driven:
                    display_manager.shutdown()
                    from cli_agent.core.display_manager import JSONDisplayManager

                    display_manager = JSONDisplayManager(host.event_bus)
                else:
                    display_manager = host.display_manager

    try:
        # Run the chat session loop - let the global interrupt manager handle Ctrl+C
        await chat_session_loop()

    finally:
        # Allow pending events to be processed before shutdown
        await asyncio.sleep(0.1)

        # Clean up HTTP clients to prevent "Event loop is closed" errors
        try:
            from cli_agent.utils.http_client import http_client_manager

            await http_client_manager.cleanup_all()
        except Exception as e:
            # Don't fail cleanup if HTTP client cleanup fails
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"HTTP client cleanup failed: {e}")

        # Clean up display manager
        if display_manager and hasattr(display_manager, "shutdown"):
            display_manager.shutdown()

        if "host" in locals():
            host_instance = locals()["host"]
            if hasattr(
                host_instance.shutdown, "__call__"
            ) and asyncio.iscoroutinefunction(host_instance.shutdown):
                await host_instance.shutdown()
            else:
                host_instance.shutdown()


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
