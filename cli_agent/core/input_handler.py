#!/usr/bin/env python3
"""
Professional input handler using prompt_toolkit for robust terminal interaction.

This module provides the InterruptibleInput class which offers advanced terminal
input handling with support for interruption, multiline input, command history,
and fallback mechanisms for environments where prompt_toolkit is not available.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from cli_agent.core.global_interrupt import get_global_interrupt_manager
from config import get_config_dir

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Suppress WARNING messages during interactive chat
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class InterruptibleInput:
    """Professional input handler using prompt_toolkit for robust terminal interaction.

    This class provides a sophisticated input handling system that supports:
    - Keyboard interruption handling (Ctrl+C and optionally ESC)
    - Multiline input modes
    - Graceful fallback when prompt_toolkit is unavailable
    - Asyncio event loop compatibility
    - Professional terminal interaction patterns
    - Non-blocking interrupt detection for tool execution loops

    Attributes:
        interrupted (bool): Flag indicating if input was interrupted
        _available (bool): Whether prompt_toolkit is available
        _prompt: Reference to prompt_toolkit prompt function
        _patch_stdout: Reference to prompt_toolkit patch_stdout
        _bindings: Key bindings for custom keyboard shortcuts
        _allow_escape_interrupt (bool): Whether ESC key should trigger interruption

    Key Features:
        - check_for_interrupt(): Non-blocking check for ESC/Ctrl+C during operations
        - Used by tool execution loops to allow user interruption between tool calls
        - Prevents getting stuck in long tool execution sequences
    """

    def __init__(self, agent=None):
        """Initialize the InterruptibleInput handler.

        Sets up prompt_toolkit components if available, otherwise prepares
        for fallback to basic input() function. Also initializes command history.
        
        Args:
            agent: Optional reference to the agent for accessing tool names
        """
        self.interrupted = False
        self.immediate_exit_requested = False  # Flag for immediate exit from empty prompt
        self.global_interrupt_manager = get_global_interrupt_manager()
        self.agent = agent  # Store agent reference for tool name completion
        self._current_input_text = ""  # Track current input text for empty prompt detection
        self._waiting_for_input = False  # Track if we're currently waiting for user input
        self._setup_history()
        self._setup_prompt_toolkit()

    def _setup_history(self):
        """Setup command history storage.
        
        Creates a persistent history file in the user's config directory.
        """
        # Create config directory for agent
        self.config_dir = get_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up history file path
        self.history_file = self.config_dir / "command_history.txt"
        
        # Initialize history storage for fallback mode
        self._fallback_history = []
        self._history_position = 0
        
        # Load existing history
        self._load_history()

    def _load_history(self):
        """Load command history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self._fallback_history = [line.strip() for line in f.readlines() if line.strip()]
                # Limit history size to last 1000 commands
                if len(self._fallback_history) > 1000:
                    self._fallback_history = self._fallback_history[-1000:]
                    self._save_history()
        except Exception as e:
            logger.warning(f"Could not load command history: {e}")
            self._fallback_history = []

    def _save_history(self):
        """Save command history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                for command in self._fallback_history:
                    f.write(f"{command}\n")
        except Exception as e:
            logger.warning(f"Could not save command history: {e}")

    def _add_to_history(self, command: str):
        """Add a command to history (avoiding duplicates of last command)."""
        if command and command.strip():
            command = command.strip()
            # Don't add duplicate of the last command
            if not self._fallback_history or self._fallback_history[-1] != command:
                self._fallback_history.append(command)
                # Limit history size
                if len(self._fallback_history) > 1000:
                    self._fallback_history = self._fallback_history[-1000:]
                self._save_history()

    def _setup_prompt_toolkit(self):
        """Setup prompt_toolkit components.

        Attempts to import and configure prompt_toolkit for advanced terminal
        interaction. If import fails, sets up for fallback mode.
        """
        try:
            import asyncio

            from prompt_toolkit import prompt
            from prompt_toolkit.completion import Completer, Completion, PathCompleter
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.keys import Keys
            from prompt_toolkit.patch_stdout import patch_stdout

            class AdvancedCompleter(Completer):
                def __init__(self, agent=None):
                    self.agent = agent

                def get_completions(self, document, complete_event):
                    text = document.text_before_cursor
                    
                    # Handle file path completion after @
                    if "@" in text:
                        yield from self._get_file_completions(text)
                    
                    # Handle slash command completion
                    elif text.strip().startswith("/"):
                        yield from self._get_slash_command_completions(text)

                def _get_file_completions(self, text):
                    """Get file path completions for @ references."""
                    idx = text.rfind("@")
                    if idx == -1:
                        return
                    fragment = text[idx + 1 :]

                    # Allow both absolute paths (/...) and relative paths (anything else)
                    # Skip empty fragments
                    if not fragment:
                        # For empty fragment after @, show current directory contents
                        fragment = "./"

                    # Use glob to get file completions directly
                    import glob
                    import os

                    # Handle the pattern - if it ends with incomplete name, add *
                    pattern = fragment
                    if not pattern.endswith("/") and not pattern.endswith("*"):
                        pattern += "*"

                    try:
                        matches = glob.glob(pattern)
                        # Also try with expanduser for ~ paths
                        if pattern.startswith("~/"):
                            expanded = os.path.expanduser(pattern)
                            matches.extend(glob.glob(expanded))

                        for match in matches:
                            if os.path.isdir(match) and not match.endswith("/"):
                                match += "/"

                            # The completion text should be just the match
                            # start_position should replace from @ to cursor
                            yield Completion(
                                "@" + match, start_position=-(len(text) - idx)
                            )
                    except (OSError, Exception):
                        # Fallback - no completions
                        pass

                def _get_slash_command_completions(self, text):
                    """Get slash command completions including tool names for permissions."""
                    # Parse the command
                    parts = text.strip().split()
                    if not parts:
                        return

                    command = parts[0]  # e.g. "/permissions"
                    
                    # Handle /permissions command with tool completion
                    if command == "/permissions" and len(parts) >= 2:
                        if len(parts) == 2:
                            # Check if we have trailing space (indicates ready for next argument)
                            if text.endswith(" ") and parts[1] in ["allow", "deny"]:
                                # Show all tool names for allow/deny after space
                                tool_names = self._get_available_tool_names()
                                for tool_name in tool_names:
                                    yield Completion(tool_name, start_position=0)
                            else:
                                # Complete allow/deny/auto/reset
                                subcommands = ["allow", "deny", "auto", "reset"]
                                current = parts[1]
                                for sub in subcommands:
                                    if sub.startswith(current):
                                        yield Completion(sub, start_position=-len(current))
                        elif len(parts) == 3 and parts[1] in ["allow", "deny"]:
                            # Complete tool names for allow/deny
                            current_tool = parts[2]
                            tool_names = self._get_available_tool_names()
                            for tool_name in tool_names:
                                if tool_name.startswith(current_tool):
                                    yield Completion(tool_name, start_position=-len(current_tool))
                    
                    # Handle /models command with provider completion
                    elif command == "/models":
                        providers = ["anthropic", "openai", "deepseek", "gemini", "openrouter", "ollama"]
                        if len(parts) == 1 and text.endswith(" "):
                            # Show all providers after space: "/models "
                            for provider in providers:
                                yield Completion(provider, start_position=0)
                        elif len(parts) == 2:
                            # Complete partial provider name: "/models anth"
                            current = parts[1]
                            for provider in providers:
                                if provider.startswith(current):
                                    yield Completion(provider, start_position=-len(current))
                    
                    # Handle /switch command with provider:model completion
                    elif command == "/switch":
                        if len(parts) == 1 and text.endswith(" "):
                            # Show all provider:model combinations after space: "/switch "
                            provider_models = self._get_available_provider_models()
                            for provider, models in provider_models.items():
                                for model in models:
                                    provider_model = f"{provider}:{model}"
                                    yield Completion(provider_model, start_position=0)
                        elif len(parts) == 2:
                            # Complete partial provider:model: "/switch anth" or "/switch anthropic:cl"
                            current = parts[1]
                            provider_models = self._get_available_provider_models()
                            
                            if ":" in current:
                                # Completing model part: "/switch anthropic:cl"
                                provider_part, model_part = current.split(":", 1)
                                if provider_part in provider_models:
                                    for model in provider_models[provider_part]:
                                        if model.startswith(model_part):
                                            provider_model = f"{provider_part}:{model}"
                                            yield Completion(provider_model, start_position=-len(current))
                            else:
                                # Completing provider part or showing provider:model combinations
                                for provider, models in provider_models.items():
                                    if provider.startswith(current):
                                        # Complete just the provider part with colon
                                        yield Completion(f"{provider}:", start_position=-len(current))
                                        # Also show specific models for this provider
                                        for model in models[:3]:  # Limit to first 3 models to avoid clutter
                                            provider_model = f"{provider}:{model}"
                                            if provider_model.startswith(current):
                                                yield Completion(provider_model, start_position=-len(current))
                    
                    # Handle other slash commands (basic completion)
                    elif len(parts) == 1 and not text.endswith(" "):
                        # Only complete slash command names if not followed by space
                        commands = [
                            "/help", "/clear", "/compact", "/tokens", "/tools", "/permissions",
                            "/model", "/models", "/switch", "/provider", "/review", "/truncate",
                            "/refresh-models", "/history", "/init", "/quit", "/exit"
                        ]
                        current = command
                        for cmd in commands:
                            if cmd.startswith(current):
                                yield Completion(cmd, start_position=-len(current))

                def _get_available_tool_names(self):
                    """Get list of available tool names for completion."""
                    if not self.agent or not hasattr(self.agent, 'available_tools'):
                        return []
                    
                    tool_names = []
                    for tool_key in self.agent.available_tools.keys():
                        # Add the full tool key (e.g., "builtin:bash_execute")
                        tool_names.append(tool_key)
                        
                        # Also add just the tool name part for convenience
                        if ":" in tool_key:
                            tool_name = tool_key.split(":", 1)[1]
                            tool_names.append(tool_name)
                    
                    # Remove duplicates and sort
                    return sorted(list(set(tool_names)))

                def _get_available_provider_models(self):
                    """Get available provider-model combinations for completion."""
                    if not self.agent or not hasattr(self.agent, 'config'):
                        # Fallback to basic providers and models if no agent config
                        return {
                            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
                            "openai": ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo", "o1-preview"],
                            "deepseek": ["deepseek-chat", "deepseek-reasoner"],
                            "google": ["gemini-2.5-flash", "gemini-1.5-pro"],
                            "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo-preview"],
                            "ollama": ["llama2", "llama3"]
                        }
                    
                    try:
                        # Try to get actual available models from config
                        return self.agent.config.get_available_provider_models()
                    except Exception:
                        # Fallback if config method fails
                        return {
                            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
                            "openai": ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo", "o1-preview"],
                            "deepseek": ["deepseek-chat", "deepseek-reasoner"],
                            "google": ["gemini-2.5-flash", "gemini-1.5-pro"],
                            "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo-preview"],
                            "ollama": ["llama2", "llama3"]
                        }

            self._prompt = prompt
            self._patch_stdout = patch_stdout
            self._available = True

            # Set up command history with prompt_toolkit
            try:
                self._history = FileHistory(str(self.history_file))
                logger.debug(f"Command history initialized: {self.history_file}")
            except Exception as e:
                logger.warning(f"Could not initialize prompt_toolkit history: {e}")
                self._history = None

            # Create key bindings for interruption
            self._bindings = KeyBindings()

            @self._bindings.add(Keys.Escape)
            def handle_escape(event):
                """Handle escape key for interruption when enabled.

                Args:
                    event: The key event from prompt_toolkit
                """
                if getattr(self, "_allow_escape_interrupt", False):
                    self.interrupted = True
                    event.app.exit(exception=KeyboardInterrupt)

            @self._bindings.add(Keys.ControlC)
            def handle_ctrl_c(event):
                """Handle Ctrl+C with buffer content awareness.

                Args:
                    event: The key event from prompt_toolkit
                """
                # Get current buffer text
                current_text = event.app.current_buffer.text
                
                # Check if we have active operations
                has_active_operations = self.global_interrupt_manager.is_interrupted()
                
                # Only exit immediately if prompt is empty AND no operations are running
                if not current_text.strip() and not has_active_operations:
                    # Empty prompt and no operations - exit immediately
                    event.app.exit(exception=KeyboardInterrupt("immediate_exit"))
                else:
                    # Has content or operations are running - clear input and continue
                    event.app.current_buffer.reset()
                    # Raise KeyboardInterrupt to trigger operation cancellation if needed
                    event.app.exit(exception=KeyboardInterrupt("clear_input"))

            self._completer = AdvancedCompleter(self.agent)

        except ImportError:
            self._available = False
            logger.warning("prompt_toolkit not available, falling back to basic input")

    def get_input(
        self,
        prompt_text: str,
        multiline_mode: bool = False,
        allow_escape_interrupt: bool = False,
    ) -> Optional[str]:
        """Get input using prompt_toolkit for professional terminal interaction.

        This method provides sophisticated input handling with support for
        interruption, multiline modes, and asyncio compatibility.

        Args:
            prompt_text (str): The prompt to display to the user
            multiline_mode (bool): If True, requires empty line to send. If False, sends on Enter.
            allow_escape_interrupt (bool): If True, pressing ESC alone will interrupt. If False, ESC is ignored.

        Returns:
            Optional[str]: The user's input string, or None if interrupted or EOF

        Raises:
            None: All exceptions are caught and handled gracefully
        """
        # Register this input handler for empty prompt detection
        self.global_interrupt_manager.set_current_input_handler(self)
        
        # Track that we're waiting for input
        self._waiting_for_input = True
        self._current_input_text = ""
        
        try:
            if not self._available:
                # Fallback to basic input if prompt_toolkit unavailable
                try:
                    result = input(prompt_text)
                    if result:
                        self._add_to_history(result)
                    return result
                except KeyboardInterrupt:
                    # Manually trigger the global interrupt manager since basic input intercepted the signal
                    import os
                    import signal

                    os.kill(
                        os.getpid(), signal.SIGINT
                    )  # Re-send SIGINT to trigger our global handler
                    return None
                except EOFError:
                    # End of input (e.g., pipe closed)
                    self.interrupted = True
                    return None

            # Set up escape interrupt behavior
            self._allow_escape_interrupt = allow_escape_interrupt

            # Check if we're in an asyncio event loop
            import asyncio

            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # We're in an async context, need to run in a thread
                import concurrent.futures
                import threading

                def run_prompt():
                    """Run prompt_toolkit in a separate thread to avoid event loop conflicts.

                    Returns:
                        str: The user's input
                    """
                    return self._prompt(
                        prompt_text,
                        key_bindings=self._bindings,  # Always use our key bindings for Ctrl+C handling
                        multiline=multiline_mode,
                        wrap_lines=True,
                        enable_history_search=True,
                        history=getattr(self, "_history", None),
                        completer=getattr(self, "_completer", None),
                        bottom_toolbar=None,  # Ensure no bottom toolbar
                        reserve_space_for_menu=0,  # Don't reserve space for completion menu
                    )

                # Run the prompt in a thread pool to avoid asyncio conflicts
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_prompt)
                    result = future.result()
                    return result

            except RuntimeError:
                # No event loop running, safe to use prompt_toolkit directly
                result = self._prompt(
                    prompt_text,
                    key_bindings=self._bindings,  # Always use our key bindings for Ctrl+C handling
                    multiline=multiline_mode,
                    wrap_lines=True,
                    enable_history_search=True,
                    history=getattr(self, "_history", None),
                    completer=getattr(self, "_completer", None),
                    bottom_toolbar=None,  # Ensure no bottom toolbar
                    reserve_space_for_menu=0,  # Don't reserve space for completion menu
                )
                return result

        except KeyboardInterrupt as e:
            # Check if this is an immediate exit request
            if str(e) == "immediate_exit":
                # Empty prompt - set immediate exit flag and return None
                self.immediate_exit_requested = True
                return None
            elif str(e) == "clear_input":
                # Had content - just clear and continue (don't set any flags)
                return None
            else:
                # Fallback - re-send signal to global handler
                import os
                import signal
                os.kill(os.getpid(), signal.SIGINT)
                return None
        except EOFError:
            # Handle Ctrl+D gracefully
            self.interrupted = True
            return None
        except Exception as e:
            logger.error(f"Error in prompt_toolkit input: {e}")
            # Fallback to basic input
            try:
                result = input(prompt_text)
                if result:
                    self._add_to_history(result)
                return result
            except KeyboardInterrupt:
                # Let global interrupt manager handle the logic
                return None
            except EOFError:
                self.interrupted = True
                return None
        finally:
            # Clean up input state
            self._waiting_for_input = False
            self._current_input_text = ""

    def check_for_interrupt(self) -> bool:
        """Check if an interrupt signal is pending without blocking.

        This method checks for keyboard input (ESC or Ctrl+C) without waiting
        for user input. It's useful for checking interrupts during long operations
        like tool execution loops.

        Returns:
            bool: True if an interrupt is detected, False otherwise
        """
        # Check global interrupt state first
        if self.global_interrupt_manager.is_interrupted():
            self.interrupted = True
            return True

        # Check local interrupted state
        if self.interrupted:
            return True

        if not self._available:
            # Without prompt_toolkit, we can't easily check for pending input
            # Return current interrupted state
            return self.interrupted

        try:
            import select
            import sys

            # Check if there's pending input on stdin (Unix-like systems)
            if hasattr(select, "select"):
                # Check for pending input with 0 timeout (non-blocking)
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    # Read the character to check what it is
                    import termios
                    import tty

                    # Save terminal settings
                    old_settings = termios.tcgetattr(sys.stdin)
                    try:
                        # Set terminal to raw mode to read single characters
                        tty.setraw(sys.stdin.fileno())
                        char = sys.stdin.read(1)

                        # Check for ESC (27) or Ctrl+C (3)
                        if ord(char) == 27 or ord(char) == 3:  # ESC or Ctrl+C
                            self.interrupted = True
                            # Also set global interrupt
                            self.global_interrupt_manager.set_interrupted(True)
                            return True

                    finally:
                        # Restore terminal settings
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        except (ImportError, OSError, Exception):
            # Fall back to just checking the current state
            pass

        return self.interrupted

    def get_multiline_input(
        self, initial_prompt: str, allow_escape_interrupt: bool = False
    ) -> Optional[str]:
        """Get input with smart multiline detection using prompt_toolkit.

        This method provides a simplified interface for multiline input handling.
        For normal chat interactions, it uses single-line input by default as
        users can paste multiline content which will be handled automatically.

        Args:
            initial_prompt (str): The initial prompt to display
            allow_escape_interrupt (bool): If True, pressing ESC alone will interrupt

        Returns:
            Optional[str]: The user's input string, or None if interrupted
        """
        # Register this input handler for empty prompt detection
        self.global_interrupt_manager.set_current_input_handler(self)
        
        if not self._available:
            # Fallback behavior
            try:
                result = input(initial_prompt)
                if result:
                    self._add_to_history(result)
                return result
            except KeyboardInterrupt:
                # Let global interrupt manager handle the logic
                return None
            except EOFError:
                # End of input (e.g., pipe closed)
                self.interrupted = True
                return None

        # For normal chat, just use single-line input by default
        # Users can paste multiline content and it will be handled automatically
        user_input = self.get_input(
            initial_prompt,
            multiline_mode=False,
            allow_escape_interrupt=allow_escape_interrupt,
        )
        return user_input

    def get_history_info(self) -> str:
        """Get information about command history functionality.
        
        Returns:
            str: Information about how to use command history
        """
        if self._available:
            return (
                "📚 Command History: Use ↑/↓ arrow keys to navigate previous commands. "
                f"History stored in: {self.history_file}"
            )
        else:
            return (
                "📚 Command History: Available in fallback mode. "
                f"History stored in: {self.history_file}"
            )
