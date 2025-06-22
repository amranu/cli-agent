#!/usr/bin/env python3
"""
Professional input handler using prompt_toolkit for robust terminal interaction.

This module provides the InterruptibleInput class which offers advanced terminal
input handling with support for interruption, multiline input, and fallback
mechanisms for environments where prompt_toolkit is not available.
"""

import logging
from typing import Optional

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

    Attributes:
        interrupted (bool): Flag indicating if input was interrupted
        _available (bool): Whether prompt_toolkit is available
        _prompt: Reference to prompt_toolkit prompt function
        _patch_stdout: Reference to prompt_toolkit patch_stdout
        _bindings: Key bindings for custom keyboard shortcuts
        _allow_escape_interrupt (bool): Whether ESC key should trigger interruption
    """

    def __init__(self):
        """Initialize the InterruptibleInput handler.

        Sets up prompt_toolkit components if available, otherwise prepares
        for fallback to basic input() function.
        """
        self.interrupted = False
        self._setup_prompt_toolkit()

    def _setup_prompt_toolkit(self):
        """Setup prompt_toolkit components.

        Attempts to import and configure prompt_toolkit for advanced terminal
        interaction. If import fails, sets up for fallback mode.
        """
        try:
            import asyncio

            from prompt_toolkit import prompt
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.keys import Keys
            from prompt_toolkit.patch_stdout import patch_stdout

            self._prompt = prompt
            self._patch_stdout = patch_stdout
            self._available = True

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
        if not self._available:
            # Fallback to basic input if prompt_toolkit unavailable
            try:
                return input(prompt_text)
            except KeyboardInterrupt:
                self.interrupted = True
                return None

        try:
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
                        key_bindings=self._bindings if allow_escape_interrupt else None,
                        multiline=multiline_mode,
                        wrap_lines=True,
                        enable_history_search=False,
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
                    key_bindings=self._bindings if allow_escape_interrupt else None,
                    multiline=multiline_mode,
                    wrap_lines=True,
                    enable_history_search=False,
                )
                return result

        except KeyboardInterrupt:
            self.interrupted = True
            return None
        except EOFError:
            # Handle Ctrl+D gracefully
            return None
        except Exception as e:
            logger.error(f"Error in prompt_toolkit input: {e}")
            # Fallback to basic input
            try:
                return input(prompt_text)
            except KeyboardInterrupt:
                self.interrupted = True
                return None

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
        if not self._available:
            # Fallback behavior
            try:
                return input(initial_prompt)
            except KeyboardInterrupt:
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
