"""Simple, clean interactive prompt using prompt_toolkit only."""

import asyncio
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText


class SimplePromptManager:
    """Clean, simple prompt manager using just prompt_toolkit."""
    
    def __init__(self):
        self.session = None
        self.history = InMemoryHistory()
        self.interrupted = False
        self._setup_session()
    
    def _setup_session(self):
        """Setup the prompt session with key bindings."""
        bindings = KeyBindings()
        
        @bindings.add('escape')
        def _(event):
            """Handle escape key."""
            self.interrupted = True
            get_app().exit()
        
        @bindings.add('c-c')
        def _(event):
            """Handle Ctrl+C."""
            self.interrupted = True
            get_app().exit()
        
        # Command completion
        commands = [
            '/help', '/clear', '/model', '/review', '/tokens', '/compact',
            '/switch-chat', '/switch-reason', '/switch-gemini', '/switch-gemini-pro',
            'quit', 'exit', 'tools'
        ]
        completer = WordCompleter(commands, ignore_case=True)
        
        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            key_bindings=bindings,
            multiline=False,
            wrap_lines=True,
            complete_style='multi-column'
        )
    
    def get_input(self, prompt: str = "You: ") -> Optional[str]:
        """Get user input."""
        self.interrupted = False
        try:
            result = self.session.prompt(prompt)
            return result
        except (EOFError, KeyboardInterrupt):
            self.interrupted = True
            return None
    
    def get_multiline_input(self, prompt: str = "You: ") -> Optional[str]:
        """Get potentially multiline input with smart detection."""
        first_line = self.get_input(prompt)
        if first_line is None:
            return None
        
        # Check if this looks like multiline content
        is_likely_multiline = (
            len(first_line) > 300 or
            first_line.startswith(('def ', 'class ', 'import ', 'from ')) or
            '```' in first_line or
            (first_line.endswith(':') and len(first_line) > 80) or
            first_line.endswith('{') or
            first_line.endswith('\\')
        )
        
        if is_likely_multiline:
            print("(Detected multiline content. Continue typing, empty line to finish)")
            lines = [first_line]
            
            while True:
                line = self.get_input("... ")
                if line is None:  # Interrupted
                    return None
                if line.strip() == "":  # Empty line ends input
                    break
                lines.append(line)
            
            return '\n'.join(lines)
        
        return first_line


# Replacement function for the existing InterruptibleInput
class ModernInterruptibleInput:
    """Drop-in replacement for the existing InterruptibleInput class."""
    
    def __init__(self):
        self.prompt_manager = SimplePromptManager()
        self.interrupted = False
    
    def get_input(self, prompt: str, multiline_mode: bool = False) -> Optional[str]:
        """Get input - compatibility method."""
        result = self.prompt_manager.get_input(prompt)
        self.interrupted = self.prompt_manager.interrupted
        return result
    
    def get_multiline_input(self, prompt: str) -> Optional[str]:
        """Get multiline input - compatibility method."""
        result = self.prompt_manager.get_multiline_input(prompt)
        self.interrupted = self.prompt_manager.interrupted
        return result
    
    def setup_terminal(self):
        """No-op for compatibility."""
        pass
    
    def restore_terminal(self):
        """No-op for compatibility."""
        pass