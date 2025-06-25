"""Terminal management for persistent bottom prompt display."""

import asyncio
import os
import sys
import termios
import tty
from typing import Optional


class TerminalManager:
    """Manages terminal display with persistent bottom prompt."""
    
    def __init__(self):
        self.is_terminal = sys.stdout.isatty()
        self.prompt_text = ""
        self.prompt_active = False
        self.terminal_height = 24  # Default fallback
        self.original_settings = None
        
        if self.is_terminal:
            try:
                # Get terminal size
                self.terminal_height, self.terminal_width = os.get_terminal_size()
            except OSError:
                self.terminal_height, self.terminal_width = 24, 80
                
    def start_persistent_prompt(self, prompt_text: str):
        """Start displaying a persistent prompt at the bottom of the terminal."""
        if not self.is_terminal:
            return
            
        self.prompt_text = prompt_text
        self.prompt_active = True
        
        # Save cursor position and move to bottom
        self._save_cursor()
        self._move_to_bottom()
        self._clear_line()
        sys.stdout.write(prompt_text)
        sys.stdout.flush()
        
    def stop_persistent_prompt(self):
        """Stop displaying the persistent prompt."""
        if not self.is_terminal or not self.prompt_active:
            return
            
        self.prompt_active = False
        self._move_to_bottom()
        self._clear_line()
        sys.stdout.flush()
        
    def write_above_prompt(self, text: str):
        """Write text above the persistent prompt."""
        if not self.is_terminal:
            # Fallback for non-terminal environments
            sys.stdout.write(text)
            sys.stdout.flush()
            return
            
        if self.prompt_active:
            # Save current cursor position
            self._save_cursor()
            
            # Move to the line above the prompt
            self._move_cursor_up(1)
            
            # Write the text and ensure it ends with newline
            if not text.endswith('\n'):
                text += '\n'
            sys.stdout.write(text)
            
            # Move to bottom and redraw prompt
            self._move_to_bottom()
            self._clear_line()
            sys.stdout.write(self.prompt_text)
            sys.stdout.flush()
        else:
            # No prompt active, write normally
            sys.stdout.write(text)
            sys.stdout.flush()
            
    def update_prompt(self, new_prompt_text: str):
        """Update the prompt text."""
        if not self.is_terminal:
            return
            
        self.prompt_text = new_prompt_text
        if self.prompt_active:
            self._move_to_bottom()
            self._clear_line()
            sys.stdout.write(new_prompt_text)
            sys.stdout.flush()
            
    def _save_cursor(self):
        """Save current cursor position."""
        sys.stdout.write('\033[s')  # Save cursor position
        
    def _restore_cursor(self):
        """Restore saved cursor position."""
        sys.stdout.write('\033[u')  # Restore cursor position
        
    def _move_to_bottom(self):
        """Move cursor to bottom line."""
        sys.stdout.write(f'\033[{self.terminal_height};1H')  # Move to bottom line, column 1
        
    def _move_cursor_up(self, lines: int):
        """Move cursor up by specified number of lines."""
        sys.stdout.write(f'\033[{lines}A')
        
    def _clear_line(self):
        """Clear the current line."""
        sys.stdout.write('\033[K')  # Clear from cursor to end of line
        
    def _scroll_up(self, lines: int = 1):
        """Scroll the terminal up by specified lines."""
        for _ in range(lines):
            sys.stdout.write('\033[S')  # Scroll up one line
            
    def get_terminal_size(self) -> tuple[int, int]:
        """Get current terminal size."""
        if self.is_terminal:
            try:
                return os.get_terminal_size()
            except OSError:
                pass
        return self.terminal_height, self.terminal_width
        
    def setup_terminal_raw_mode(self):
        """Set up terminal for raw input mode."""
        if not self.is_terminal:
            return
            
        try:
            self.original_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setraw(sys.stdin.fileno())
        except (termios.error, OSError):
            self.original_settings = None
            
    def restore_terminal_mode(self):
        """Restore terminal to original mode."""
        if self.original_settings and self.is_terminal:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original_settings)
            except (termios.error, OSError):
                pass
            finally:
                self.original_settings = None


# Global terminal manager instance
_terminal_manager = None

def get_terminal_manager() -> TerminalManager:
    """Get the global terminal manager instance."""
    global _terminal_manager
    if _terminal_manager is None:
        _terminal_manager = TerminalManager()
    return _terminal_manager