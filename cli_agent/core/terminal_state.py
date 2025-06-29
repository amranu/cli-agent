"""Terminal state management for clean permission prompts.

This module provides functionality to capture, clear, and restore terminal state
to enable clean permission prompts that don't interfere with ongoing conversations.
"""

import os
import subprocess
import sys
import time
from typing import Optional, Tuple


class TerminalState:
    """Manages terminal state capture and restoration for clean UI elements."""

    def __init__(self):
        self.is_terminal = sys.stdout.isatty()
        self.supports_clearing = self._detect_clear_support()
        self.supports_alternate_screen = self._detect_alternate_screen_support()
        self.terminal_height, self.terminal_width = self._get_terminal_size()
        self._saved_state = None

    def _detect_clear_support(self) -> bool:
        """Detect if terminal supports clearing operations."""
        if not self.is_terminal:
            return False
        
        # Check common terminal types that support clearing
        term = os.environ.get('TERM', '').lower()
        supported_terms = ['xterm', 'vt100', 'ansi', 'linux', 'screen', 'tmux']
        
        return any(supported in term for supported in supported_terms)

    def _detect_alternate_screen_support(self) -> bool:
        """Detect if terminal supports alternate screen buffer."""
        if not self.is_terminal:
            return False
            
        # Most modern terminals support alternate screen
        term = os.environ.get('TERM', '').lower()
        unsupported_terms = ['dumb', 'unknown']
        
        return not any(unsupported in term for unsupported in unsupported_terms)

    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get current terminal dimensions."""
        try:
            if self.is_terminal:
                return os.get_terminal_size()
        except OSError:
            pass
        return 24, 80  # Default fallback

    def can_clear_screen(self) -> bool:
        """Check if we can safely clear the screen."""
        return self.is_terminal and self.supports_clearing

    def clear_screen(self) -> bool:
        """Clear the entire screen and move cursor to top-left.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        if not self.can_clear_screen():
            return False
            
        try:
            # Use alternate screen buffer if supported for better restoration
            if self.supports_alternate_screen:
                sys.stdout.write('\033[?47h')  # Enter alternate screen
                sys.stdout.write('\033[2J')    # Clear screen
                sys.stdout.write('\033[H')     # Move to top-left
            else:
                sys.stdout.write('\033[2J')    # Clear screen
                sys.stdout.write('\033[H')     # Move to top-left
            sys.stdout.flush()
            return True
        except (OSError, UnicodeEncodeError):
            return False

    def restore_screen(self) -> bool:
        """Restore the previous screen state.
        
        Returns:
            bool: True if restoration was successful, False otherwise
        """
        if not self.can_clear_screen():
            return False
            
        try:
            if self.supports_alternate_screen:
                sys.stdout.write('\033[?47l')  # Exit alternate screen
            else:
                # For terminals without alternate screen, just clear and 
                # let the conversation continue normally
                sys.stdout.write('\033[2J')    # Clear screen
                sys.stdout.write('\033[H')     # Move to top-left
            sys.stdout.flush()
            return True
        except (OSError, UnicodeEncodeError):
            return False

    def center_text(self, text: str, width: Optional[int] = None) -> str:
        """Center text within the terminal width.
        
        Args:
            text: Text to center
            width: Width to center within (defaults to terminal width)
            
        Returns:
            Centered text with appropriate padding
        """
        if width is None:
            width = self.terminal_width
            
        lines = text.split('\n')
        centered_lines = []
        
        for line in lines:
            # Remove ANSI color codes for length calculation
            clean_line = self._remove_ansi_codes(line)
            line_length = len(clean_line)
            
            if line_length >= width:
                centered_lines.append(line)
            else:
                padding = (width - line_length) // 2
                centered_lines.append(' ' * padding + line)
                
        return '\n'.join(centered_lines)

    def _remove_ansi_codes(self, text: str) -> str:
        """Remove ANSI escape codes from text for length calculation."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def create_box(self, content: str, title: str = "", width: Optional[int] = None) -> str:
        """Create a bordered box around content.
        
        Args:
            content: Text content to box
            title: Optional title for the box
            width: Width of the box (defaults to fit content + padding)
            
        Returns:
            Formatted box with content
        """
        if width is None:
            # Calculate width based on content
            lines = content.split('\n')
            if title:
                lines.append(title)
            max_length = max(len(self._remove_ansi_codes(line)) for line in lines)
            width = min(max_length + 4, self.terminal_width - 4)
        
        # Box drawing characters
        horizontal = '─'
        vertical = '│'
        top_left = '┌'
        top_right = '┐'
        bottom_left = '└'
        bottom_right = '┘'
        
        # Build the box
        box_lines = []
        
        # Top border with optional title
        if title:
            title_padded = f" {title} "
            title_length = len(self._remove_ansi_codes(title_padded))
            border_length = width - title_length
            left_border = border_length // 2
            right_border = border_length - left_border
            top_line = top_left + horizontal * left_border + title_padded + horizontal * right_border + top_right
        else:
            top_line = top_left + horizontal * (width - 2) + top_right
        box_lines.append(top_line)
        
        # Content lines
        content_lines = content.split('\n')
        for line in content_lines:
            clean_length = len(self._remove_ansi_codes(line))
            padding = width - 2 - clean_length
            if padding > 0:
                padded_line = vertical + line + ' ' * padding + vertical
            else:
                # Truncate if too long
                truncated = line[:width-5] + '...' if clean_length > width-2 else line
                padded_line = vertical + truncated + ' ' * (width - 2 - len(self._remove_ansi_codes(truncated))) + vertical
            box_lines.append(padded_line)
        
        # Bottom border
        bottom_line = bottom_left + horizontal * (width - 2) + bottom_right
        box_lines.append(bottom_line)
        
        return '\n'.join(box_lines)

    def position_vertically_centered(self, content: str) -> str:
        """Position content vertically centered in terminal.
        
        Args:
            content: Content to center vertically
            
        Returns:
            Content with appropriate vertical padding
        """
        content_lines = content.split('\n')
        content_height = len(content_lines)
        
        if content_height >= self.terminal_height:
            return content
            
        vertical_padding = (self.terminal_height - content_height) // 2
        
        # Add empty lines before content
        padded_lines = [' '] * vertical_padding + content_lines
        
        return '\n'.join(padded_lines)

    def get_terminal_info(self) -> dict:
        """Get information about terminal capabilities.
        
        Returns:
            Dictionary with terminal capability information
        """
        return {
            'is_terminal': self.is_terminal,
            'supports_clearing': self.supports_clearing,
            'supports_alternate_screen': self.supports_alternate_screen,
            'height': self.terminal_height,
            'width': self.terminal_width,
            'term_type': os.environ.get('TERM', 'unknown')
        }


# Global terminal state instance
_terminal_state = None


def get_terminal_state() -> TerminalState:
    """Get the global terminal state instance."""
    global _terminal_state
    if _terminal_state is None:
        _terminal_state = TerminalState()
    return _terminal_state