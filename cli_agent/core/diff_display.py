"""Clean diff display manager for terminal-based diff previews.

This module provides a clean, focused diff preview system that clears
the terminal, shows only the diff and confirmation prompt, and restores 
the terminal state after the user responds.
"""

import logging
import sys
from typing import Dict, Any, Optional, List

from cli_agent.core.terminal_state import get_terminal_state
from cli_agent.utils.diff_display import ColoredDiffDisplay

logger = logging.getLogger(__name__)


class CleanDiffDisplay:
    """Manages clean, focused diff previews that clear the terminal."""

    def __init__(self):
        self.terminal_state = get_terminal_state()
        self.in_clean_mode = False

    def can_use_clean_display(self) -> bool:
        """Check if we can use clean terminal display."""
        return self.terminal_state.can_clear_screen()

    def show_single_edit_diff(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        file_content: Optional[str] = None,
        context_lines: int = 3,
    ) -> Optional[bool]:
        """Show a clean diff preview for a single edit with user confirmation.
        
        Args:
            file_path: Path to the file being modified
            old_text: Text to be replaced
            new_text: Text to replace with
            file_content: Current file content (if None, will read from file)
            context_lines: Number of context lines to show around changes
            
        Returns:
            bool: True if user confirms, False if user denies, None if error/fallback needed
        """
        if not self.can_use_clean_display():
            return None

        # Clear the screen and enter clean mode
        if not self.terminal_state.clear_screen():
            return None

        self.in_clean_mode = True

        try:
            # Generate diff content using existing ColoredDiffDisplay
            diff_content = self._capture_diff_output(
                file_path, old_text, new_text, file_content, context_lines
            )
            
            if not diff_content:
                self.restore_terminal()
                return None

            # Create the full content with diff + confirmation prompt
            content = self._create_diff_content(diff_content, file_path, "single edit")

            # Position and display the content
            centered_content = self.terminal_state.position_vertically_centered(content)
            print(centered_content, end='', flush=True)
            
            # Get user input
            response = self._get_user_confirmation()
            
            return response

        except Exception as e:
            logger.error(f"Error showing clean diff display: {e}")
            return None
        finally:
            self.restore_terminal()

    def show_multiedit_diff(
        self,
        file_path: str,
        edits: List[Dict[str, Any]],
        file_content: Optional[str] = None,
        context_lines: int = 3,
    ) -> Optional[bool]:
        """Show a clean diff preview for multiple edits with user confirmation.
        
        Args:
            file_path: Path to the file being modified
            edits: List of edit operations
            file_content: Current file content (if None, will read from file)
            context_lines: Number of context lines to show around changes
            
        Returns:
            bool: True if user confirms, False if user denies, None if error/fallback needed
        """
        if not self.can_use_clean_display():
            return None

        # Clear the screen and enter clean mode
        if not self.terminal_state.clear_screen():
            return None

        self.in_clean_mode = True

        try:
            # Read file content if not provided
            if file_content is None:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    logger.error(f"Could not read file for diff preview: {e}")
                    self.restore_terminal()
                    return None

            # Generate combined diff content for all edits
            all_diffs = []
            current_content = file_content
            
            for i, edit in enumerate(edits):
                old_string = edit.get("old_string", "")
                new_string = edit.get("new_string", "")
                
                if old_string:
                    edit_diff = self._capture_edit_diff_output(
                        file_path, old_string, new_string, current_content, context_lines, i + 1
                    )
                    if edit_diff:
                        all_diffs.append(edit_diff)
                    
                    # Update content for next edit preview
                    if old_string in current_content:
                        replace_all = edit.get("replace_all", False)
                        if replace_all:
                            current_content = current_content.replace(old_string, new_string)
                        else:
                            current_content = current_content.replace(old_string, new_string, 1)

            if not all_diffs:
                self.restore_terminal()
                return None

            # Combine all diffs
            combined_diff = "\n".join(all_diffs)
            
            # Create the full content with diff + confirmation prompt
            content = self._create_diff_content(combined_diff, file_path, f"{len(edits)} edits")

            # Position and display the content
            centered_content = self.terminal_state.position_vertically_centered(content)
            print(centered_content, end='', flush=True)
            
            # Get user input
            response = self._get_user_confirmation()
            
            return response

        except Exception as e:
            logger.error(f"Error showing clean multiedit diff display: {e}")
            return None
        finally:
            self.restore_terminal()

    def restore_terminal(self) -> bool:
        """Restore the terminal to its previous state.
        
        Returns:
            bool: True if restoration was successful, False otherwise
        """
        if not self.in_clean_mode:
            return True

        try:
            success = self.terminal_state.restore_screen()
            self.in_clean_mode = False
            return success
        except Exception as e:
            logger.error(f"Error restoring terminal: {e}")
            self.in_clean_mode = False
            return False

    def _capture_diff_output(
        self, file_path: str, old_text: str, new_text: str, 
        file_content: Optional[str], context_lines: int
    ) -> Optional[str]:
        """Capture the colored diff output as a string."""
        import io
        import contextlib
        
        # Capture stdout to get the diff content
        captured_output = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output):
            success = ColoredDiffDisplay.show_replace_diff(
                file_path=file_path,
                old_text=old_text,
                new_text=new_text,
                file_content=file_content,
                context_lines=context_lines
            )
        
        if success:
            return captured_output.getvalue()
        return None

    def _capture_edit_diff_output(
        self, file_path: str, old_text: str, new_text: str,
        file_content: str, context_lines: int, edit_number: int
    ) -> Optional[str]:
        """Capture diff output for a single edit in a multiedit operation."""
        import io
        import contextlib
        
        # Capture stdout to get the diff content
        captured_output = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output):
            print(f"\n--- Edit {edit_number} Preview ---")
            success = ColoredDiffDisplay.show_replace_diff(
                file_path=file_path,
                old_text=old_text,
                new_text=new_text,
                file_content=file_content,
                context_lines=context_lines
            )
        
        if success:
            return captured_output.getvalue()
        return None

    def _create_diff_content(self, diff_output: str, file_path: str, operation_type: str) -> str:
        """Create the complete content including diff and confirmation prompt."""
        # ANSI color codes
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        YELLOW = "\033[33m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        RESET = "\033[0m"
        
        lines = [
            f"{BOLD}{CYAN}📝 File Edit Preview{RESET}",
            "",
            f"{YELLOW}File:{RESET} {file_path}",
            f"{YELLOW}Operation:{RESET} {operation_type}",
            "",
            "━" * 80,
            "",
            diff_output.rstrip(),
            "",
            "━" * 80,
            "",
            f"{BOLD}Do you want to proceed with these changes?{RESET}",
            "",
            f"  {GREEN}y{RESET} / {GREEN}yes{RESET}  - Apply the changes",
            f"  {RED}n{RESET} / {RED}no{RESET}   - Cancel the operation",
            "",
            "Your choice: ",
        ]
        
        return "\n".join(lines)

    def _get_user_confirmation(self) -> bool:
        """Get user confirmation for the diff changes."""
        try:
            while True:
                # Move cursor to end of prompt and get input
                response = input().strip().lower()
                
                if response in ('y', 'yes'):
                    return True
                elif response in ('n', 'no'):
                    return False
                else:
                    # Invalid input, ask again
                    print(f"\nPlease enter 'y' for yes or 'n' for no: ", end='', flush=True)
                    
        except (KeyboardInterrupt, EOFError):
            # User pressed Ctrl+C or Ctrl+D, treat as "no"
            return False


# Global instance for easy access
_clean_diff_display = None

def get_clean_diff_display() -> CleanDiffDisplay:
    """Get the global CleanDiffDisplay instance."""
    global _clean_diff_display
    if _clean_diff_display is None:
        _clean_diff_display = CleanDiffDisplay()
    return _clean_diff_display