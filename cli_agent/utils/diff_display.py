"""Utilities for displaying colored diffs in terminal."""

import difflib
import sys
from typing import Optional


class ColoredDiffDisplay:
    """Display colored diffs in terminal."""

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bold": "\033[1m",
        "dim": "\033[2m",
    }

    @classmethod
    def show_replace_diff(
        cls,
        file_path: str,
        old_text: str,
        new_text: str,
        file_content: Optional[str] = None,
        context_lines: int = 3,
    ) -> bool:
        """
        Display a colored diff for replace_in_file operation.

        Args:
            file_path: Path to the file being modified
            old_text: Text to be replaced
            new_text: Text to replace with
            file_content: Current file content (if None, will read from file)
            context_lines: Number of context lines to show around changes

        Returns:
            bool: True if diff was displayed successfully, False otherwise
        """
        try:
            # Read file content if not provided
            if file_content is None:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except FileNotFoundError:
                    print(
                        f"{cls.COLORS['red']}Error: File not found: {file_path}{cls.COLORS['reset']}\r"
                    )
                    return False
                except Exception as e:
                    print(
                        f"{cls.COLORS['red']}Error reading file: {e}{cls.COLORS['reset']}\r"
                    )
                    return False

            # Check if old text exists in file
            if old_text not in file_content:
                print(
                    f"{cls.COLORS['yellow']}Warning: Text to replace not found in {file_path}{cls.COLORS['reset']}\r"
                )
                return False

            # Generate content after replacement
            new_content = file_content.replace(old_text, new_text)

            # Split into lines for difflib
            old_lines = file_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)

            # Generate unified diff
            diff = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                    n=context_lines,
                )
            )

            if not diff:
                print(
                    f"{cls.COLORS['yellow']}No changes detected{cls.COLORS['reset']}\r"
                )
                return False

            # Display header
            print(
                f"\r\n{cls.COLORS['bold']}{cls.COLORS['cyan']}📝 Diff Preview: {file_path}{cls.COLORS['reset']}\r"
            )
            print(f"{cls.COLORS['dim']}{'─' * 60}{cls.COLORS['reset']}\r")

            # Display colored diff
            for line in diff:
                if line.startswith("+++") or line.startswith("---"):
                    # File headers
                    print(
                        f"{cls.COLORS['bold']}{cls.COLORS['white']}{line.rstrip()}{cls.COLORS['reset']}\r"
                    )
                elif line.startswith("@@"):
                    # Hunk headers
                    print(
                        f"{cls.COLORS['bold']}{cls.COLORS['magenta']}{line.rstrip()}{cls.COLORS['reset']}\r"
                    )
                elif line.startswith("+"):
                    # Added lines
                    print(
                        f"{cls.COLORS['green']}{line.rstrip()}{cls.COLORS['reset']}\r"
                    )
                elif line.startswith("-"):
                    # Removed lines
                    print(f"{cls.COLORS['red']}{line.rstrip()}{cls.COLORS['reset']}\r")
                else:
                    # Context lines
                    print(f"{cls.COLORS['dim']}{line.rstrip()}{cls.COLORS['reset']}\r")

            print(f"{cls.COLORS['dim']}{'─' * 60}{cls.COLORS['reset']}\r")

            # Show summary
            changes_count = file_content.count(old_text)
            if changes_count == 1:
                print(
                    f"{cls.COLORS['blue']}Will replace 1 occurrence{cls.COLORS['reset']}\r"
                )
            else:
                print(
                    f"{cls.COLORS['blue']}Will replace {changes_count} occurrences{cls.COLORS['reset']}\r"
                )

            print("\r")  # Add spacing after diff
            return True

        except Exception as e:
            print(
                f"{cls.COLORS['red']}Error generating diff: {e}{cls.COLORS['reset']}\r"
            )
            return False

    @classmethod
    def disable_colors(cls):
        """Disable colors (useful for non-terminal output)."""
        for key in cls.COLORS:
            cls.COLORS[key] = ""

    @classmethod
    def is_terminal_capable(cls) -> bool:
        """Check if terminal supports colors."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
