#!/usr/bin/env python3
"""Response formatting and display utilities for MCP Agent."""

import logging
import re
import shutil
from io import StringIO

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Handles text formatting, markdown rendering, and tool execution display."""

    def __init__(self, config=None):
        """Initialize ResponseFormatter with optional config."""
        self.config = config

    def format_chunk_safely(self, chunk: str) -> str:
        """Apply basic formatting to streaming chunks without losing spaces."""
        if not chunk:
            return chunk

        try:
            # Don't try to format chunks that are just whitespace or very short
            if len(chunk.strip()) < 2:
                return chunk

            # Only apply formatting if the chunk contains complete markdown patterns
            if self._chunk_has_complete_markdown(chunk):
                return self._apply_simple_formatting(chunk)
            else:
                return chunk

        except Exception:
            # Always fall back to original chunk to preserve spaces
            return chunk

    def _apply_simple_formatting(self, chunk: str) -> str:
        """Apply simple visible formatting that definitely works."""
        formatted = chunk

        # Make bold text UPPERCASE and remove asterisks
        formatted = re.sub(r"\*\*([^*]+)\*\*", lambda m: m.group(1).upper(), formatted)

        # Make italic text with underscores and remove asterisks
        formatted = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"_\1_", formatted)

        # Make code text with brackets and remove backticks
        formatted = re.sub(r"`([^`]+)`", r"[\1]", formatted)

        return formatted

    def _apply_rich_console_formatting(self, chunk: str) -> str:
        """Use Rich Console to apply formatting directly to stdout."""
        import sys

        try:
            from rich.console import Console
            from rich.text import Text
        except ImportError:
            # Fallback to simple formatting if Rich not available
            return self._apply_simple_formatting(chunk)

        # Create a console that writes directly to our stdout
        console = Console(
            file=sys.stdout,
            force_terminal=True,
            width=None,
            legacy_windows=False,
            no_color=False,
            color_system="auto",
        )

        # Create formatted text
        text = Text()
        last_pos = 0

        # Find markdown patterns and apply Rich formatting
        patterns = [
            (r"\*\*([^*]+)\*\*", "bold"),  # Bold
            (r"(?<!\*)\*([^*]+)\*(?!\*)", "italic"),  # Italic
            (r"`([^`]+)`", "on bright_black"),  # Code
        ]

        matches = []
        for pattern, style in patterns:
            for match in re.finditer(pattern, chunk):
                matches.append((match.start(), match.end(), match.group(1), style))

        # Sort matches by position
        matches.sort()

        # Apply formatting
        for start, end, content, style in matches:
            # Add text before match
            if start > last_pos:
                text.append(chunk[last_pos:start])

            # Add formatted content
            text.append(content, style=style)
            last_pos = end

        # Add remaining text
        if last_pos < len(chunk):
            text.append(chunk[last_pos:])

        # Capture output to string buffer instead of printing directly
        try:
            capture = StringIO()
            temp_console = Console(file=capture, force_terminal=True, width=200)
            temp_console.print(text, end="")
            return capture.getvalue()
        except Exception:
            # Fallback to simple formatting
            return self._apply_simple_formatting(chunk)

    def _apply_direct_ansi_formatting(self, chunk: str) -> str:
        """Apply ANSI escape codes for terminal formatting."""
        formatted = chunk

        # Bold formatting
        formatted = re.sub(r"\*\*([^*]+)\*\*", r"\033[1m\1\033[0m", formatted)  # Bold

        # Italic formatting
        formatted = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)", r"\033[3m\1\033[0m", formatted
        )  # Italic

        # Code formatting (background color)
        formatted = re.sub(
            r"`([^`]+)`", r"\033[47m\033[30m\1\033[0m", formatted
        )  # Code with background

        return formatted

    def _chunk_has_complete_markdown(self, chunk: str) -> bool:
        """Check if chunk contains complete markdown patterns (not partial)."""
        # Check for complete bold patterns
        bold_pattern = r"\*\*[^*]+\*\*"
        if re.search(bold_pattern, chunk):
            return True

        # Check for complete italic patterns (but not part of bold)
        italic_pattern = r"(?<!\*)\*[^*]+\*(?!\*)"
        if re.search(italic_pattern, chunk):
            return True

        # Check for complete code patterns
        code_pattern = r"`[^`]+`"
        if re.search(code_pattern, chunk):
            return True

        return False

    def _apply_basic_markdown_to_text(self, text, chunk: str):
        """Apply basic markdown formatting using Rich Text objects."""
        try:
            from rich.text import Text
        except ImportError:
            return chunk

        # This is a placeholder for more sophisticated text formatting
        # For now, we'll use the simple formatting approach
        return self._apply_simple_formatting(chunk)

    def _find_safe_markdown_boundary(self, text: str, start_pos: int) -> int:
        """Find a safe position to split text that doesn't break markdown patterns."""
        # Look for natural break points
        safe_chars = [" ", "\n", ".", "!", "?", ";", ","]

        # Start from the desired position and work backwards
        for i in range(start_pos, max(0, start_pos - 50), -1):
            if i < len(text) and text[i] in safe_chars:
                # Make sure we're not in the middle of a markdown pattern
                substring = text[max(0, i - 10) : i + 10]
                if not self._is_inside_markdown_pattern(substring, 10):
                    return i

        # If no safe boundary found, return the original position
        return start_pos

    def _is_inside_markdown_pattern(self, text: str, pos: int) -> bool:
        """Check if position is inside a markdown pattern."""
        if pos >= len(text):
            return False

        # Check for bold patterns
        before = text[:pos]
        after = text[pos:]

        # Count asterisks before and after
        asterisks_before = before.count("**") % 2
        asterisks_after = after.count("**") % 2

        if asterisks_before == 1:  # Inside bold pattern
            return True

        # Check for code patterns
        backticks_before = before.count("`") % 2
        if backticks_before == 1:  # Inside code pattern
            return True

        return False

    def format_markdown(self, text: str) -> str:
        """Format markdown text for terminal display using Rich."""
        if not text:
            return text

        try:
            from rich.console import Console
            from rich.markdown import Markdown

            # Detect actual terminal width, fallback to 80 if detection fails
            try:
                terminal_width = shutil.get_terminal_size().columns
                # Ensure reasonable bounds (minimum 40, maximum 200)
                terminal_width = max(40, min(200, terminal_width))
            except (OSError, AttributeError):
                terminal_width = 80  # Fallback for environments without terminal

            # Create a console that outputs to a string buffer
            console = Console(
                file=StringIO(), force_terminal=True, width=terminal_width
            )

            # Parse and render markdown
            md = Markdown(text)
            console.print(md)

            # Get the formatted output
            formatted = console.file.getvalue()

            # Clean up any extra newlines at the end
            return formatted.rstrip()

        except ImportError:
            # Fallback to basic formatting if Rich is not available
            logger.warning(
                "Rich library not available, using basic markdown formatting"
            )
            return self._basic_markdown_format(text)

    def _basic_markdown_format(self, text: str) -> str:
        """Basic fallback markdown formatting."""
        if not text:
            return text

        # Apply basic ANSI formatting
        formatted = self._apply_direct_ansi_formatting(text)
        return formatted

    def display_tool_execution_start(
        self, tool_count: int, is_subagent: bool = False, interactive: bool = True
    ) -> str:
        """Display tool execution start message."""
        if is_subagent:
            return f"🤖 [SUBAGENT] Executing {tool_count} tool(s)..."
        else:
            return f"🔧 Using {tool_count} tool(s)..."

    def display_tool_execution_step(
        self,
        step_num: int,
        tool_name: str,
        arguments: dict,
        is_subagent: bool = False,
        interactive: bool = True,
    ) -> str:
        """Display individual tool execution step."""
        if is_subagent:
            return f"🤖 [SUBAGENT] Step {step_num}: Executing {tool_name}..."
        else:
            return f"{step_num}. Executing {tool_name}..."

    def display_tool_execution_result(
        self,
        result: str,
        is_error: bool = False,
        is_subagent: bool = False,
        interactive: bool = True,
    ) -> str:
        """Display tool execution result."""
        if is_error:
            prefix = "❌ [SUBAGENT] Error:" if is_subagent else "❌ Error:"
        else:
            prefix = "✅ [SUBAGENT] Result:" if is_subagent else "✅ Result:"

        # Don't truncate results - show full content
        result_preview = result

        # Handle multi-line results properly with newlines and carriage returns
        if "\n" in result_preview:
            lines = result_preview.split("\n")
            formatted_lines = [f"{prefix} {lines[0]}"]
            for line in lines[1:]:
                formatted_lines.append(f"\r{line}")  # Carriage return to start of line
            return "\n".join(formatted_lines)
        else:
            return f"{prefix} {result_preview}"

    def display_tool_processing(
        self, is_subagent: bool = False, interactive: bool = True
    ) -> str:
        """Display tool processing message."""
        if is_subagent:
            return "🤖 [SUBAGENT] Processing tool results..."
        else:
            return "⚙️ Processing tool results..."

    def format_streaming_chunk(self, chunk: str, use_rich: bool = True) -> str:
        """Format a streaming chunk with appropriate method based on capabilities."""
        if not chunk:
            return chunk

        if use_rich:
            try:
                return self._apply_rich_console_formatting(chunk)
            except ImportError:
                # Fall back to simple formatting if Rich not available
                return self.format_chunk_safely(chunk)
        else:
            return self.format_chunk_safely(chunk)

    def clean_terminal_output(self, text: str) -> str:
        """Clean text for proper terminal display with carriage returns."""
        if not text:
            return text

        # Replace newlines with proper carriage return + newline for terminal
        return text.replace("\n", "\r\n")
