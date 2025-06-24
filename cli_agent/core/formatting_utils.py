"""Formatting and display utilities for BaseMCPAgent."""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FormattingUtils:
    """Handles text formatting, markdown processing, and display utilities."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    def format_chunk_safely(self, chunk: str) -> str:
        """Format a chunk of text safely, with fallbacks for errors."""
        try:
            # Try rich formatting first if available
            return self.apply_rich_console_formatting(chunk)
        except Exception:
            try:
                # Fall back to simple formatting
                return self.apply_simple_formatting(chunk)
            except Exception:
                # Final fallback: return as-is
                return chunk

    def apply_simple_formatting(self, chunk: str) -> str:
        """Apply simple text formatting without external dependencies."""
        # Basic markdown-style formatting
        chunk = re.sub(r"\*\*(.*?)\*\*", r"\033[1m\1\033[0m", chunk)  # Bold
        chunk = re.sub(r"\*(.*?)\*", r"\033[3m\1\033[0m", chunk)  # Italic
        chunk = re.sub(r"`(.*?)`", r"\033[96m\1\033[0m", chunk)  # Code
        return chunk

    def apply_rich_console_formatting(self, chunk: str) -> str:
        """Apply rich console formatting if rich is available."""
        try:
            import io

            from rich.console import Console
            from rich.markdown import Markdown

            console = Console(file=io.StringIO(), force_terminal=True, width=80)
            md = Markdown(chunk)
            console.print(md)
            return console.file.getvalue()
        except ImportError:
            return self.apply_simple_formatting(chunk)

    def apply_direct_ansi_formatting(self, chunk: str) -> str:
        """Apply direct ANSI formatting for markdown-like syntax."""
        return self.apply_simple_formatting(chunk)

    def chunk_has_complete_markdown(self, chunk: str) -> bool:
        """Check if a chunk contains complete markdown elements."""
        # Check for complete code blocks
        if chunk.count("```") % 2 != 0:
            return False

        # Check for complete bold/italic formatting
        if chunk.count("**") % 2 != 0 or chunk.count("*") % 2 != 0:
            return False

        return True

    def apply_basic_markdown_to_text(self, text, chunk: str):
        """Apply basic markdown formatting to text."""
        # This is a placeholder for more complex markdown processing
        return self.apply_simple_formatting(text)

    def find_safe_markdown_boundary(self, text: str, start_pos: int) -> int:
        """Find a safe boundary to break markdown text."""
        # Look for natural break points like newlines, periods, etc.
        for i in range(start_pos, min(start_pos + 100, len(text))):
            if text[i] in ["\n", ".", "!", "?"]:
                return i + 1
        return start_pos + 50  # Fallback

    def format_markdown(self, text: str) -> str:
        """Format markdown text with available formatting library."""
        return self.basic_markdown_format(text)

    def basic_markdown_format(self, text: str) -> str:
        """Basic markdown formatting using ANSI escape codes."""
        return self.apply_simple_formatting(text)

    def display_tool_execution_start(
        self, tool_name: str, args: Dict[str, Any], interactive: bool = True
    ):
        """Display the start of tool execution."""
        if not interactive:
            return

        args_preview = str(args)[:100] + "..." if len(str(args)) > 100 else str(args)
        print(f"\n🔧 Executing {tool_name} with args: {args_preview}", flush=True)

    def display_tool_execution_step(
        self,
        step_number: int,
        total_steps: int,
        tool_name: str,
        args: Dict[str, Any],
        interactive: bool = True,
    ):
        """Display progress during multi-step tool execution."""
        if not interactive:
            return

        args_preview = str(args)[:50] + "..." if len(str(args)) > 50 else str(args)
        print(
            f"\r🔧 Step {step_number}/{total_steps}: {tool_name}({args_preview})",
            end="",
            flush=True,
        )

    def truncate_tool_result(self, result: str, max_length: int = None) -> str:
        """Truncate tool result for display if configured to do so."""
        if (
            not hasattr(self.agent, "config")
            or not self.agent.config.truncate_tool_results
        ):
            return result

        max_len = max_length or self.agent.config.tool_result_max_length
        if len(result) <= max_len:
            return result

        # Truncate and add indicator
        truncated = result[:max_len]
        return f"{truncated}... [output truncated - {len(result)} total chars]"

    def display_tool_execution_result(
        self,
        tool_name: str,
        result: str,
        success: bool = True,
        interactive: bool = True,
    ):
        """Display the result of tool execution."""
        if not interactive:
            return

        # Truncate result for display only (not for the model)
        display_result = self.truncate_tool_result(result)
        status = "✅" if success else "❌"
        print(f"\n{status} {tool_name}: {display_result}", flush=True)

    def display_tool_processing(self, message: str, interactive: bool = True):
        """Display tool processing status."""
        if interactive:
            print(f"\r⚡ {message}", end="", flush=True)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count total tokens in conversation messages."""
        total = 0
        for message in messages:
            content = message.get("content", "")
            total += self.estimate_tokens(str(content))
        return total

    def should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if conversation should be compacted based on token count."""
        if hasattr(self.agent, "get_token_limit"):
            token_limit = self.agent.get_token_limit()
            current_tokens = self.count_conversation_tokens(messages)
            # Compact if we're using more than 80% of token limit
            return current_tokens > (token_limit * 0.8)
        return False

    def compact_conversation(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compact conversation by summarizing older messages."""
        if len(messages) <= 3:
            return messages

        # Keep first message (usually system) and last 2 messages
        # Summarize everything in between
        first_msg = messages[0]
        last_msgs = messages[-2:]
        middle_msgs = messages[1:-2]

        if not middle_msgs:
            return messages

        # Create summary of middle messages
        summary_content = f"[CONVERSATION SUMMARY: {len(middle_msgs)} messages exchanged involving tool usage and responses]"
        summary_msg = {"role": "system", "content": summary_content}

        return [first_msg, summary_msg] + last_msgs
