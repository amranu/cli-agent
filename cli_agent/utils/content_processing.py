"""
Content processing utilities for extracting text and handling different LLM response formats.

This module provides utilities for processing LLM response content, extracting text
before tool calls, and handling different content patterns across LLM types.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Base content processing utilities."""

    @staticmethod
    def clean_code_blocks(text: str) -> str:
        """Remove code block markers from text."""
        # Remove markdown code block markers
        text = re.sub(r"^```\w*\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def extract_text_before_patterns(content: str, patterns: List[str]) -> str:
        """Extract text that appears before any of the given patterns."""
        for pattern in patterns:
            try:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    text_before = match.group(1).strip()
                    if text_before:
                        return ContentProcessor.clean_code_blocks(text_before)
            except Exception as e:
                logger.debug(f"Pattern {pattern} failed: {e}")
                continue

        return ""

    @staticmethod
    def split_content_and_tools(
        content: str, tool_patterns: List[str]
    ) -> tuple[str, List[str]]:
        """Split content into text part and tool call parts."""
        text_part = ContentProcessor.extract_text_before_patterns(
            content, tool_patterns
        )

        # Extract tool call sections
        tool_parts = []
        for pattern in tool_patterns:
            try:
                matches = re.findall(pattern, content, re.DOTALL)
                tool_parts.extend(matches)
            except Exception as e:
                logger.debug(f"Tool extraction pattern {pattern} failed: {e}")
                continue

        return text_part, tool_parts


class DeepSeekContentProcessor(ContentProcessor):
    """Content processor for DeepSeek-specific patterns."""

    # DeepSeek-specific patterns for extracting text before tool calls
    TEXT_EXTRACTION_PATTERNS = [
        r"^(.*?)(?=<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>)",
        r'^(.*?)(?=```json\s*\{\s*"function")',
        r"^(.*?)(?=```python\s*<｜tool▁calls▁begin｜>)",
        r'^(.*?)(?=\{[\s\S]*?"function"\s*:)',
    ]

    # Patterns for extracting tool call sections
    TOOL_CALL_PATTERNS = [
        r"<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>",
        r"<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>",
        r'```json\s*(\{.*?"function".*?\})\s*```',
        r"```python\s*<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>",
    ]

    @classmethod
    def extract_text_before_tool_calls(cls, content: str) -> str:
        """Extract text that appears before DeepSeek tool calls."""
        return cls.extract_text_before_patterns(content, cls.TEXT_EXTRACTION_PATTERNS)

    @classmethod
    def split_content_and_tool_calls(cls, content: str) -> tuple[str, List[str]]:
        """Split DeepSeek content into text and tool call parts."""
        return cls.split_content_and_tools(content, cls.TOOL_CALL_PATTERNS)


class GeminiContentProcessor(ContentProcessor):
    """Content processor for Gemini-specific patterns."""

    # Gemini-specific patterns for extracting text before tool calls
    TEXT_EXTRACTION_PATTERNS = [
        r"^(.*?)(?=<execute_tool>)",
        r"^(.*?)(?=\w+\s*\([^)]*\)\s*$)",  # Before function calls
        r"^(.*?)(?=Tool:\s*\w+:\w+)",
    ]

    # Patterns for extracting tool call sections
    TOOL_CALL_PATTERNS = [
        r"<execute_tool>\s*(\{.*?\})\s*</execute_tool>",
        r"(\w+\s*\([^)]*\))",  # Python-style function calls
        r"(Tool:\s*\w+:\w+.*?)(?=\n|$)",
    ]

    @classmethod
    def extract_text_before_tool_calls(cls, content: str) -> str:
        """Extract text that appears before Gemini tool calls."""
        return cls.extract_text_before_patterns(content, cls.TEXT_EXTRACTION_PATTERNS)

    @classmethod
    def split_content_and_tool_calls(cls, content: str) -> tuple[str, List[str]]:
        """Split Gemini content into text and tool call parts."""
        return cls.split_content_and_tools(content, cls.TOOL_CALL_PATTERNS)


class ContentProcessorFactory:
    """Factory for creating content processors for different LLM types."""

    _processors = {
        "deepseek": DeepSeekContentProcessor,
        "openai": ContentProcessor,  # Generic processor
        "gemini": GeminiContentProcessor,
    }

    @classmethod
    def create_processor(cls, llm_type: str) -> ContentProcessor:
        """Create appropriate content processor for LLM type."""
        processor_class = cls._processors.get(llm_type.lower(), ContentProcessor)
        return processor_class()

    @classmethod
    def extract_text_before_tool_calls(cls, content: str, llm_type: str) -> str:
        """Convenience method to extract text before tool calls for specific LLM."""
        if llm_type.lower() == "deepseek":
            return DeepSeekContentProcessor.extract_text_before_tool_calls(content)
        elif llm_type.lower() == "gemini":
            return GeminiContentProcessor.extract_text_before_tool_calls(content)
        else:
            # Generic extraction - look for common patterns
            generic_patterns = [
                r"^(.*?)(?=```json\s*\{)",  # Before JSON blocks
                r"^(.*?)(?=\w+\s*\()",  # Before function calls
                r"^(.*?)(?=<\w+>)",  # Before XML tags
            ]
            return ContentProcessor.extract_text_before_patterns(
                content, generic_patterns
            )

    @classmethod
    def split_content_and_tool_calls(
        cls, content: str, llm_type: str
    ) -> tuple[str, List[str]]:
        """Convenience method to split content for specific LLM type."""
        if llm_type.lower() == "deepseek":
            return DeepSeekContentProcessor.split_content_and_tool_calls(content)
        elif llm_type.lower() == "gemini":
            return GeminiContentProcessor.split_content_and_tool_calls(content)
        else:
            # Generic splitting
            generic_patterns = [
                r"```json\s*(\{.*?\})\s*```",
                r"(\w+\s*\([^)]*\))",
                r"<\w+>\s*(\{.*?\})\s*</\w+>",
            ]
            return ContentProcessor.split_content_and_tools(content, generic_patterns)


# Convenience functions for common operations


def extract_text_before_tool_calls(content: str, llm_type: str = "generic") -> str:
    """Extract text that appears before tool calls for the specified LLM type."""
    return ContentProcessorFactory.extract_text_before_tool_calls(content, llm_type)


def split_response_content(
    content: str, llm_type: str = "generic"
) -> tuple[str, List[str]]:
    """Split response content into text and tool call parts for the specified LLM type."""
    return ContentProcessorFactory.split_content_and_tool_calls(content, llm_type)


def clean_response_text(text: str) -> str:
    """Clean response text by removing code blocks and extra whitespace."""
    return ContentProcessor.clean_code_blocks(text)
