"""
Tool call parsing utilities for different LLM response formats.

This module provides utilities for parsing tool calls from various LLM response
formats and creating standardized tool call objects.
"""

import json
import logging
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolCallParser:
    """Base utilities for parsing tool calls from different formats."""

    @staticmethod
    def create_tool_call_object(
        name: str, args: str, call_id: str = None
    ) -> SimpleNamespace:
        """Create a standardized tool call object similar to OpenAI's format."""
        tool_call = SimpleNamespace()
        tool_call.function = SimpleNamespace()
        tool_call.function.name = name
        tool_call.function.arguments = args
        tool_call.id = call_id or f"call_{name}_{int(time.time())}"
        tool_call.type = "function"

        return tool_call

    @staticmethod
    def validate_json_arguments(args_json: str) -> bool:
        """Validate that a string contains valid JSON."""
        try:
            json.loads(args_json)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def validate_tool_name(tool_name: str) -> bool:
        """Validate tool name format."""
        return tool_name and (tool_name.startswith("builtin_") or "_" in tool_name)

    @staticmethod
    def extract_text_before_tool_calls(content: str, patterns: List[str]) -> str:
        """Extract text that appears before tool calls using multiple patterns."""
        for pattern in patterns:
            try:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    text_before = match.group(1).strip()
                    if text_before:
                        # Clean up code block markers
                        text_before = re.sub(r"^```\w*\s*", "", text_before)
                        text_before = re.sub(r"\s*```$", "", text_before)
                        return text_before
            except Exception as e:
                logger.debug(f"Pattern {pattern} failed: {e}")
                continue

        return ""


class DeepSeekToolCallParser(ToolCallParser):
    """Parser for DeepSeek's custom tool calling formats."""

    # DeepSeek tool call patterns
    PATTERNS = [
        r"^(.*?)(?=<｜tool▁calls▁begin｜>|<｜tool▁call▁begin｜>)",
        r'^(.*?)(?=```json\s*\{\s*"function")',
        r"^(.*?)(?=```python\s*<｜tool▁calls▁begin｜>)",
        r'^(.*?)(?=\{[\s\S]*?"function"\s*:)',
    ]

    @classmethod
    def extract_text_before_calls(cls, content: str) -> str:
        """Extract text before DeepSeek tool calls."""
        return cls.extract_text_before_tool_calls(content, cls.PATTERNS)

    @classmethod
    def parse_tool_calls(cls, content: str) -> List[SimpleNamespace]:
        """Parse tool calls from DeepSeek response content."""
        tool_calls = []

        # Try to find tool call markers
        tool_call_patterns = [
            r"<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>",
            r"<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>",
            r'```json\s*(\{.*?"function".*?\})\s*```',
            r"```python\s*<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>",
        ]

        for pattern in tool_call_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    # Parse the JSON content
                    if match.strip().startswith("{"):
                        tool_data = json.loads(match.strip())
                    else:
                        # Handle multi-line tool calls
                        lines = match.strip().split("\n")
                        for line in lines:
                            line = line.strip()
                            if line.startswith("{") and line.endswith("}"):
                                tool_data = json.loads(line)
                                break
                        else:
                            continue

                    # Extract function information
                    if "function" in tool_data:
                        func_info = tool_data["function"]
                        function_name = func_info.get("name", "")
                        arguments = func_info.get("arguments", "{}")

                        # Validate function name format
                        if cls.validate_tool_name(function_name):
                            # Convert to tool call object
                            tool_call = cls.create_tool_call_object(
                                name=function_name,
                                args=(
                                    arguments
                                    if isinstance(arguments, str)
                                    else json.dumps(arguments)
                                ),
                                call_id=tool_data.get("id"),
                            )
                            tool_calls.append(tool_call)
                        else:
                            logger.warning(f"Invalid tool name format: {function_name}")

                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    logger.debug(f"Failed to parse tool call: {e}")
                    continue

        logger.debug(f"Parsed {len(tool_calls)} tool calls from DeepSeek response")
        return tool_calls


class GeminiToolCallParser(ToolCallParser):
    """Parser for Gemini's various tool calling formats."""

    @classmethod
    def parse_structured_tool_calls(cls, response: Any) -> List[SimpleNamespace]:
        """Parse tool calls from Gemini's structured response format."""
        tool_calls = []

        try:
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(
                    candidate.content, "parts"
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            func_call = part.function_call
                            tool_call = cls.create_tool_call_object(
                                name=func_call.name,
                                args=(
                                    json.dumps(dict(func_call.args))
                                    if func_call.args
                                    else "{}"
                                ),
                                call_id=f"gemini_{func_call.name}_{int(time.time())}",
                            )
                            tool_calls.append(tool_call)
        except Exception as e:
            logger.debug(f"Failed to parse structured Gemini tool calls: {e}")

        return tool_calls

    @classmethod
    def parse_python_style_calls(cls, text: str) -> List[SimpleNamespace]:
        """Parse Python-style function calls from text."""
        tool_calls = []

        # Pattern for Python-style function calls: function_name(arg1='value', arg2='value')
        pattern = r"(\w+)\s*\(\s*([^)]*)\s*\)"
        matches = re.findall(pattern, text)

        for func_name, args_str in matches:
            if cls.validate_tool_name(func_name):
                try:
                    # Parse arguments string into JSON
                    if args_str.strip():
                        # Simple parsing for key=value pairs
                        args_dict = {}
                        arg_pairs = re.findall(
                            r'(\w+)\s*=\s*[\'"]([^\'"]*)[\'"]', args_str
                        )
                        for key, value in arg_pairs:
                            args_dict[key] = value
                        args_json = json.dumps(args_dict)
                    else:
                        args_json = "{}"

                    tool_call = cls.create_tool_call_object(
                        name=func_name,
                        args=args_json,
                        call_id=f"python_{func_name}_{int(time.time())}",
                    )
                    tool_calls.append(tool_call)

                except Exception as e:
                    logger.debug(f"Failed to parse Python-style call {func_name}: {e}")

        return tool_calls

    @classmethod
    def parse_xml_style_calls(cls, content: str) -> List[SimpleNamespace]:
        """Parse XML-style tool calls from content."""
        tool_calls = []

        # Pattern for XML-style: <execute_tool>{"tool_name": "...", "parameters": {...}}</execute_tool>
        pattern = r"<execute_tool>\s*(\{.*?\})\s*</execute_tool>"
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            try:
                tool_data = json.loads(match)
                tool_name = tool_data.get("tool_name", "")
                parameters = tool_data.get("parameters", {})

                if cls.validate_tool_name(tool_name):
                    tool_call = cls.create_tool_call_object(
                        name=tool_name,
                        args=json.dumps(parameters),
                        call_id=f"xml_{tool_name}_{int(time.time())}",
                    )
                    tool_calls.append(tool_call)

            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Failed to parse XML-style tool call: {e}")

        return tool_calls

    @classmethod
    def parse_all_formats(
        cls, response: Any, text_content: str = ""
    ) -> List[SimpleNamespace]:
        """Parse tool calls from all supported Gemini formats."""
        tool_calls = []

        # Try structured format first
        structured_calls = cls.parse_structured_tool_calls(response)
        tool_calls.extend(structured_calls)

        # If no structured calls found, try parsing text content
        if not tool_calls and text_content:
            # Try Python-style calls
            python_calls = cls.parse_python_style_calls(text_content)
            tool_calls.extend(python_calls)

            # Try XML-style calls
            xml_calls = cls.parse_xml_style_calls(text_content)
            tool_calls.extend(xml_calls)

        logger.debug(f"Parsed {len(tool_calls)} tool calls from Gemini response")
        return tool_calls


class ToolCallParserFactory:
    """Factory for creating appropriate tool call parsers."""

    @staticmethod
    def create_parser(llm_type: str) -> ToolCallParser:
        """Create appropriate parser for LLM type."""
        parsers = {
            "deepseek": DeepSeekToolCallParser,
            "openai": ToolCallParser,  # Basic parser for standard OpenAI format
            "gemini": GeminiToolCallParser,
        }

        parser_class = parsers.get(llm_type.lower(), ToolCallParser)
        return parser_class()

    @staticmethod
    def parse_for_llm(
        llm_type: str, response: Any, content: str = ""
    ) -> List[SimpleNamespace]:
        """Convenience function to parse tool calls for specific LLM type."""
        if llm_type.lower() == "deepseek":
            return DeepSeekToolCallParser.parse_tool_calls(content)
        elif llm_type.lower() == "gemini":
            return GeminiToolCallParser.parse_all_formats(response, content)
        else:
            # For other LLMs, assume standard format
            return []
