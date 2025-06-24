"""
Tool conversion utilities for different LLM formats.

This module provides base classes and utilities for converting tools between
different LLM API formats while sharing common logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseToolConverter(ABC):
    """Base class for tool conversion with shared logic."""

    def normalize_tool_name(self, tool_key: str) -> str:
        """Normalize tool name by replacing colons with underscores."""
        return tool_key.replace(":", "_")

    def generate_description(self, tool_info: dict) -> str:
        """Generate tool description with fallback."""
        return (
            tool_info.get("description")
            or f"Execute {tool_info.get('name', 'unknown')} tool"
        )

    def get_base_schema(self, tool_info: dict) -> dict:
        """Get tool schema with fallback to basic object schema."""
        return tool_info.get("schema") or {"type": "object", "properties": {}}

    def validate_tool_info(self, tool_key: str, tool_info: dict) -> bool:
        """Validate that tool_info has required fields."""
        if not isinstance(tool_info, dict):
            logger.warning(
                f"Tool {tool_key} has invalid tool_info type: {type(tool_info)}"
            )
            return False

        required_fields = ["name"]
        for field in required_fields:
            if field not in tool_info:
                logger.warning(f"Tool {tool_key} missing required field: {field}")
                return False

        return True

    @abstractmethod
    def convert_tools(self, available_tools: Dict[str, Dict]) -> Any:
        """Convert tools to LLM-specific format."""
        pass


class OpenAIStyleToolConverter(BaseToolConverter):
    """Converter for OpenAI-style tools (DeepSeek, OpenAI, etc.)."""

    def convert_tools(self, available_tools: Dict[str, Dict]) -> List[Dict]:
        """Convert tools to OpenAI function calling format."""
        tools = []

        for tool_key, tool_info in available_tools.items():
            if not self.validate_tool_info(tool_key, tool_info):
                continue

            tool = {
                "type": "function",
                "function": {
                    "name": self.normalize_tool_name(tool_key),
                    "description": self.generate_description(tool_info),
                    "parameters": self.get_base_schema(tool_info),
                },
            }
            tools.append(tool)

        logger.debug(f"Converted {len(tools)} tools to OpenAI format")
        return tools


class GeminiToolConverter(BaseToolConverter):
    """Converter for Gemini tools with schema sanitization."""

    # Gemini unsupported properties
    UNSUPPORTED_PROPERTIES = {
        "additionalProperties",
        "additional_properties",
        "patternProperties",
        "dependencies",
        "definitions",
        "$ref",
        "$schema",
        "$id",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
    }

    def sanitize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize schema for Gemini API compatibility."""
        if not isinstance(schema, dict):
            return {"type": "OBJECT", "properties": {}}

        sanitized = {}

        for key, value in schema.items():
            # Skip unsupported properties
            if key in self.UNSUPPORTED_PROPERTIES:
                logger.debug(f"Skipping unsupported Gemini property: {key}")
                continue

            # Handle special properties
            if key == "type":
                # Convert to Gemini type format
                type_value = value.upper() if isinstance(value, str) else "OBJECT"
                sanitized[key] = type_value
            elif key == "properties" and isinstance(value, dict):
                # Recursively sanitize properties
                sanitized[key] = {
                    prop_name: self.sanitize_schema(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                # Sanitize array item schema
                sanitized[key] = self.sanitize_schema(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_schema(value)
            elif isinstance(value, list):
                # Sanitize array items if they're objects
                sanitized[key] = [
                    self.sanitize_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        # Ensure basic structure exists
        if "type" not in sanitized:
            sanitized["type"] = "OBJECT"

        if sanitized.get("type") == "OBJECT" and "properties" not in sanitized:
            sanitized["properties"] = {}

        return sanitized

    def convert_tools(self, available_tools: Dict[str, Dict]) -> List[Dict]:
        """Convert tools to Gemini function calling format."""
        function_declarations = []

        for tool_key, tool_info in available_tools.items():
            if not self.validate_tool_info(tool_key, tool_info):
                continue

            # Sanitize schema for Gemini
            raw_schema = self.get_base_schema(tool_info)
            sanitized_schema = self.sanitize_schema(raw_schema)

            function_declaration = {
                "name": self.normalize_tool_name(tool_key),
                "description": self.generate_description(tool_info),
                "parameters": sanitized_schema,
            }
            function_declarations.append(function_declaration)

        logger.debug(f"Converted {len(function_declarations)} tools to Gemini format")

        # Return as Gemini expects (would need types.Tool wrapper in actual usage)
        return function_declarations


class AnthropicToolConverter(BaseToolConverter):
    """Converter for Anthropic/Claude tools."""

    def convert_tools(self, available_tools: Dict[str, Dict]) -> List[Dict]:
        """Convert tools to Anthropic tool format."""
        tools = []

        for tool_key, tool_info in available_tools.items():
            if not self.validate_tool_info(tool_key, tool_info):
                continue

            tool = {
                "name": self.normalize_tool_name(tool_key),
                "description": self.generate_description(tool_info),
                "input_schema": self.get_base_schema(tool_info),
            }
            tools.append(tool)

        logger.debug(f"Converted {len(tools)} tools to Anthropic format")
        return tools


class ToolConverterFactory:
    """Factory for creating appropriate tool converters."""

    @staticmethod
    def create_converter(llm_type: str) -> BaseToolConverter:
        """Create appropriate tool converter for LLM type."""
        converters = {
            "openai": OpenAIStyleToolConverter,
            "deepseek": OpenAIStyleToolConverter,
            "gemini": GeminiToolConverter,
            "anthropic": AnthropicToolConverter,
        }

        converter_class = converters.get(llm_type.lower())
        if not converter_class:
            logger.warning(f"Unknown LLM type: {llm_type}, using OpenAI style")
            converter_class = OpenAIStyleToolConverter

        return converter_class()

    @staticmethod
    def get_supported_llm_types() -> List[str]:
        """Get list of supported LLM types."""
        return ["openai", "deepseek", "gemini", "anthropic"]


def convert_tools_for_llm(available_tools: Dict[str, Dict], llm_type: str) -> Any:
    """Convenience function to convert tools for specific LLM type."""
    converter = ToolConverterFactory.create_converter(llm_type)
    return converter.convert_tools(available_tools)
