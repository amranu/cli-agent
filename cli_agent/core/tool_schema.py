#!/usr/bin/env python3
"""Tool schema management and validation for MCP Agent."""

import json
import logging
import time
import types
from typing import Any, Dict, List, Optional

from cli_agent.utils.tool_name_utils import ToolNameUtils

logger = logging.getLogger(__name__)


class ToolSchemaManager:
    """Manages tool schema validation, normalization, and object creation."""

    @staticmethod
    def normalize_tool_name(tool_key: str) -> str:
        """Normalize tool name by replacing colons with underscores."""
        return ToolNameUtils.normalize_tool_name(tool_key)

    @staticmethod
    def generate_default_description(tool_info: dict) -> str:
        """Generate a default description for a tool if none exists."""
        return tool_info.get("description") or f"Execute {tool_info['name']} tool"

    @staticmethod
    def get_tool_schema(tool_info: dict) -> dict:
        """Get tool schema with fallback to basic object schema."""
        return tool_info.get("schema") or {"type": "object", "properties": {}}

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
    def create_tool_call_object(name: str, args: str, call_id: str = None):
        """Create a standardized tool call object."""
        # Create a SimpleNamespace object similar to OpenAI's format
        tool_call = types.SimpleNamespace()
        tool_call.function = types.SimpleNamespace()
        tool_call.function.name = name
        tool_call.function.arguments = args
        tool_call.id = call_id or f"call_{name}_{int(time.time())}"
        tool_call.type = "function"

        return tool_call

    @staticmethod
    def parse_tool_arguments(args: Any) -> Dict[str, Any]:
        """Parse tool arguments from various formats into a dict."""
        if args is None:
            return {}

        if isinstance(args, dict):
            return args

        if isinstance(args, str):
            try:
                return json.loads(args)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse tool arguments as JSON: {args}")
                return {}

        # Try to convert other types to dict
        try:
            if hasattr(args, "__dict__"):
                return vars(args)
            elif hasattr(args, "items"):
                return dict(args)
            else:
                logger.warning(f"Unknown argument type: {type(args)}")
                return {}
        except Exception as e:
            logger.warning(f"Failed to convert arguments to dict: {e}")
            return {}

    @staticmethod
    def format_tool_arguments(args: Dict[str, Any]) -> str:
        """Format tool arguments as JSON string."""
        try:
            return json.dumps(args, indent=None, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to format arguments as JSON: {e}")
            return "{}"

    @staticmethod
    def validate_tool_schema(schema: Dict[str, Any]) -> bool:
        """Validate that a tool schema is properly formatted."""
        if not isinstance(schema, dict):
            return False

        # Basic schema validation - should have type and properties
        if "type" not in schema:
            return False

        if schema["type"] == "object" and "properties" not in schema:
            return False

        return True

    @staticmethod
    def get_required_parameters(schema: Dict[str, Any]) -> List[str]:
        """Extract required parameters from tool schema."""
        if not isinstance(schema, dict):
            return []

        return schema.get("required", [])

    @staticmethod
    def get_tool_parameter_info(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract parameter information from tool schema."""
        if not isinstance(schema, dict) or "properties" not in schema:
            return {}

        return schema["properties"]
