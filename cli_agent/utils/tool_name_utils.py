"""Utilities for tool name handling and normalization."""

from typing import Dict, List, Optional


class ToolNameUtils:
    """Centralized utilities for tool name normalization and resolution."""

    @staticmethod
    def normalize_tool_name(tool_name: str) -> str:
        """Normalize tool name for OpenAI API compatibility (replace colons with underscores)."""
        return tool_name.replace(":", "_")

    @staticmethod
    def denormalize_tool_name(normalized_name: str) -> str:
        """Convert normalized tool name back to original format."""
        # This is a simple reverse operation, but could be enhanced with mapping
        return normalized_name.replace("_", ":", 1)

    @staticmethod
    def find_tool_key_candidates(tool_key: str) -> List[str]:
        """Generate candidate tool keys for lookup when exact match fails."""
        candidates = []

        # Strategy 1: Add builtin: prefix if it doesn't exist
        if not tool_key.startswith("builtin:") and not tool_key.startswith("mcp:"):
            candidates.append(f"builtin:{tool_key}")

        # Strategy 2: Replace first underscore with colon (for MCP tools)
        if "_" in tool_key:
            candidates.append(tool_key.replace("_", ":", 1))

        # Strategy 3: Replace all underscores with colons
        if "_" in tool_key:
            candidates.append(tool_key.replace("_", ":"))
            
        # Strategy 4: Handle partial tool names for common built-ins
        # This prevents LLMs from calling partial tool names like "result" instead of "emit_result"
        if tool_key == "result":
            candidates.extend(["builtin:emit_result", "emit_result", "builtin_emit_result"])

        return candidates

    @staticmethod
    def resolve_tool_key(
        tool_key: str, available_tools: Dict[str, Dict]
    ) -> Optional[str]:
        """Resolve tool key by trying the key itself and various candidates."""
        # First try exact match
        if tool_key in available_tools:
            return tool_key

        # Try candidates
        candidates = ToolNameUtils.find_tool_key_candidates(tool_key)
        for candidate in candidates:
            if candidate in available_tools:
                return candidate

        return None

    @staticmethod
    def extract_tool_name_from_call(tool_call) -> str:
        """Extract tool name from various tool call formats with fallback."""
        if isinstance(tool_call, dict):
            if "function" in tool_call and isinstance(tool_call["function"], dict):
                # Standard OpenAI format: tc["function"]["name"]
                return tool_call["function"].get(
                    "name", f"<missing_function_name_{id(tool_call)}>"
                )
            else:
                # Fallback for other formats: tc["name"]
                return tool_call.get("name", f"<missing_dict_name_{id(tool_call)}>")
        else:
            # Object format
            if hasattr(tool_call, "function"):
                return getattr(
                    tool_call.function,
                    "name",
                    f"<missing_obj_function_name_{id(tool_call)}>",
                )
            else:
                return getattr(tool_call, "name", f"<missing_obj_name_{id(tool_call)}>")

    @staticmethod
    def create_normalized_tools_mapping(
        available_tools: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Create a mapping that includes both original and normalized tool names."""
        normalized_tools = available_tools.copy()

        for tool_key, tool_info in available_tools.items():
            normalized_key = ToolNameUtils.normalize_tool_name(tool_key)
            if normalized_key != tool_key:
                normalized_tools[normalized_key] = tool_info.copy()

        return normalized_tools
