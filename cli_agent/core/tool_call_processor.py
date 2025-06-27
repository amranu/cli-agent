"""Tool call processing framework for MCP agents."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ToolCallProcessor:
    """Handles tool call parsing, normalization, and validation."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    def parse_tool_calls_generic(
        self, response: Any, text_content: str = ""
    ) -> List[Dict[str, Any]]:
        """Unified tool call parsing pipeline with provider hooks."""
        tool_calls = []

        # Step 1: Extract structured tool calls (provider-specific)
        structured_calls = self.extract_structured_calls(response)
        tool_calls.extend(structured_calls)

        # Step 2: Parse text-based tool calls (provider-specific patterns)
        if text_content:
            text_calls = self.parse_text_based_calls(text_content)
            tool_calls.extend(text_calls)

        # Step 3: Normalize all tool calls to standard format
        normalized_calls = self.normalize_tool_calls_unified(tool_calls)

        return normalized_calls

    def extract_structured_calls(self, response: Any) -> List[Any]:
        """Extract structured tool calls from response - delegates to agent."""
        return self.agent._extract_structured_calls(response)

    def parse_text_based_calls(self, text_content: str) -> List[Any]:
        """Parse text-based tool calls using provider patterns - delegates to agent."""
        return self.agent._parse_text_based_calls(text_content)

    def normalize_tool_calls_unified(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Unified tool call normalization."""
        normalized_calls = []

        for i, tool_call in enumerate(tool_calls):
            if hasattr(tool_call, "name") and hasattr(tool_call, "args"):
                # SimpleNamespace or object format
                normalized_calls.append(
                    {
                        "id": getattr(tool_call, "id", f"call_{i}"),
                        "name": tool_call.name,
                        "arguments": tool_call.args,
                    }
                )
            elif isinstance(tool_call, dict):
                if "function" in tool_call:
                    # OpenAI-style format - preserve original name even if None/empty
                    function_name = tool_call["function"].get("name")
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": (
                                function_name
                                if function_name is not None
                                else f"<missing_name_in_function_{i}>"
                            ),
                            "arguments": tool_call["function"].get("arguments", {}),
                        }
                    )
                else:
                    # Simple dict format - preserve original name even if None/empty
                    tool_name = tool_call.get("name")
                    normalized_calls.append(
                        {
                            "id": tool_call.get("id", f"call_{i}"),
                            "name": (
                                tool_name
                                if tool_name is not None
                                else f"<missing_name_{i}>"
                            ),
                            "arguments": tool_call.get("arguments", {}),
                        }
                    )
            else:
                # Fallback
                normalized_calls.append(
                    {
                        "id": f"call_{i}",
                        "name": str(tool_call),
                        "arguments": {},
                    }
                )

        return normalized_calls

    def extract_and_normalize_tool_calls(
        self, response: Any, content: str = ""
    ) -> List[Dict[str, Any]]:
        """Extract and normalize tool calls from response."""
        # Use agent's existing method if available, otherwise use generic parsing
        if hasattr(self.agent, "_extract_and_normalize_tool_calls"):
            return self.agent._extract_and_normalize_tool_calls(response, content)
        else:
            return self.parse_tool_calls_generic(response, content)

    def validate_and_convert_tool_calls(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """Validate and convert tool calls to standard format."""
        # Use agent's existing method if available
        if hasattr(self.agent, "_validate_and_convert_tool_calls"):
            return self.agent._validate_and_convert_tool_calls(tool_calls)
        else:
            return self.normalize_tool_calls_unified(tool_calls)
