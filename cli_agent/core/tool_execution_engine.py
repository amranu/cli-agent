"""Tool execution engine for MCP agents."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToolExecutionEngine:
    """Handles tool execution, validation, and coordination."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    async def execute_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool (built-in or external) and return the result."""
        import json
        import sys

        try:
            if tool_key not in self.agent.available_tools:
                # Try reverse normalization: convert underscores back to colons
                # This handles cases where LLM calls "ai-models_deepseek_chat" but tool is stored as "ai-models:deepseek_chat"
                denormalized_key = (
                    tool_key.replace("_", ":", 1) if "_" in tool_key else tool_key
                )

                if denormalized_key in self.agent.available_tools:
                    tool_key = denormalized_key  # Use the original key format
                else:
                    # Debug: show available tools when tool not found
                    available_list = list(self.agent.available_tools.keys())[
                        :10
                    ]  # First 10 tools
                    return f"Error: Tool {tool_key} not found. Available tools: {available_list}"

            tool_info = self.agent.available_tools[tool_key]
            tool_name = tool_info["name"]

            # Show diff preview for replace_in_file before permission check
            if tool_name == "replace_in_file" and not self.agent.is_subagent:
                try:
                    from cli_agent.utils.diff_display import ColoredDiffDisplay

                    file_path = arguments.get("file_path", "")
                    old_text = arguments.get("old_text", "")
                    new_text = arguments.get("new_text", "")

                    if file_path and old_text:
                        # Show colored diff preview
                        ColoredDiffDisplay.show_replace_diff(
                            file_path=file_path, old_text=old_text, new_text=new_text
                        )
                except Exception as e:
                    # Don't fail tool execution if diff display fails
                    logger.warning(f"Failed to display diff preview: {e}")

            # Check tool permissions (both main agent and subagents)
            if (
                hasattr(self.agent, "permission_manager")
                and self.agent.permission_manager
            ):
                from cli_agent.core.tool_permissions import (
                    ToolDeniedReturnToPrompt,
                    ToolPermissionResult,
                )

                input_handler = getattr(self.agent, "_input_handler", None)
                permission_result = (
                    await self.agent.permission_manager.check_tool_permission(
                        tool_name, arguments, input_handler
                    )
                )

                if not permission_result.allowed:
                    if (
                        permission_result.return_to_prompt
                        and not self.agent.is_subagent
                    ):
                        # Only return to prompt for main agent, not subagents
                        raise ToolDeniedReturnToPrompt(permission_result.reason)
                    else:
                        # For subagents or config-based denials, return error message
                        return f"Tool execution denied: {permission_result.reason}"

            # Forward to parent if this is a subagent (except for subagent management tools)
            if self.agent.is_subagent and self.agent.comm_socket:
                excluded_tools = ["task", "task_status", "task_results"]
                if tool_name not in excluded_tools:
                    # Tool forwarding happens silently
                    return await self.agent._forward_tool_to_parent(
                        tool_key, tool_name, arguments
                    )
            elif self.agent.is_subagent:
                sys.stderr.write(
                    f"🤖 [SUBAGENT] WARNING: is_subagent=True but no comm_socket for tool {tool_name}\n"
                )
                sys.stderr.flush()

            # Check if it's a built-in tool
            if tool_info["server"] == "builtin":
                logger.info(f"Executing built-in tool: {tool_name}")
                return await self.agent._execute_builtin_tool(tool_name, arguments)

            # Handle external MCP tools with FastMCP
            client = tool_info["client"]
            if client is None:
                return f"Error: No client session for tool {tool_key}"

            logger.info(f"Executing MCP tool: {tool_name}")
            result = await client.call_tool(tool_name, arguments)

            # Format the result for FastMCP
            if hasattr(result, "content") and result.content:
                content_parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        content_parts.append(content.text)
                    elif hasattr(content, "data"):
                        content_parts.append(str(content.data))
                    else:
                        content_parts.append(str(content))
                return "\n".join(content_parts)
            elif isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return f"Tool executed successfully. Result type: {type(result)}, Content: {result}"

        except Exception as e:
            # Re-raise tool permission denials so they can be handled at the chat level
            from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt

            if isinstance(e, ToolDeniedReturnToPrompt):
                raise  # Re-raise the exception to bubble up to interactive chat

            logger.error(f"Error executing tool {tool_key}: {e}")
            return f"Error executing tool {tool_key}: {str(e)}"

    async def execute_tool_calls_batch(
        self, tool_calls: List[Dict[str, Any]], messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls in batch and update messages."""
        # Delegate to agent's existing method
        if hasattr(self.agent, "_execute_tool_calls_batch"):
            return await self.agent._execute_tool_calls_batch(tool_calls, messages)
        else:
            # Fallback implementation
            for tool_call in tool_calls:
                try:
                    result = await self.execute_mcp_tool(
                        tool_call["name"], tool_call.get("arguments", {})
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.get("id"),
                        }
                    )
                except Exception as e:
                    messages.append(
                        {
                            "role": "tool",
                            "content": f"Error executing {tool_call['name']}: {str(e)}",
                            "tool_call_id": tool_call.get("id"),
                        }
                    )
            return messages

    def validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """Validate a tool call has required fields."""
        required_fields = ["name"]
        return all(field in tool_call for field in required_fields)

    def get_available_tools(self) -> Dict[str, Any]:
        """Get available tools - delegates to agent."""
        return self.agent.available_tools

    def format_tool_result(
        self, tool_name: str, result: Any, error: Optional[str] = None
    ) -> str:
        """Format tool execution result for conversation."""
        if error:
            return f"Error executing {tool_name}: {error}"

        if isinstance(result, dict):
            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)

        return str(result)

    async def handle_tool_permission_check(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> bool:
        """Check if tool execution is permitted - delegates to agent."""
        if hasattr(self.agent, "_check_tool_permissions"):
            return await self.agent._check_tool_permissions(tool_name, arguments)
        return True  # Default to allowing all tools

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool."""
        tools = self.get_available_tools()
        for tool_key, tool_info in tools.items():
            if tool_info.get("name") == tool_name or tool_key == tool_name:
                return tool_info.get("schema")
        return None

    def validate_tool_arguments(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> bool:
        """Validate tool arguments against schema."""
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return True  # No schema, assume valid

        # Basic validation - could be enhanced with jsonschema
        required_params = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required parameters
        for param in required_params:
            if param not in arguments:
                logger.warning(
                    f"Missing required parameter '{param}' for tool '{tool_name}'"
                )
                return False

        # Check parameter types
        for param, value in arguments.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type and not self._check_type_match(value, expected_type):
                    logger.warning(
                        f"Parameter '{param}' type mismatch for tool '{tool_name}'"
                    )
                    return False

        return True

    def _check_type_match(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type in type_mapping:
            return isinstance(value, type_mapping[expected_type])

        return True  # Unknown type, assume valid
