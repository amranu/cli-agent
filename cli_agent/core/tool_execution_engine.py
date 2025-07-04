"""Tool execution engine for MCP agents."""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from cli_agent.utils.tool_name_utils import ToolNameUtils

logger = logging.getLogger(__name__)


class ToolExecutionEngine:
    """Handles tool execution, validation, and coordination."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent
        # Debug: Log when tool execution engine is created
        try:
            with open("/tmp/tool_engine_init.txt", "a") as f:
                f.write(f"=== ToolExecutionEngine created ===\n")
                f.write(f"agent type: {type(agent)}\n")
                f.write(f"is_subagent: {getattr(agent, 'is_subagent', 'N/A')}\n")
                f.write("==================================\n")
        except:
            pass

    async def execute_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool (built-in or external) and return the result."""
        import json
        import os
        import sys

        try:
            # Debug logging for tool calls
            logger.debug(f"Tool call: {tool_key} with args: {arguments}")
            
            # Log tool execution for subagents
            if self.agent.is_subagent:
                logger.debug(f"Subagent tool execution: tool_key='{tool_key}', available_tools_count={len(self.agent.available_tools)}")
                logger.debug(f"Available tool keys: {list(self.agent.available_tools.keys())[:5]}...")
            
            # Handle special case for emit:result calls from subagents
            if tool_key == "emit:result" and self.agent.is_subagent:
                # Redirect to the proper emit_result tool
                tool_key = "builtin:emit_result"
                logger.debug(f"Redirected emit:result to {tool_key}")

            if tool_key not in self.agent.available_tools:
                # Check if this might be a role restriction
                role_restricted = False
                if hasattr(self.agent, '_role') and self.agent._role:
                    # Check if tool would be available without role filtering
                    from cli_agent.tools.builtin_tools import get_all_builtin_tools
                    all_builtin_tools = get_all_builtin_tools()
                    if tool_key in all_builtin_tools:
                        role_restricted = True
                        return f"Error: Tool '{tool_key}' is not available for role '{self.agent._role}'. This role only has access to specific tools."
                
                # Always try to resolve tool key using the centralized resolver
                resolved_key = ToolNameUtils.resolve_tool_key(
                    tool_key, self.agent.available_tools
                )
                if resolved_key:
                    logger.debug(f"Resolved tool '{tool_key}' to '{resolved_key}'")
                    tool_key = resolved_key

                # Final check if tool still not found
                if tool_key not in self.agent.available_tools:
                    # Show first 10 available tools when tool not found
                    available_list = list(self.agent.available_tools.keys())[:10]
                    if role_restricted:
                        return f"Error: Tool {tool_key} not available for role '{self.agent._role}'. Available tools: {available_list}"
                    else:
                        return f"Error: Tool {tool_key} not found. Available tools: {available_list}"

            tool_info = self.agent.available_tools[tool_key]
            tool_name = tool_info["name"]
            
            # EXECUTE PRE-TOOL HOOKS
            hook_manager = getattr(self.agent, 'hook_manager', None)
            if hook_manager and not self.agent.is_subagent:
                try:
                    from cli_agent.core.hooks.hook_config import HookType
                    
                    # Build context for pre-tool hooks
                    pre_context = {
                        "tool_name": tool_name,
                        "tool_args": arguments,
                        "timestamp": datetime.now().isoformat(),
                        "session_id": getattr(self.agent, 'session_id', 'unknown')
                    }
                    
                    # Execute pre-tool hooks
                    pre_results = await hook_manager.execute_hooks(HookType.PRE_TOOL_USE, pre_context)
                    
                    # Check if any hooks want to block the operation
                    should_block, reason = hook_manager.should_block_operation(pre_results)
                    if should_block:
                        return f"Tool execution blocked by hook: {reason}"
                    
                    # Check for modified arguments from hooks
                    modified_args = hook_manager.extract_modified_arguments(pre_results)
                    if modified_args:
                        arguments.update(modified_args)
                        logger.info(f"Hook modified tool arguments: {modified_args}")
                        
                except Exception as e:
                    logger.warning(f"Error executing pre-tool hooks: {e}")
                    # Continue with tool execution despite hook errors
            
            # Show clean diff preview and get user confirmation for file edits
            # This also serves as the permission check for edit tools
            edit_confirmed_by_diff = False
            if tool_name == "replace_in_file" and not self.agent.is_subagent:
                try:
                    from cli_agent.core.diff_display import get_clean_diff_display
                    from cli_agent.utils.diff_display import ColoredDiffDisplay

                    file_path = arguments.get("file_path", "")
                    old_text = arguments.get("old_text", "")
                    new_text = arguments.get("new_text", "")

                    if file_path and old_text:
                        # Try clean diff display first
                        clean_display = get_clean_diff_display()
                        user_confirmed = clean_display.show_single_edit_diff(
                            file_path=file_path,
                            old_text=old_text,
                            new_text=new_text
                        )
                        
                        if user_confirmed is None:
                            # Fallback to regular diff display if clean display not available
                            print("\n📝 Diff Preview:")
                            ColoredDiffDisplay.show_replace_diff(
                                file_path=file_path, old_text=old_text, new_text=new_text
                            )
                            # Continue with normal permission flow for fallback
                        elif user_confirmed is False:
                            # User declined the changes, raise exception to return to prompt
                            from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt
                            raise ToolDeniedReturnToPrompt("Edit cancelled by user")
                        elif user_confirmed is True:
                            # User confirmed via diff display, skip permission check
                            edit_confirmed_by_diff = True
                        
                except Exception as e:
                    # Don't fail tool execution if diff display fails
                    logger.warning(f"Failed to display diff preview: {e}")

            # Show clean diff preview and get user confirmation for multiedit
            # This also serves as the permission check for edit tools
            elif tool_name == "multiedit" and not self.agent.is_subagent:
                try:
                    from cli_agent.core.diff_display import get_clean_diff_display
                    from cli_agent.utils.diff_display import ColoredDiffDisplay

                    file_path = arguments.get("file_path", "")
                    edits = arguments.get("edits", [])

                    if file_path and edits:
                        # Try clean diff display first
                        clean_display = get_clean_diff_display()
                        user_confirmed = clean_display.show_multiedit_diff(
                            file_path=file_path,
                            edits=edits
                        )
                        
                        if user_confirmed is None:
                            # Fallback to regular diff display if clean display not available
                            print("\n📝 Multi-Edit Preview:")
                            
                            # Read file content once for all diff previews
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    current_content = f.read()
                            except FileNotFoundError:
                                print(f"Warning: File not found for diff preview: {file_path}")
                                current_content = ""
                            except Exception as e:
                                print(f"Warning: Could not read file for diff preview: {e}")
                                current_content = ""

                            if current_content:
                                # Show preview for each edit
                                for i, edit in enumerate(edits):
                                    old_string = edit.get("old_string", "")
                                    new_string = edit.get("new_string", "")

                                    if old_string:
                                        print(f"\n--- Edit {i+1} Preview ---")
                                        ColoredDiffDisplay.show_replace_diff(
                                            file_path=file_path,
                                            old_text=old_string,
                                            new_text=new_string,
                                            file_content=current_content,
                                            context_lines=3,
                                        )
                                        # Update content for next edit preview
                                        if old_string in current_content:
                                            replace_all = edit.get("replace_all", False)
                                            if replace_all:
                                                current_content = current_content.replace(
                                                    old_string, new_string
                                                )
                                            else:
                                                current_content = current_content.replace(
                                                    old_string, new_string, 1
                                                )
                            # Continue with normal permission flow for fallback
                        elif user_confirmed is False:
                            # User declined the changes, raise exception to return to prompt
                            from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt
                            raise ToolDeniedReturnToPrompt("Multi-edit cancelled by user")
                        elif user_confirmed is True:
                            # User confirmed via diff display, skip permission check
                            edit_confirmed_by_diff = True
                        
                except Exception as e:
                    # Don't fail tool execution if diff display fails
                    logger.warning(f"Failed to display multiedit diff preview: {e}")

            # Forward to parent if this is a subagent (except for subagent management tools)
            # This must happen BEFORE permission checks so parent can handle permissions
            if self.agent.is_subagent and self.agent.comm_socket:
                excluded_tools = ["task", "task_status", "task_results", "websearch", "webfetch"]
                if tool_name not in excluded_tools:
                    # Tool forwarding happens silently - parent will handle permissions
                    return await self.agent._forward_tool_to_parent(
                        tool_key, tool_name, arguments
                    )

            # Check tool permissions (main agent only, since subagents forward to parent)
            # Skip permission checks for subagents if bypass is enabled
            bypass_subagent_permissions = (
                self.agent.is_subagent
                and hasattr(self.agent, "config")
                and getattr(self.agent.config, "subagent_permissions_bypass", False)
            )

            if bypass_subagent_permissions:
                logger.info(
                    f"Bypassing permission check for subagent tool: {tool_name} (SUBAGENT_PERMISSIONS_BYPASS=true)"
                )

            # Skip ALL permission checks if in stream-json mode or edit already confirmed by diff
            if os.environ.get("STREAM_JSON_MODE") == "true":
                logger.info(f"Skipping all permission checks for tool {tool_name} (stream-json mode)")
            elif edit_confirmed_by_diff:
                logger.info(f"Skipping permission check for {tool_name} (already confirmed via diff display)")
            elif (
                hasattr(self.agent, "permission_manager")
                and self.agent.permission_manager
                and not bypass_subagent_permissions
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
                    # Only main agents reach here (subagents forward to parent)
                    raise ToolDeniedReturnToPrompt(permission_result.reason)
            elif self.agent.is_subagent:
                # Subagents should always check permissions if they have a permission manager
                # The SubagentInputHandler will communicate with the main process for permission prompts
                if hasattr(self.agent, "permission_manager") and self.agent.permission_manager:
                    from cli_agent.core.tool_permissions import (
                        ToolDeniedReturnToPrompt,
                        ToolPermissionResult,
                    )

                    input_handler = getattr(self.agent, "_input_handler", None)
                    
                    # Debug: Show file path to confirm we're using local code
                    import inspect
                    current_file = inspect.getfile(inspect.currentframe())
                    sys.stderr.write(f"🤖 [SUBAGENT] PERMISSION CHECK from {current_file}\n")
                    sys.stderr.write(f"🤖 [SUBAGENT] Checking permissions for {tool_name} with handler {type(input_handler).__name__ if input_handler else 'None'}\n")
                    sys.stderr.flush()
                    
                    permission_result = (
                        await self.agent.permission_manager.check_tool_permission(
                            tool_name, arguments, input_handler
                        )
                    )

                    sys.stderr.write(f"🤖 [SUBAGENT] Permission result: {permission_result.allowed} - {permission_result.reason}\n")
                    sys.stderr.flush()

                    if not permission_result.allowed:
                        raise ToolDeniedReturnToPrompt(permission_result.reason)
                else:
                    sys.stderr.write(f"🤖 [SUBAGENT] No permission manager, executing directly: {tool_name}\n")
                    sys.stderr.flush()

            # EXECUTE TOOL WITH POST-HOOK INTEGRATION
            start_time = time.time()
            tool_result = None
            tool_error = None
            
            try:
                # Check if it's a built-in tool
                if tool_info["server"] == "builtin":
                    logger.info(
                        f"Executing built-in tool: {tool_name} with arguments: {arguments}"
                    )
                    tool_result = await self.agent._execute_builtin_tool(tool_name, arguments)
                else:
                    # Handle external MCP tools with FastMCP
                    client = tool_info["client"]
                    if client is None:
                        return f"Error: No client session for tool {tool_key}"

                    logger.info(f"Executing MCP tool: {tool_name} with arguments: {arguments}")
                    result = await client.call_tool(tool_name, arguments)
                    tool_result = result
                    
            except Exception as e:
                tool_error = e
                execution_time = time.time() - start_time
                
                # EXECUTE POST-TOOL HOOKS (ERROR CASE)
                if hook_manager and not self.agent.is_subagent:
                    try:
                        post_context = {
                            "tool_name": tool_name,
                            "tool_args": arguments,
                            "error": str(e),
                            "execution_time": execution_time,
                            "timestamp": datetime.now().isoformat(),
                            "session_id": getattr(self.agent, 'session_id', 'unknown')
                        }
                        
                        await hook_manager.execute_hooks(HookType.POST_TOOL_USE, post_context)
                    except Exception as hook_error:
                        logger.warning(f"Error executing post-tool hooks (error case): {hook_error}")
                
                # Re-raise the original tool error
                raise e
            
            execution_time = time.time() - start_time
            
            # EXECUTE POST-TOOL HOOKS (SUCCESS CASE)
            if hook_manager and not self.agent.is_subagent:
                try:
                    post_context = {
                        "tool_name": tool_name,
                        "tool_args": arguments,
                        "result": str(tool_result),
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat(),
                        "session_id": getattr(self.agent, 'session_id', 'unknown')
                    }
                    
                    await hook_manager.execute_hooks(HookType.POST_TOOL_USE, post_context)
                except Exception as e:
                    logger.warning(f"Error executing post-tool hooks (success case): {e}")
            
            # Return built-in tool result directly
            if tool_info["server"] == "builtin":
                return tool_result
            
            # Handle MCP tool result formatting
            result = tool_result
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
            elif isinstance(result, list):
                # Handle case where result is directly a list of content objects (FastMCP format)
                content_parts = []
                for content in result:
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
            from cli_agent.core.tool_permissions import ToolDeniedReturnToPrompt, ToolExecutionErrorReturnToPrompt

            if isinstance(e, ToolDeniedReturnToPrompt):
                raise  # Re-raise the exception to bubble up to interactive chat

            # Check for common error types that should clear conversation and add error context
            error_str = str(e)
            if any(error_indicator in error_str.lower() for error_indicator in [
                "expecting value", "json", "parse", "decode", "invalid", "syntax error",
                "connection", "timeout", "network", "api", "authentication"
            ]):
                # These errors indicate API/parsing issues that corrupt conversation state
                logger.error(f"Tool execution error that requires conversation cleanup: {tool_key}: {e}")
                raise ToolExecutionErrorReturnToPrompt(
                    tool_name=tool_key, 
                    error_message=error_str, 
                    original_error=e
                )

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
