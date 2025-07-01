"""
Hook manager for coordinating hook execution and integration with the event system.

This module provides the main interface for the hooks system, coordinating
between hook configuration, execution, and event emission.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from .hook_config import HookConfig, HookType
from .hook_executor import HookExecutor, HookResult
from .hook_events import (
    HookExecutionStartEvent,
    HookExecutionCompleteEvent,
    HookExecutionErrorEvent,
)

logger = logging.getLogger(__name__)


class HookManager:
    """Manages hook execution and coordinates with the event system."""
    
    def __init__(self, config: HookConfig, event_bus=None):
        """Initialize hook manager with configuration and optional event bus."""
        self.config = config
        self.event_bus = event_bus
        self.executor = HookExecutor()
        self._hooks_enabled = True
        
        # Validate configuration on startup
        validation_errors = config.validate()
        if validation_errors:
            logger.warning(f"Hook configuration validation errors: {validation_errors}")
            
        logger.info(f"Hook manager initialized with {len(config.hooks)} hook types")
    
    def disable_hooks(self):
        """Temporarily disable all hook execution."""
        self._hooks_enabled = False
        logger.info("Hooks disabled")
    
    def enable_hooks(self):
        """Re-enable hook execution."""
        self._hooks_enabled = True
        logger.info("Hooks enabled")
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        context: Dict[str, Any]
    ) -> List[HookResult]:
        """Execute all matching hooks for a given type and context."""
        if not self._hooks_enabled:
            logger.debug(f"Hooks disabled, skipping {hook_type.value} hooks")
            return []
            
        tool_name = context.get("tool_name", "*")
        matching_hooks = self.config.get_matching_hooks(hook_type, tool_name)
        
        if not matching_hooks:
            logger.debug(f"No matching {hook_type.value} hooks for tool: {tool_name}")
            return []
            
        # Log hook execution (avoid print statements that can trigger more events)
        hook_type_display = hook_type.value.replace("ToolUse", "-tool").lower()
        logger.info(f"Executing {len(matching_hooks)} {hook_type_display} hook{'s' if len(matching_hooks) != 1 else ''} for tool: {tool_name}")
        
        results = []
        for hook_def in matching_hooks:
            try:
                # Emit hook execution start event
                if self.event_bus:
                    start_event = HookExecutionStartEvent(
                        hook_type=hook_type.value,
                        hook_command=hook_def.command,
                        context=context
                    )
                    # Add hook name to the event data
                    if hook_def.name:
                        start_event.data["hook_name"] = hook_def.name
                    await self.event_bus.emit(start_event)
                
                # Execute the hook
                result = await self.executor.execute_hook(hook_def, context)
                results.append(result)
                
                # Log hook result
                if result.success:
                    logger.debug(
                        f"Hook executed successfully: {hook_def.command[:50]}... "
                        f"(exit_code={result.exit_code}, time={result.execution_time:.3f}s)"
                    )
                else:
                    logger.warning(
                        f"Hook execution failed: {hook_def.command[:50]}... "
                        f"(exit_code={result.exit_code}, error={result.stderr[:100]})"
                    )
                
                # Emit hook execution complete event
                if self.event_bus:
                    complete_event = HookExecutionCompleteEvent(
                        hook_type=hook_type.value,
                        hook_command=hook_def.command,
                        result=result
                    )
                    await self.event_bus.emit(complete_event)
                    
            except Exception as e:
                logger.error(f"Unexpected error executing hook: {e}")
                
                # Create error result
                error_result = HookResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Hook manager error: {str(e)}",
                    execution_time=0.0
                )
                results.append(error_result)
                
                # Emit hook execution error event
                if self.event_bus:
                    error_event = HookExecutionErrorEvent(
                        hook_type=hook_type.value,
                        hook_command=hook_def.command,
                        error=str(e)
                    )
                    await self.event_bus.emit(error_event)
        
        return results
    
    def should_block_operation(self, results: List[HookResult]) -> Tuple[bool, str]:
        """Determine if hook results indicate the operation should be blocked."""
        blocking_reasons = []
        
        for result in results:
            if not result.should_continue:
                reason = result.reason or "Hook requested operation be blocked"
                blocking_reasons.append(reason)
                logger.info(f"Hook blocking operation: {reason}")
        
        if blocking_reasons:
            combined_reason = "; ".join(blocking_reasons)
            return True, combined_reason
            
        return False, ""
    
    def extract_modified_arguments(
        self, 
        results: List[HookResult]
    ) -> Optional[Dict[str, Any]]:
        """Extract modified arguments from hook results (PreToolUse only)."""
        for result in results:
            if result.json_output and "modified_args" in result.json_output:
                modified_args = result.json_output["modified_args"]
                if isinstance(modified_args, dict):
                    logger.info(f"Hook provided modified arguments: {modified_args}")
                    return modified_args
                    
        return None
    
    def get_hook_summary(self) -> Dict[str, Any]:
        """Get a summary of the current hook configuration."""
        summary = {
            "enabled": self._hooks_enabled,
            "hook_types": {},
            "total_hooks": 0
        }
        
        for hook_type, matchers in self.config.hooks.items():
            hook_count = sum(len(matcher.hooks) for matcher in matchers)
            summary["hook_types"][hook_type.value] = {
                "matchers": len(matchers),
                "hooks": hook_count
            }
            summary["total_hooks"] += hook_count
            
        return summary
    
    def match_hooks_for_tool(
        self,
        hook_type: HookType, 
        tool_name: str
    ) -> List[str]:
        """Get list of hook commands that would match for a tool (for debugging)."""
        matching_hooks = self.config.get_matching_hooks(hook_type, tool_name)
        return [hook.command for hook in matching_hooks]
    
async def create_hook_manager_from_config() -> Optional[HookManager]:
    """Create a hook manager by loading configuration from standard sources."""
    try:
        config = HookConfig.load_from_multiple_sources()
        
        if not config.has_hooks():
            logger.debug("No hook configuration found")
            return None
            
        return HookManager(config)
        
    except Exception as e:
        logger.error(f"Failed to create hook manager: {e}")
        return None