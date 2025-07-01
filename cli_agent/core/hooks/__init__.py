"""
Hooks system for CLI-Agent.

This module provides a hooks system similar to Claude Code's, allowing users
to execute shell commands at specific points in the application lifecycle.
"""

from .hook_config import HookConfig, HookType, HookDefinition, HookMatcher
from .hook_executor import HookExecutor, HookResult
from .hook_manager import HookManager
from .hook_events import (
    HookEventType,
    HookExecutionStartEvent,
    HookExecutionCompleteEvent,
    HookExecutionErrorEvent,
)

__all__ = [
    "HookConfig",
    "HookType", 
    "HookDefinition",
    "HookMatcher",
    "HookExecutor",
    "HookResult",
    "HookManager",
    "HookEventType",
    "HookExecutionStartEvent",
    "HookExecutionCompleteEvent", 
    "HookExecutionErrorEvent",
]