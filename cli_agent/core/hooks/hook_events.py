"""
Hook-specific event types for the CLI-Agent event system.

This module extends the core event system with events specifically
related to hook execution and management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from ..event_system import Event, EventType


class HookEventType(Enum):
    """Event types specific to the hooks system."""
    
    HOOK_EXECUTION_START = "hook_execution_start"
    HOOK_EXECUTION_COMPLETE = "hook_execution_complete"  
    HOOK_EXECUTION_ERROR = "hook_execution_error"


@dataclass
class HookExecutionStartEvent(Event):
    """Event emitted when a hook starts executing."""
    
    hook_type: str = ""  # PreToolUse, PostToolUse, etc.
    hook_command: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    system_type: str = "hook_execution_start"  # Required for SYSTEM events
    data: Dict[str, Any] = field(default_factory=dict)  # Required for SYSTEM events
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.SYSTEM  # Use SYSTEM event type for hook events


@dataclass
class HookExecutionCompleteEvent(Event):
    """Event emitted when a hook completes execution."""
    
    hook_type: str = ""
    hook_command: str = ""
    result: Any = None  # HookResult object
    system_type: str = "hook_execution_complete"  # Required for SYSTEM events
    data: Dict[str, Any] = field(default_factory=dict)  # Required for SYSTEM events
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.SYSTEM


@dataclass
class HookExecutionErrorEvent(Event):
    """Event emitted when a hook execution encounters an error."""
    
    hook_type: str = ""
    hook_command: str = ""
    error: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.ERROR