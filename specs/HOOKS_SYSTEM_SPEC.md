# CLI-Agent Hooks System Specification

## Overview

This specification defines a hooks system for cli-agent that allows users to execute shell commands at specific points in the application's lifecycle. The system is designed to be compatible with Claude Code's hooks format while integrating seamlessly with cli-agent's existing event-driven architecture.

## Architecture

### Hook Types

The hooks system supports the following event types:

1. **PreToolUse** - Executed before any tool is called
2. **PostToolUse** - Executed after a tool completes (success or failure)  
3. **Notification** - Executed when the system emits notifications or status messages
4. **Stop** - Executed when the agent finishes responding to a user message

### Configuration Format

Hooks are configured using JSON files with the following structure:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "bash_execute",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'About to execute bash command'"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command", 
            "command": "notify-send 'Tool completed: {{tool_name}}'"
          }
        ]
      }
    ],
    "Notification": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "logger 'CLI-Agent: {{message}}'"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Response complete' >> ~/.cli-agent-log"
          }
        ]
      }
    ]
  }
}
```

### Configuration Locations

Hook configurations are loaded from the following locations (in order of precedence):

1. `.config/agent/settings.local.json` (project-local, gitignored)
2. `.config/agent/settings.json` (project-wide)
3. `~/.config/agent/settings.json` (user-global)
4. `~/.config/cli-agent/settings.json` (cli-agent specific)

### Matchers

Matchers determine which tools or events trigger hooks:

- `"*"` - Matches all tools/events
- `"bash_execute"` - Matches specific tool name
- `"builtin:*"` - Matches all built-in tools
- `"mcp__*"` - Matches all MCP tools
- `"write_file|replace_in_file"` - Pipe-separated alternatives (regex OR)

### Template Variables

Hook commands support template variable substitution:

- `{{tool_name}}` - Name of the tool being executed
- `{{tool_args}}` - JSON representation of tool arguments
- `{{result}}` - Tool execution result (PostToolUse only)
- `{{error}}` - Error message if tool failed (PostToolUse only)
- `{{message}}` - Notification message (Notification only)
- `{{timestamp}}` - Current timestamp (ISO format)
- `{{session_id}}` - Current session identifier
- `{{user_input}}` - Last user input text

### Hook Execution

#### Exit Codes

Hook commands communicate results via exit codes:

- `0` - Success, continue normal operation
- `2` - Blocking error, halt current operation
- `Other` - Non-blocking error, log warning and continue

#### JSON Output

For advanced control, hooks can output JSON to stdout:

```json
{
  "continue": true,
  "decision": "approve",
  "reason": "Command is safe to execute",
  "metadata": {
    "hook_name": "security_check",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

JSON Output Fields:
- `continue` (boolean) - Whether to continue operation
- `decision` (string) - "approve", "block", or "modify"
- `reason` (string) - Human-readable explanation
- `metadata` (object) - Additional context information
- `modified_args` (object) - Modified tool arguments (PreToolUse only)

## Implementation Components

### 1. Hook Configuration (`cli_agent/core/hooks/hook_config.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class HookType(Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse" 
    NOTIFICATION = "Notification"
    STOP = "Stop"

@dataclass
class HookDefinition:
    type: str  # "command"
    command: str
    timeout: Optional[int] = 30
    working_directory: Optional[str] = None
    env: Optional[Dict[str, str]] = None

@dataclass
class HookMatcher:
    matcher: str  # Pattern to match against
    hooks: List[HookDefinition]

@dataclass
class HookConfig:
    hooks: Dict[HookType, List[HookMatcher]]
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'HookConfig':
        """Load hook configuration from JSON file."""
        pass
    
    @classmethod  
    def load_from_multiple_sources(cls) -> 'HookConfig':
        """Load and merge hook configurations from all sources."""
        pass
```

### 2. Hook Executor (`cli_agent/core/hooks/hook_executor.py`)

```python
import asyncio
import json
import subprocess
from typing import Dict, Any, Optional, Tuple

@dataclass
class HookResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    json_output: Optional[Dict[str, Any]] = None
    should_continue: bool = True
    decision: str = "approve"
    reason: Optional[str] = None

class HookExecutor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        
    async def execute_hook(
        self,
        hook: HookDefinition,
        context: Dict[str, Any]
    ) -> HookResult:
        """Execute a single hook with template substitution."""
        
    def substitute_template_variables(
        self, 
        command: str, 
        context: Dict[str, Any]
    ) -> str:
        """Replace template variables in hook command."""
        
    def parse_json_output(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Parse JSON output from hook command."""
```

### 3. Hook Manager (`cli_agent/core/hooks/hook_manager.py`)

```python
from .hook_config import HookConfig, HookType
from .hook_executor import HookExecutor, HookResult
from ..event_system import EventBus, Event

class HookManager:
    def __init__(self, config: HookConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.executor = HookExecutor()
        
    async def execute_hooks(
        self,
        hook_type: HookType,
        context: Dict[str, Any]
    ) -> List[HookResult]:
        """Execute all matching hooks for a given type."""
        
    def match_hooks(
        self,
        hook_type: HookType, 
        tool_name: str
    ) -> List[HookDefinition]:
        """Find all hooks that match the given tool name."""
        
    def should_block_operation(self, results: List[HookResult]) -> Tuple[bool, str]:
        """Determine if hooks results indicate operation should be blocked."""
```

### 4. Hook Events (`cli_agent/core/hooks/hook_events.py`)

```python
from dataclasses import dataclass
from ..event_system import Event, EventType

class HookEventType(Enum):
    HOOK_EXECUTION_START = "hook_execution_start"
    HOOK_EXECUTION_COMPLETE = "hook_execution_complete"
    HOOK_EXECUTION_ERROR = "hook_execution_error"

@dataclass  
class HookExecutionStartEvent(Event):
    hook_type: str
    hook_command: str
    context: Dict[str, Any]
    
@dataclass
class HookExecutionCompleteEvent(Event):
    hook_type: str
    hook_command: str
    result: HookResult
    
@dataclass
class HookExecutionErrorEvent(Event):
    hook_type: str
    hook_command: str
    error: str
```

## Integration Points

### 1. Tool Execution Integration

In `cli_agent/core/tool_execution_engine.py`:

```python
class ToolExecutionEngine:
    def __init__(self, agent):
        self.agent = agent
        self.hook_manager = agent.hook_manager  # Initialize from agent
        
    async def execute_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        # Existing validation code...
        
        # PRE-TOOL HOOK EXECUTION
        if self.hook_manager:
            pre_context = {
                "tool_name": tool_name,
                "tool_args": arguments,
                "timestamp": datetime.now().isoformat(),
                "session_id": getattr(self.agent, 'session_id', 'unknown')
            }
            
            pre_results = await self.hook_manager.execute_hooks(
                HookType.PRE_TOOL_USE, pre_context
            )
            
            should_block, reason = self.hook_manager.should_block_operation(pre_results)
            if should_block:
                return f"Tool execution blocked by hook: {reason}"
                
            # Check for modified arguments
            for result in pre_results:
                if result.json_output and 'modified_args' in result.json_output:
                    arguments.update(result.json_output['modified_args'])
        
        # EXISTING TOOL EXECUTION CODE
        start_time = time.time()
        try:
            # ... existing execution logic ...
            result = await self.existing_tool_execution_logic(tool_name, arguments)
            execution_time = time.time() - start_time
            
            # POST-TOOL HOOK EXECUTION  
            if self.hook_manager:
                post_context = {
                    "tool_name": tool_name,
                    "tool_args": arguments,
                    "result": result,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": getattr(self.agent, 'session_id', 'unknown')
                }
                
                await self.hook_manager.execute_hooks(
                    HookType.POST_TOOL_USE, post_context
                )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # POST-TOOL HOOK EXECUTION (ERROR CASE)
            if self.hook_manager:
                post_context = {
                    "tool_name": tool_name,
                    "tool_args": arguments,
                    "error": str(e),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": getattr(self.agent, 'session_id', 'unknown')
                }
                
                await self.hook_manager.execute_hooks(
                    HookType.POST_TOOL_USE, post_context
                )
            
            raise
```

### 2. Event System Integration

In `cli_agent/core/event_system.py`:

```python
class EventBus:
    def __init__(self, session_id: Optional[str] = None):
        # ... existing initialization ...
        self.hook_manager: Optional[HookManager] = None
        
    def set_hook_manager(self, hook_manager: HookManager):
        """Set the hook manager for this event bus."""
        self.hook_manager = hook_manager
        
    async def emit(self, event: Event):
        # ... existing emit logic ...
        
        # Execute notification hooks for certain event types
        if (self.hook_manager and 
            event.event_type in {EventType.STATUS, EventType.SYSTEM_MESSAGE, EventType.ERROR}):
            
            context = {
                "message": self._extract_message_from_event(event),
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "session_id": event.session_id or self.session_id
            }
            
            asyncio.create_task(
                self.hook_manager.execute_hooks(HookType.NOTIFICATION, context)
            )
```

### 3. Configuration Integration

In `config.py`:

```python
class HostConfig(BaseSettings):
    # ... existing configuration ...
    
    hooks_enabled: bool = Field(default=True, alias="HOOKS_ENABLED")
    hooks_timeout: int = Field(default=30, alias="HOOKS_TIMEOUT")
    
    def load_hook_config(self) -> Optional[HookConfig]:
        """Load hook configuration from all sources."""
        if not self.hooks_enabled:
            return None
            
        return HookConfig.load_from_multiple_sources()
```

### 4. Agent Integration

In `cli_agent/core/base_agent.py`:

```python
class BaseMCPAgent(ABC):
    def __init__(self, config: HostConfig, is_subagent: bool = False):
        # ... existing initialization ...
        
        # Initialize hooks system
        self.hook_manager = None
        if not is_subagent:  # Only main agent uses hooks
            hook_config = config.load_hook_config()
            if hook_config:
                from cli_agent.core.hooks.hook_manager import HookManager
                self.hook_manager = HookManager(hook_config, self.event_bus)
                
                # Connect to event bus for Stop hooks
                self.event_bus.set_hook_manager(self.hook_manager)
                
    async def _finish_response(self):
        """Called when agent finishes responding to user."""
        if self.hook_manager:
            context = {
                "timestamp": datetime.now().isoformat(),
                "session_id": getattr(self, 'session_id', 'unknown'),
                "conversation_length": len(self.conversation_history)
            }
            
            await self.hook_manager.execute_hooks(HookType.STOP, context)
```

## Security Considerations

### Warnings and Documentation

1. **Security Warning**: Hooks execute with full user permissions and can perform any action the user can perform
2. **Configuration Security**: Hook configurations should be carefully reviewed before use
3. **Command Injection**: Template variable substitution is vulnerable to injection attacks if user input is not sanitized
4. **Resource Limits**: Hooks have configurable timeouts to prevent indefinite hanging

### Best Practices

1. **Minimal Permissions**: Use hooks for logging and notifications rather than system modifications
2. **Input Validation**: Sanitize template variables before substitution
3. **Error Handling**: Graceful handling of hook failures to prevent disrupting normal operation
4. **Logging**: Comprehensive logging of hook execution for debugging and security auditing

## Usage Examples

### 1. Development Workflow Automation

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "write_file|replace_in_file",
        "hooks": [
          {
            "type": "command",
            "command": "black {{tool_args.file_path}} 2>/dev/null || true"
          },
          {
            "type": "command", 
            "command": "git add {{tool_args.file_path}} 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

### 2. Security and Audit Logging

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "bash_execute",
        "hooks": [
          {
            "type": "command",
            "command": "echo '[$(date)] BASH: {{tool_args.command}}' >> ~/.cli-agent-audit.log"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '[$(date)] TOOL: {{tool_name}} - {{result}}' | head -c 200 >> ~/.cli-agent-audit.log"
          }
        ]
      }
    ]
  }
}
```

### 3. Notification Integration

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"CLI-Agent finished\" with title \"Task Complete\"'"
          }
        ]
      }
    ],
    "Notification": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "slack-cli send '#ai-agents' '{{message}}'"
          }
        ]
      }
    ]
  }
}
```

### 4. Advanced Control with JSON Output

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "bash_execute",
        "hooks": [
          {
            "type": "command",
            "command": "/usr/local/bin/security-check '{{tool_args.command}}'"
          }
        ]
      }
    ]
  }
}
```

Where `/usr/local/bin/security-check` outputs:

```json
{
  "continue": false,
  "decision": "block",
  "reason": "Command contains potentially dangerous operations",
  "metadata": {
    "risk_level": "high",
    "blocked_patterns": ["rm -rf", "sudo"]
  }
}
```

## Testing Strategy

### Unit Tests

1. **Hook Configuration Loading**: Test JSON parsing and validation
2. **Template Variable Substitution**: Test all variable types and edge cases  
3. **Hook Execution**: Test command execution, timeouts, and result parsing
4. **Hook Matching**: Test pattern matching logic
5. **Integration Points**: Mock testing of tool execution integration

### Integration Tests

1. **End-to-End Hook Flow**: Test complete hook execution in tool pipeline
2. **Configuration Sources**: Test precedence and merging of multiple config files
3. **Error Handling**: Test hook failures and recovery
4. **Performance**: Test hook execution performance impact

### Security Tests

1. **Command Injection**: Test resistance to malicious template variables
2. **Resource Exhaustion**: Test timeout and resource limit enforcement
3. **Permission Boundaries**: Test hook execution within expected permissions

## Future Enhancements

1. **Hook Debugging**: Debug mode for hook development and troubleshooting
2. **Hook Marketplace**: Sharable hook configurations for common workflows
3. **Conditional Hooks**: More sophisticated matching with conditions
4. **Async Hooks**: Background hook execution for non-blocking operations
5. **Hook Chaining**: Ability for hooks to trigger other hooks
6. **Web Hooks**: HTTP endpoint triggers in addition to shell commands

## Conclusion

This hooks system provides powerful extensibility for cli-agent while maintaining security awareness and integration with the existing architecture. The system is designed to be familiar to Claude Code users while taking advantage of cli-agent's superior event-driven architecture for more robust and flexible hook execution.