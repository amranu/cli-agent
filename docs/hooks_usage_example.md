# CLI-Agent Hooks System Usage Guide

## Configuration Setup

### 1. Create Configuration Directory

Create the hooks configuration directory:

```bash
mkdir -p .config/agent
```

### 2. Create Hooks Configuration

Create `.config/agent/settings.json` with your hook definitions:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "bash_execute",
        "hooks": [
          {
            "type": "command",
            "command": "echo '[SECURITY] About to execute: {{tool_args}}' >> ~/.cli-agent-audit.log"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "write_file|replace_in_file",
        "hooks": [
          {
            "type": "command",
            "command": "black '{{tool_args.file_path}}' 2>/dev/null || true",
            "timeout": 15
          },
          {
            "type": "command",
            "command": "git add '{{tool_args.file_path}}' 2>/dev/null || true",
            "timeout": 10
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
            "command": "notify-send 'CLI-Agent' '{{message}}' 2>/dev/null || true"
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
            "command": "echo '[SESSION] {{timestamp}} - Response complete' >> ~/.cli-agent-audit.log"
          }
        ]
      }
    ]
  }
}
```

## Configuration Locations

The hooks system checks these locations in order:

1. **Project-local (gitignored)**: `.config/agent/settings.local.json`
2. **Project-wide**: `.config/agent/settings.json`
3. **User-global**: `~/.config/agent/settings.json`
4. **CLI-Agent specific**: `~/.config/cli-agent/settings.json`

## Hook Types

### PreToolUse
Executed before any tool runs. Use for:
- Security auditing
- Parameter validation
- Resource preparation

### PostToolUse
Executed after tool completion. Use for:
- File formatting (black, prettier)
- Git operations
- Cleanup tasks
- Result processing

### Notification
Triggered on system messages. Use for:
- Desktop notifications
- Slack/Teams integration
- Logging

### Stop
Executed when agent finishes responding. Use for:
- Session logging
- Cleanup
- Statistics collection

## Template Variables

Available in all hook commands:

| Variable | Description | Hook Types |
|----------|-------------|------------|
| `{{tool_name}}` | Name of the tool | All |
| `{{tool_args}}` | JSON tool arguments | All |
| `{{timestamp}}` | ISO timestamp | All |
| `{{session_id}}` | Session identifier | All |
| `{{result}}` | Tool execution result | PostToolUse |
| `{{error}}` | Error message | PostToolUse |
| `{{execution_time}}` | Time in seconds | PostToolUse |
| `{{message}}` | Notification text | Notification |
| `{{conversation_length}}` | Message count | Stop |

## Pattern Matching

### Exact Match
```json
{"matcher": "bash_execute"}
```

### Wildcard (all tools)
```json
{"matcher": "*"}
```

### Prefix Match
```json
{"matcher": "builtin:*"}
```

### Multiple Tools
```json
{"matcher": "write_file|replace_in_file|multiedit"}
```

## Advanced Features

### JSON Output Control

Hooks can output JSON for advanced control:

```bash
#!/bin/bash
# security-check.sh
if echo "$1" | grep -q "rm -rf"; then
  echo '{"continue": false, "decision": "block", "reason": "Dangerous command detected"}'
  exit 2
else
  echo '{"continue": true, "decision": "approve"}'
  exit 0
fi
```

### Environment Variables

```json
{
  "type": "command",
  "command": "my-script.sh",
  "env": {
    "TOOL_NAME": "{{tool_name}}",
    "LOG_LEVEL": "INFO"
  }
}
```

### Working Directory

```json
{
  "type": "command", 
  "command": "git status",
  "working_directory": "{{tool_args.file_path | dirname}}"
}
```

## Common Use Cases

### 1. Development Workflow
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "write_file|replace_in_file",
        "hooks": [
          {
            "type": "command",
            "command": "black '{{tool_args.file_path}}' 2>/dev/null || true"
          },
          {
            "type": "command", 
            "command": "git add '{{tool_args.file_path}}' 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

### 2. Security Auditing
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "bash_execute",
        "hooks": [
          {
            "type": "command",
            "command": "echo '[AUDIT] {{timestamp}} - {{tool_args}}' >> ~/.security.log"
          }
        ]
      }
    ]
  }
}
```

### 3. Notifications
```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Task completed\" with title \"CLI-Agent\"'"
          }
        ]
      }
    ]
  }
}
```

## Environment Variables

Control hooks behavior with environment variables:

```bash
# Disable hooks entirely
export HOOKS_ENABLED=false

# Set custom timeout
export HOOKS_TIMEOUT=60
```

## User Feedback and Status

### Startup Confirmation
When hooks are loaded, you'll see a confirmation message:
```
🪝 Hooks enabled: 5 hooks loaded (PreToolUse, PostToolUse, Notification, Stop)
```

### Hook Execution Feedback
During tool execution, you'll see brief notifications:
```
🪝 Running 2 pre-tool hooks...
🪝 Running 1 post-tool hook...
```

### Status Command
Use the `/hooks` slash command to check status:
```
/hooks                    # Show current status
/hooks disable           # Temporarily disable hooks
/hooks enable            # Re-enable hooks
```

Example output:
```
🪝 Hooks System Status:
  Enabled: ✅ Yes
  Total hooks: 8

Hook Types:
  PreToolUse: 2 hooks (2 matchers)
  PostToolUse: 4 hooks (2 matchers)
  Notification: 1 hooks (1 matchers)
  Stop: 1 hooks (1 matchers)
```

## Troubleshooting

### Debug Hook Execution
Enable debug logging to see hook execution:

```bash
export LOG_LEVEL=DEBUG
```

### Validate Configuration
Use the test script to validate your configuration:

```bash
python test_hooks_system.py
```

### Common Issues

1. **Hooks not executing**: Check file paths and permissions
2. **Template variables not substituted**: Verify variable names match exactly
3. **Commands failing**: Test commands manually first
4. **Performance impact**: Use shorter timeouts for hooks

## Security Considerations

⚠️ **WARNING**: Hooks execute with full user permissions

- Review all hook commands carefully
- Avoid user input in hook commands
- Use absolute paths when possible
- Set appropriate timeouts
- Test hooks in safe environments first