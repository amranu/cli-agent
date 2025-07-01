# CLI-Agent Hooks System

A powerful workflow automation system that allows you to execute shell commands at specific points in the agent lifecycle.

## Overview

Hooks enable you to:
- **Audit** all bash commands for security compliance
- **Format** code automatically after file edits  
- **Stage** files to git after modifications
- **Send** desktop notifications for important events
- **Log** session statistics and activity

## Hook Types

| Type | When | Use Cases |
|------|------|-----------|
| **PreToolUse** | Before tool execution | Security auditing, validation, backups |
| **PostToolUse** | After tool completion | Code formatting, git operations, cleanup |
| **Notification** | On system messages/errors | Desktop notifications, alerts |
| **Stop** | When agent finishes response | Session logging, statistics |

## Configuration

### Directory Structure

Hooks are stored in individual files (recommended):

```
~/.config/agent/hooks/
├── pre-security-audit.json      # Audit bash commands
├── post-format-code.json        # Auto-format Python files
├── post-git-add.json           # Auto-stage modified files
├── notify-desktop.json         # Desktop notifications
└── stop-session-log.json       # Session logging
```

### Individual Hook File Format

Create `.json` files with this simple format:

```json
{
  "matcher": "tool_pattern",
  "command": "command to execute",
  "timeout": 30
}
```

**Automatic Type Detection** from filename:
- `pre-*.json` → PreToolUse
- `post-*.json` → PostToolUse  
- `notify-*.json` → Notification
- `stop-*.json` → Stop

## Examples

### Security Auditing
**`~/.config/agent/hooks/pre-security-audit.json`**
```json
{
  "matcher": "bash_execute",
  "command": "echo '[SECURITY] About to execute: {{tool_args}}' >> ~/.config/agent/audit.log",
  "timeout": 5
}
```

### Code Formatting
**`~/.config/agent/hooks/post-format-code.json`**
```json
{
  "matcher": "write_file|replace_in_file|multiedit",
  "command": "black '{{tool_args.file_path}}' 2>/dev/null || true",
  "timeout": 15
}
```

### Git Integration
**`~/.config/agent/hooks/post-git-add.json`**
```json
{
  "matcher": "write_file|replace_in_file|multiedit",
  "command": "git add '{{tool_args.file_path}}' 2>/dev/null || true",
  "timeout": 10
}
```

### Desktop Notifications
**`~/.config/agent/hooks/notify-desktop.json`**
```json
{
  "matcher": "system_message|error",
  "command": "osascript -e 'display notification \"{{message}}\" with title \"CLI-Agent\"' 2>/dev/null || notify-send 'CLI-Agent' '{{message}}' 2>/dev/null || true",
  "timeout": 5
}
```

### Session Logging
**`~/.config/agent/hooks/stop-session-log.json`**
```json
{
  "matcher": "*",
  "command": "echo '[SESSION] {{timestamp}} - Response complete. Messages: {{conversation_length}}' >> ~/.config/agent/session.log",
  "timeout": 5
}
```

## Template Variables

Use these variables in your hook commands:

| Variable | Description | Available In |
|----------|-------------|--------------|
| `{{tool_name}}` | Name of the tool | All hooks |
| `{{tool_args}}` | Tool arguments (JSON) | Pre/PostToolUse |
| `{{tool_args.file_path}}` | File path argument | File-related tools |
| `{{result}}` | Tool execution result | PostToolUse |
| `{{error}}` | Error message | PostToolUse (on failure) |
| `{{timestamp}}` | Current timestamp | All hooks |
| `{{message}}` | Event message | Notification |
| `{{conversation_length}}` | Number of messages | Stop |

## Pattern Matching

| Pattern | Matches |
|---------|---------|
| `*` | All tools |
| `bash_execute` | Exact tool name |
| `write_file\|replace_in_file` | Multiple tools (OR) |
| `builtin:*` | All built-in tools |

## Management

### Status & Control
```bash
/hooks              # Show detailed status
/hooks disable      # Temporarily disable all hooks
/hooks enable       # Re-enable hooks
```

### Enable/Disable Individual Hooks
```bash
# Disable temporarily
mv pre-security-audit.json pre-security-audit.json.disabled

# Re-enable
mv pre-security-audit.json.disabled pre-security-audit.json
```

## Advanced Features

### Multiple Hooks Per File
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "write_file",
        "hooks": [
          {
            "type": "command",
            "command": "black '{{tool_args.file_path}}'"
          },
          {
            "type": "command", 
            "command": "git add '{{tool_args.file_path}}'"
          }
        ]
      }
    ]
  }
}
```

### YAML Format Support
**`~/.config/agent/hooks/pre-validation.yaml`**
```yaml
matcher: "*"
command: echo 'Validating {{tool_name}} with {{tool_args}}'
timeout: 10
env:
  VALIDATION_MODE: "strict"
```

### Conditional Execution
Use shell logic for conditional hooks:
```json
{
  "matcher": "write_file",
  "command": "[ '{{tool_args.file_path}}' != '*.py' ] || black '{{tool_args.file_path}}'",
  "timeout": 15
}
```

## Best Practices

### 1. Robust Commands
Make commands fail gracefully:
```json
{
  "command": "your-command 2>/dev/null || true"
}
```

### 2. Descriptive Filenames
```
✅ pre-security-audit.json
✅ post-format-python.json  
✅ notify-slack-completion.json

❌ hook1.json
❌ test.json
```

### 3. Reasonable Timeouts
- Quick commands: 5-10 seconds
- File operations: 15-30 seconds
- Network operations: 30-60 seconds

### 4. Use Specific Matchers
Avoid `*` matcher except for logging/notification hooks.

## Troubleshooting

### Hooks Not Loading
1. Check file syntax: `cat ~/.config/agent/hooks/your-hook.json | jq .`
2. Verify filename pattern matches hook type
3. Check logs: `export LOG_LEVEL=DEBUG`

### Hook Execution Issues
1. Test command manually in shell
2. Check timeout is sufficient
3. Ensure proper escaping of file paths
4. Use absolute paths when possible

### Debug Individual Hooks
Create test hook:
```json
{
  "matcher": "*",
  "command": "echo 'Debug: {{tool_name}} called with {{tool_args}}'",
  "timeout": 5
}
```

## Example Workflows

### Development Setup
```
~/.config/agent/hooks/
├── pre-security-audit.json     # Log all bash commands
├── post-format-python.json     # Auto-format with black
├── post-format-js.json         # Auto-format with prettier  
├── post-git-add.json           # Auto-stage changes
└── notify-completion.json      # Desktop notification
```

### Security & Compliance
```
~/.config/agent/hooks/
├── pre-command-audit.json      # Audit all commands
├── pre-file-backup.json        # Backup before edits
├── post-scan-files.json        # Security scan
└── stop-session-report.json   # Generate compliance report
```

The hooks system provides powerful automation while maintaining security and transparency through comprehensive logging and user feedback.