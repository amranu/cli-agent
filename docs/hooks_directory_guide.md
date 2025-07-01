# Individual Hook Files Guide

## Overview

CLI-Agent now supports organizing hooks in individual files within a `hooks/` directory, making hook management much more organized and maintainable.

## Directory Structure

Create hooks in these directories (checked in order):

```
.config/agent/hooks/           # Project-specific hooks
~/.config/agent/hooks/         # User-global hooks  
~/.config/cli-agent/hooks/     # CLI-Agent specific hooks
```

## Individual Hook File Format

### Simple Format (Recommended)

Create individual `.json` files with this simple format:

```json
{
  "matcher": "tool_pattern",
  "command": "command to execute",
  "timeout": 30,
  "working_directory": "/optional/path",
  "env": {
    "OPTIONAL": "environment_vars"
  }
}
```

### Automatic Type Detection

Hook types are automatically detected from filenames:

- **PreToolUse**: `pre-*.json`, `before-*.json`
- **PostToolUse**: `post-*.json`, `after-*.json`  
- **Notification**: `notify-*.json`, `notification-*.json`
- **Stop**: `stop-*.json`, `end-*.json`, `finish-*.json`

### Example Hook Files

#### `.config/agent/hooks/pre-security-audit.json`
```json
{
  "matcher": "bash_execute",
  "command": "echo '[SECURITY] About to execute: {{tool_args}}' >> ~/.audit.log",
  "timeout": 5
}
```

#### `.config/agent/hooks/post-format-code.json`
```json
{
  "matcher": "write_file|replace_in_file|multiedit",
  "command": "black '{{tool_args.file_path}}' 2>/dev/null || true",
  "timeout": 15
}
```

#### `.config/agent/hooks/post-git-add.json`
```json
{
  "matcher": "write_file|replace_in_file|multiedit", 
  "command": "git add '{{tool_args.file_path}}' 2>/dev/null || true",
  "timeout": 10
}
```

#### `.config/agent/hooks/notify-desktop.json`
```json
{
  "matcher": "*",
  "command": "osascript -e 'display notification \"{{message}}\" with title \"CLI-Agent\"' 2>/dev/null || notify-send 'CLI-Agent' '{{message}}' 2>/dev/null || true",
  "timeout": 5
}
```

#### `.config/agent/hooks/stop-session-log.json`
```json
{
  "matcher": "*",
  "command": "echo '[SESSION] {{timestamp}} - Response complete. Messages: {{conversation_length}}' >> ~/.session.log",
  "timeout": 5
}
```

## Advanced Formats

### Explicit Type Specification

If filename detection doesn't work, specify the type explicitly:

```json
{
  "type": "PreToolUse",
  "matcher": "bash_execute",
  "command": "echo 'Custom pre-hook'"
}
```

### Multiple Hooks in One File

Use the full configuration format for multiple hooks:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "bash_execute",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'First hook'"
          },
          {
            "type": "command", 
            "command": "echo 'Second hook'"
          }
        ]
      }
    ]
  }
}
```

### YAML Support

Hooks can also be written in YAML format (if PyYAML is installed):

#### `.config/agent/hooks/pre-validation.yaml`
```yaml
matcher: "*"
command: echo 'Validating {{tool_name}} with {{tool_args}}'
timeout: 10
env:
  VALIDATION_MODE: "strict"
```

## Benefits of Individual Hook Files

### 1. **Organization**
- Each hook has its own file
- Easy to find and modify specific hooks
- Clear purpose from filename

### 2. **Maintainability**  
- Enable/disable hooks by renaming files
- Version control individual hooks
- Share specific hooks between projects

### 3. **Flexibility**
- Mix individual files with traditional config
- Different hooks in different directories
- Easy to add/remove hooks

### 4. **Debugging**
- Isolate problematic hooks
- Test individual hooks separately
- Clear error reporting per file

## File Management

### Enable/Disable Hooks

Rename files to disable temporarily:
```bash
# Disable
mv pre-security-audit.json pre-security-audit.json.disabled

# Re-enable  
mv pre-security-audit.json.disabled pre-security-audit.json
```

### Organize by Category

Create subdirectories for organization:
```
.config/agent/hooks/
├── security/
│   ├── pre-audit.json
│   └── post-scan.json
├── formatting/
│   ├── post-black.json
│   └── post-prettier.json
└── git/
    ├── post-add.json
    └── post-commit.json
```

### Share Hooks

Copy hook files between projects:
```bash
# Copy useful hooks to new project
cp ~/.config/agent/hooks/post-format-code.json .config/agent/hooks/
```

## Migration from Single File

### Convert Existing Configuration

If you have hooks in `settings.json`:

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
          }
        ]
      }
    ]
  }
}
```

Create individual file `.config/agent/hooks/post-format.json`:
```json
{
  "matcher": "write_file", 
  "command": "black '{{tool_args.file_path}}'"
}
```

### Gradual Migration

Both formats work together:
- Keep existing `settings.json` hooks
- Add new hooks as individual files
- Gradually move hooks to individual files

## Best Practices

### 1. **Descriptive Filenames**
```
✅ pre-security-audit.json
✅ post-format-python.json  
✅ notify-slack-completion.json

❌ hook1.json
❌ test.json
❌ my-hook.json
```

### 2. **Consistent Naming**
- Use consistent prefixes: `pre-`, `post-`, `notify-`, `stop-`
- Include tool or purpose: `security`, `format`, `git`
- Use kebab-case: `pre-security-audit.json`

### 3. **Documentation**
Add comments in JSON (non-standard but helpful):
```json
{
  "_description": "Automatically format Python files after editing",
  "matcher": "write_file|replace_in_file",
  "command": "black '{{tool_args.file_path}}' 2>/dev/null || true",
  "timeout": 15
}
```

### 4. **Error Handling**
Make commands robust:
```json
{
  "matcher": "*",
  "command": "your-command 2>/dev/null || true",
  "timeout": 10
}
```

## Status and Debugging

### Check Hook Loading
```bash
# See which hooks are loaded
/hooks

# Check for errors in logs
export LOG_LEVEL=DEBUG
```

### Test Individual Hooks
Create minimal test hooks:
```json
{
  "matcher": "*",
  "command": "echo 'Test hook executed: {{tool_name}}'",
  "timeout": 5
}
```

## Example Workflows

### Development Workflow
```
.config/agent/hooks/
├── pre-security-check.json     # Audit dangerous commands
├── post-format-python.json     # Auto-format Python files
├── post-format-js.json         # Auto-format JavaScript files
├── post-git-add.json           # Auto-add modified files
└── notify-completion.json      # Desktop notification on completion
```

### Security & Auditing
```
.config/agent/hooks/
├── pre-command-audit.json      # Log all bash commands
├── pre-file-backup.json        # Backup before file edits
├── post-scan-files.json        # Security scan modified files
└── stop-session-summary.json  # Log session statistics
```

### Integration Workflow
```
.config/agent/hooks/
├── post-run-tests.json         # Run tests after code changes
├── post-update-docs.json       # Update documentation
├── notify-slack.json           # Notify team channel
└── stop-deploy-check.json      # Check if ready for deployment
```

The individual hook files system provides much better organization and maintainability while preserving all the power and flexibility of the hooks system.