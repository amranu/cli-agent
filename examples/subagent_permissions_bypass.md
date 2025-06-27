# Subagent Permissions Bypass Example

This example demonstrates how to use the `SUBAGENT_PERMISSIONS_BYPASS` feature to allow subagents to execute tools without requiring user permission for each tool call.

## Use Case

This feature is particularly useful for:
- Automated workflows where subagents need to execute multiple tools
- CI/CD environments where interactive permission prompts are not feasible  
- Batch processing tasks where trusted subagents should run without interruption
- Development environments where permission prompts slow down iteration

## Configuration

Set the environment variable to enable bypass for all subagents:

```bash
export SUBAGENT_PERMISSIONS_BYPASS=true
```

Or add it to your `.env` file:

```bash
# .env
SUBAGENT_PERMISSIONS_BYPASS=true
```

## Example Usage

### Without Bypass (Default Behavior)

```bash
# Normal interactive mode - each tool requires permission
agent chat
> Spawn a subagent to analyze the current directory and create a summary report

# Subagent will prompt for permission for each tool:
# - Permission to execute `ls -la`? (y/n): 
# - Permission to execute `file *`? (y/n):
# - Permission to execute `du -sh *`? (y/n):
```

### With Bypass Enabled

```bash
# Set the bypass flag
export SUBAGENT_PERMISSIONS_BYPASS=true

agent chat
> Spawn a subagent to analyze the current directory and create a summary report

# Subagent executes all tools automatically without prompts:
# ✅ Executing bash_execute: ls -la
# ✅ Executing bash_execute: file *  
# ✅ Executing bash_execute: du -sh *
# ✅ Task completed - emit_result called
```

## Security Considerations

- **Trust Level**: Only enable this in trusted environments where subagent actions are acceptable
- **Scope**: The bypass affects ALL subagents, not individual ones
- **Main Agent**: The main agent still requires permissions - this only affects subagents
- **Audit Trail**: Tool execution is still logged even when permissions are bypassed

## Implementation Details

When `SUBAGENT_PERMISSIONS_BYPASS=true`:
- Subagents skip the `ToolPermissionManager.check_tool_permission()` call
- Tool execution proceeds directly to the tool implementation
- A log message is generated: `"Bypassing permission check for subagent tool: {tool_name}"`
- Main agent permission checks remain unchanged

## Best Practices

1. **Development vs Production**: Enable bypass in development, disable in production
2. **Specific Tools**: Consider which tools truly need bypass (file operations vs system commands)
3. **Monitoring**: Watch logs for bypass events to understand subagent behavior
4. **Documentation**: Document when and why bypass is enabled for your workflows

## Example .env Configuration

```bash
# API Keys
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Model Selection
DEFAULT_PROVIDER_MODEL=anthropic:claude-3.5-sonnet

# Subagent Configuration
SUBAGENT_PERMISSIONS_BYPASS=true

# Other Configuration
AUTO_APPROVE_TOOLS=false  # Main agent still requires approval
LOG_LEVEL=INFO
```