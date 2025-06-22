# Tool Permission System

The MCP Agent includes a comprehensive tool permission system that provides user prompts for tool execution with configurable allowed/disallowed tools, session-based permission tracking, and pattern matching similar to Claude Code.

## Features

### 🔒 **User Prompts for Tool Execution**
Every tool execution (except when explicitly allowed) prompts the user for permission with options to:
- Execute once
- Allow tool for the rest of the session  
- Auto-approve ALL tools for the session
- Deny once
- Deny tool for the rest of the session

### 📝 **Configurable Tool Permissions**
Configure tool permissions via:
- Command line arguments: `--allowed-tools`, `--disallowed-tools`, `--auto-approve-tools`
- Environment variables: `ALLOWED_TOOLS`, `DISALLOWED_TOOLS`, `AUTO_APPROVE_TOOLS`
- Configuration files

### 🎯 **Pattern Matching Support**
Supports Claude Code-style patterns:
- `"Bash"` - Allow/deny bash command execution
- `"Bash(*)"` - Allow/deny any bash command
- `"Bash(git:*)"` - Allow/deny bash commands starting with "git:"
- `"*"` - Wildcard for any tool
- Exact tool names: `"bash_execute"`, `"read_file"`, etc.

### 💾 **Session-Based Tracking**
- Permissions persist across the session
- Session data stored in `.agent_sessions/.tool_permissions.json`
- Interactive slash commands for managing permissions

## Usage Examples

### Command Line Usage

#### Allow Specific Tools
```bash
# Allow only read operations
python agent.py chat --allowed-tools "Read Write"

# Allow bash commands with git
python agent.py chat --allowed-tools "Bash(git:*)"

# Allow multiple tool types
python agent.py chat --allowed-tools "Bash Edit Read"
```

#### Deny Specific Tools
```bash
# Deny file writing and bash execution
python agent.py chat --disallowed-tools "Write Bash"

# Deny specific bash commands
python agent.py chat --disallowed-tools "Bash(rm:*) Bash(sudo:*)"

# Mixed patterns
python agent.py chat --disallowed-tools "Bash(rm:*)" --allowed-tools "Read Write"
```

#### Auto-Approve Mode
```bash
# Auto-approve all tools for the session
python agent.py chat --auto-approve-tools

# Also works with ask command
python agent.py ask "What files are in this directory?" --auto-approve-tools
```

### Environment Variables

Set persistent tool permissions via environment variables:

```bash
# In .env file
ALLOWED_TOOLS=Read,Write,Bash(git:*)
DISALLOWED_TOOLS=Bash(rm:*),Bash(sudo:*)
AUTO_APPROVE_TOOLS=false
```

### Interactive Session Management

Use slash commands during chat sessions:

```bash
# Show current permission status
/permissions

# Allow a tool for the session
/permissions allow bash_execute

# Deny a tool for the session  
/permissions deny write_file

# Enable auto-approval for all tools
/permissions auto

# Reset all session permissions
/permissions reset
```

## Tool Name Mappings

The system supports both friendly names (Claude Code style) and internal tool names:

| Friendly Name | Internal Tool Name | Description |
|---------------|-------------------|-------------|
| `Bash` | `bash_execute` | Execute shell commands |
| `Read` | `read_file` | Read file contents |
| `Write` | `write_file` | Write file contents |
| `Edit` | `replace_in_file` | Edit files in-place |
| `List` | `list_directory` | List directory contents |
| `WebFetch` | `webfetch` | Fetch web content |
| `Task` | `task` | Spawn subagent tasks |
| `Todo` | `todo_read`, `todo_write` | Manage todo lists |
| `Directory` | `get_current_directory` | Get current directory |

## Permission Flow

1. **Tool Execution Request**: When the AI wants to execute a tool
2. **Permission Check**: System checks in this order:
   - Session auto-approval enabled? → Allow
   - Tool in session approvals? → Allow  
   - Tool in session denials? → Deny
   - Tool in config disallowed list? → Deny
   - Tool in config allowed list? → Allow
   - No specific config? → Prompt user
3. **User Prompt**: If prompting is needed:
   ```
   🔧 Tool Execution Request:
      Tool: bash_execute
      Action: Execute bash command: ls -la
      Arguments: {"command": "ls -la"}
   
   Allow this tool to execute?
     [y] Yes, execute once
     [a] Yes, and allow 'bash_execute' for the rest of this session
     [A] Yes, and auto-approve ALL tools for this session
     [n] No, deny this execution
     [d] No, and deny 'bash_execute' for the rest of this session
   
   Choice [y/a/A/n/d]:
   ```

## Pattern Examples

### Basic Patterns
```bash
# Allow all tools
--allowed-tools "*"

# Allow only safe read operations
--allowed-tools "Read List Directory"

# Deny dangerous operations
--disallowed-tools "Bash Write"
```

### Advanced Patterns
```bash
# Allow git operations only
--allowed-tools "Bash(git:*)"

# Allow safe bash commands, deny dangerous ones
--allowed-tools "Bash(ls:*) Bash(cat:*) Bash(grep:*)" --disallowed-tools "Bash(rm:*) Bash(sudo:*)"

# Complex permission setup
--allowed-tools "Read Write Edit List" --disallowed-tools "Bash(rm:*) Bash(sudo:*) Task"
```

### Configuration File Example

```json
{
  "allowed_tools": ["Read", "Write", "Bash(git:*)"],
  "disallowed_tools": ["Bash(rm:*)", "Bash(sudo:*)"],
  "auto_approve_tools": false
}
```

## Security Considerations

### 🔒 **Default Behavior**
- **Prompts by default**: No tools execute without user awareness
- **Session isolation**: Permissions don't persist across different sessions
- **Explicit approval**: Users must explicitly allow dangerous operations

### ⚠️ **Safety Features**
- **Disallowed list takes precedence**: If a tool is in both allowed and disallowed lists, it's denied
- **Pattern validation**: Invalid patterns are logged and treated as non-matches
- **Graceful fallback**: System defaults to prompting if configuration is invalid

### 🛡️ **Best Practices**
1. **Use specific patterns**: Prefer `"Bash(git:*)"` over `"Bash(*)"` 
2. **Deny dangerous commands**: Always include patterns like `"Bash(rm:*)"`, `"Bash(sudo:*)"`
3. **Review session permissions**: Use `/permissions` to check current state
4. **Test in safe environments**: Verify patterns work as expected before production use

## Implementation Details

### Session Storage
- Permissions stored in `.agent_sessions/.tool_permissions.json`
- Format:
  ```json
  {
    "approvals": ["bash_execute", "read_file"],
    "denials": ["write_file"],
    "auto_approve": false
  }
  ```

### Pattern Matching Algorithm
1. Exact name match (`tool_name == pattern`)
2. Wildcard match (`pattern == "*"`)
3. Friendly name mapping (`"Bash"` → `"bash_execute"`)
4. Parameterized patterns (`"Bash(*)"`, `"Bash(git:*)"`)
5. Glob pattern matching (using `fnmatch`)

### Integration Points
- **CLI Arguments**: Parsed and merged with configuration
- **Base Agent**: Permission checks in `_execute_mcp_tool()` method
- **Slash Commands**: Interactive management via `/permissions`
- **Session Management**: Persistent storage and retrieval

## Troubleshooting

### Common Issues

**Tool still prompts despite being in allowed list**
- Check pattern matching with exact tool name
- Verify configuration is loaded correctly
- Use `/permissions` to see current status

**Pattern not matching expected tools**
- Test patterns with the mapping table
- Remember friendly names vs internal names
- Check for typos in pattern syntax

**Permissions not persisting**
- Verify `.agent_sessions/` directory exists
- Check file permissions on session directory
- Look for error messages in output

### Debug Commands
```bash
# Check current permission status
/permissions

# Test with verbose output
python agent.py chat --allowed-tools "Bash" --auto-approve-tools

# Reset and start fresh
/permissions reset
```

The tool permission system provides comprehensive control over tool execution while maintaining usability and security. It's designed to give users full visibility and control over what tools the AI can execute, with sensible defaults and flexible configuration options.