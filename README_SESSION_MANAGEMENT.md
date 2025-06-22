# Session Management for MCP Agent

The MCP Agent now supports conversation persistence and resumption through a comprehensive session management system.

## Features

### 🔄 **Continue Last Conversation**
```bash
# Continue the most recent conversation
python agent.py chat -c
python agent.py chat --continue
```

### 📝 **Resume Specific Session**
```bash
# Resume a specific session by ID
python agent.py chat --resume <session-id>
```

### 📋 **Session Management Commands**
```bash
# List recent sessions
python agent.py sessions list

# Show detailed session information
python agent.py sessions show <session-id>

# Delete a specific session
python agent.py sessions delete <session-id>

# Clear all sessions
python agent.py sessions clear
```

## How It Works

### Session Storage
- Sessions are stored in `.agent_sessions/` directory
- Each session is saved as a JSON file with UUID filename
- Sessions include metadata: creation time, last updated, message count
- The last accessed session is tracked in `last_session.json`

### Automatic Persistence
- Every user message and assistant response is automatically saved
- Sessions are updated in real-time during conversations
- Message history is preserved across application restarts
- Host switching (DeepSeek ↔ Gemini) preserves session state

### Session Data Structure
```json
{
  "session_id": "uuid-string",
  "created_at": "2025-06-22T09:07:58.178581",
  "last_updated": "2025-06-22T09:08:15.234567",
  "message_count": 5,
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
    ...
  ]
}
```

## Usage Examples

### Starting a New Conversation
```bash
# Start fresh conversation (creates new session automatically)
python agent.py chat
# Output: Started new session: a1b2c3d4...
```

### Continuing Work
```bash
# Continue where you left off
python agent.py chat --continue
# Output: Continuing session: a1b2c3d4...
#         Restored 8 messages
```

### Managing Sessions
```bash
# See all your conversations
python agent.py sessions list
# Output:
# Recent sessions:
# 1. a1b2c3d4... (8 messages) - 2025-06-22 09:15:30
#    "How do I create a REST API with FastAPI?"
# 2. b2c3d4e5... (3 messages) - 2025-06-22 08:45:12
#    "Explain Python decorators"

# Get detailed info about a session
python agent.py sessions show a1b2c3d4-5678-9abc-def0-123456789abc

# Resume an older session
python agent.py chat --resume b2c3d4e5-6789-abcd-ef01-23456789abcd
```

### Working with Streaming JSON
Session management works seamlessly with streaming JSON mode:
```bash
# Continue last session in JSON mode
python agent.py chat --continue --output-format stream-json

# Resume specific session with JSON I/O
python agent.py chat --resume <session-id> --input-format stream-json --output-format stream-json
```

## Integration with Existing Features

### ✅ **Compatible with All Features**
- **MCP Servers**: Sessions work with all connected MCP servers
- **Streaming JSON**: Full compatibility with Claude Code protocol
- **Model Switching**: Sessions persist across model changes
- **Tool Execution**: Tool calls and results are saved in sessions
- **Subagents**: Subagent interactions are included in session history

### ✅ **Smart Behavior**
- **Automatic Creation**: New sessions created when needed
- **Graceful Fallback**: If resume fails, starts new session
- **Efficient Storage**: Only changed sessions are written to disk
- **Preview Text**: First message shown in session lists for easy identification

## Session Lifecycle

1. **Creation**: `python agent.py chat` → New session UUID generated
2. **Active Use**: Messages automatically saved during conversation
3. **Continuation**: `python agent.py chat -c` → Last session restored
4. **Resumption**: `python agent.py chat --resume <id>` → Specific session restored
5. **Management**: Use `sessions` commands to view/delete old sessions

## File Structure

```
.agent_sessions/
├── a1b2c3d4-5678-9abc-def0-123456789abc.json    # Session 1
├── b2c3d4e5-6789-abcd-ef01-23456789abcd.json    # Session 2
├── c3d4e5f6-789a-bcde-f012-3456789abcde.json    # Session 3
└── last_session.json                             # Last session pointer
```

## Benefits

### 🎯 **Productivity**
- Never lose conversation context
- Pick up exactly where you left off
- Manage multiple project conversations

### 🔒 **Data Persistence**
- All conversations safely stored locally
- No data loss from application crashes
- Easy backup of conversation history

### 🚀 **Workflow Integration**
- Works with existing CI/CD and automation
- Compatible with streaming JSON for external tools
- Seamless integration with all agent features

## Advanced Usage

### Script Integration
```python
from session_manager import SessionManager

# Programmatic session management
manager = SessionManager()
session_id = manager.create_new_session()
manager.add_message({"role": "user", "content": "Hello"})

# Resume sessions in scripts
last_session = manager.continue_last_session()
if last_session:
    messages = manager.get_messages()
    print(f"Resumed {len(messages)} messages")
```

### Batch Operations
```bash
# List all sessions and pipe to other commands
python agent.py sessions list | grep "Python" 

# Export session data
python -c "
from session_manager import SessionManager
import json
manager = SessionManager()
sessions = manager.list_sessions()
print(json.dumps(sessions, indent=2))
"
```

The session management system provides a robust foundation for maintaining conversation context and enables sophisticated workflows while maintaining simplicity for everyday use.