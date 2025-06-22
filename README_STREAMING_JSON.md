# Streaming JSON Support for MCP Agent

This codebase now supports streaming JSON input/output compatible with Claude Code's `--input-format stream-json` and `--output-format stream-json` options.

## Features Implemented

### 1. Streaming JSON Protocol
- **System Init Messages**: Initialization with session ID, tools, and model info
- **Assistant Messages**: Text responses and tool use requests  
- **User Messages**: Tool results and user input
- **Tool Execution**: Full tool call and result streaming

### 2. Command Line Options
```bash
# Streaming JSON output (text input, JSON output)
python agent.py chat --output-format stream-json

# Streaming JSON input (JSON input, text output)  
python agent.py chat --input-format stream-json

# Full streaming JSON mode (JSON input and output)
python agent.py chat --input-format stream-json --output-format stream-json

# Also available for ask command
python agent.py ask "Hello" --output-format stream-json
```

### 3. Compatible Message Format

Based on analysis of Claude Code logs, the messages follow this format:

**System Init:**
```json
{
  "type": "system",
  "subtype": "init", 
  "cwd": "/path/to/working/dir",
  "session_id": "uuid",
  "tools": ["Bash", "Read", "Write", "todo_read", "todo_write"],
  "mcp_servers": ["server1", "server2"],
  "model": "deepseek-chat",
  "permissionMode": "default",
  "apiKeySource": "none"
}
```

**Assistant Text Response:**
```json
{
  "type": "assistant",
  "message": {
    "id": "msg_...",
    "type": "message",
    "role": "assistant", 
    "model": "deepseek-chat",
    "content": [{"type": "text", "text": "Response text"}],
    "stop_reason": null,
    "stop_sequence": null,
    "usage": {...}
  },
  "parent_tool_use_id": null,
  "session_id": "uuid"
}
```

**Assistant Tool Use:**
```json
{
  "type": "assistant",
  "message": {
    "id": "msg_...",
    "content": [{
      "type": "tool_use",
      "id": "toolu_...", 
      "name": "Bash",
      "input": {"command": "ls -la", "description": "List files"}
    }],
    ...
  },
  "session_id": "uuid"
}
```

**User Tool Result:**
```json
{
  "type": "user",
  "message": {
    "role": "user",
    "content": [{
      "tool_use_id": "toolu_...",
      "type": "tool_result",
      "content": "command output",
      "is_error": false
    }]
  },
  "session_id": "uuid"
}
```

## Testing

### Unit Tests
```bash
# Test JSON output format
python test_streaming_json.py output

# Test JSON input parsing  
python test_streaming_json.py input

# Create sample conversation
python test_streaming_json.py sample
```

### Integration Tests
```bash
# Create test input
echo '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Hello"}]},"session_id":"test"}' > input.jsonl

# Test streaming JSON mode
python agent.py chat --input-format stream-json --output-format stream-json < input.jsonl
```

## Implementation Details

### Core Components

1. **`streaming_json.py`**: Main streaming JSON handler
   - `StreamingJSONHandler`: Core class for JSON I/O
   - Message classes: `SystemInitMessage`, `AssistantMessage`, `UserMessage`
   - Tool execution support

2. **`agent.py`**: CLI integration
   - Added `--input-format` and `--output-format` options
   - `handle_streaming_json_chat()`: Main streaming JSON chat loop
   - `stream_json_response()`: Response streaming handler

### Key Features

- **Session Management**: UUID-based session tracking
- **Tool Integration**: Full tool call and result streaming 
- **Error Handling**: Proper error propagation in JSON format
- **Compatibility**: Matches Claude Code's streaming JSON protocol

### Usage in External Tools

This makes the MCP Agent compatible with tools that expect Claude Code's streaming JSON format, enabling integration with:

- CI/CD pipelines
- External automation tools  
- Custom chat interfaces
- API integrations

The streaming JSON format provides a structured, parseable interface while maintaining all the functionality of the interactive text mode.