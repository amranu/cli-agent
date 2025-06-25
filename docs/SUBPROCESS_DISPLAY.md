# Subprocess Display System

This document describes the multi-line subprocess display system that provides dedicated terminal lines for each active subagent.

## Overview

The subprocess display system gives each spawned subagent 3 dedicated lines in the terminal for clean, non-overlapping output. This allows multiple subagents to run simultaneously while maintaining a readable interface.

## Architecture

### Components

1. **TerminalManager** (`cli_agent/core/terminal_manager.py`)
   - Manages line allocation and positioning
   - Places subprocess lines just above the prompt area (not at the top)
   - Supports up to 6 concurrent subprocesses by default

2. **SubprocessDisplayCoordinator** (`cli_agent/core/subprocess_display.py`) 
   - Coordinates subprocess registration and message routing
   - Auto-registers subprocesses when they send messages
   - Handles cleanup when subprocesses complete

3. **SubagentCoordinator** (`cli_agent/core/subagent_coordinator.py`)
   - Integrates subprocess display with existing subagent system
   - Routes subagent messages to dedicated display lines

## Features

### Line Allocation
- **3 lines per subprocess**: Each subagent gets 3 dedicated terminal lines
- **Bottom-up positioning**: Lines allocated just above the prompt, not at terminal top
- **Dynamic capacity**: Max subprocesses calculated based on terminal size
- **Clean separation**: Visual separator line between chat and subprocess areas

### Visual Layout
```
┌─────────────────────────────────────┐
│ Chat content and main agent output  │
│ (scrollable area)                   │
├─────────────────────────────────────┤ ← separator line
│ [task_001] 🤖 Processing files...   │ ← subprocess 1, line 1
│ [task_001] 💬 Found 15 Python files │ ← subprocess 1, line 2  
│ [task_001] 📊 Status: analyzing...  │ ← subprocess 1, line 3
│ [task_002] 🚀 Starting generation...│ ← subprocess 2, line 1
│ [task_002] ✅ Created functions     │ ← subprocess 2, line 2
│ [task_002] 💬 Adding error handling │ ← subprocess 2, line 3
├─────────────────────────────────────┤
│ You: [prompt stays at bottom]       │ ← persistent prompt
└─────────────────────────────────────┘
```

### Message Types
- **🤖 Status updates**: General subprocess activity
- **💬 Progress messages**: Detailed progress information  
- **❌ Error messages**: Error notifications with red emoji
- **✅ Completion**: Success/completion indicators
- **📊 Status info**: System status updates

## Usage

### Automatic Integration
The system automatically integrates with existing subagent spawning:

```python
# When spawning a subagent, it's automatically registered for display
await agent.subagent_manager.spawn_subagent("Analyze files", prompt)
```

### Manual Control
For advanced use cases:

```python
from cli_agent.core.subprocess_display import get_subprocess_display_coordinator

coordinator = get_subprocess_display_coordinator()

# Register a process
success = await coordinator.register_subprocess("task_123", "File Analysis")

# Send messages
await coordinator.display_subprocess_message("task_123", "Processing...", "info")
await coordinator.display_subprocess_error("task_123", "File not found")

# Clean up when done
await coordinator.unregister_subprocess("task_123")
```

## Configuration

### Terminal Layout
- **Lines per subprocess**: 3 (configurable via `terminal_manager.lines_per_subprocess`)
- **Reserved lines**: 5 for prompt and spacing (configurable via `terminal_manager.reserved_lines`)
- **Max subprocesses**: Calculated as `(terminal_height - reserved_lines) // lines_per_subprocess`

### Capacity Examples
- **24-line terminal**: Up to 6 subprocesses (24 - 5 = 19, 19 ÷ 3 = 6)
- **30-line terminal**: Up to 8 subprocesses (30 - 5 = 25, 25 ÷ 3 = 8)
- **Small terminal (15 lines)**: Up to 3 subprocesses (15 - 5 = 10, 10 ÷ 3 = 3)

## Testing

Run the test suite:
```bash
python -m pytest tests/unit/test_subprocess_display.py -v
python -m pytest tests/unit/test_terminal_manager.py -v
```

Demo the system:
```bash
python test_subprocess_display.py
```

## Benefits

1. **No Overlap**: Each subprocess has dedicated lines, preventing message collision
2. **Visual Clarity**: Clean separation between main chat and subprocess activity  
3. **Scalable**: Supports multiple concurrent subprocesses based on terminal size
4. **Persistent Prompt**: User prompt always stays at bottom, never gets overwritten
5. **Auto-Management**: Automatic registration, cleanup, and lifecycle management
6. **Rich Display**: Emoji indicators and formatted messages for better UX

## Implementation Notes

- Works only in terminal environments (graceful fallback to regular display otherwise)
- Uses ANSI escape sequences for cursor positioning and line control
- Integrates with existing event system for consistent message handling
- Thread-safe and async-compatible for concurrent subprocess management