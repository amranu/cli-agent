# CLAUDE.md - MCP Agent Architecture Documentation

This document provides comprehensive technical documentation for the MCP Agent codebase, specifically designed to help Claude Code and other AI assistants understand the modular architecture, components, and development patterns.

## 🏗️ Architecture Overview

The MCP Agent has been refactored from a monolithic 3,237-line file into a clean modular architecture. The system provides an extensible framework for creating language model-powered agents with tool integration, interactive chat, and multi-backend support.

### Key Architectural Principles

- **Modular Design**: Components separated by responsibility
- **Model Agnosticism**: Core logic independent of LLM implementation  
- **Tool Extensibility**: Support for both built-in and MCP protocol tools
- **Interactive & Programmatic**: CLI and library usage patterns
- **Centralized Management**: Single source of truth for shared functionality

## 📁 Modular File Structure

```
cli-agent/
├── cli_agent/                    # Main package - modular architecture
│   ├── __init__.py              # Package exports and version
│   ├── core/                    # Core agent functionality
│   │   ├── __init__.py         # Core component exports
│   │   ├── base_agent.py       # BaseMCPAgent abstract class (1,891 lines)
│   │   ├── input_handler.py    # InterruptibleInput terminal handling (194 lines)
│   │   └── slash_commands.py   # SlashCommandManager command system (330 lines)
│   ├── tools/                   # Tool integration and execution
│   │   ├── __init__.py         # Tool exports
│   │   └── builtin_tools.py    # Built-in tool definitions (292 lines)
│   ├── cli/                     # Command-line interface (future expansion)
│   │   └── __init__.py
│   ├── subagents/              # Subagent management (future expansion)
│   │   └── __init__.py
│   └── utils/                   # Utility functions (future expansion)
│       └── __init__.py
├── agent.py                     # Main CLI entry point (505 lines)
├── mcp_deepseek_host.py        # DeepSeek LLM implementation
├── mcp_gemini_host.py          # Google Gemini LLM implementation
├── config.py                   # Configuration management
├── subagent.py                 # Subagent subprocess management
└── README.md                   # Project documentation
```

## 🧩 Core Components

### 1. BaseMCPAgent (`cli_agent/core/base_agent.py`)

**Primary Role**: Abstract base class providing shared agent functionality

**Key Responsibilities**:
- Tool management and execution (built-in + MCP)
- Conversation history and token management
- Interactive chat with slash command integration
- Subagent task spawning and communication
- Abstract methods for LLM-specific implementations

**Critical Methods**:
```python
# Abstract methods - must be implemented by subclasses
async def generate_response(messages, tools=None) -> Union[str, Any]
def convert_tools_to_llm_format() -> List[Dict]
def parse_tool_calls(response) -> List[Dict[str, Any]]

# Concrete shared functionality
async def interactive_chat(input_handler, existing_messages=None)
async def _execute_mcp_tool(tool_key, arguments) -> str
def get_token_limit() -> int
def compact_conversation(messages) -> List[Dict[str, Any]]
```

**Tool Integration**:
- Built-in tools: `bash_execute`, `read_file`, `write_file`, `web_fetch`, `task`, etc.
- MCP external tools via protocol integration
- Unified execution interface: `_execute_mcp_tool()`

**Subagent System**:
- Event-driven subagent communication
- Task spawning: `_task()`, status tracking: `_task_status()`, results: `_task_results()`
- Interrupt-safe streaming with subagent result integration

### 2. InterruptibleInput (`cli_agent/core/input_handler.py`)

**Primary Role**: Professional terminal input handling with interruption support

**Key Features**:
- Multiline input detection and handling
- ESC key interruption during operations
- Raw terminal mode management
- Prompt_toolkit integration with graceful fallbacks
- Asyncio-compatible threading for event loop safety

**Usage Pattern**:
```python
input_handler = InterruptibleInput()
user_input = input_handler.get_multiline_input("You: ")
if user_input is None:  # User interrupted
    handle_interruption()
```

### 3. SlashCommandManager (`cli_agent/core/slash_commands.py`)

**Primary Role**: Slash command system similar to Claude Code

**Supported Commands**:
- **Built-in**: `/help`, `/clear`, `/compact`, `/tokens`, `/tools`, `/quit`
- **Model switching**: `/switch-chat`, `/switch-reason`, `/switch-gemini`, `/switch-gemini-pro`
- **Custom commands**: Loaded from `.claude/commands/` directories
- **MCP commands**: `mcp__<server>__<command>` format

**Extensibility**:
- Project-specific commands: `.claude/commands/*.md`
- Personal commands: `~/.claude/commands/*.md`
- Dynamic MCP command discovery

### 4. Built-in Tools (`cli_agent/tools/builtin_tools.py`)

**Primary Role**: Core tool definitions and schemas

**Available Tools**:
```python
def get_all_builtin_tools() -> Dict[str, Dict]:
    return {
        "builtin:bash_execute": {...},      # Execute bash commands
        "builtin:read_file": {...},         # Read file contents
        "builtin:write_file": {...},        # Write files
        "builtin:list_directory": {...},    # Directory listing
        "builtin:get_current_directory": {...}, # Current directory
        "builtin:todo_read": {...},         # Read todo list
        "builtin:todo_write": {...},        # Update todo list
        "builtin:replace_in_file": {...},   # Text replacement
        "builtin:webfetch": {...},          # Web content fetching
        "builtin:task": {...},              # Spawn subagent tasks
        "builtin:task_status": {...},       # Check task status
        "builtin:task_results": {...},      # Get task results
    }
```

## 🔌 LLM Implementation Pattern

### DeepSeek Implementation (`mcp_deepseek_host.py`)

```python
class MCPDeepseekHost(BaseMCPAgent):
    def __init__(self, config: HostConfig, is_subagent: bool = False):
        super().__init__(config, is_subagent)
        self.deepseek_config = config.get_deepseek_config()
        self.stream = self.deepseek_config.stream  # For centralized generate_response
        
    def convert_tools_to_llm_format(self) -> List[Dict]:
        return self._convert_tools_to_deepseek_format()
        
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        return self._parse_deepseek_tool_calls(response)
        
    async def chat_completion(self, messages, stream=True, interactive=True):
        # DeepSeek-specific API integration
```

### Gemini Implementation (`mcp_gemini_host.py`)

```python
class MCPGeminiHost(BaseMCPAgent):
    def __init__(self, config: HostConfig, is_subagent: bool = False):
        super().__init__(config, is_subagent)
        self.gemini_config = config.get_gemini_config()
        self.stream = True  # For centralized generate_response
        
    def convert_tools_to_llm_format(self) -> List[Dict]:
        return self._convert_tools_to_gemini_format()
        
    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        # Parse from response.candidates[0].content.parts
```

## 🔄 Execution Flow

### Interactive Chat Session

1. **Initialization**:
   ```python
   host = MCPDeepseekHost(config)  # or MCPGeminiHost
   input_handler = InterruptibleInput()
   await host.interactive_chat(input_handler)
   ```

2. **User Input Processing**:
   - Input captured via `InterruptibleInput`
   - Slash commands handled by `SlashCommandManager`
   - Regular messages added to conversation history

3. **Response Generation**:
   - `host.generate_response()` called (centralized implementation)
   - Calls `host.chat_completion()` with streaming/interactive settings
   - LLM-specific API integration in subclass

4. **Tool Execution** (if requested):
   - Tools parsed via `parse_tool_calls()`
   - Executed through `_execute_mcp_tool()`
   - Results added to conversation, process repeats

5. **Streaming Output**:
   - Real-time response streaming
   - ESC key interruption support
   - Subagent result integration during streaming

### Subagent Task Flow

1. **Task Spawning**: `/task` or `task` tool call
2. **Background Execution**: Separate subprocess with event communication
3. **Result Integration**: Automatic collection and conversation injection
4. **Status Tracking**: `/task-status` for monitoring active tasks

## 🛠️ Development Patterns

### Adding New LLM Backend

1. **Create Host Class**:
   ```python
   class MCPNewLLMHost(BaseMCPAgent):
       def convert_tools_to_llm_format(self) -> List[Dict]:
           # Format tools for your LLM's API
           
       def parse_tool_calls(self, response) -> List[Dict[str, Any]]:
           # Parse your LLM's tool call format
           
       async def chat_completion(self, messages, stream=True, interactive=True):
           # Your LLM's API integration
   ```

2. **Update CLI Integration** (in `agent.py`):
   ```python
   # Add model switching command
   elif config.model_type == "new-llm":
       from new_llm_host import MCPNewLLMHost
       host = MCPNewLLMHost(config)
   ```

### Adding Custom Tools

1. **Built-in Tools** (add to `cli_agent/tools/builtin_tools.py`):
   ```python
   def get_my_custom_tool() -> Dict:
       return {
           "server": "builtin",
           "name": "my_custom_tool",
           "description": "Description of what it does",
           "schema": {...},  # JSON schema
           "client": None
       }
   ```

2. **MCP External Tools**: Create MCP server and connect via CLI

### Adding Slash Commands

1. **Built-in Commands** (in `SlashCommandManager`):
   ```python
   def _handle_my_command(self, args: str) -> str:
       # Command implementation
       return "Result message"
   ```

2. **Custom Commands**: Create `.claude/commands/my-command.md`

## 🧪 Testing Patterns

### Component Testing
```python
# Test individual components
from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from cli_agent.tools.builtin_tools import get_all_builtin_tools

# Test tool definitions
tools = get_all_builtin_tools()
assert "builtin:bash_execute" in tools
```

### Integration Testing
```python
# Test LLM implementations
from mcp_deepseek_host import MCPDeepseekHost
host = MCPDeepseekHost(config)
tools = host.convert_tools_to_llm_format()
```

## 🎯 Key Design Patterns

- **Abstract Base Class**: `BaseMCPAgent` enforces interface contracts
- **Strategy Pattern**: Interchangeable LLM implementations
- **Command Pattern**: Slash commands and CLI structure
- **Event-Driven**: Subagent communication via async queues
- **Dependency Injection**: Configuration and tool injection
- **Template Method**: `interactive_chat()` defines flow, subclasses customize steps

## 📊 Refactoring Benefits

**Before**: 3,237-line monolithic `agent.py`
**After**: Modular architecture with focused components

**Maintainability**: ✅ Each module has single responsibility
**Testability**: ✅ Components can be unit tested
**Reusability**: ✅ Core modules work across implementations  
**Scalability**: ✅ Easy to add features without touching entire codebase
**Developer Experience**: ✅ Much easier to navigate and understand

## 🚀 Future Expansion Areas

- **`cli_agent/cli/`**: Extract CLI commands to separate modules
- **`cli_agent/subagents/`**: Enhanced subagent management
- **`cli_agent/utils/`**: Common utilities and helpers
- **Plugin System**: Dynamic tool loading and discovery
- **Multi-Agent**: Agent-to-agent communication protocols

This modular architecture provides a solid foundation for continued development and extension of the MCP Agent system.