# AGENT.md

This document provides a technical overview of the MCP Agent codebase, designed to help a coding agent understand the system's architecture, components, and core functionalities.

## 1. Overview

The MCP Agent is an extensible framework for creating language model-powered agents that can interact with a variety of tools. The codebase has been refactored from a monolithic structure into a clean modular architecture for improved maintainability and extensibility.

The system provides:
- **Command-line interface (CLI)** for interactive and programmatic usage
- **Interactive chat** with advanced features like multiline input and interruption
- **Tool management** with built-in and MCP protocol integration  
- **Conversation handling** with automatic compaction and token management
- **Model-agnostic design** supporting multiple LLM backends

The key features of the agent are:

- **Modular Architecture:** Clean separation of concerns across focused modules
- **Extensible Tool Integration:** Built-in tools and external tools via MCP protocol
- **Model Agnosticism:** Core logic independent of LLM implementation
- **Interactive and Non-Interactive Modes:** Chat sessions and single-turn requests
- **Advanced Chat Features:** Multiline input, slash commands, interruption handling
- **Conversation Management:** Automatic compaction within token limits
- **Subagent System:** Spawn background tasks with event-driven communication

## 2. Modular Architecture

The codebase is organized into a clean modular structure:

### `cli_agent/` Package: Modular Core Framework

The main functionality has been extracted into focused modules:

#### `cli_agent/core/base_agent.py`: The Core Agent Framework

Contains the foundational `BaseMCPAgent` class - an abstract base class defining core agent functionality.

- **`BaseMCPAgent` class (1,891 lines):**
    - **Initialization:** Loads configuration, built-in tools, and centralized subagent management
    - **Tool Management:**
        - Uses `cli_agent.tools.builtin_tools` for tool definitions
        - `_execute_mcp_tool()`: Unified execution for built-in and external tools
        - `start_mcp_server()`: Connects to external MCP tool servers
    - **Conversation Management:**
        - `conversation_history`: Message storage with token tracking
        - `compact_conversation()`: Intelligent conversation summarization
        - `get_token_limit()`: Centralized token limit management with model configuration
    - **Interactive Chat:**
        - `interactive_chat()`: Centralized chat session management
        - Integrates with `SlashCommandManager` for command handling
        - Subagent task spawning and result integration
    - **Centralized Methods** (newly consolidated):
        - `generate_response()`: Centralized LLM response generation with streaming support
        - Tool conversion helpers: `normalize_tool_name()`, `generate_default_description()`, etc.
    - **Abstract Methods:**
        - `convert_tools_to_llm_format()`: Format tools for specific LLM APIs
        - `parse_tool_calls()`: Parse tool calls from LLM responses
        - `chat_completion()`: LLM-specific API integration

#### `cli_agent/core/slash_commands.py`: Command System

- **`SlashCommandManager` class (330 lines):**
  - Manages slash commands like `/help`, `/clear`, `/compact`, `/tokens`
  - Model switching: `/switch-chat`, `/switch-reason`, `/switch-gemini`
  - Custom command loading from `.claude/commands/` directories
  - MCP command integration with `mcp__<server>__<command>` format

#### `cli_agent/core/input_handler.py`: Terminal Interaction

- **`InterruptibleInput` class (194 lines):**
  - Professional terminal input with prompt_toolkit integration
  - Multiline input detection and smart handling
  - ESC key interruption support during operations
  - Asyncio-compatible threading for event loop safety

#### `cli_agent/tools/builtin_tools.py`: Tool Definitions

- **Built-in Tool Registry (292 lines):**
  - Complete tool schemas: `bash_execute`, `read_file`, `write_file`, `webfetch`
  - Task management: `task`, `task_status`, `task_results`
  - Utility tools: `todo_read`, `todo_write`, `replace_in_file`
  - Centralized tool access functions

### `agent.py`: CLI Interface (505 lines)

- **Streamlined Entry Point:** Main CLI using Click framework
- **Commands:** `chat`, `ask`, `init`, `mcp`, model switching commands
- **Integration:** Imports from modular `cli_agent` package
- **Host Selection:** Dynamic selection between DeepSeek and Gemini backends

### `mcp_deepseek_host.py`: Deepseek Model Implementation

This file provides the `MCPDeepseekHost` class, which is a concrete implementation of `BaseMCPAgent` for the Deepseek language model.

- **`MCPDeepseekHost` class:**
    - **Deepseek API Integration:** It uses the `openai` library to communicate with the Deepseek API.
    - **Tool Formatting:**
        - `_convert_tools_to_deepseek_format()`: Implements the `convert_tools_to_llm_format` method to format tools as expected by the Deepseek API.
    - **Tool Call Parsing:**
        - `_parse_deepseek_tool_calls()`: Implements the `parse_tool_calls` method to parse the tool-calling syntax used by Deepseek.
    - **System Prompt:**
        - `_create_system_prompt()`: Customizes the system prompt with instructions specific to the Deepseek model.
    - **Streaming and Tool Use:**
        - `chat_completion()`: Manages the interaction with the Deepseek API, including handling streaming responses and iterative tool calls.

### `mcp_gemini_host.py`: Gemini Model Implementation

This file provides the `MCPGeminiHost` class, which is a concrete implementation of `BaseMCPAgent` for the Google Gemini language model.

- **`MCPGeminiHost` class:**
    - **Gemini API Integration:** It uses the `google-generativeai` library to communicate with the Gemini API.
    - **Tool Formatting:**
        - `_convert_tools_to_gemini_format()`: Implements the `convert_tools_to_llm_format` method to format tools for the Gemini API.
    - **Tool Call Parsing:**
        - `_parse_response_content()`: Implements the `parse_tool_calls` method to parse tool calls from the Gemini API's response. It is designed to handle multiple response formats, including standard function calls, Python-style calls, and XML-style calls.
    - **System Prompt:**
        - `_create_system_prompt()`: Customizes the system prompt for the Gemini model.
    - **Streaming and Tool Use:**
        - `chat_completion()`: Manages the interaction with the Gemini API, including handling streaming responses and iterative tool calls. It also includes retry logic for API requests.

## 3. Execution Flow

The modular architecture provides a streamlined execution flow:

1.  **Initialization:**
    - CLI entry point in `agent.py` using Click framework
    - Host class selection (`MCPDeepseekHost` or `MCPGeminiHost`) based on configuration
    - `BaseMCPAgent.__init__()` loads built-in tools from `cli_agent.tools.builtin_tools`
    - Centralized subagent management system initialization
    - MCP server connections established

2.  **User Input Handling:**
    - **Interactive mode:** `InterruptibleInput` class captures user input with multiline support
    - **Slash commands:** `SlashCommandManager` processes commands like `/help`, `/compact`, `/switch-*`
    - **Non-interactive mode:** Direct message processing via `ask` command

3.  **Message Processing:**
    - User messages added to conversation history
    - Automatic conversation compaction when approaching token limits
    - Centralized `generate_response()` method called with streaming/interactive settings

4.  **Model-Specific Processing:**
    - `convert_tools_to_llm_format()` formats tools for specific LLM APIs
    - `chat_completion()` handles LLM-specific API integration
    - Tools and conversation history sent to language model

5.  **Tool Execution:**
    - `parse_tool_calls()` extracts tool requests from LLM responses
    - `_execute_mcp_tool()` provides unified execution for built-in and external tools
    - Tool results added to conversation, process repeats until final response

6.  **Streaming Output:**
    - Real-time response streaming with ESC interruption support
    - Subagent result integration during streaming pauses
    - Professional terminal output handling via `InterruptibleInput`

7.  **Subagent Integration:**
    - Background task spawning via `task` tool or `/task` command
    - Event-driven communication with subprocess subagents
    - Automatic result collection and conversation integration

## 4. Tool Integration

The agent's tool integration is a core feature and is based on the following principles:

- **Built-in Tools:** A set of essential tools is defined directly in `agent.py`. These tools are available to all agent implementations.
- **External Tools (MCP):** The agent can connect to external tool servers using the MCP protocol. This allows for the integration of tools written in any language.
- **Tool Abstraction:** The `BaseMCPAgent` class provides a unified interface for executing both built-in and external tools, making the tool execution process transparent to the agent's core logic.
- **Model-Specific Formatting:** Each model implementation is responsible for formatting the tool specifications in the way that its API expects.

## 5. Modular Architecture Benefits

The refactoring from a monolithic 3,237-line file to a modular architecture provides significant benefits:

### Before & After Comparison
- **Before:** Single `agent.py` file (3,237 lines)
- **After:** Focused modules with clear responsibilities
  - `base_agent.py`: 1,891 lines (core functionality)
  - `slash_commands.py`: 330 lines (command system)  
  - `input_handler.py`: 194 lines (terminal handling)
  - `builtin_tools.py`: 292 lines (tool definitions)
  - `agent.py`: 505 lines (CLI interface only)

### Key Improvements
- **Maintainability**: Each module has a single, clear responsibility
- **Testability**: Components can be unit tested independently
- **Reusability**: Core modules work across different LLM implementations
- **Scalability**: New features can be added without touching entire codebase
- **Developer Experience**: Much easier to navigate and understand
- **Code Quality**: Enforced separation of concerns and reduced coupling

## 6. Key Design Patterns

The modular codebase employs several key design patterns:

- **Abstract Base Classes:** `BaseMCPAgent` enforces clear separation between core logic and LLM-specific implementations
- **Strategy Pattern:** Interchangeable model implementations (`MCPDeepseekHost`, `MCPGeminiHost`) as different response generation strategies
- **Command Pattern:** 
  - CLI implementation using Click library
  - Slash command system in `SlashCommandManager`
- **Template Method:** `interactive_chat()` defines the flow, subclasses customize specific steps
- **Dependency Injection:** Configuration and tool injection into agent constructors
- **Event-Driven Architecture:** Subagent communication via asyncio queues
- **Module Pattern:** Clean separation of concerns across focused modules
- **Asynchronous Programming:** Non-blocking I/O operations for API requests and MCP communication

## 7. Development Guidelines

### Adding New Components
- **New LLM Backend:** Extend `BaseMCPAgent` and implement abstract methods
- **New Tools:** Add to `cli_agent/tools/builtin_tools.py` or create MCP server
- **New Commands:** Add handlers to `SlashCommandManager` or create custom command files
- **New Utilities:** Add to `cli_agent/utils/` for shared functionality

### Import Patterns
```python
# Core components
from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from cli_agent.tools.builtin_tools import get_all_builtin_tools

# LLM implementations  
from mcp_deepseek_host import MCPDeepseekHost
from mcp_gemini_host import MCPGeminiHost
```

This modular architecture provides a solid, maintainable foundation for any coding agent to understand and extend the MCP Agent system.
