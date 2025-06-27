# AGENT.md

This document provides a comprehensive technical overview of the MCP Agent codebase, designed to help coding agents understand the system's sophisticated architecture, components, and core functionalities.

## 1. Overview

The MCP Agent is a sophisticated, extensible framework for creating language model-powered agents with advanced tool integration, subagent management, and multi-backend support. The codebase has evolved from a monolithic structure into a clean, modular architecture emphasizing maintainability, extensibility, and production-ready features.

### Core System Capabilities
- **Multi-LLM Backend Support:** Runtime-switchable support for 6 providers (Anthropic, OpenAI, DeepSeek, Google Gemini, OpenRouter, Ollama) with unified provider-model architecture
- **Advanced Subagent System:** Process-isolated subagents with event-driven communication and automatic result integration
- **Comprehensive Tool Ecosystem:** 16 built-in tools + external MCP protocol integration with unified execution pipeline
- **Context Management:** Intelligent context preservation with automatic conversation compaction and subagent delegation strategies
- **Professional UI/UX:** Event-driven terminal interface with streaming responses, interruption support, and slash commands
- **Production Features:** Exponential backoff retry logic, fault tolerance, persistent configuration management, and comprehensive error handling
- **Provider-Model Separation:** Same models accessible through different providers with flexible configuration

### Key Architectural Features

- **Provider-Model Architecture:** Clean separation between API providers and model characteristics via composition
- **Event-Driven Design:** Centralized event bus transforming streaming into discrete JSON events with interrupt support
- **Modular Core System:** 29 core modules with specialized responsibilities (base_agent, chat_interface, tool_execution_engine, etc.)
- **Strategy Pattern Implementation:** Interchangeable tool conversion and parsing strategies per LLM format
- **Context-Aware Operation:** Smart delegation of context-heavy tasks to preserve main agent efficiency
- **Stream-First Design:** Unified streaming interface across all backends with comprehensive interruption capabilities
- **Fault-Tolerant Design:** Comprehensive error handling, exponential backoff retry logic, and graceful degradation
- **Tool Extensibility:** Built-in tools (glob, grep, multiedit, etc.) + external MCP tool integration

## 2. Modular Architecture

The codebase implements a sophisticated modular architecture with clear separation of concerns and plugin-style extensibility:

```
cli_agent/
├── __init__.py                    # Package exports and version info
├── core/                          # Core agent framework (29 core files)
│   ├── base_agent.py              # Abstract base agent (2,122 lines)
│   ├── base_llm_provider.py       # Centralized LLM provider functionality (456 lines)
│   ├── base_provider.py           # Provider API abstraction (279 lines)
│   ├── model_config.py            # Model-specific configurations (531 lines)
│   ├── mcp_host.py                # Provider-model composition (607 lines)
│   ├── chat_interface.py          # Interactive chat management (912 lines)
│   ├── input_handler.py           # Terminal interaction utilities (303 lines)
│   ├── slash_commands.py          # Command system management (806 lines)
│   ├── tool_execution_engine.py   # Tool execution and validation (280 lines)
│   ├── builtin_tool_executor.py   # Tool implementations (685 lines)
│   ├── event_system.py            # Central event bus architecture (523 lines)
│   ├── display_manager.py         # Event-driven display coordination (400 lines)
│   ├── response_handler.py        # Response processing framework (682 lines)
│   ├── subagent_coordinator.py    # Subagent lifecycle management
│   ├── tool_permissions.py        # Tool access control system (421 lines)
│   └── [15 additional specialized modules]
├── providers/                     # API provider implementations
│   ├── anthropic_provider.py      # Anthropic API provider
│   ├── openai_provider.py         # OpenAI API provider
│   ├── deepseek_provider.py       # DeepSeek API provider
│   ├── google_provider.py         # Google Gemini API provider
│   ├── openrouter_provider.py     # OpenRouter multi-model API
│   └── ollama_provider.py         # Ollama local inference provider
├── tools/                         # Built-in tool definitions
│   ├── __init__.py                # Tool exports
│   └── builtin_tools.py           # 16 built-in tools with JSON schemas (475 lines)
├── utils/                         # Shared utility modules (7 utility modules)
│   ├── __init__.py                # Centralized utility exports
│   ├── tool_conversion.py         # Multi-format tool schema conversion
│   ├── tool_parsing.py            # Tool call parsing from LLM responses
│   ├── content_processing.py      # Content extraction and cleaning utilities
│   ├── http_client.py             # HTTP client factory and lifecycle management
│   ├── retry.py                   # Exponential backoff retry logic
│   └── diff_display.py            # Terminal diff visualization with colors
├── mcp/                           # MCP protocol implementations
│   ├── __init__.py                # MCP module initialization
│   └── model_server.py            # MCP model server implementation
└── tests/                         # Testing framework (109 tests passing)
    ├── unit/                      # Unit tests
    ├── integration/               # Integration tests
    └── fixtures/                  # Test fixtures
```

### `cli_agent/` Package: Modular Core Framework

#### `cli_agent/core/base_agent.py`: The Core Agent Framework

Contains the foundational `BaseMCPAgent` class - a sophisticated abstract base class implementing the core agent functionality with centralized subagent management.

**Key Architecture:**
```python
BaseMCPAgent (Abstract Base Class)
└── MCPHost (Provider-Model Composition)
    ├── BaseProvider + ModelConfig
    └── Tool Conversion & Event-Driven Streaming
```

**Core Responsibilities:**
- **MCP Integration:** FastMCP client management for external tool servers
- **Subagent System:** Centralized `SubagentManager` with event-driven communication
- **Tool Ecosystem:** Unified execution pipeline for built-in and external tools
- **Context Management:** Token tracking, conversation compaction, and smart delegation
- **Interactive Framework:** Chat session management with interruption support

**Major Components:**
- **Initialization & Configuration:**
  - Loads built-in tools from `cli_agent.tools.builtin_tools`
  - Initializes centralized `SubagentManager` for main agents (excluded for subagents)
  - Establishes MCP server connections with automatic tool discovery
  - Sets up slash command integration via `SlashCommandManager`

- **Tool Management System:**
  - `_execute_mcp_tool()`: Unified execution dispatcher for built-in vs external tools
  - `start_mcp_server()`: Automated MCP server connection and tool registration
  - Built-in tool access control (subagents have `task` tools removed to prevent recursion)
  - Tool forwarding for subagents via communication sockets

- **Conversation & Context Management:**
  - `conversation_history`: Message storage with automatic token tracking
  - `compact_conversation()`: AI-powered conversation summarization when approaching limits
  - `get_token_limit()`: Model-specific token limit management (centralized)
  - Context preservation strategies with subagent delegation guidance

- **Interactive Chat Framework:**
  - `interactive_chat()`: Centralized session management with streaming support
  - Subagent task spawning, monitoring, and automatic result integration
  - Real-time interruption handling with ESC key support
  - Keep-alive message system during long tool executions

- **Centralized Methods (Recently Consolidated):**
  - `generate_response()`: Unified LLM response interface with streaming/interactive modes
  - Subagent lifecycle management: spawning, monitoring, result collection
  - System prompt customization (different for main agents vs subagents)

- **Abstract Interface (LLM-Specific Implementation Required):**
  - `convert_tools_to_llm_format()`: Transform tools to LLM-specific API format
  - `parse_tool_calls()`: Extract and parse tool calls from LLM responses
  - `chat_completion()`: LLM-specific API integration with streaming support

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

- **Built-in Tool Registry (475 lines):**
  - **File Operations:** `read_file`, `write_file`, `replace_in_file`, `multiedit`, `list_directory`, `get_current_directory`
  - **Search & Discovery:** `glob` (pattern matching), `grep` (content search)
  - **System Integration:** `bash_execute` for shell command execution with interrupt support
  - **Web Access:** `webfetch` for HTTP requests and content retrieval
  - **Task Management:** `todo_read`, `todo_write` for session-specific task tracking
  - **Subagent Control:** `task`, `task_status`, `task_results`, `emit_result` (filtered appropriately for subagents)
  - **Tool Schema:** Complete JSON schemas with validation and parameter definitions
  - **Centralized Access:** `get_all_builtin_tools()` function for tool registration
  - **Advanced Features:** Whitespace-preserving edits, modification time sorting, regex search patterns

#### `cli_agent/utils/`: Shared Utility Framework

**Comprehensive utility system supporting the modular architecture:**

##### `cli_agent/utils/tool_conversion.py`: LLM Format Conversion (188 lines)
- **Abstract Factory Pattern:** `BaseToolConverter` → `OpenAIStyleToolConverter` / `GeminiToolConverter`
- **OpenAI Format:** For DeepSeek and OpenAI-compatible APIs with function calling schema
- **Gemini Format:** Specialized conversion with schema sanitization and type normalization
- **ToolConverterFactory:** Automatic converter selection based on LLM type
- **Features:** Name normalization (`server:tool` → `server_tool`), validation, fallback handling

##### `cli_agent/utils/tool_parsing.py`: Multi-Format Response Parsing (280 lines)
- **Unified Interface:** `ToolCallParser` base class with LLM-specific implementations
- **DeepSeek Parser:** Custom format (`<｜tool▁calls▁begin｜>`), JSON blocks, Python-style calls
- **Gemini Parser:** Structured function calls, XML-style calls, Python-style calls
- **Smart Parsing:** Automatic format detection and multi-pattern support
- **Consistent Output:** Normalized tool call objects regardless of source format

##### `cli_agent/utils/content_processing.py`: Content Extraction
- **Pattern-Based Extraction:** Extract text content before tool calls for streaming
- **LLM-Specific Processors:** Custom patterns for DeepSeek vs Gemini content formatting
- **Code Block Cleaning:** Remove markdown formatting and extract clean content
- **Content Splitting:** Separate conversational text from tool execution sections

##### `cli_agent/utils/http_client.py`: HTTP Client Management
- **HTTPClientFactory:** Standardized client creation with timeout and connection pooling
- **LLM-Specific Clients:** Optimized configurations for OpenAI, DeepSeek, Gemini
- **Lifecycle Management:** `HTTPClientManager` for resource tracking and cleanup
- **Connection Pooling:** Configurable limits and keepalive settings

##### `cli_agent/utils/retry.py`: Retry Logic with Backoff
- **RetryHandler:** Exponential backoff with jitter for API resilience
- **Smart Error Detection:** Identifies retryable errors (timeouts, rate limits, 5xx)
- **Async/Sync Support:** Works with both synchronous and asynchronous functions
- **Configurable:** Base delay, max delay, backoff factor, max attempts

### `agent.py`: CLI Interface (505 lines)

- **Streamlined Entry Point:** Main CLI using Click framework
- **Commands:** `chat`, `ask`, `init`, `mcp`, model switching commands
- **Integration:** Imports from modular `cli_agent` package
- **Host Selection:** Dynamic provider-model selection via configuration

### Provider-Model Architecture

The system implements a sophisticated provider-model separation architecture that replaces the legacy host implementations:

#### **BaseProvider Abstraction:**
- **AnthropicProvider:** Direct Anthropic API integration with native message format
- **OpenAIProvider:** Direct OpenAI API integration with function calling support
- **DeepSeekProvider:** OpenAI-compatible format with reasoning content extraction
- **GoogleProvider:** Gemini API integration with complex message-to-content conversion
- **OpenRouterProvider:** Multi-model aggregator with comprehensive model metadata
- **OllamaProvider:** Local inference provider with dynamic model discovery

#### **ModelConfig Classes:**
- **ClaudeModel:** Anthropic tool format, parameter system prompts, reasoning support
- **GPTModel:** OpenAI tool format, message system prompts, o1 model special handling
- **GeminiModel:** Gemini tool format, prepend system prompts, function calling modes
- **DeepSeekModel:** OpenAI format with reasoning content parsing for chain-of-thought models

#### **MCPHost Composition:**
- **Unified Interface:** Combines Provider + Model via composition pattern
- **Tool Conversion:** Automatic format conversion based on model.get_tool_format()
- **Backward Compatibility:** Inherits from BaseLLMProvider for seamless integration
- **Event-Driven Streaming:** Interrupt-aware streaming with event bus integration

#### **Key Benefits:**
- **Provider Flexibility:** Same model accessible through different providers (e.g., Claude via Anthropic or OpenRouter)
- **Easy Extensibility:** New providers or models require minimal code changes
- **Tool Format Abstraction:** Automatic conversion between OpenAI, Anthropic, and Gemini formats
- **Configuration Management:** Unified configuration system with provider-model combinations

### Subagent System Architecture

**Complete process-isolated subagent system with event-driven communication:**

#### Core Components:

**`subagent.py`: Subagent Management Framework**
- **SubagentManager:** Orchestrates multiple concurrent subagent processes
  - Async process spawning with UUID-based unique task IDs
  - Event-driven message collection via async queues and callbacks
  - Real-time message display during execution
  - Automatic result collection and conversation restart
- **SubagentProcess:** Individual subagent lifecycle management
  - JSON-based task definition and communication protocol
  - Async stdout monitoring with structured message parsing (`SUBAGENT_MSG:` prefix)
  - Process cleanup and graceful termination handling
- **Communication Protocol:** JSON-over-stdout with message types (`output`, `status`, `error`, `result`)

**`subagent_runner.py`: Subprocess Execution Environment**
- **Isolated Execution:** Creates independent host instances with `is_subagent=True`
- **Tool Execution Monitoring:** Intercepts and reports tool execution progress
- **Enhanced Prompting:** Adds specific instructions for focused task execution
- **Result Emission:** Structured result reporting via `emit_result_with_id()`
- **Resource Management:** Automatic cleanup and error handling

#### Key Design Decisions:
- **Process Isolation:** Subagents run in separate processes (not threads) for fault tolerance
- **Tool Access Control:** Subagents have `task` tools removed to prevent infinite recursion
- **Communication Model:** Event-driven async messaging for real-time feedback
- **Result Integration:** Automatic collection and main conversation restart with results

## 3. Execution Flow

The sophisticated modular architecture provides a comprehensive execution flow with advanced features:

### Primary Execution Pipeline:

1.  **System Initialization:**
    - **CLI Bootstrap:** `agent.py` entry point using Click framework with comprehensive command set
    - **Configuration Loading:** Pydantic-based config with environment variable integration
    - **Host Selection:** Dynamic instantiation of `MCPHost` with provider-model configuration
    - **Base Agent Setup:** `BaseMCPAgent.__init__()` with centralized subagent management and tool loading
    - **External Integration:** Automatic MCP server connections with tool discovery and registration

2.  **Input Processing & Command Handling:**
    - **Interactive Mode:** `InterruptibleInput` with prompt_toolkit for professional terminal interaction
    - **Multiline Support:** Smart detection and handling of multiline inputs with proper formatting
    - **Slash Commands:** `SlashCommandManager` processes meta-commands (`/help`, `/compact`, `/switch-*`, `/tools`)
    - **Model Switching:** Runtime backend switching with configuration persistence
    - **Non-Interactive Mode:** Single-turn processing via `ask` command with full tool support

3.  **Context-Aware Message Processing:**
    - **History Management:** Conversation storage with automatic token tracking per model
    - **Smart Compaction:** AI-powered conversation summarization when approaching token limits
    - **Context Delegation:** Intelligent subagent spawning for context-heavy tasks
    - **Response Generation:** Centralized `generate_response()` with streaming/interactive mode detection

4.  **LLM-Specific Processing Pipeline:**
    - **Tool Format Conversion:** Factory-based conversion using `ToolConverterFactory` for LLM-specific schemas
    - **API Integration:** Model-specific `chat_completion()` with retry logic and streaming support
    - **Response Parsing:** Multi-format tool call extraction using `ToolCallParserFactory`
    - **Content Processing:** LLM-specific content extraction and cleaning utilities

5.  **Unified Tool Execution Framework:**
    - **Tool Call Parsing:** Multi-format parsing (structured, XML, Python-style) with validation
    - **Execution Routing:** Built-in tools executed locally, external tools via MCP clients
    - **Parallel Execution:** Concurrent tool execution using `asyncio.gather()` for performance
    - **Result Integration:** Structured result formatting and conversation continuation

6.  **Advanced Streaming & UI:**
    - **Real-Time Streaming:** Response streaming with ESC interruption and keep-alive messages
    - **Terminal Management:** Professional output handling with carriage return management
    - **Subagent Integration:** Real-time message display during subagent execution
    - **Progress Indication:** Tool execution progress with visual indicators

7.  **Subagent Orchestration:**
    - **Process Spawning:** Isolated subprocess creation with unique task IDs
    - **Event-Driven Communication:** Async message queues with real-time callbacks
    - **Resource Management:** Automatic process cleanup and error handling
    - **Result Collection:** Automatic gathering and main conversation restart with results

### Advanced Features:

8.  **Fault Tolerance & Recovery:**
    - **Retry Logic:** Exponential backoff for API calls with smart error detection
    - **Stream Recovery:** Automatic retry for failed streaming responses (Gemini-specific)
    - **Graceful Degradation:** Fallback mechanisms for missing dependencies or failures
    - **Process Isolation:** Subagent failures don't crash main agent

9.  **Context Management Strategy:**
    - **Token Awareness:** Model-specific limits with automatic monitoring
    - **Delegation Patterns:** Smart identification of context-heavy tasks for subagent delegation
    - **Memory Efficiency:** Strategic conversation compaction and cleanup
    - **Subagent Context:** Separate context spaces for focused task execution

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
- **After:** Modular architecture with 29 core files and specialized modules
  - `base_agent.py`: 2,122 lines (core functionality)
  - `chat_interface.py`: 912 lines (interactive chat management)
  - `slash_commands.py`: 806 lines (enhanced command system)
  - `builtin_tool_executor.py`: 685 lines (tool implementations)
  - `response_handler.py`: 682 lines (response processing)
  - `mcp_host.py`: 607 lines (provider-model composition)
  - `event_system.py`: 523 lines (event bus architecture)
  - `builtin_tools.py`: 475 lines (16 tool definitions)
  - `agent.py`: 1,447 lines (comprehensive CLI interface)

### Key Improvements
- **Maintainability**: Each module has a single, clear responsibility with defined interfaces
- **Testability**: Components can be unit tested independently (109 tests passing)
- **Reusability**: Core modules work across different LLM implementations via provider-model architecture
- **Scalability**: New features can be added without touching entire codebase (16 tools, 5 providers)
- **Developer Experience**: Much easier to navigate and understand with comprehensive documentation
- **Code Quality**: Enforced separation of concerns, reduced coupling, and event-driven patterns
- **Event Architecture**: Clean separation via centralized event bus for display and processing
- **Tool Extensibility**: Built-in tools + external MCP integration with unified execution pipeline
- **Provider Flexibility**: Same models accessible through different providers with seamless switching

## 6. Key Design Patterns & Architectural Decisions

The sophisticated codebase employs numerous advanced design patterns and architectural decisions:

### Core Design Patterns:
- **Abstract Base Classes:** `BaseMCPAgent` provides centralized functionality with LLM-specific abstract methods
- **Strategy Pattern:** Interchangeable LLM implementations with unified interfaces for tool conversion and parsing
- **Factory Pattern:** `ToolConverterFactory`, `ContentProcessorFactory`, `HTTPClientFactory` for component creation
- **Observer Pattern:** Event-driven subagent communication with callback registration and async message queues
- **Command Pattern:** CLI framework with Click and slash command system for interactive operations
- **Template Method:** `interactive_chat()` and `generate_response()` define flows with customizable steps
- **Facade Pattern:** `BaseMCPAgent` provides simplified interface to complex tool and conversation management

### Advanced Architectural Patterns:
- **Plugin Architecture:** Modular tool integration with built-in tools + external MCP protocol support
- **Event-Driven Architecture:** Async subagent messaging with real-time callbacks and queue-based communication
- **Process-Based Concurrency:** Subagent isolation using subprocess rather than threading for fault tolerance
- **Stream-First Design:** Unified streaming interface across all backends with interruption support
- **Context-Aware Operation:** Smart delegation patterns for context preservation and efficiency
- **Dependency Injection:** Configuration-driven component instantiation with environment variable integration

### Notable Architectural Decisions:

#### 1. **Centralized vs Distributed Tool Execution**
- **Built-in Tools:** Direct execution within agent process for core functionality
- **External Tools:** MCP server delegation for extensibility without code changes
- **Unified Interface:** Transparent execution pipeline regardless of tool source

#### 2. **Process Isolation for Subagents**
- **Design Choice:** Subprocess isolation rather than threading for fault tolerance
- **Benefits:** Independent resource management, graceful failure handling, scalable execution
- **Trade-offs:** Higher overhead but better reliability and isolation

#### 3. **Multi-Format Tool Call Support**
- **Flexibility:** Support for structured, XML-style, and Python-style tool calls per LLM
- **Robustness:** Multiple parsing strategies with automatic format detection
- **Extensibility:** Easy addition of new formats without breaking existing functionality

#### 4. **Streaming-First with Fallback**
- **Primary Mode:** Real-time streaming responses with user interruption support
- **Reliability:** Automatic retry logic for stream failures (especially Gemini)
- **User Experience:** Immediate feedback with professional terminal handling

#### 5. **Context Management Strategy**
- **Token Awareness:** Model-specific limits with automatic monitoring and compaction
- **Smart Delegation:** Guidance for agents to delegate context-heavy tasks to subagents
- **Memory Efficiency:** Conversation summarization while preserving key context

## 7. Development Guidelines

### Adding New Components
- **New LLM Backend:** Create new `BaseProvider` and `ModelConfig` classes, then compose via `MCPHost`
- **New Tools:** Add to `cli_agent/tools/builtin_tools.py` or create MCP server
- **New Commands:** Add handlers to `SlashCommandManager` or create custom command files
- **New Utilities:** Add to `cli_agent/utils/` for shared functionality

### Import Patterns
```python
# Core components
from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from cli_agent.tools.builtin_tools import get_all_builtin_tools

# Provider-Model Architecture
from cli_agent.core.mcp_host import MCPHost
from cli_agent.providers.anthropic_provider import AnthropicProvider
from cli_agent.core.model_config import ClaudeModel
```

### Recent Architectural Improvements:

**Context Management & Subagent Integration:**
- Enhanced system prompts with context preservation guidance for main agents vs focused execution for subagents
- Tool access control preventing subagent recursion (task tools filtered out for subagents)
- Smart delegation patterns with specific criteria for subagent spawning (>200 lines, multiple files, complex investigations)

**Reliability & Production Features:**
- Comprehensive retry logic with exponential backoff across all API interactions
- Stream failure recovery with automatic retry for Gemini streaming responses
- Enhanced error handling and graceful degradation for missing dependencies
- Professional terminal interaction with proper carriage return handling

**Tool System Enhancements:**
- Multi-format tool call parsing with automatic format detection and validation
- Parallel tool execution using asyncio.gather() for improved performance
- Enhanced argument parsing fixing issues with tools like bash_execute
- Unified tool conversion pipeline supporting multiple LLM formats

**Development Experience:**
- Comprehensive utility framework in cli_agent/utils/ for shared functionality
- Factory pattern implementation for tool converters, content processors, and HTTP clients
- Centralized configuration management with Pydantic validation and environment integration
- Professional documentation and clear separation of concerns

This sophisticated modular architecture provides a production-ready, maintainable foundation that demonstrates advanced software engineering principles. The system successfully balances complexity with usability, offering both powerful capabilities and clean interfaces for developers working with the MCP Agent system.
