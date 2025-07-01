# CLI Agent

A powerful, modular command-line interface for interacting with AI models enhanced with Model Context Protocol (MCP) tool integration. Features a centralized architecture that makes it easy to add new LLM providers while providing robust tool integration and subagent management capabilities.

## 🚀 Features

- **Multiple AI Backends**: Support for Anthropic Claude, OpenAI GPT, DeepSeek, Google Gemini, OpenRouter, and ollama with easy extensibility
- **Hooks System**: Powerful workflow automation with security controls - audit commands, block dangerous operations, auto-format code, and integrate with external tools
- **MCP Model Server**: Expose all AI models as standardized MCP tools with persistent conversations
- **Modular Architecture**: Provider-model separation with centralized base agent for maximum flexibility
- **MCP Server Integration**: Connect to multiple MCP servers for extended functionality
- **Persistent Conversations**: Maintain conversation context across multiple tool calls for each AI model
- **Interactive Chat**: Real-time conversation with AI models and comprehensive tool access
- **Subagent System**: Spawn focused subagents for complex tasks with automatic coordination
- **Command-Line Tools**: Manage MCP servers and query models directly
- **Built-in Tools**: File operations, bash execution, web fetching, todo management, and task delegation
- **Enhanced Tool Display**: Full parameter visibility and complete response output (no truncation)

## 📦 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install mcp-cli-agent
```

### Option 2: Install from Source

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/amranu/cli-agent.git
    cd cli-agent
    ```

2.  **Install the package**:
    ```bash
    pip install -e .
    ```

## ⚙️ Configuration

**Configure API keys** (environment variables):

```bash
    # Set environment variables for the providers you want to use
    export OPENAI_API_KEY=your_openai_api_key_here
    export DEEPSEEK_API_KEY=your_deepseek_api_key_here
    export ANTHROPIC_API_KEY=your_anthropic_api_key_here
    export GEMINI_API_KEY=your_gemini_api_key_here
    export OPENROUTER_API_KEY=your_openrouter_api_key_here

    # Start with automatic provider selection
    agent chat
    
    # Or specify a particular provider-model combination
    agent chat --model openai:gpt-4-turbo-preview
```

**Smart Provider Selection**: The agent automatically selects a configured provider based on available API keys.

Configuration is automatically saved to `~/.config/cli-agent/config.json` and persists across sessions.

**Migration from MCP Agent**: If you previously used MCP Agent, your configuration will be automatically migrated from `~/.config/mcp-agent/` to `~/.config/cli-agent/` on first run.

## 🛠️ Usage

### Interactive Chat

Start an interactive chat session with your configured AI model and MCP tools:

```bash
agent chat --model deepseek:deepseek-chat
```

### MCP Model Server

Start the MCP model server to expose all AI models as standardized MCP tools with persistent conversations:

```bash
# Start via stdio transport (recommended for MCP clients)
python mcp_server.py --stdio

# Or start via agent CLI (defaults to stdio)
agent mcp serve
agent mcp serve --stdio  # explicit stdio

# Start with TCP transport (useful for debugging)
python mcp_server.py --tcp --port 3000 --host localhost
agent mcp serve --tcp --port 3000 --host localhost
```

The model server exposes AI models from 6 providers:
- **Anthropic**: Claude models
- **OpenAI**: GPT models
- **DeepSeek**: Chat and reasoning models
- **Gemini**: Google's Gemini models
- **OpenRouter**: Multi-provider access
- **Ollama**: Local models via Ollama server


### MCP Server Management

#### Add a new MCP server

```bash
# Format: name:command:arg1:arg2:...
agent mcp add myserver:node:/path/to/server.js
agent mcp add filesystem:python:-m:mcp.server.stdio:filesystem:--root:.

# Add the AI models server to your MCP configuration
agent mcp add ai-models:python:mcp_server.py:--stdio
```

#### List configured servers

```bash
agent mcp list
```

#### Remove a server

```bash
agent mcp remove myserver
```

### Single Query

Ask a one-time question without entering interactive mode:

```bash
agent ask "What's the weather like today?"
```

### Model Switching

Switch between different AI models using the provider-model format (configuration persists automatically):

```bash
# Provider-model format switching
agent switch anthropic:claude-3.5-sonnet
agent switch openai:gpt-4-turbo-preview
agent switch deepseek:deepseek-chat
agent switch gemini:gemini-2.5-flash

```

Or use slash commands within interactive chat:

```
/switch anthropic:claude-3.5-sonnet
/switch openai:gpt-4-turbo-preview
/switch deepseek:deepseek-reasoner
/switch gemini:gemini-2.5-pro

```

## 🔧 Configuration

### Persistent Configuration System

The agent uses an automatic persistent configuration system that saves settings to `~/.config/cli-agent/config.json`:

-   **API Keys**: Set via environment variables
-   **Model Preferences**: Automatically saved when using switch commands
-   **MCP Servers**: Managed through the CLI and persisted across sessions
-   **Tool Permissions**: Configurable with session-based approval system

### Environment Variables

Configure the agent through environment variables:

```bash
# Anthropic Configuration (required for Claude models)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022     # optional, defaults to claude-3-5-sonnet-20241022
ANTHROPIC_TEMPERATURE=0.7                      # optional, defaults to 0.7

# OpenAI Configuration (required for GPT models)  
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4-turbo-preview               # optional, defaults to gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7                         # optional, defaults to 0.7

# DeepSeek Configuration (required for DeepSeek models)
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-chat                   # optional, defaults to deepseek-chat
DEEPSEEK_TEMPERATURE=0.6                       # optional, defaults to 0.6

# Gemini Configuration (required for Gemini models)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash                  # optional, defaults to gemini-2.5-flash
GEMINI_TEMPERATURE=0.7                         # optional, defaults to 0.7

# OpenRouter Configuration (optional for multi-provider access)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet   # optional
OPENROUTER_TEMPERATURE=0.7                     # optional, defaults to 0.7

# Provider-Model Selection (new format)
DEFAULT_PROVIDER_MODEL=anthropic:claude-3.5-sonnet  # defaults to deepseek:deepseek-chat

# Subagent Configuration
SUBAGENT_PERMISSIONS_BYPASS=false             # bypass permission checks for subagents, defaults to false

# Host Configuration (optional)
HOST_NAME=cli-agent                            # defaults to 'cli-agent'
LOG_LEVEL=INFO                                 # defaults to INFO
```

Configuration changes made via commands (like model switching) are automatically persisted and don't require manual `.env` file editing.

## 🎯 Available Tools

### Built-in Tools

The agent comes with comprehensive built-in tools:

-   **File Operations**: Read, write, edit, and search files with surgical precision
-   **Directory Operations**: List directories, get current path, navigate filesystem
-   **Shell Execution**: Run bash commands with full output capture
-   **Web Fetching**: Download and process web content
-   **Todo Management**: Organize and track tasks across sessions
-   **Task Delegation**: Spawn focused subagents for complex or context-heavy tasks
-   **Text Processing**: Search, replace, and manipulate text content


### AI Model Tools (via MCP Server)

All model tools support:
- **Persistent Conversations**: Maintain context across calls
- **Conversation Management**: Create, continue, or clear conversations
- **Full Parameter Control**: Temperature, max_tokens, system prompts

## 🔍 Interactive Chat Commands

Within the interactive chat, use these slash commands:

-   `/help` - Show available commands
-   `/tools` - List all available tools
-   `/clear` - Clear conversation history
-   `/model` - Show current model
-   `/tokens` - Show token usage
-   `/compact` - Compact conversation history
-   `/switch <provider>:<model>` - Switch to any provider-model combination
-   `/task` - Spawn a subagent for complex tasks

## 📚 Examples


### Example: Basic File Operations

```bash
agent chat --model deepseek:deepseek-chat
```

In chat:

```
You: List all files in this directory
You: Read the contents of agent.py
You: Create a new file called hello.py with a simple function
```

### Example: System Operations

In chat:

```
You: Show me the current directory
You: Run "git status" to check repository status
You: What's the disk usage of this folder?
```

### Example: Subagent Task Delegation

For complex or context-heavy tasks, delegate to focused subagents:

```
You: /task Analyze all Python files in the src/ directory and create a summary of the class structure and dependencies

You: Can you analyze this large log file and find any error patterns?
     [Agent automatically spawns subagent for file analysis]

```

Subagents work independently and automatically return results to the main conversation.

## 🏗️ Architecture

### Provider-Model Architecture

```
┌─────────────────┐
│   CLI Interface │    
│   (agent.py)    │    
└─────────┬───────┘    
          │            
          ▼            
┌─────────────────────┐    ┌─────────────────────────────────────────┐
│      MCPHost        │◄───│              Provider + Model           │
│ (BaseLLMProvider)   │    │                Composition              │
└─────────┬───────────┘    │ ┌─────────────────┐ ┌─────────────────┐ │
          │                │ │   BaseProvider  │ │   ModelConfig   │ │
          ▼                │ │   Subclasses:   │ │   Subclasses:   │ │
┌─────────────────────┐    │ │ • Anthropic     │ │ • ClaudeModel   │ │
│   BaseLLMProvider   │    │ │ • OpenAI        │ │ • GPTModel      │ │
│  (Centralized LLM   │    │ │ • DeepSeek      │ │ • GeminiModel   │ │
│   functionality)    │    │ │ • Google        │ │ • DeepSeekModel │ │
└─────────┬───────────┘    │ │ • OpenRouter    │ │                 │ │
          │                │ └─────────────────┘ └─────────────────┘ │
          ▼                └─────────────────────────────────────────┘
┌─────────────────────┐                         │
│   BaseMCPAgent      │                         ▼
│  (Abstract base)    │               ┌─────────────────┐
└─────────┬───────────┘               │ Tool Converters │
          │                           │ • OpenAI Format │
┌─────────▼───────────┐               │ • Anthropic     │
│ Core Components:    │               │ • Gemini        │
│ • SubagentCoordinator│              └─────────────────┘
│ • BuiltinToolExecutor│                
│ • ChatInterface     │                ┌─────────────────┐
│ • SlashCommands     │◄───────────────│ Built-in Tools  │
│ • ToolExecutionEngine│               │ • File Ops      │
│ • TokenManager      │               │ • Bash Execute  │
│ • SystemPromptBuilder│               │ • Web Fetch     │
│ • MessageProcessor  │               │ • Todo Mgmt     │
└─────────┬───────────┘               │ • Task Spawn    │
          │                           │ • Glob/Grep     │
          ▼                           │ • MultiEdit     │
┌─────────────────────┐               └─────────────────┘
│ External MCP Servers │                      
│ • AI Model Server   │               ┌─────────────────┐
│ • File System       │               │ Subagent System │
│ • APIs & Databases  │               │ • Focused Tasks │
│ • Custom Tools      │               │ • Auto Cleanup  │
└─────────────────────┘               │ • Event-Driven  │
                                      └─────────────────┘
```

### Key Architectural Benefits

-   **Provider-Model Separation**: API providers decoupled from model characteristics
-   **MCP Model Server**: Standardized access to all AI models via MCP protocol
-   **Persistent Conversations**: Conversation context maintained across tool calls
-   **Easy Extensibility**: Adding new providers or models requires minimal code
-   **Robust Tool Integration**: Unified tool execution with provider-specific optimizations
-   **Intelligent Subagent System**: Automatic task delegation and coordination
-   **Multi-Provider Access**: Same model accessible through different providers
-   **Enhanced Visibility**: Full parameter display and complete response output

## 🤝 Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on our code of conduct and the process for submitting pull requests.

1.  Fork the repository
2.  Create a feature branch: `git checkout -b feature-name`
3.  Make your changes
4.  Add tests if applicable
5.  Commit your changes: `git commit -m 'Add feature'`
6.  Push to the branch: `git push origin feature-name`
7.  Submit a pull request

## 📋 Requirements

-   Python 3.10+
-   API keys for desired providers:
    -   Anthropic API key (for Claude models)
    -   OpenAI API key (for GPT models)
    -   DeepSeek API key (for DeepSeek models)
    -   Google AI Studio API key (for Gemini models)
    -   OpenRouter API key (for multi-provider access)
-   FastMCP for MCP server functionality
-   Node.js (for MCP servers that require it)

## 🔒 Security

-   **API Keys**: Stored as environment variables
-   **Configuration**: Automatically managed in user home directory (`~/.config/cli-agent/`)
-   **MCP Servers**: Local configurations with session-based tool permissions
-   **Tool Execution**: Built-in permission system for sensitive operations
-   **Subagent Isolation**: Subagents run in controlled environments with specific tool access
-   **Subagent Permissions**: Can be configured to bypass permission checks for automated workflows via `SUBAGENT_PERMISSIONS_BYPASS=true`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for the extensible tool integration framework
-   [DeepSeek](https://www.deepseek.com/) for the powerful reasoning models
-   [Google AI](https://ai.google.dev/) for Gemini model access
-   [FastMCP](https://github.com/jlowin/fastmcp) for the Python MCP client implementation

## 📞 Support

-   🐛 [Report Issues](https://github.com/amranu/cli-agent/issues)
-   💬 [Discussions](https://github.com/amranu/cli-agent/discussions)
-   📖 [Wiki](https://github.com/amranu/cli-agent/wiki)

---

**Happy coding with CLI Agent! 🤖✨**
