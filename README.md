# MCP Agent

A powerful, modular command-line interface for interacting with AI models enhanced with Model Context Protocol (MCP) tool integration. Features a centralized architecture that makes it easy to add new LLM providers while providing robust tool integration and subagent management capabilities.

## 🚀 Features

- **Multiple AI Backends**: Support for DeepSeek and Google Gemini models with easy extensibility
- **Modular Architecture**: Centralized base agent with provider-specific implementations
- **MCP Server Integration**: Connect to multiple MCP servers for extended functionality
- **Persistent Configuration**: Automatic configuration management in `~/.config/agent/`
- **Interactive Chat**: Real-time conversation with AI models and comprehensive tool access
- **Subagent System**: Spawn focused subagents for complex tasks with automatic coordination
- **Command-Line Tools**: Manage MCP servers and query models directly
- **Built-in Tools**: File operations, bash execution, web fetching, todo management, and task delegation

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/amranu/agent.git
    cd agent
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API keys** (environment variables or interactive setup):
    ```bash
    # Set environment variables (recommended)
    export DEEPSEEK_API_KEY=your_deepseek_api_key_here
    export GEMINI_API_KEY=your_gemini_api_key_here

    # Or use interactive configuration
    python agent.py chat  # Will prompt for missing keys
    ```

    Configuration is automatically saved to `~/.config/agent/config.py` and persists across sessions.

## 🛠️ Usage

### Interactive Chat

Start an interactive chat session with your configured AI model and MCP tools:

```bash
python agent.py chat
```

### MCP Server Management

#### Add a new MCP server

```bash
# Format: name:command:arg1:arg2:...
python agent.py mcp add myserver:node:/path/to/server.js
python agent.py mcp add filesystem:python:-m:mcp.server.stdio:filesystem:--root:.
```

#### List configured servers

```bash
python agent.py mcp list
```

#### Remove a server

```bash
python agent.py mcp remove myserver
```

### Single Query

Ask a one-time question without entering interactive mode:

```bash
python agent.py ask "What's the weather like today?"
```

### Model Switching

Switch between different AI models (configuration persists automatically):

```bash
python agent.py switch-deepseek      # DeepSeek Chat model
python agent.py switch-reason        # DeepSeek Reasoner model
python agent.py switch-gemini-flash  # Google Gemini Flash
python agent.py switch-gemini-pro    # Google Gemini Pro
```

Or use slash commands within interactive chat:

```
/switch-deepseek
/switch-reason
/switch-gemini-flash
/switch-gemini-pro
```

## 🔧 Configuration

### Persistent Configuration System

The agent uses an automatic persistent configuration system that saves settings to `~/.config/agent/config.py`:

-   **API Keys**: Set via environment variables or interactive prompts
-   **Model Preferences**: Automatically saved when using switch commands
-   **MCP Servers**: Managed through the CLI and persisted across sessions
-   **Tool Permissions**: Configurable with session-based approval system

### Environment Variables

Configure the agent through environment variables:

```bash
# DeepSeek Configuration (required for DeepSeek models)
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-chat                    # optional, defaults to deepseek-chat
DEEPSEEK_TEMPERATURE=0.7                        # optional, defaults to 0.7

# Gemini Configuration (required for Gemini models)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash                   # optional, defaults to gemini-2.5-flash
GEMINI_TEMPERATURE=0.7                          # optional, defaults to 0.7

# Host Configuration (optional)
HOST_NAME=mcp-agent                             # defaults to 'mcp-agent'
LOG_LEVEL=INFO                                  # defaults to INFO
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

### MCP Server Tools

Connect external MCP servers to add functionality like:

-   **API Integrations**: Connect to various web APIs
-   **File System**: Advanced file operations
-   **Database Connectors**: PostgreSQL, MySQL, SQLite
-   **Development Tools**: Git operations, code analysis
-   **Custom Services**: Your own MCP-compatible tools

## 🔍 Interactive Chat Commands

Within the interactive chat, use these slash commands:

-   `/help` - Show available commands
-   `/tools` - List all available tools
-   `/clear` - Clear conversation history
-   `/model` - Show current model
-   `/tokens` - Show token usage
-   `/compact` - Compact conversation history
-   `/switch-deepseek` - Switch to DeepSeek Chat
-   `/switch-reason` - Switch to DeepSeek Reasoner
-   `/switch-gemini-flash` - Switch to Gemini Flash
-   `/switch-gemini-pro` - Switch to Gemini Pro
-   `/task` - Spawn a subagent for complex tasks
-   `/task-status` - Check status of running subagents

## 📚 Examples

### Example: Basic File Operations

```bash
python agent.py chat
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

You: /task-status
     [Shows: "1 subagent running: log-analysis-task"]
```

Subagents work independently and automatically return results to the main conversation.

## 🏗️ Architecture

### Modular Agent Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   CLI Interface │────│   BaseMCPAgent       │────│   AI Backends   │
│                 │    │   (Centralized)      │    │                 │
└─────────────────┘    └──────────────────────┘    │ ┌─────────────┐ │
                               │                    │ │DeepSeek Host│ │
                               │                    │ │Gemini Host  │ │
                    ┌─────────────────┐              │ │[Easy to    │ │
                    │  Subagent Mgr   │              │ │ extend]     │ │
                    │                 │              │ └─────────────┘ │
                    │ ┌─────────────┐ │              └─────────────────┘
                    │ │Focused Tasks│ │                        │
                    │ │Auto Cleanup │ │              ┌─────────────────┐
                    │ │Parallel Exec│ │              │  MCP Servers    │
                    │ └─────────────┘ │              │                 │
                    └─────────────────┘              │ ┌─────────────┐ │
                               │                     │ │ File System │ │
                    ┌─────────────────┐              │ │ APIs        │ │
                    │ Built-in Tools  │              │ │ Database    │ │
                    │                 │              │ │ Custom Tools│ │
                    │ ┌─────────────┐ │              │ └─────────────┘ │
                    │ │File Ops     │ │              └─────────────────┘
                    │ │Bash Execute │ │
                    │ │Todo Mgmt    │ │
                    │ │Web Fetch    │ │
                    │ │Task Spawn   │ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

### Key Architectural Benefits

-   **Centralized Base Agent**: Shared functionality across all LLM providers
-   **Easy Extensibility**: Adding new LLM backends requires minimal code
-   **Robust Tool Integration**: Unified tool execution with provider-specific optimizations
-   **Intelligent Subagent System**: Automatic task delegation and coordination
-   **Persistent Configuration**: No manual file editing required

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
-   DeepSeek API key (for DeepSeek models)
-   Google AI Studio API key (for Gemini models)
-   Node.js (for MCP servers that require it)

## 🔒 Security

-   **API Keys**: Stored as environment variables, never committed to git
-   **Configuration**: Automatically managed in user home directory (`~/.config/agent/`)
-   **MCP Servers**: Local configurations with session-based tool permissions
-   **Tool Execution**: Built-in permission system for sensitive operations
-   **Subagent Isolation**: Subagents run in controlled environments with specific tool access

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for the extensible tool integration framework
-   [DeepSeek](https://www.deepseek.com/) for the powerful reasoning models
-   [Google AI](https://ai.google.dev/) for Gemini model access
-   [FastMCP](https://github.com/jlowin/fastmcp) for the Python MCP client implementation

## 📞 Support

-   🐛 [Report Issues](https://github.com/amranu/agent/issues)
-   💬 [Discussions](https://github.com/amranu/agent/discussions)
-   📖 [Wiki](https://github.com/amranu/agent/wiki)

---

**Happy coding with MCP Agent! 🤖✨**
