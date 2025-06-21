# MCP Agent

A powerful command-line interface for interacting with AI models enhanced with Model Context Protocol (MCP) tool integration. Connect multiple MCP servers to extend your AI agent with external APIs, file systems, databases, and more.

## 🚀 Features

- **Multiple AI Backends**: Support for DeepSeek and Google Gemini models
- **MCP Server Integration**: Connect to multiple MCP servers for extended functionality
- **Persistent Configuration**: Save and manage MCP server configurations
- **Interactive Chat**: Real-time conversation with AI models and tool access
- **Command-Line Tools**: Manage MCP servers and query models directly
- **Built-in Tools**: File operations, bash execution, web fetching, and todo management

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amranu/agent.git
   cd agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize configuration**:
   ```bash
   python agent.py init
   ```

4. **Configure API keys** by editing the generated `.env` file:
   ```bash
   # DeepSeek API Configuration
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   # Gemini API Configuration  
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

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

Switch between different AI models:

```bash
python agent.py switch-chat      # DeepSeek Chat model
python agent.py switch-reason    # DeepSeek Reasoner model
python agent.py switch-gemini    # Google Gemini Flash
python agent.py switch-gemini-pro # Google Gemini Pro
```

## 🔧 Configuration

### Environment Variables

Configure the agent through environment variables in your `.env` file:

```bash
# DeepSeek Configuration
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.7

# Gemini Configuration
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7

# Host Configuration
HOST_NAME=mcp-agent
LOG_LEVEL=INFO
```

## 🎯 Available Tools

### Built-in Tools

The agent comes with several built-in tools:

- **File Operations**: Read, write, edit files
- **Directory Operations**: List directories, get current path  
- **Shell Execution**: Run bash commands
- **Web Fetching**: Download web content
- **Todo Management**: Manage task lists

### MCP Server Tools

Connect external MCP servers to add functionality like:

- **API Integrations**: Connect to various web APIs
- **File System**: Advanced file operations
- **Database Connectors**: PostgreSQL, MySQL, SQLite
- **Development Tools**: Git operations, code analysis
- **Custom Services**: Your own MCP-compatible tools

## 🔍 Interactive Chat Commands

Within the interactive chat, use these slash commands:

- `/help` - Show available commands
- `/tools` - List all available tools
- `/clear` - Clear conversation history
- `/model` - Show current model
- `/tokens` - Show token usage
- `/compact` - Compact conversation history
- `/switch-chat` - Switch to DeepSeek Chat
- `/switch-reason` - Switch to DeepSeek Reasoner
- `/switch-gemini` - Switch to Gemini Flash
- `/switch-gemini-pro` - Switch to Gemini Pro

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

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │────│   Agent Core    │────│   AI Backends   │
│                 │    │                 │    │ (DeepSeek/Gemini)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               │
                    ┌─────────────────┐
                    │  MCP Servers    │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ File System │ │
                    │ │ APIs        │ │
                    │ │ Database    │ │
                    │ │ Custom Tools│ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -m 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📋 Requirements

- Python 3.8+
- DeepSeek API key (for DeepSeek models)
- Google AI Studio API key (for Gemini models)
- Node.js (for MCP servers that require it)

## 🔒 Security

- API keys are stored in `.env` files (not committed to git)
- MCP server configurations are stored locally
- All sensitive files are excluded via `.gitignore`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for the extensible tool integration framework
- [DeepSeek](https://www.deepseek.com/) for the powerful reasoning models
- [Google AI](https://ai.google.dev/) for Gemini model access
- [FastMCP](https://github.com/jlowin/fastmcp) for the Python MCP client implementation

## 📞 Support

- 🐛 [Report Issues](https://github.com/amranu/agent/issues)
- 💬 [Discussions](https://github.com/amranu/agent/discussions)
- 📖 [Wiki](https://github.com/amranu/agent/wiki)

---

**Happy coding with MCP Agent! 🤖✨**