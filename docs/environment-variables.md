# CLI-Agent Environment Variables

Complete reference for all environment variables used by the CLI-Agent system.

## 🎯 Quick Start

**Essential Variables** - Set these to get started:
```bash
# Choose your primary provider
export ANTHROPIC_API_KEY=your_anthropic_key
# OR
export OPENAI_API_KEY=your_openai_key  
# OR
export DEEPSEEK_API_KEY=your_deepseek_key

# Optional: Set default provider-model
export DEFAULT_PROVIDER_MODEL=anthropic:claude-3.5-sonnet
```

## 📚 API Provider Configuration

### Anthropic (Claude)
```bash
ANTHROPIC_API_KEY=""                    # Required: Anthropic API key
ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"  # Optional: Model name
ANTHROPIC_TEMPERATURE=0.7               # Optional: Temperature (0.0-1.0)
ANTHROPIC_MAX_TOKENS=8192              # Optional: Max response tokens
```

### OpenAI (GPT)
```bash
OPENAI_API_KEY=""                      # Required: OpenAI API key
OPENAI_MODEL="gpt-4-turbo-preview"     # Optional: Model name
OPENAI_TEMPERATURE=0.7                 # Optional: Temperature (0.0-1.0)
OPENAI_MAX_TOKENS=4096                 # Optional: Max response tokens
```

### DeepSeek (Chat & Reasoning)
```bash
DEEPSEEK_API_KEY=""                    # Required: DeepSeek API key
DEEPSEEK_MODEL="deepseek-chat"         # Optional: Model name
DEEPSEEK_TEMPERATURE=0.6               # Optional: Temperature (0.0-1.0)
DEEPSEEK_MAX_TOKENS=4096               # Optional: Max response tokens
DEEPSEEK_STREAM=true                   # Optional: Enable streaming
```

### Google Gemini
```bash
GEMINI_API_KEY=""                      # Required: Google AI Studio API key
GEMINI_MODEL="gemini-2.5-flash"       # Optional: Model name
GEMINI_TEMPERATURE=0.7                 # Optional: Temperature (0.0-1.0)
GEMINI_MAX_TOKENS=8192                 # Optional: Max response tokens
GEMINI_TOP_P=0.9                       # Optional: Top-p parameter
GEMINI_TOP_K=40                        # Optional: Top-k parameter
GEMINI_STREAM=false                    # Optional: Enable streaming
GEMINI_FORCE_FUNCTION_CALLING=false    # Optional: Force function calling
GEMINI_FUNCTION_CALLING_MODE="AUTO"    # Optional: AUTO, ANY, or NONE
```

### OpenRouter (Multi-Provider)
```bash
OPENROUTER_API_KEY=""                  # Required: OpenRouter API key
OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"  # Optional: Model name
OPENROUTER_TEMPERATURE=0.7             # Optional: Temperature (0.0-1.0)
OPENROUTER_MAX_TOKENS=8192             # Optional: Max response tokens
```

### Ollama (Local Models)
```bash
OLLAMA_BASE_URL="http://localhost:11434"  # Optional: Ollama server URL
OLLAMA_MODEL="llama2"                  # Optional: Model name
OLLAMA_TEMPERATURE=0.7                 # Optional: Temperature (0.0-1.0)
OLLAMA_MAX_TOKENS=4096                 # Optional: Max response tokens
OLLAMA_TIMEOUT=180.0                   # Optional: Request timeout (seconds)
```

## ⚙️ Core System Configuration

### Provider-Model Selection
```bash
DEFAULT_PROVIDER_MODEL=""              # Default: intelligent auto-selection
                                      # Format: "provider:model"
                                      # Examples: "anthropic:claude-3.5-sonnet"
                                      #          "openai:gpt-4-turbo-preview"
                                      #          "deepseek:deepseek-chat"

MODEL_TYPE="deepseek"                  # Legacy: backward compatibility only
```

### Host Configuration
```bash
HOST_NAME="mcp-agent"                  # Agent identifier for sessions
LOG_LEVEL="INFO"                       # Logging level: DEBUG, INFO, WARNING, ERROR
LAST_SESSION_ID=""                     # Last active session for continuation
```

## 🔧 Tool System Configuration

### Tool Permissions
```bash
# Security: Control which tools are available
ALLOWED_TOOLS="task,task_status,task_results,emit_result,read_file,list_directory,get_current_directory,todo_read,todo_write"
DISALLOWED_TOOLS=""                    # Comma-separated list to block

# Automation: Skip permission prompts
AUTO_APPROVE_TOOLS=false               # WARNING: Security risk if enabled
```

### Tool Display
```bash
TRUNCATE_TOOL_RESULTS=true             # Truncate long tool outputs
TOOL_RESULT_MAX_LENGTH=1000            # Max characters to display
```

## 🤖 Subagent System

### Subagent Configuration
```bash
BACKGROUND_SUBAGENTS=false             # Enable background task execution
SUBAGENT_PERMISSIONS_BYPASS=false     # Allow subagents to bypass permissions
                                      # WARNING: Security consideration
```

### Special Operation Modes
```bash
STREAM_JSON_MODE=""                    # Internal: Enable JSON streaming mode
                                      # Used for machine-to-machine communication
```

## 🪝 Hooks System

### Hook Configuration
```bash
HOOKS_ENABLED=true                     # Enable/disable hooks system
HOOKS_TIMEOUT=30                       # Hook execution timeout (seconds)
```

## 🌐 MCP Server Configuration

### MCP Model Server
```bash
MCP_SERVER_ENABLED=false               # Enable MCP model server
MCP_SERVER_PORT=3000                   # Server port
MCP_SERVER_HOST="localhost"            # Server host address
MCP_SERVER_AUTH_TOKEN=""               # Authentication token
MCP_SERVER_LOG_LEVEL="INFO"            # Server log level
```

## 🔄 Configuration Loading Priority

Variables are loaded in this order (highest to lowest priority):

1. **Environment Variables** (highest priority)
2. **`.env` file** in current directory
3. **Persistent Config** `~/.config/agent/config.json`
4. **Default Values** (lowest priority)

## 📂 Configuration File Locations

| Purpose | Location |
|---------|----------|
| **Main Config** | `~/.config/agent/config.json` |
| **Environment File** | `./.env` (project-specific) |
| **MCP Servers** | `~/.config/agent/mcp_servers.json` |
| **Hooks** | `~/.config/agent/hooks/` |
| **Sessions** | `~/.config/agent/sessions/` |
| **Todos** | `~/.config/agent/todos/` |

## 🎯 Common Configuration Patterns

### Development Setup
```bash
# .env file for development
ANTHROPIC_API_KEY=your_key_here
DEFAULT_PROVIDER_MODEL=anthropic:claude-3.5-sonnet
LOG_LEVEL=DEBUG
HOOKS_ENABLED=true
```

### Production Setup
```bash
# Production environment
DEEPSEEK_API_KEY=your_production_key
DEFAULT_PROVIDER_MODEL=deepseek:deepseek-chat
LOG_LEVEL=INFO
AUTO_APPROVE_TOOLS=false
SUBAGENT_PERMISSIONS_BYPASS=false
```

### Local Development with Ollama
```bash
# Using local models
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
DEFAULT_PROVIDER_MODEL=ollama:llama3.1
```

### Multi-Provider Setup
```bash
# All providers configured
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key

# Switch between them using /switch command
DEFAULT_PROVIDER_MODEL=anthropic:claude-3.5-sonnet
```

## 🛡️ Security Considerations

### API Keys
- **Never commit API keys** to version control
- **Use `.env` files** for local development
- **Use environment variables** in production
- **Rotate keys regularly**

### Permission Settings
```bash
# Secure defaults (recommended)
AUTO_APPROVE_TOOLS=false               # Always prompt for tool usage
SUBAGENT_PERMISSIONS_BYPASS=false     # Subagents follow permission rules
HOOKS_ENABLED=true                     # Allow hooks for security auditing
```

### Automation Settings (Use with caution)
```bash
# For automated workflows only
AUTO_APPROVE_TOOLS=true                # WARNING: Skips all permission checks
SUBAGENT_PERMISSIONS_BYPASS=true      # WARNING: Subagents bypass security
```

## 🔍 Debugging Variables

### Verbose Logging
```bash
LOG_LEVEL=DEBUG                        # Detailed logging
```

### Tool Debugging
```bash
TRUNCATE_TOOL_RESULTS=false           # Show full tool outputs
TOOL_RESULT_MAX_LENGTH=10000          # Increase display limit
```

## 📋 Environment Variable Validation

The agent validates environment variables on startup:

- **API Keys**: Checked for basic format
- **Numeric Values**: Validated for type and range
- **Enum Values**: Checked against allowed options
- **URLs**: Validated for basic format

Invalid values fall back to defaults with warnings in logs.

## 🔗 Related Documentation

- [**Configuration Guide**](configuration.md) - General configuration
- [**Hooks System**](hooks.md) - Workflow automation
- [**MCP Integration**](mcp-integration.md) - Server setup
- [**Security Guide**](security.md) - Security best practices

## 💡 Tips

1. **Start Simple**: Only set API keys initially
2. **Use .env Files**: For project-specific settings
3. **Check Logs**: Set `LOG_LEVEL=DEBUG` for troubleshooting
4. **Validate Settings**: Use `/model` command to verify configuration
5. **Security First**: Never enable auto-approval in production