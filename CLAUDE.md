# CLAUDE.md - MCP Agent Architecture Documentation

This document provides comprehensive technical documentation for the MCP Agent codebase, specifically designed to help Claude Code and other AI assistants understand the modular architecture, components, and development patterns.

## 🏗️ Architecture Overview

The MCP Agent has been refactored from a monolithic 3,237-line file into a clean modular architecture with a sophisticated Provider-Model system. The system provides an extensible framework for creating language model-powered agents with tool integration, interactive chat, and multi-backend support via a flexible provider-model architecture.

### Key Architectural Principles

- **Modular Design**: Components separated by responsibility
- **Provider-Model Separation**: API providers separated from model characteristics
- **Model Agnosticism**: Core logic independent of LLM implementation  
- **Tool Extensibility**: Support for both built-in and MCP protocol tools
- **Interactive & Programmatic**: CLI and library usage patterns
- **Centralized Management**: Single source of truth for shared functionality
- **Multi-Provider Support**: Same model accessible through different providers

## 📁 Modular File Structure

```
agent/
├── cli_agent/                    # Main package - modular architecture
│   ├── __init__.py              # Package exports and version
│   ├── core/                    # Core agent functionality (29 core files)
│   │   ├── __init__.py         # Core component exports
│   │   ├── base_agent.py       # BaseMCPAgent abstract class (2,122 lines)
│   │   ├── base_llm_provider.py # BaseLLMProvider shared functionality (456 lines)
│   │   ├── base_provider.py    # BaseProvider API abstraction (279 lines)
│   │   ├── model_config.py     # ModelConfig classes for LLM characteristics (531 lines)
│   │   ├── mcp_host.py         # MCPHost unified provider+model interface (607 lines)
│   │   ├── chat_interface.py   # Interactive chat management (912 lines)
│   │   ├── input_handler.py    # InterruptibleInput terminal handling (303 lines)
│   │   ├── slash_commands.py   # SlashCommandManager command system (806 lines)
│   │   ├── tool_execution_engine.py # Tool execution and validation (280 lines)
│   │   ├── builtin_tool_executor.py # Built-in tool implementations (685 lines)
│   │   ├── event_system.py     # Central event bus architecture (523 lines)
│   │   ├── display_manager.py  # Event-driven display coordination (400 lines)
│   │   ├── response_handler.py # Response processing framework (682 lines)
│   │   ├── subagent_coordinator.py # Subagent lifecycle management
│   │   ├── tool_permissions.py # Tool access control system (421 lines)
│   │   ├── terminal_manager.py # Terminal state management
│   │   ├── global_interrupt.py # Centralized interrupt handling
│   │   └── [15 additional core modules]
│   ├── providers/               # API provider implementations
│   │   ├── __init__.py         # Provider exports
│   │   ├── anthropic_provider.py # Anthropic API provider
│   │   ├── openai_provider.py  # OpenAI API provider
│   │   ├── openrouter_provider.py # OpenRouter API provider
│   │   ├── deepseek_provider.py # DeepSeek API provider
│   │   └── google_provider.py  # Google Gemini API provider
│   ├── tools/                   # Tool integration and execution
│   │   ├── __init__.py         # Tool exports
│   │   └── builtin_tools.py    # Built-in tool definitions (475 lines)
│   ├── utils/                   # Utility functions (7 utility modules)
│   │   ├── __init__.py         # Utility exports
│   │   ├── tool_conversion.py  # Multi-format tool schema conversion
│   │   ├── content_processing.py # Content extraction and cleaning utilities
│   │   ├── diff_display.py     # Terminal diff visualization with colors
│   │   ├── http_client.py      # HTTP client factory and lifecycle management
│   │   ├── retry.py            # Exponential backoff retry logic
│   │   └── tool_parsing.py     # Tool call parsing from LLM responses
│   ├── mcp/                     # MCP protocol implementations
│   │   ├── __init__.py         # MCP module initialization
│   │   └── model_server.py     # MCP model server implementation
│   │   ├── retry.py            # Retry logic utilities
│   │   └── tool_parsing.py     # Tool parsing utilities
│   ├── cli/                     # Command-line interface (future expansion)
│   │   └── __init__.py
│   └── subagents/              # Subagent management (future expansion)
│       └── __init__.py
├── agent.py                     # Main CLI entry point (505 lines)
├── mcp_deepseek_host.py        # Legacy DeepSeek implementation (deprecated)
├── mcp_gemini_host.py          # Legacy Gemini implementation (deprecated)
├── config.py                   # Configuration management with provider-model support
├── subagent.py                 # Subagent subprocess management
└── README.md                   # Project documentation
```

## 🏗️ Provider-Model Architecture

The system uses a sophisticated Provider-Model architecture that separates API integration concerns from model-specific behavior:

### Architecture Layers

1. **Provider Layer** (`BaseProvider`): Handles API integration
   - Authentication and request formatting
   - Streaming response processing
   - Error handling and retry logic
   - Rate limit management

2. **Model Layer** (`ModelConfig`): Handles LLM characteristics
   - Tool calling formats (OpenAI, Anthropic, Gemini)
   - System prompt styles (message, parameter, prepend)
   - Special features (reasoning content, thinking)
   - Token limits and parameters

3. **Host Layer** (`MCPHost`): Combines provider + model
   - Unified interface inheriting from `BaseLLMProvider`
   - Delegates API calls to provider
   - Delegates formatting to model
   - Maintains backward compatibility

### Provider-Model Combinations

The same model can be accessed through different providers:

```python
# Claude via different providers
config.create_host_from_provider_model("anthropic:claude-3.5-sonnet")
config.create_host_from_provider_model("openrouter:anthropic/claude-3.5-sonnet")

# GPT-4 via different providers  
config.create_host_from_provider_model("openai:gpt-4-turbo-preview")
config.create_host_from_provider_model("openrouter:openai/gpt-4-turbo-preview")
```

## 🧩 Core Components

### 1. BaseProvider (`cli_agent/core/base_provider.py`)

**Primary Role**: Abstract base class for API provider implementations

**Key Responsibilities**:
- API authentication and client management
- Request formatting and response parsing
- Streaming response processing
- Error handling and retry logic
- Rate limit information extraction

**Abstract Methods**:
```python
@abstractmethod
async def make_request(self, messages, model_name, tools=None, stream=False, **params) -> Any
@abstractmethod  
def extract_response_content(self, response) -> Tuple[str, List[Any], Dict[str, Any]]
@abstractmethod
async def process_streaming_response(self, response) -> Tuple[str, List[Any], Dict[str, Any]]
@abstractmethod
def is_retryable_error(self, error: Exception) -> bool
@abstractmethod
def get_error_message(self, error: Exception) -> str
@abstractmethod
def get_rate_limit_info(self, response: Any) -> Dict[str, Any]
```

**Concrete Providers**:
- `AnthropicProvider`: Direct Anthropic API integration
- `OpenAIProvider`: OpenAI API integration
- `OpenRouterProvider`: OpenRouter multi-model API
- `DeepSeekProvider`: DeepSeek API (OpenAI-compatible)
- `GoogleProvider`: Google Gemini API integration

### 2. ModelConfig (`cli_agent/core/model_config.py`)

**Primary Role**: Model-specific behavior and characteristics

**Key Responsibilities**:
- Tool calling format specification
- System prompt style management
- Message formatting for specific models
- Parameter validation and defaults
- Special content parsing (reasoning, thinking)

**Abstract Methods**:
```python
@abstractmethod
def get_tool_format(self) -> str  # "openai", "anthropic", "gemini"
@abstractmethod
def get_system_prompt_style(self) -> str  # "message", "parameter", "prepend"
@abstractmethod
def format_messages_for_model(self, messages) -> List[Dict[str, Any]]
@abstractmethod
def parse_special_content(self, text_content: str) -> Dict[str, Any]
```

**Concrete Models**:
- `ClaudeModel`: Anthropic tool format, parameter system prompts
- `GPTModel`: OpenAI tool format, message system prompts  
- `GeminiModel`: Gemini tool format, prepend system prompts
- `DeepSeekModel`: OpenAI tool format, reasoning content support

### 3. MCPHost (`cli_agent/core/mcp_host.py`)

**Primary Role**: Unified interface combining Provider + Model

**Key Features**:
- Inherits from `BaseLLMProvider` for compatibility
- Delegates API calls to provider instance
- Delegates formatting to model instance
- Handles tool conversion via format-specific converters
- Manages provider-specific features (reasoning, thinking)

```python
class MCPHost(BaseLLMProvider):
    def __init__(self, provider: BaseProvider, model: ModelConfig, config: HostConfig):
        self.provider = provider
        self.model = model
        super().__init__(config, is_subagent)
        
    def convert_tools_to_llm_format(self) -> List[Any]:
        # Uses model.get_tool_format() to select appropriate converter
        
    async def _make_api_request(self, messages, tools=None, stream=True) -> Any:
        # Delegates to provider.make_request()
```

### 4. BaseMCPAgent (`cli_agent/core/base_agent.py`)

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

### 5. InterruptibleInput (`cli_agent/core/input_handler.py`)

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

### 6. SlashCommandManager (`cli_agent/core/slash_commands.py`)

**Primary Role**: Slash command system similar to Claude Code

**Supported Commands**:
- **Built-in**: `/help`, `/clear`, `/compact`, `/tokens`, `/tools`, `/quit`
- **Provider-Model switching**: `/switch <provider>:<model>` (e.g., `/switch anthropic:claude-3.5-sonnet`)
- **Legacy model switching**: `/switch-chat`, `/switch-reason`, `/switch-gemini`, `/switch-gemini-pro`
- **Custom commands**: Loaded from `.claude/commands/` directories
- **MCP commands**: `mcp__<server>__<command>` format

**Extensibility**:
- Project-specific commands: `.claude/commands/*.md`
- Personal commands: `~/.claude/commands/*.md`
- Dynamic MCP command discovery

### 7. Built-in Tools (`cli_agent/tools/builtin_tools.py`)

**Primary Role**: Core tool definitions and schemas

**Available Tools**:
```python
def get_all_builtin_tools() -> Dict[str, Dict]:
    return {
        "builtin:bash_execute": {...},      # Execute bash commands with interrupt support
        "builtin:read_file": {...},         # Read file contents with offset/limit
        "builtin:write_file": {...},        # Write files with directory creation
        "builtin:list_directory": {...},    # Directory listing with file type indicators
        "builtin:get_current_directory": {...}, # Current directory
        "builtin:replace_in_file": {...},   # Exact string replacement with whitespace validation
        "builtin:multiedit": {...},         # Multiple sequential edits in one operation
        "builtin:glob": {...},              # File pattern matching with modification time sorting
        "builtin:grep": {...},              # Content search using regex patterns
        "builtin:todo_read": {...},         # Read session-specific todo lists
        "builtin:todo_write": {...},        # Write/update todo lists with structured format
        "builtin:webfetch": {...},          # Web content fetching and processing
        "builtin:task": {...},              # Spawn subagent for specific tasks
        "builtin:task_status": {...},       # Check running subagent task status
        "builtin:task_results": {...},      # Retrieve completed task results
        "builtin:emit_result": {...},       # Emit subagent results (subagents only)
    }
```

### 8. Tool Conversion System (`cli_agent/utils/tool_conversion.py`)

**Primary Role**: Convert tools between different LLM API formats

**Available Converters**:
- `OpenAIStyleToolConverter`: For OpenAI, DeepSeek, and compatible APIs
- `AnthropicToolConverter`: For Claude/Anthropic tool format
- `GeminiToolConverter`: For Google Gemini function calling
- `ToolConverterFactory`: Factory for creating appropriate converters

```python
# Usage via factory
converter = ToolConverterFactory.create_converter("anthropic")
tools = converter.convert_tools(available_tools)

# Direct usage in MCPHost
tool_format = self.model.get_tool_format()
if tool_format == "anthropic":
    converter = AnthropicToolConverter()
tools = converter.convert_tools(self.available_tools)
```

## 🔌 Provider-Model Implementation Patterns

### Provider Implementation Pattern

```python
class MyCustomProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my-provider"
        
    def get_default_base_url(self) -> str:
        return "https://api.my-provider.com"
        
    def _create_client(self, **kwargs) -> Any:
        # Create API client (AsyncOpenAI, httpx, etc.)
        
    async def make_request(self, messages, model_name, tools=None, stream=False, **params):
        # Format request and call API
        
    def extract_response_content(self, response) -> Tuple[str, List[Any], Dict[str, Any]]:
        # Parse response into text, tool_calls, metadata
        
    def is_retryable_error(self, error: Exception) -> bool:
        # Define retry logic for provider-specific errors
```

### Model Implementation Pattern

```python
@dataclass  
class MyCustomModel(ModelConfig):
    def __init__(self, variant: str = "my-model-default"):
        super().__init__(
            name=variant,
            provider_model_name=variant,
            context_length=128000,
            supports_tools=True,
            temperature=0.7
        )
    
    @property
    def model_family(self) -> str:
        return "my-model"
        
    def get_tool_format(self) -> str:
        return "openai"  # or "anthropic", "gemini"
        
    def get_system_prompt_style(self) -> str:
        return "message"  # or "parameter", "prepend"
        
    def format_messages_for_model(self, messages) -> List[Dict[str, Any]]:
        # Apply model-specific message formatting
```

### Configuration Integration

```python
# In config.py
class HostConfig(BaseSettings):
    # Add provider configuration
    my_provider_api_key: str = Field(default="", alias="MY_PROVIDER_API_KEY")
    my_provider_model: str = Field(default="my-model-v1", alias="MY_PROVIDER_MODEL")
    
    def create_host_from_provider_model(self, provider_model: str):
        # Add case for your provider
        if pm_config.provider_name == "my-provider":
            provider = MyCustomProvider(api_key=pm_config.provider_config.api_key)
            model = MyCustomModel(variant=pm_config.model_name)
            return MCPHost(provider=provider, model=model, config=self)
```

## 🔄 Execution Flow

### Provider-Model Host Creation

1. **Configuration-Based Creation**:
   ```python
   config = load_config()
   
   # Create hosts for different provider-model combinations
   deepseek_host = config.create_host_from_provider_model("deepseek:deepseek-chat")
   claude_anthropic = config.create_host_from_provider_model("anthropic:claude-3.5-sonnet")
   claude_openrouter = config.create_host_from_provider_model("openrouter:anthropic/claude-3.5-sonnet")
   gpt4_openai = config.create_host_from_provider_model("openai:gpt-4-turbo-preview")
   ```

2. **Direct Creation**:
   ```python
   from cli_agent.core.mcp_host import MCPHost
   from cli_agent.providers.anthropic_provider import AnthropicProvider
   from cli_agent.core.model_config import ClaudeModel
   
   provider = AnthropicProvider(api_key="your-key")
   model = ClaudeModel(variant="claude-3.5-sonnet")
   host = MCPHost(provider=provider, model=model, config=config)
   ```

### Interactive Chat Session

1. **Initialization**:
   ```python
   host = config.create_host_from_provider_model("anthropic:claude-3.5-sonnet")
   input_handler = InterruptibleInput()
   await host.interactive_chat(input_handler)
   ```

2. **User Input Processing**:
   - Input captured via `InterruptibleInput`
   - Slash commands handled by `SlashCommandManager`
   - Regular messages added to conversation history

3. **Response Generation**:
   - `host.generate_response()` called (centralized implementation)
   - `MCPHost` delegates to `provider.make_request()` for API call
   - `model.format_messages_for_model()` formats messages appropriately
   - Provider handles streaming/non-streaming responses

4. **Tool Execution** (if requested):
   - Tools converted via `model.get_tool_format()` and appropriate converter
   - Tool calls parsed via `provider.extract_response_content()`
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

### Adding New Provider-Model Combination

1. **Create Provider** (if new API):
   ```python
   class NewAPIProvider(BaseProvider):
       @property
       def name(self) -> str:
           return "new-api"
           
       async def make_request(self, messages, model_name, tools=None, stream=False, **params):
           # Implement API integration
           
       def extract_response_content(self, response):
           # Parse API response format
   ```

2. **Create Model Config** (if new model family):
   ```python
   @dataclass
   class NewLLMModel(ModelConfig):
       @property
       def model_family(self) -> str:
           return "new-llm"
           
       def get_tool_format(self) -> str:
           return "openai"  # or create custom converter
           
       def format_messages_for_model(self, messages):
           # Handle model-specific message formatting
   ```

3. **Update Configuration**:
   ```python
   # Add to HostConfig in config.py
   new_api_key: str = Field(default="", alias="NEW_API_KEY")
   
   def create_host_from_provider_model(self, provider_model: str):
       # Add case for new provider
       elif pm_config.provider_name == "new-api":
           provider = NewAPIProvider(api_key=pm_config.provider_config.api_key)
           model = NewLLMModel(variant=pm_config.model_name)
           return MCPHost(provider=provider, model=model, config=self)
   ```

4. **Usage**:
   ```python
   # Use new provider-model combination
   host = config.create_host_from_provider_model("new-api:new-llm-model")
   ```

### Adding Tool Format Converter

1. **Create Custom Converter**:
   ```python
   class MyCustomToolConverter(BaseToolConverter):
       def convert_tools(self, available_tools: Dict[str, Dict]) -> List[Dict]:
           tools = []
           for tool_key, tool_info in available_tools.items():
               if not self.validate_tool_info(tool_key, tool_info):
                   continue
               # Convert to your custom format
               tool = {
                   "function_name": self.normalize_tool_name(tool_key),
                   "description": self.generate_description(tool_info),
                   "parameters": self.get_base_schema(tool_info)
               }
               tools.append(tool)
           return tools
   ```

2. **Register in Factory**:
   ```python
   # Update ToolConverterFactory.create_converter()
   converters = {
       "openai": OpenAIStyleToolConverter,
       "anthropic": AnthropicToolConverter, 
       "gemini": GeminiToolConverter,
       "my-custom": MyCustomToolConverter,
   }
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

### Provider-Model Testing
```python
# Test provider-model combinations
from config import load_config

config = load_config()

# Test configuration parsing
provider, model = config.parse_provider_model_string("anthropic:claude-3.5-sonnet")
assert provider == "anthropic"
assert model == "claude-3.5-sonnet"

# Test host creation
host = config.create_host_from_provider_model("deepseek:deepseek-chat")
assert host.provider.name == "deepseek"
assert host.model.name == "deepseek-chat"
```

### Tool Conversion Testing
```python
# Test tool converters
from cli_agent.utils.tool_conversion import ToolConverterFactory

# Test different formats
for llm_type in ["openai", "anthropic", "gemini"]:
    converter = ToolConverterFactory.create_converter(llm_type)
    tools = converter.convert_tools(sample_tools)
    assert len(tools) > 0
```

### Component Testing
```python
# Test individual components
from cli_agent.core.base_agent import BaseMCPAgent
from cli_agent.core.input_handler import InterruptibleInput
from cli_agent.tools.builtin_tools import get_all_builtin_tools
from cli_agent.core.model_config import ClaudeModel, GPTModel
from cli_agent.providers.deepseek_provider import DeepSeekProvider

# Test tool definitions
tools = get_all_builtin_tools()
assert "builtin:bash_execute" in tools

# Test model configs
model = ClaudeModel(variant="claude-3.5-sonnet")
assert model.get_tool_format() == "anthropic"
assert model.get_system_prompt_style() == "parameter"

# Test providers
provider = DeepSeekProvider(api_key="test")
assert provider.name == "deepseek"
assert provider.supports_streaming() == True
```

## 🎯 Key Design Patterns

- **Abstract Base Class**: `BaseMCPAgent` and `BaseProvider` enforce interface contracts
- **Strategy Pattern**: Interchangeable provider and model implementations
- **Composition Pattern**: `MCPHost` combines provider + model via composition
- **Factory Pattern**: `ToolConverterFactory` creates appropriate converters
- **Command Pattern**: Slash commands and CLI structure
- **Event-Driven**: Subagent communication via async queues
- **Dependency Injection**: Configuration and tool injection
- **Template Method**: `interactive_chat()` defines flow, subclasses customize steps
- **Separation of Concerns**: Providers handle APIs, models handle behavior

## 📊 Refactoring Benefits

**Before**: 3,237-line monolithic `agent.py`
**After**: Modular provider-model architecture with focused components

**Maintainability**: ✅ Each module has single responsibility
**Testability**: ✅ Components can be unit tested independently
**Reusability**: ✅ Providers and models can be mixed and matched
**Scalability**: ✅ Easy to add new providers or models without touching existing code
**Flexibility**: ✅ Same model accessible through multiple providers
**Developer Experience**: ✅ Much easier to navigate, understand, and extend
**Multi-Provider Support**: ✅ Reduces vendor lock-in and increases reliability

## 🚀 Usage Examples

### Configuration-Based Usage

```env
# .env file - multiple provider configurations
DEEPSEEK_API_KEY=your_deepseek_key
ANTHROPIC_API_KEY=your_anthropic_key  
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key

# Default provider-model selection
DEFAULT_PROVIDER_MODEL=anthropic:claude-3.5-sonnet
```

```python
# Python usage
from config import load_config

config = load_config()

# Create hosts for different combinations
default_host = config.create_host_from_provider_model()  # Uses DEFAULT_PROVIDER_MODEL
claude_anthropic = config.create_host_from_provider_model("anthropic:claude-3.5-sonnet")
claude_openrouter = config.create_host_from_provider_model("openrouter:anthropic/claude-3.5-sonnet")
gpt4_openai = config.create_host_from_provider_model("openai:gpt-4-turbo-preview")
deepseek_reasoning = config.create_host_from_provider_model("deepseek:deepseek-reasoner")

# Use any host with the same interface
await claude_anthropic.interactive_chat(input_handler)
```

### Available Provider-Model Combinations

```python
# Get available combinations
available = config.get_available_provider_models()
print(available)
# {
#   "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
#   "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo-preview"],
#   "openai": ["gpt-4-turbo-preview", "gpt-3.5-turbo", "o1-preview"],
#   "deepseek": ["deepseek-chat", "deepseek-reasoner"],
#   "gemini": ["gemini-2.5-flash", "gemini-1.5-pro"]
# }
```

### CLI Integration

```bash
# Switch between provider-model combinations
/switch anthropic:claude-3.5-sonnet
/switch openrouter:anthropic/claude-3.5-sonnet
/switch deepseek:deepseek-reasoner
/switch openai:gpt-4-turbo-preview
```

## 🚀 Future Expansion Areas

- **`cli_agent/cli/`**: Extract CLI commands to separate modules
- **`cli_agent/subagents/`**: Enhanced subagent management
- **Additional Providers**: Azure OpenAI, Cohere, Mistral, local model servers
- **Additional Models**: Support for new model families and variants
- **Plugin System**: Dynamic tool loading and discovery
- **Multi-Agent**: Agent-to-agent communication protocols
- **Provider Fallbacks**: Automatic failover between providers
- **Cost Tracking**: Per-provider usage and cost monitoring

This provider-model architecture provides a solid, flexible foundation for continued development and extension of the MCP Agent system, enabling users to leverage the best combination of providers and models for their specific needs.