# Provider-Model Architecture Design

## Overview

This document outlines the new architecture that separates **Provider** (API endpoint/service) logic from **Model** (LLM characteristics) logic, enabling the same model to be used through multiple providers.

## Current Problem

The existing architecture conflates provider and model:
```python
MCPDeepseekHost = DeepSeek API + DeepSeek Model Logic
MCPGeminiHost = Gemini API + Gemini Model Logic
```

This makes it impossible to use:
- Claude through Anthropic vs OpenRouter
- GPT-4 through OpenAI vs Azure 
- Any model through multiple API providers

## New Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BaseProvider  │    │   ModelConfig   │    │    MCPHost      │
│                 │    │                 │    │                 │
│ • API Integration│    │ • Tool Formats  │    │ • Orchestrates  │
│ • Authentication│    │ • Prompt Styles │    │ • Provider +    │
│ • Request Format│    │ • Special Feats │    │   Model         │
│ • Error Handling│    │ • Token Limits  │    │ • Inherits from │
│ • Rate Limiting │    │ • Defaults      │    │   BaseLLMProvider│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1. BaseProvider (API Integration Layer)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseProvider(ABC):
    """Base class for API providers (Anthropic, OpenAI, OpenRouter, etc.)"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url or self.get_default_base_url()
        self.client = self._create_client(**kwargs)
    
    @abstractmethod
    def get_default_base_url(self) -> str:
        """Get default API base URL for this provider."""
        pass
    
    @abstractmethod 
    def _create_client(self, **kwargs) -> Any:
        """Create provider-specific HTTP client."""
        pass
    
    @abstractmethod
    async def make_request(self, 
                          messages: List[Dict], 
                          model_name: str,
                          tools: Optional[List] = None,
                          stream: bool = False,
                          **model_params) -> Any:
        """Make API request using provider's format."""
        pass
    
    @abstractmethod
    def extract_response_content(self, response: Any) -> tuple[str, List[Any], Dict[str, Any]]:
        """Extract (text, tool_calls, metadata) from provider's response format."""
        pass
    
    @abstractmethod
    async def process_streaming_response(self, response: Any) -> tuple[str, List[Any], Dict[str, Any]]:
        """Process streaming response from provider."""
        pass
    
    @abstractmethod
    def is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable for this provider."""
        pass
    
    @abstractmethod
    def get_error_message(self, error: Exception) -> str:
        """Extract meaningful error message from provider error."""
        pass
```

### 2. ModelConfig (Model Characteristics Layer)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ModelConfig(ABC):
    """Base configuration for model-specific behavior."""
    
    name: str                    # e.g., "claude-3.5-sonnet"
    provider_model_name: str     # e.g., "claude-3-5-sonnet-20241022" 
    context_length: int          # Token limit
    supports_tools: bool = True
    supports_streaming: bool = True
    temperature: float = 0.7
    max_tokens: int = 4000
    
    @abstractmethod
    def get_tool_format(self) -> str:
        """Get tool calling format: 'openai', 'anthropic', 'gemini', etc."""
        pass
    
    @abstractmethod
    def get_system_prompt_style(self) -> str:
        """Get system prompt style: 'message', 'parameter', 'prepend'"""
        pass
    
    @abstractmethod
    def format_messages_for_model(self, messages: List[Dict]) -> List[Dict]:
        """Apply model-specific message formatting."""
        pass
    
    @abstractmethod
    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        """Get instructions optimized for this model."""
        pass
    
    @abstractmethod
    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        """Parse model-specific content like <thinking> or <reasoning>."""
        pass
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for API requests."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
```

### 3. Concrete Provider Implementations

```python
class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""
    
    def get_default_base_url(self) -> str:
        return "https://api.anthropic.com"
    
    def _create_client(self, **kwargs):
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
    
    async def make_request(self, messages, model_name, tools=None, stream=False, **params):
        # Convert to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else None
        
        return await self.client.messages.create(
            model=model_name,
            messages=anthropic_messages,
            tools=anthropic_tools,
            stream=stream,
            **params
        )
    
    def extract_response_content(self, response):
        # Parse Anthropic response format
        text = response.content[0].text if response.content else ""
        tool_calls = []  # Extract from response.content if tool use blocks
        metadata = {}
        return text, tool_calls, metadata

class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider (OpenAI-compatible)."""
    
    def get_default_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"
    
    def _create_client(self, **kwargs):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    async def make_request(self, messages, model_name, tools=None, stream=False, **params):
        # OpenAI-compatible format
        return await self.client.chat.completions.create(
            model=model_name,
            messages=messages,  # Already in OpenAI format
            tools=tools,
            stream=stream,
            **params
        )
    
    def extract_response_content(self, response):
        # Parse OpenAI response format
        message = response.choices[0].message
        text = message.content or ""
        tool_calls = message.tool_calls or []
        metadata = {}
        return text, tool_calls, metadata
```

### 4. Concrete Model Configurations

```python
class ClaudeModel(ModelConfig):
    """Claude model configuration."""
    
    def __init__(self, variant: str = "claude-3.5-sonnet"):
        model_map = {
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3.5-haiku": "claude-3-5-haiku-20241022",
            "claude-3-opus": "claude-3-opus-20240229",
        }
        
        super().__init__(
            name=variant,
            provider_model_name=model_map.get(variant, variant),
            context_length=200000,
            supports_tools=True,
            supports_streaming=True,
        )
    
    def get_tool_format(self) -> str:
        return "anthropic"
    
    def get_system_prompt_style(self) -> str:
        return "parameter"  # Claude uses system parameter, not system message
    
    def format_messages_for_model(self, messages: List[Dict]) -> List[Dict]:
        # Remove system messages, handle alternating user/assistant pattern
        formatted = []
        for msg in messages:
            if msg["role"] != "system":  # System handled separately
                formatted.append(msg)
        return formatted
    
    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        if is_subagent:
            return """You are a focused subagent. Use tools immediately when needed.
End with <result>your findings</result>."""
        return """You are Claude. Use tools when actions are needed.
Think step by step and be helpful."""
    
    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        # Parse <thinking> blocks, <result> blocks, etc.
        import re
        special_content = {}
        
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
        if thinking_match:
            special_content["thinking"] = thinking_match.group(1).strip()
        
        return special_content

class GPTModel(ModelConfig):
    """GPT model configuration."""
    
    def __init__(self, variant: str = "gpt-4"):
        model_map = {
            "gpt-4": "gpt-4-0613",
            "gpt-4-turbo": "gpt-4-turbo-2024-04-09", 
            "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        }
        
        super().__init__(
            name=variant,
            provider_model_name=model_map.get(variant, variant),
            context_length=128000 if "gpt-4" in variant else 16000,
            supports_tools=True,
            supports_streaming=True,
        )
    
    def get_tool_format(self) -> str:
        return "openai"
    
    def get_system_prompt_style(self) -> str:
        return "message"  # GPT uses system messages
    
    def format_messages_for_model(self, messages: List[Dict]) -> List[Dict]:
        # GPT messages are already in correct format
        return messages
    
    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        if is_subagent:
            return """You are a subagent focused on a specific task. Use tools when needed.
Provide clear, actionable results."""
        return """You are GPT-4. Use available tools to help users effectively.
Be precise and helpful."""
    
    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        # GPT doesn't have special content blocks
        return {}
```

### 5. New MCPHost Implementation

```python
class MCPHost(BaseLLMProvider):
    """Unified MCP Host that combines a Provider with a Model."""
    
    def __init__(self, 
                 provider: BaseProvider, 
                 model: ModelConfig, 
                 config: HostConfig, 
                 is_subagent: bool = False):
        
        self.provider = provider
        self.model = model
        
        # Call parent initialization
        super().__init__(config, is_subagent)
    
    # Implement required methods by delegating to provider/model
    def convert_tools_to_llm_format(self) -> List[Any]:
        tool_format = self.model.get_tool_format()
        if tool_format == "openai":
            converter = OpenAIStyleToolConverter()
        elif tool_format == "anthropic":
            converter = AnthropicToolConverter()
        elif tool_format == "gemini":
            converter = GeminiToolConverter()
        else:
            raise ValueError(f"Unknown tool format: {tool_format}")
        
        return converter.convert_tools(self.available_tools)
    
    def _extract_structured_calls_impl(self, response: Any) -> List[Any]:
        # Delegate to provider to extract tool calls
        _, tool_calls, _ = self.provider.extract_response_content(response)
        
        # Convert to standard format
        from types import SimpleNamespace
        calls = []
        for tc in tool_calls:
            call = SimpleNamespace()
            if hasattr(tc, 'function'):  # OpenAI format
                call.name = tc.function.name
                call.args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                call.id = tc.id
            else:  # Other formats
                call.name = tc.name
                call.args = tc.args
                call.id = getattr(tc, 'id', None)
            calls.append(call)
        
        return calls
    
    async def _make_api_request(self, messages, tools=None, stream=True) -> Any:
        # Format messages for the specific model
        formatted_messages = self.model.format_messages_for_model(messages)
        
        # Get model parameters
        model_params = self.model.get_default_parameters()
        
        # Make request through provider
        return await self.provider.make_request(
            messages=formatted_messages,
            model_name=self.model.provider_model_name,
            tools=tools,
            stream=stream,
            **model_params
        )
    
    def _extract_response_content(self, response: Any) -> tuple[str, List[Any], Dict[str, Any]]:
        # Get content from provider
        text, tool_calls, metadata = self.provider.extract_response_content(response)
        
        # Parse model-specific content
        special_content = self.model.parse_special_content(text)
        metadata.update(special_content)
        
        return text, tool_calls, metadata
    
    def _get_current_runtime_model(self) -> str:
        return self.model.name
    
    # ... implement other required methods
```

## Usage Examples

### Same Model, Different Providers

```python
# Claude through Anthropic
anthropic_claude = MCPHost(
    provider=AnthropicProvider(api_key="ant-key"),
    model=ClaudeModel("claude-3.5-sonnet"),
    config=config
)

# Claude through OpenRouter  
openrouter_claude = MCPHost(
    provider=OpenRouterProvider(api_key="or-key"),
    model=ClaudeModel("claude-3.5-sonnet"),  # Same model config!
    config=config
)

# GPT-4 through OpenAI
openai_gpt = MCPHost(
    provider=OpenAIProvider(api_key="sk-key"),
    model=GPTModel("gpt-4"),
    config=config
)

# GPT-4 through Azure
azure_gpt = MCPHost(
    provider=AzureProvider(api_key="azure-key", endpoint="https://myazure.openai.azure.com"),
    model=GPTModel("gpt-4"),  # Same model config!
    config=config
)
```

### Configuration-Driven Selection

```python
def create_host(provider_name: str, model_name: str, config: HostConfig) -> MCPHost:
    # Create provider
    if provider_name == "anthropic":
        provider = AnthropicProvider(api_key=config.anthropic_api_key)
    elif provider_name == "openrouter":
        provider = OpenRouterProvider(api_key=config.openrouter_api_key)
    elif provider_name == "openai":
        provider = OpenAIProvider(api_key=config.openai_api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    # Create model
    if "claude" in model_name:
        model = ClaudeModel(model_name)
    elif "gpt" in model_name:
        model = GPTModel(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MCPHost(provider=provider, model=model, config=config)

# Usage
host = create_host("openrouter", "claude-3.5-sonnet", config)
```

## Benefits

1. **Flexibility**: Same model through multiple providers
2. **Maintainability**: Provider logic separate from model logic  
3. **Scalability**: Easy to add new providers or models independently
4. **Reusability**: Model configs work across all compatible providers
5. **Testing**: Can test models with mock providers
6. **Cost Optimization**: Switch providers based on pricing/availability

## Migration Path

1. Create new architecture alongside existing hosts
2. Add provider/model implementations gradually  
3. Update configuration system to support provider+model selection
4. Migrate existing hosts to use new architecture
5. Remove old monolithic hosts

This architecture provides a clean separation of concerns while maintaining all existing functionality.