# LLM Provider Implementation Specification

This document provides a comprehensive guide for adding new LLM providers to the MCP Agent system. After implementing the `BaseLLMProvider` abstraction, adding a new LLM provider requires implementing only 6 core methods plus some standard configuration methods.

## Overview

The MCP Agent uses a layered architecture where `BaseLLMProvider` handles all common functionality (streaming, tool execution, error handling, normalization) while LLM-specific providers only implement their unique API integration and response parsing logic.

## Required Implementation

### 1. Class Structure

```python
#!/usr/bin/env python3
"""MCP Host implementation using [YourLLM] as the language model backend."""

import logging
from typing import Any, Dict, List, Optional, Union

# Import your LLM's SDK
from your_llm_sdk import YourLLMClient

from cli_agent.core.base_llm_provider import BaseLLMProvider
from cli_agent.utils.tool_conversion import YourLLMToolConverter  # Create this
from cli_agent.utils.tool_parsing import YourLLMToolCallParser   # Create this
from config import HostConfig

logger = logging.getLogger(__name__)

class MCPYourLLMHost(BaseLLMProvider):
    """MCP Host that uses [YourLLM] as the language model backend."""
    
    def __init__(self, config: HostConfig, is_subagent: bool = False):
        # Store any provider-specific state (must be before super().__init__)
        self.your_llm_specific_state = None
        
        # Call parent initialization (which will call our abstract methods)
        super().__init__(config, is_subagent)
```

### 2. Required Configuration Methods

These methods integrate with the existing configuration system:

```python
def _get_provider_config(self):
    """Get provider-specific configuration from HostConfig."""
    return self.config.get_your_llm_config()

def _get_streaming_preference(self, provider_config) -> bool:
    """Get streaming preference for your LLM."""
    return provider_config.stream  # or True/False if always the same

def _calculate_timeout(self, provider_config) -> float:
    """Calculate timeout based on your LLM model characteristics."""
    return 120.0  # seconds, adjust based on your provider's typical response time

def _create_llm_client(self, provider_config, timeout_seconds):
    """Create and configure your LLM client."""
    self.your_llm_config = provider_config  # Store for later use
    
    client = YourLLMClient(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,  # if applicable
        timeout=timeout_seconds,
        # ... other provider-specific settings
    )
    
    # Store as both _client (from base class) and your_llm_client (for compatibility)
    self.your_llm_client = client
    return client

def convert_tools_to_llm_format(self) -> List[Any]:
    """Convert tools to your LLM's expected format."""
    converter = YourLLMToolConverter()
    return converter.convert_tools(self.available_tools)

def _get_current_runtime_model(self) -> str:
    """Get the actual model being used at runtime."""
    return self.your_llm_config.model
```

### 3. Required Abstract Methods

These 6 methods must be implemented for your LLM provider:

#### A. Tool Call Extraction Methods

```python
def _extract_structured_calls_impl(self, response: Any) -> List[Any]:
    """Extract structured tool calls from your LLM's response object.
    
    Args:
        response: Your LLM's raw response object
        
    Returns:
        List of tool call objects in SimpleNamespace format with:
        - .name: tool name
        - .args: tool arguments dict  
        - .id: tool call ID (optional)
    """
    from types import SimpleNamespace
    
    structured_calls = []
    
    # Example for OpenAI-style responses:
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                call = SimpleNamespace()
                call.name = tc.function.name
                call.args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                call.id = getattr(tc, "id", None)
                structured_calls.append(call)
    
    # Example for Gemini-style responses:
    elif hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and candidate.content:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    call = SimpleNamespace()
                    call.name = part.function_call.name
                    call.args = dict(part.function_call.args) if part.function_call.args else {}
                    call.id = getattr(part.function_call, "id", None)
                    structured_calls.append(call)
    
    return structured_calls

def _parse_text_based_calls_impl(self, text_content: str) -> List[Any]:
    """Parse text-based tool calls from response content.
    
    Args:
        text_content: Raw text content from the response
        
    Returns:
        List of tool call objects in SimpleNamespace format
    """
    from types import SimpleNamespace
    
    text_calls = []
    
    if text_content:
        # Use your provider's text parser
        tool_calls = YourLLMToolCallParser.parse_tool_calls(text_content)
        
        for tc in tool_calls:
            call = SimpleNamespace()
            call.name = tc.function.name
            call.args = tc.function.arguments
            call.id = getattr(tc, "id", None)
            text_calls.append(call)
    
    return text_calls

def _get_text_extraction_patterns(self) -> List[str]:
    """Get regex patterns for extracting text that appears before tool calls.
    
    Returns:
        List of regex patterns specific to your provider's tool call format
    """
    return [
        # Add patterns specific to your LLM's tool call format
        r"^(.*?)(?=<tool_call>)",           # XML-style
        r"^(.*?)(?=```json\s*\{)",          # JSON code blocks
        r"^(.*?)(?=\w+\s*\()",              # Function call style
        # Add more patterns as needed for your provider
    ]
```

#### B. Response Processing Methods

```python
def _extract_response_content(self, response: Any) -> tuple[str, List[Any], Dict[str, Any]]:
    """Extract text content, tool calls, and provider-specific data from response.
    
    Args:
        response: Your LLM's raw response object
        
    Returns:
        tuple: (text_content, tool_calls, provider_specific_data)
    """
    # Example for OpenAI-style:
    if not hasattr(response, "choices") or not response.choices:
        return "", [], {}
    
    message = response.choices[0].message
    text_content = message.content or ""
    tool_calls = message.tool_calls if hasattr(message, "tool_calls") else []
    
    # Handle provider-specific features (e.g., DeepSeek reasoning, Claude thinking)
    provider_data = {}
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        provider_data["reasoning_content"] = message.reasoning_content
    
    return text_content, tool_calls, provider_data

async def _process_streaming_chunks(self, response) -> tuple[str, List[Any], Dict[str, Any]]:
    """Process streaming response chunks and accumulate content.
    
    Args:
        response: Your LLM's streaming response object
        
    Returns:
        tuple: (accumulated_content, tool_calls, provider_specific_data)
    """
    accumulated_content = ""
    accumulated_tool_calls = []
    provider_data = {}
    
    # Example for OpenAI-style streaming:
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            
            # Accumulate text content
            if delta.content:
                accumulated_content += delta.content
            
            # Handle streaming tool calls
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                # Implement streaming tool call accumulation logic
                # (This varies significantly by provider)
                pass
            
            # Handle provider-specific streaming data
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if "reasoning_content" not in provider_data:
                    provider_data["reasoning_content"] = ""
                provider_data["reasoning_content"] += delta.reasoning_content
    
    return accumulated_content, accumulated_tool_calls, provider_data
```

#### C. API Integration Methods

```python
async def _make_api_request(self, messages: List[Dict[str, Any]], tools: Optional[List] = None, stream: bool = True) -> Any:
    """Make an API request to your LLM provider.
    
    Args:
        messages: Processed messages ready for your provider
        tools: Tools formatted for your provider (if any)
        stream: Whether to use streaming
        
    Returns:
        Your provider's response object
    """
    # Use centralized retry logic from base class
    return await self._make_api_request_with_retry(
        lambda: self.your_llm_client.chat.completions.create(
            model=self.your_llm_config.model,
            messages=messages,  # Already processed by centralized pipeline
            tools=tools,
            temperature=self.your_llm_config.temperature,
            max_tokens=self.your_llm_config.max_tokens,
            stream=stream,
            # Add other provider-specific parameters
        )
    )

def _create_mock_response(self, content: str, tool_calls: List[Any]) -> Any:
    """Create a mock response object for centralized processing.
    
    Args:
        content: Text content
        tool_calls: List of tool calls
        
    Returns:
        Mock response object compatible with your provider's format
    """
    # Create a mock object that matches your provider's response structure
    mock_response = type("MockResponse", (), {})()
    mock_response.choices = [type("MockChoice", (), {})()]
    mock_response.choices[0].message = type("MockMessage", (), {})()
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = tool_calls
    return mock_response
```

#### D. Error Handling Method

```python
def _is_provider_retryable_error(self, error_str: str) -> bool:
    """Check if error is retryable according to your provider's error patterns.
    
    Args:
        error_str: Lowercase error message string
        
    Returns:
        True if error should be retried for this provider
    """
    # Add provider-specific retryable error patterns
    return (
        "your_provider_name" in error_str
        or "model overloaded" in error_str
        or "service unavailable" in error_str
        or "quota exceeded" in error_str
        # Add more patterns specific to your provider
    )
```

### 4. Optional Override Methods

These methods have default implementations but can be overridden for provider-specific behavior:

```python
def _get_llm_specific_instructions(self) -> str:
    """Provide provider-specific instructions for optimal performance."""
    if self.is_subagent:
        return """**Instructions for [YourLLM] Subagents:**
1. You are a subagent - focus on your specific task
2. Use tools immediately when needed
3. MANDATORY: End with `emit_result` containing your findings
"""
    else:
        return """**Instructions for [YourLLM]:**
1. Use tools immediately for actions (don't just describe)
2. Delegate complex tasks to subagents when appropriate
"""

def _handle_provider_specific_features(self, provider_data: Dict[str, Any]) -> None:
    """Handle provider-specific features like reasoning content."""
    if "reasoning_content" in provider_data:
        reasoning = provider_data["reasoning_content"]
        print(f"\n<reasoning>{reasoning}</reasoning>", flush=True)

def _format_provider_specific_content(self, provider_data: Dict[str, Any]) -> str:
    """Format provider-specific content for output."""
    if "reasoning_content" in provider_data:
        return f"<reasoning>{provider_data['reasoning_content']}</reasoning>\n\n"
    return ""

def _enhance_messages_for_model(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Provider-specific message enhancement."""
    # Example: Add system prompt for first message
    enhanced_messages = messages
    is_first_message = len(messages) == 1 and messages[0].get("role") == "user"
    
    if self.is_subagent or is_first_message:
        system_prompt = self._create_system_prompt(for_first_message=True)
        enhanced_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
    
    return enhanced_messages

def _clean_message_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Provider-specific message cleaning."""
    # Implement any provider-specific message format requirements
    return messages  # Default: no cleaning needed
```

## Supporting Files to Create

### 1. Tool Converter (`cli_agent/utils/tool_conversion.py`)

Add a tool converter class for your provider:

```python
class YourLLMToolConverter:
    """Convert tools to YourLLM format."""
    
    def convert_tools(self, available_tools: Dict[str, Dict]) -> List[Any]:
        """Convert tools to your provider's expected format."""
        # Implement conversion logic
        pass
```

### 2. Tool Parser (`cli_agent/utils/tool_parsing.py`)

Add a tool parser class for your provider:

```python
class YourLLMToolCallParser:
    """Parse tool calls from YourLLM text responses."""
    
    @staticmethod
    def parse_tool_calls(text_content: str) -> List[Any]:
        """Parse tool calls from text using provider-specific patterns."""
        # Implement parsing logic
        pass
```

### 3. Configuration (`config.py`)

Add configuration support for your provider:

```python
@dataclass
class YourLLMConfig:
    """Configuration for YourLLM provider."""
    api_key: str = ""
    model: str = "your-default-model"
    base_url: str = "https://api.yourprovider.com"
    temperature: float = 0.7
    max_tokens: int = 4000
    stream: bool = True
    # Add provider-specific settings

class HostConfig:
    # Add method to existing class
    def get_your_llm_config(self) -> YourLLMConfig:
        """Get YourLLM configuration."""
        return YourLLMConfig(
            api_key=os.getenv("YOUR_LLM_API_KEY", ""),
            model=os.getenv("YOUR_LLM_MODEL", "your-default-model"),
            # ... other configuration
        )
```

## Integration with Main Agent

Add your provider to the main agent (`agent.py`):

```python
# In the model detection logic
elif config.model_type == "your-llm":
    from mcp_your_llm_host import MCPYourLLMHost
    host = MCPYourLLMHost(config, is_subagent)
```

## Example Provider Patterns

### OpenAI-Compatible APIs
- Use `OpenAIStyleToolConverter`
- Response structure: `response.choices[0].message`
- Tool calls: `message.tool_calls[].function`

### Google Gemini Style
- Custom tool converter for function declarations
- Response structure: `response.candidates[0].content.parts`
- Tool calls: `part.function_call`

### Claude Style (Anthropic)
- Text-based tool parsing with XML tags
- Response structure: simple text with embedded tool calls
- Custom patterns for `<thinking>` and tool blocks

## Testing Your Implementation

1. **Import Test**: Verify your module imports correctly
   ```bash
   python -c "from mcp_your_llm_host import MCPYourLLMHost; print('Import successful')"
   ```

2. **Basic Tool Test**: Test basic tool execution
   ```bash
   echo "run echo test" | python agent.py chat --model your-llm --auto-approve-tools
   ```

3. **Subagent Test**: Test subagent functionality
   ```bash
   echo "Spawn a subagent to run ls" | python agent.py chat --model your-llm
   ```

## Common Patterns and Best Practices

1. **Error Handling**: Always use the centralized retry logic from `_make_api_request_with_retry`
2. **Tool Calls**: Convert to SimpleNamespace format for consistency across the framework
3. **Streaming**: Handle both streaming and non-streaming modes
4. **Provider Features**: Use `provider_data` dict for provider-specific features like reasoning
5. **Message Enhancement**: Add system prompts and provider-specific formatting in `_enhance_messages_for_model`
6. **Timeouts**: Set appropriate timeouts based on your provider's characteristics

## Benefits of This Architecture

- **Minimal Implementation**: Only 6 core methods required
- **Automatic Features**: Inherit streaming, tool execution, error handling, subagents
- **Consistent Behavior**: All providers work identically from user perspective
- **Easy Maintenance**: Bug fixes and new features apply to all providers
- **Type Safety**: Clear interfaces and expected return types

This specification enables adding new LLM providers with minimal code while maintaining full compatibility with the MCP Agent's advanced features like subagents, tool permissions, and streaming responses.