# Dynamic Model Listing Documentation

## Overview

The MCP Agent now supports dynamic model listing for providers that expose model discovery APIs. This allows users to see what models are available without hardcoding model lists.

## Supported Providers

### OpenRouter

The OpenRouter provider now includes comprehensive model listing functionality:

```python
from cli_agent.providers.openrouter_provider import OpenRouterProvider

# Initialize provider
provider = OpenRouterProvider(api_key="your-api-key")

# Get detailed model information
models = await provider.get_available_models()
# Returns list of dicts with: id, name, context_length, description, pricing, top_provider

# Get just model IDs
model_ids = await provider.get_available_models_summary()
# Returns: ['anthropic/claude-3.5-sonnet', 'openai/gpt-4-turbo', ...]

# Static method (doesn't require persistent client)
models = await OpenRouterProvider.fetch_available_models_static(api_key="your-api-key")
```

Features:
- Automatic caching (1 hour) to avoid API rate limits
- Retry logic for transient failures
- Detailed model metadata including context length and pricing
- Error handling with graceful fallbacks

### OpenAI

The OpenAI provider already supports model listing:

```python
from cli_agent.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider(api_key="your-api-key")
models = await provider.get_available_models()
# Returns: ['gpt-4-turbo-preview', 'gpt-3.5-turbo', 'o1-preview', ...]
```

### DeepSeek

The DeepSeek provider now includes model listing with fallback:

```python
from cli_agent.providers.deepseek_provider import DeepSeekProvider

provider = DeepSeekProvider(api_key="your-api-key")
models = await provider.get_available_models()
# Attempts to use /v1/models endpoint, falls back to known models if not supported
```

## Example Usage

```python
import asyncio
from config import load_config

async def list_all_available_models():
    config = load_config()
    
    # List models from different providers
    providers_to_check = [
        ("openrouter", config.openrouter_api_key),
        ("openai", config.openai_api_key),
        ("deepseek", config.deepseek_api_key),
    ]
    
    for provider_name, api_key in providers_to_check:
        if not api_key:
            print(f"{provider_name}: No API key configured")
            continue
            
        try:
            if provider_name == "openrouter":
                from cli_agent.providers.openrouter_provider import OpenRouterProvider
                provider = OpenRouterProvider(api_key=api_key)
                models = await provider.get_available_models()
                print(f"\n{provider_name}: Found {len(models)} models")
                for model in models[:5]:  # Show first 5
                    print(f"  - {model['id']} (context: {model['context_length']})")
            elif provider_name == "openai":
                from cli_agent.providers.openai_provider import OpenAIProvider
                provider = OpenAIProvider(api_key=api_key)
                models = await provider.get_available_models()
                print(f"\n{provider_name}: {models}")
            elif provider_name == "deepseek":
                from cli_agent.providers.deepseek_provider import DeepSeekProvider
                provider = DeepSeekProvider(api_key=api_key)
                models = await provider.get_available_models()
                print(f"\n{provider_name}: {models}")
        except Exception as e:
            print(f"{provider_name}: Error - {e}")

# Run the example
asyncio.run(list_all_available_models())
```

## Error Handling

All model listing methods handle errors gracefully:

1. **Network errors**: Logged with appropriate error messages
2. **Authentication failures**: Return empty list with error logged
3. **Rate limits**: Handled by retry logic (OpenRouter)
4. **Missing endpoints**: Fallback to known models (DeepSeek)

## Performance Considerations

- **Caching**: OpenRouter caches results for 1 hour to reduce API calls
- **Timeouts**: All requests have reasonable timeouts (10 seconds)
- **Retry logic**: Transient failures are automatically retried
- **Async**: All methods are async for non-blocking operation

## Future Enhancements

Potential improvements:
1. Add model listing for Google Gemini provider
2. Implement unified model discovery interface
3. Add model capability detection (tools, streaming, etc.)
4. Cache model lists persistently across sessions
5. Add model filtering by capabilities or context length