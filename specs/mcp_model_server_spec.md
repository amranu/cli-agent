# MCP Model Server Specification

## Overview
Design and implement an MCP server that exposes all available AI models (Anthropic, OpenAI, DeepSeek, Gemini, OpenRouter) as MCP tools. This would allow other MCP clients to use our models through the standardized MCP protocol.

## Background
The current agent supports multiple AI providers through a provider-model architecture. By exposing these models as an MCP server, we can create a "model hub" that allows any MCP-compatible client to access our diverse model collection through a standardized interface.

## Technical Approach

### 1. Create MCP Server Infrastructure
- **New file:** `mcp_server.py` - Main MCP server implementation
- **New file:** `cli_agent/mcp/__init__.py` - MCP module initialization
- **New file:** `cli_agent/mcp/model_server.py` - Model server logic
- **Extend:** `agent.py` - Add `mcp serve` command group

### 2. Model Exposure Strategy
Expose each provider-model combination as individual MCP tools:

#### Tool Naming Convention
- Format: `{provider}_{model_name_normalized}` 
- Examples:
  - `anthropic_claude_opus_4_20250514`
  - `anthropic_claude_sonnet_4_20250514`
  - `openai_gpt_4_turbo_preview`
  - `deepseek_deepseek_chat`
  - `gemini_gemini_2_5_flash`

#### Tool Parameters
Each model tool accepts:
- `messages` (required): Array of conversation messages
- `system_prompt` (optional): System prompt override
- `temperature` (optional): Temperature parameter (0.0-1.0)
- `max_tokens` (optional): Maximum tokens to generate
- `stream` (optional): Enable streaming response

### 3. Implementation Components

#### MCP Server Core (`cli_agent/mcp/model_server.py`)
```python
from fastmcp import FastMCP
from config import load_config
import asyncio
from typing import List, Dict, Any, Optional

def create_model_server() -> FastMCP:
    """Create and configure the MCP model server."""
    app = FastMCP("AI Models Server", version="1.0.0")
    config = load_config()
    
    # Dynamic tool generation for each available model
    available_models = config.get_available_provider_models()
    
    for provider, models in available_models.items():
        for model in models:
            create_model_tool(app, provider, model, config)
    
    return app

def create_model_tool(app: FastMCP, provider: str, model: str, config):
    """Dynamically create an MCP tool for a specific model."""
    tool_name = f"{provider}_{normalize_model_name(model)}"
    provider_model = f"{provider}:{model}"
    
    @app.tool(name=tool_name)
    async def model_tool(
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        f"""Use {model} model via {provider} provider."""
        try:
            host = config.create_host_from_provider_model(provider_model)
            
            # Add system prompt if provided
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Set model parameters
            params = {}
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Generate response
            response = await host.generate_response(
                messages=messages,
                stream=stream,
                **params
            )
            
            return response
            
        except Exception as e:
            return f"Error using {provider_model}: {str(e)}"
```

#### CLI Command Extension (`agent.py`)
```python
@cli.group()
def mcp():
    """MCP server management commands."""
    pass

@mcp.command()
@click.option("--port", default=3000, help="Port to serve on")
@click.option("--host", default="localhost", help="Host to serve on") 
@click.option("--stdio", is_flag=True, help="Use stdio transport instead of TCP")
def serve(port, host, stdio):
    """Serve all available models over MCP protocol."""
    from cli_agent.mcp.model_server import create_model_server
    
    server = create_model_server()
    
    if stdio:
        # Use stdio transport for MCP clients that expect it
        server.run_stdio()
    else:
        # Use TCP transport for web-based or network clients
        click.echo(f"Starting MCP model server on {host}:{port}")
        server.run(host=host, port=port)

@mcp.command("list-models")
def list_models():
    """List all available models that would be exposed via MCP."""
    config = load_config()
    available = config.get_available_provider_models()
    
    click.echo("Available models for MCP server:")
    for provider, models in available.items():
        click.echo(f"\n{provider}:")
        for model in models:
            tool_name = f"{provider}_{normalize_model_name(model)}"
            click.echo(f"  - {tool_name} ({provider}:{model})")
```

#### Standalone Server Script (`mcp_server.py`)
```python
#!/usr/bin/env python3
"""Standalone MCP server for AI models."""

import asyncio
import sys
from cli_agent.mcp.model_server import create_model_server

def main():
    """Run the MCP model server."""
    server = create_model_server()
    
    if "--stdio" in sys.argv:
        server.run_stdio()
    else:
        # Default to stdio for MCP compatibility
        server.run_stdio()

if __name__ == "__main__":
    main()
```

### 4. Dynamic Model Discovery
- **Startup scan**: Call `config.get_available_provider_models()` to discover models
- **API key validation**: Only expose models for which API keys are configured
- **Tool generation**: Dynamically create MCP tools for each available model
- **Metadata**: Include rich descriptions and capability information

### 5. Core Features

#### Tool Metadata
Each generated tool includes:
- **Description**: Model capabilities and use cases
- **Parameters**: Detailed parameter descriptions with types and defaults
- **Examples**: Usage examples for the tool
- **Provider info**: Which provider and authentication is used

#### Error Handling
- **Missing API keys**: Graceful handling with informative error messages
- **Provider failures**: Fallback error messages for API failures
- **Parameter validation**: Validate input parameters before processing
- **Rate limiting**: Respect provider rate limits

#### Streaming Support
- **Optional streaming**: Support both streaming and non-streaming modes
- **Real-time updates**: For clients that support streaming responses
- **Backward compatibility**: Default to non-streaming for compatibility

#### Authentication & Security
- **Optional auth**: Support for API key authentication on the MCP server
- **Request logging**: Detailed logging for debugging and monitoring
- **Rate limiting**: Server-side rate limiting to prevent abuse

### 6. Configuration

#### Environment Variables
```bash
# Enable MCP server
MCP_SERVER_ENABLED=true
MCP_SERVER_PORT=3000
MCP_SERVER_HOST=localhost

# Security (optional)
MCP_SERVER_AUTH_TOKEN=your_secret_token

# Logging
MCP_SERVER_LOG_LEVEL=INFO
```

#### Config File Extensions
```python
# config.py additions
class HostConfig(BaseSettings):
    # ... existing config ...
    
    # MCP Server settings
    mcp_server_enabled: bool = Field(default=False, alias="MCP_SERVER_ENABLED")
    mcp_server_port: int = Field(default=3000, alias="MCP_SERVER_PORT")
    mcp_server_host: str = Field(default="localhost", alias="MCP_SERVER_HOST")
    mcp_server_auth_token: str = Field(default="", alias="MCP_SERVER_AUTH_TOKEN")
```

### 7. Usage Examples

#### Starting the Server
```bash
# Start with stdio transport (MCP standard)
agent mcp serve --stdio

# Start with TCP transport
agent mcp serve --port 3000 --host 0.0.0.0

# List available models
agent mcp list-models

# Run standalone server
python mcp_server.py --stdio
```

#### Client Integration
```python
# Example MCP client usage
from mcp_client import MCPClient

client = MCPClient("stdio", command=["python", "mcp_server.py", "--stdio"])
await client.connect()

# List available tools
tools = await client.list_tools()
print([tool.name for tool in tools])
# Output: ['anthropic_claude_opus_4_20250514', 'openai_gpt_4_turbo_preview', ...]

# Use a model
response = await client.call_tool(
    "anthropic_claude_opus_4_20250514",
    {
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
        "max_tokens": 100
    }
)
```

### 8. Benefits

#### For Users
- **Unified access**: Single MCP server provides access to all models
- **Standard protocol**: Works with any MCP-compatible client
- **Easy switching**: Change models without changing client code
- **Rich metadata**: Full information about model capabilities

#### For Developers
- **Interoperability**: Integrate with Claude Desktop, VSCode, etc.
- **Centralized management**: Single point for all model access
- **Protocol compliance**: Follows MCP standards for maximum compatibility
- **Extensible**: Easy to add new providers and models

#### For Integration
- **Drop-in replacement**: Can replace individual model APIs
- **Load balancing**: Distribute requests across multiple providers
- **Fallback support**: Switch providers if one fails
- **Monitoring**: Centralized logging and metrics

## Files to Create/Modify

### New Files
1. `cli_agent/mcp/__init__.py` - MCP module initialization
2. `cli_agent/mcp/model_server.py` - Core MCP server implementation
3. `mcp_server.py` - Standalone server script
4. `specs/mcp_model_server_spec.md` - This specification file
5. `docs/mcp_server.md` - User documentation

### Modified Files
1. `agent.py` - Add `mcp` command group
2. `config.py` - Add MCP server configuration options
3. `requirements.txt` - Add FastMCP or other MCP dependencies
4. `README.md` - Document MCP server capabilities

## Implementation Phases

### Phase 1: Core Infrastructure
- Set up MCP module structure
- Implement basic server with one test model
- Add CLI commands for starting/stopping server

### Phase 2: Dynamic Model Discovery
- Implement model scanning and tool generation
- Add error handling and validation
- Support for all current providers

### Phase 3: Advanced Features
- Add streaming support
- Implement authentication and security
- Add comprehensive logging and monitoring

### Phase 4: Documentation & Testing
- Write user documentation
- Create integration tests
- Add examples and tutorials

## Technical Considerations

### Dependencies
- **FastMCP**: For MCP server implementation
- **asyncio**: For async model operations
- **click**: For CLI commands (already available)

### Performance
- **Connection pooling**: Reuse provider connections when possible
- **Request queuing**: Handle multiple concurrent requests
- **Memory management**: Properly manage model instances

### Security
- **Input validation**: Sanitize all user inputs
- **API key protection**: Never expose API keys in responses
- **Rate limiting**: Prevent abuse and respect provider limits

### Compatibility
- **MCP compliance**: Follow MCP specification exactly
- **Client support**: Test with popular MCP clients
- **Transport options**: Support both stdio and TCP transports

## Success Criteria
1. **All configured models** are exposed as MCP tools
2. **Standard MCP clients** can discover and use the tools
3. **Error handling** is graceful and informative
4. **Documentation** is complete and clear
5. **Performance** is acceptable for typical usage

This implementation would transform our agent into a true "AI model hub" that can serve models to any MCP-compatible application, greatly expanding its utility and interoperability.