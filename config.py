"""Configuration management for MCP Deepseek Host."""

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str
    command: List[str]
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)


class DeepseekConfig(BaseModel):
    """Configuration for Deepseek API."""
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"  # or "deepseek-reasoner"
    temperature: float = 0.5
    max_tokens: int = 4096
    stream: bool = True
    keepalive_interval: float = 10.0  # Keep-alive interval in seconds


class GeminiConfig(BaseModel):
    """Configuration for Gemini API."""
    api_key: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 8192
    top_p: float = 0.9
    top_k: int = 40
    stream: bool = False  # Streaming disabled for Gemini (better reliability)
    keepalive_interval: float = 10.0  # Keep-alive interval in seconds
    force_function_calling: bool = False  # Force function calling when tools are available
    function_calling_mode: str = "AUTO"  # AUTO, ANY, or NONE - controls compositional function calling behavior


class HostConfig(BaseSettings):
    """Main configuration for the MCP Host."""
    
    # Deepseek configuration
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_temperature: float = Field(default=0.7, alias="DEEPSEEK_TEMPERATURE")
    deepseek_max_tokens: int = Field(default=4096, alias="DEEPSEEK_MAX_TOKENS")
    deepseek_stream: bool = Field(default=True, alias="DEEPSEEK_STREAM")
    
    # Gemini configuration
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=4096, alias="GEMINI_MAX_TOKENS")
    gemini_top_p: float = Field(default=0.9, alias="GEMINI_TOP_P")
    gemini_top_k: int = Field(default=40, alias="GEMINI_TOP_K")
    gemini_stream: bool = Field(default=False, alias="GEMINI_STREAM")
    gemini_force_function_calling: bool = Field(default=False, alias="GEMINI_FORCE_FUNCTION_CALLING")
    gemini_function_calling_mode: str = Field(default="AUTO", alias="GEMINI_FUNCTION_CALLING_MODE")
    
    # Host configuration
    host_name: str = Field(default="mcp-deepseek-host", alias="HOST_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # MCP servers configuration
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix=""
    )
    
    def get_deepseek_config(self) -> DeepseekConfig:
        """Get Deepseek configuration."""
        return DeepseekConfig(
            api_key=self.deepseek_api_key,
            model=self.deepseek_model,
            temperature=self.deepseek_temperature,
            max_tokens=self.deepseek_max_tokens,
            stream=self.deepseek_stream
        )
    
    def get_gemini_config(self) -> GeminiConfig:
        """Get Gemini configuration."""
        return GeminiConfig(
            api_key=self.gemini_api_key,
            model=self.gemini_model,
            temperature=self.gemini_temperature,
            max_output_tokens=self.gemini_max_tokens,
            top_p=self.gemini_top_p,
            top_k=self.gemini_top_k,
            stream=self.gemini_stream,
            force_function_calling=self.gemini_force_function_calling,
            function_calling_mode=self.gemini_function_calling_mode
        )
    
    def add_mcp_server(self, name: str, command: List[str], args: List[str] = None, env: Dict[str, str] = None):
        """Add an MCP server configuration."""
        self.mcp_servers[name] = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {}
        )
    
    def save_mcp_servers(self):
        """Save MCP server configurations to mcp_servers.json."""
        import json
        mcp_config_file = "mcp_servers.json"
        
        servers_data = {}
        for name, server_config in self.mcp_servers.items():
            servers_data[name] = {
                "name": server_config.name,
                "command": server_config.command,
                "args": server_config.args,
                "env": server_config.env
            }
        
        with open(mcp_config_file, 'w') as f:
            json.dump(servers_data, f, indent=2)
    
    def load_mcp_servers(self):
        """Load MCP server configurations from mcp_servers.json."""
        import json
        mcp_config_file = "mcp_servers.json"
        
        if not os.path.exists(mcp_config_file):
            return
        
        try:
            with open(mcp_config_file, 'r') as f:
                servers_data = json.load(f)
            
            for name, server_data in servers_data.items():
                self.add_mcp_server(
                    name=server_data["name"],
                    command=server_data["command"],
                    args=server_data.get("args", []),
                    env=server_data.get("env", {})
                )
        except Exception as e:
            print(f"Warning: Failed to load MCP servers configuration: {e}")
    
    def remove_mcp_server(self, name: str) -> bool:
        """Remove an MCP server configuration."""
        if name in self.mcp_servers:
            del self.mcp_servers[name]
            return True
        return False
    
    def save(self):
        """Save current configuration to .env file."""
        env_content = []
        
        # Add Deepseek configuration
        env_content.append("# Deepseek API Configuration")
        env_content.append(f"DEEPSEEK_API_KEY={self.deepseek_api_key}")
        env_content.append(f"DEEPSEEK_MODEL={self.deepseek_model}")
        env_content.append(f"DEEPSEEK_TEMPERATURE={self.deepseek_temperature}")
        env_content.append(f"DEEPSEEK_MAX_TOKENS={self.deepseek_max_tokens}")
        env_content.append(f"DEEPSEEK_STREAM={str(self.deepseek_stream).lower()}")
        env_content.append("")
        
        # Add Gemini configuration
        env_content.append("# Gemini API Configuration")
        env_content.append(f"GEMINI_API_KEY={self.gemini_api_key}")
        env_content.append(f"GEMINI_MODEL={self.gemini_model}")
        env_content.append(f"GEMINI_TEMPERATURE={self.gemini_temperature}")
        env_content.append(f"GEMINI_MAX_TOKENS={self.gemini_max_tokens}")
        env_content.append(f"GEMINI_TOP_P={self.gemini_top_p}")
        env_content.append(f"GEMINI_TOP_K={self.gemini_top_k}")
        env_content.append(f"GEMINI_STREAM={str(self.gemini_stream).lower()}")
        env_content.append("")
        
        # Add host configuration
        env_content.append("# Host Configuration")
        env_content.append(f"HOST_NAME={self.host_name}")
        env_content.append(f"LOG_LEVEL={self.log_level}")
        
        # Write to .env file
        with open(".env", "w") as f:
            f.write("\n".join(env_content) + "\n")


def load_config() -> HostConfig:
    """Load configuration from environment variables and .env file."""
    config = HostConfig()
    config.load_mcp_servers()  # Auto-load persistent MCP servers
    return config


def create_sample_env():
    """Create a sample .env file."""
    sample_content = """# Deepseek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.7
DEEPSEEK_MAX_TOKENS=4096
DEEPSEEK_STREAM=true

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=4096
GEMINI_TOP_P=0.9
GEMINI_TOP_K=40
GEMINI_STREAM=false

# Host Configuration
HOST_NAME=mcp-deepseek-host
LOG_LEVEL=INFO
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(sample_content)
        print("Created sample .env file. Please update with your actual API key.")
    else:
        print(".env file already exists.")
