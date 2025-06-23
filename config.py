"""Configuration management for MCP Deepseek Host."""

import os
import time
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
    temperature: float = 0.6
    max_tokens: int = 4096
    stream: bool = True
    keepalive_interval: float = 10.0  # Keep-alive interval in seconds


class GeminiConfig(BaseModel):
    """Configuration for Gemini API."""

    api_key: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 16384
    top_p: float = 0.9
    top_k: int = 40
    stream: bool = False  # Streaming disabled for Gemini (better reliability)
    keepalive_interval: float = 10.0  # Keep-alive interval in seconds
    force_function_calling: bool = (
        False  # Force function calling when tools are available
    )
    function_calling_mode: str = (
        "AUTO"  # AUTO, ANY, or NONE - controls compositional function calling behavior
    )


class HostConfig(BaseSettings):
    """Main configuration for the MCP Host."""

    # Deepseek configuration
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_temperature: float = Field(default=0.6, alias="DEEPSEEK_TEMPERATURE")
    deepseek_max_tokens: int = Field(default=4096, alias="DEEPSEEK_MAX_TOKENS")
    deepseek_stream: bool = Field(default=True, alias="DEEPSEEK_STREAM")

    # Gemini configuration
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=8192, alias="GEMINI_MAX_TOKENS")
    gemini_top_p: float = Field(default=0.9, alias="GEMINI_TOP_P")
    gemini_top_k: int = Field(default=40, alias="GEMINI_TOP_K")
    gemini_stream: bool = Field(default=False, alias="GEMINI_STREAM")
    gemini_force_function_calling: bool = Field(
        default=False, alias="GEMINI_FORCE_FUNCTION_CALLING"
    )
    gemini_function_calling_mode: str = Field(
        default="AUTO", alias="GEMINI_FUNCTION_CALLING_MODE"
    )

    # Host configuration
    host_name: str = Field(default="mcp-deepseek-host", alias="HOST_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # MCP servers configuration
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    # Tool permission configuration
    allowed_tools: List[str] = Field(
        default_factory=lambda: [
            "task",
            "task_status",
            "task_results",
            "emit_result",
            "read_file",
            "list_directory",
            "get_current_directory",
            "todo_read",
            "todo_write",
        ],
        alias="ALLOWED_TOOLS",
    )
    disallowed_tools: List[str] = Field(default_factory=list, alias="DISALLOWED_TOOLS")
    auto_approve_tools: bool = Field(default=False, alias="AUTO_APPROVE_TOOLS")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    def get_deepseek_config(self) -> DeepseekConfig:
        """Get Deepseek configuration."""
        return DeepseekConfig(
            api_key=self.deepseek_api_key,
            model=self.deepseek_model,
            temperature=self.deepseek_temperature,
            max_tokens=self.deepseek_max_tokens,
            stream=self.deepseek_stream,
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
            function_calling_mode=self.gemini_function_calling_mode,
        )

    def get_tool_permission_config(self):
        """Get tool permission configuration."""
        from cli_agent.core.tool_permissions import ToolPermissionConfig

        return ToolPermissionConfig(
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            auto_approve_session=self.auto_approve_tools,
        )

    def add_mcp_server(
        self,
        name: str,
        command: List[str],
        args: List[str] = None,
        env: Dict[str, str] = None,
    ):
        """Add an MCP server configuration."""
        self.mcp_servers[name] = MCPServerConfig(
            name=name, command=command, args=args or [], env=env or {}
        )

    def save_mcp_servers(self):
        """Save MCP server configurations to mcp_servers.json."""
        import json
        from pathlib import Path

        # Store MCP config in ~/.config/agent/ to persist across working directories
        config_dir = Path.home() / ".config" / "agent"
        config_dir.mkdir(parents=True, exist_ok=True)
        mcp_config_file = config_dir / "mcp_servers.json"

        servers_data = {}
        for name, server_config in self.mcp_servers.items():
            servers_data[name] = {
                "name": server_config.name,
                "command": server_config.command,
                "args": server_config.args,
                "env": server_config.env,
            }

        with open(mcp_config_file, "w") as f:
            json.dump(servers_data, f, indent=2)

    def load_mcp_servers(self):
        """Load MCP server configurations from mcp_servers.json."""
        import json
        from pathlib import Path

        # Load MCP config from ~/.config/agent/ to persist across working directories
        config_dir = Path.home() / ".config" / "agent"
        mcp_config_file = config_dir / "mcp_servers.json"

        if not mcp_config_file.exists():
            return

        try:
            with open(mcp_config_file, "r") as f:
                servers_data = json.load(f)

            for name, server_data in servers_data.items():
                self.add_mcp_server(
                    name=server_data["name"],
                    command=server_data["command"],
                    args=server_data.get("args", []),
                    env=server_data.get("env", {}),
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
        """Save current configuration to .env file, preserving existing variables."""
        import re

        # Variables that this config manages
        managed_vars = {
            "DEEPSEEK_API_KEY": self.deepseek_api_key,
            "DEEPSEEK_MODEL": self.deepseek_model,
            "DEEPSEEK_TEMPERATURE": str(self.deepseek_temperature),
            "DEEPSEEK_MAX_TOKENS": str(self.deepseek_max_tokens),
            "DEEPSEEK_STREAM": str(self.deepseek_stream).lower(),
            "GEMINI_API_KEY": self.gemini_api_key,
            "GEMINI_MODEL": self.gemini_model,
            "GEMINI_TEMPERATURE": str(self.gemini_temperature),
            "GEMINI_MAX_TOKENS": str(self.gemini_max_tokens),
            "GEMINI_TOP_P": str(self.gemini_top_p),
            "GEMINI_TOP_K": str(self.gemini_top_k),
            "GEMINI_STREAM": str(self.gemini_stream).lower(),
            "HOST_NAME": self.host_name,
            "LOG_LEVEL": self.log_level,
        }

        # Read existing .env file if it exists
        existing_lines = []
        if os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    existing_lines = f.readlines()
            except Exception as e:
                print(f"Warning: Could not read existing .env file: {e}")

        # Process existing lines and update managed variables
        updated_lines = []
        updated_vars = set()

        for line in existing_lines:
            line = line.rstrip("\n\r")

            # Check if this line sets a managed variable
            match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
            if match:
                var_name = match.group(1)
                if var_name in managed_vars:
                    # Update with new value
                    updated_lines.append(f"{var_name}={managed_vars[var_name]}")
                    updated_vars.add(var_name)
                else:
                    # Keep existing non-managed variable
                    updated_lines.append(line)
            else:
                # Keep comments, empty lines, etc.
                updated_lines.append(line)

        # Add any managed variables that weren't in the existing file
        missing_vars = set(managed_vars.keys()) - updated_vars
        if missing_vars:
            # Add a section for new variables
            if updated_lines and updated_lines[-1].strip():
                updated_lines.append("")  # Add blank line before new section

            # Group by type for better organization
            deepseek_vars = [v for v in missing_vars if v.startswith("DEEPSEEK_")]
            gemini_vars = [v for v in missing_vars if v.startswith("GEMINI_")]
            host_vars = [
                v for v in missing_vars if v.startswith("HOST_") or v in ["LOG_LEVEL"]
            ]

            if deepseek_vars:
                updated_lines.append("# Deepseek API Configuration")
                for var in sorted(deepseek_vars):
                    updated_lines.append(f"{var}={managed_vars[var]}")
                updated_lines.append("")

            if gemini_vars:
                updated_lines.append("# Gemini API Configuration")
                for var in sorted(gemini_vars):
                    updated_lines.append(f"{var}={managed_vars[var]}")
                updated_lines.append("")

            if host_vars:
                updated_lines.append("# Host Configuration")
                for var in sorted(host_vars):
                    updated_lines.append(f"{var}={managed_vars[var]}")

        # Write back to .env file
        try:
            with open(".env", "w") as f:
                for line in updated_lines:
                    f.write(line + "\n")
        except Exception as e:
            print(f"Error: Could not write to .env file: {e}")
            raise

    def save_persistent_config(self):
        """Save persistent configuration to ~/.config/agent/config.py."""
        import json
        from pathlib import Path

        # Store persistent config in ~/.config/agent/
        config_dir = Path.home() / ".config" / "agent"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.json"

        # Only save model configuration that should persist
        persistent_config = {
            "deepseek_model": self.deepseek_model,
            "gemini_model": self.gemini_model,
            "last_updated": time.time(),
        }

        try:
            with open(config_file, "w") as f:
                json.dump(persistent_config, f, indent=2)
            print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Warning: Could not save persistent configuration: {e}")

    def load_persistent_config(self):
        """Load persistent configuration from ~/.config/agent/config.py."""
        import json
        from pathlib import Path

        config_dir = Path.home() / ".config" / "agent"
        config_file = config_dir / "config.json"

        if not config_file.exists():
            return  # No persistent config exists

        try:
            with open(config_file, "r") as f:
                persistent_config = json.load(f)

            # Apply persistent configuration, overriding defaults
            if "deepseek_model" in persistent_config:
                self.deepseek_model = persistent_config["deepseek_model"]
            if "gemini_model" in persistent_config:
                self.gemini_model = persistent_config["gemini_model"]

            print(f"Loaded persistent configuration from {config_file}")
        except Exception as e:
            print(f"Warning: Could not load persistent configuration: {e}")


def load_config() -> HostConfig:
    """Load configuration from environment variables, .env file, and persistent config."""
    config = HostConfig()
    config.load_mcp_servers()  # Auto-load persistent MCP servers
    config.load_persistent_config()  # Auto-load persistent configuration
    return config


def create_sample_env():
    """Create a sample .env file."""
    sample_content = """# Deepseek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.6
DEEPSEEK_MAX_TOKENS=4096
DEEPSEEK_STREAM=true

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=8192
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
