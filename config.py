"""Configuration management for MCP Agent with Provider-Model Architecture."""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def migrate_mcp_agent_config():
    """Migrate configuration from old directories to new agent directory."""
    old_config_dir = Path.home() / ".config" / "mcp-agent"
    cli_agent_dir = Path.home() / ".config" / "cli-agent"
    new_config_dir = Path.home() / ".config" / "agent"
    
    # Skip if new directory already exists
    if new_config_dir.exists():
        return
    
    # Migrate from cli-agent directory first (most recent), then mcp-agent (oldest)
    source_dir = None
    if cli_agent_dir.exists():
        source_dir = cli_agent_dir
    elif old_config_dir.exists():
        source_dir = old_config_dir
    
    if source_dir:
        try:
            logger.info(f"Migrating configuration from {source_dir} to {new_config_dir}")
            shutil.move(str(source_dir), str(new_config_dir))
            logger.info("Configuration migration completed successfully")
        except Exception as e:
            logger.warning(f"Failed to migrate configuration: {e}")
            # If migration fails, just continue - the old config will still work


def get_config_dir() -> Path:
    """Get the configuration directory, with migration from old location if needed."""
    migrate_mcp_agent_config()
    return Path.home() / ".config" / "agent"


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


class AnthropicConfig(BaseModel):
    """Configuration for Anthropic API."""

    api_key: str
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout: float = 120.0


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter API."""

    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3.5-sonnet"
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout: float = 120.0
    app_name: Optional[str] = None
    site_url: Optional[str] = None


class OllamaConfig(BaseModel):
    """Configuration for Ollama API."""

    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 180.0  # Ollama can be slower for local inference


class ProviderModelConfig(BaseModel):
    """Configuration for provider-model combination."""

    provider_name: str  # anthropic, openai, openrouter, deepseek, google, ollama
    model_name: str  # claude-3.5-sonnet, gpt-4, llama2, etc.
    provider_config: Union[
        AnthropicConfig,
        OpenAIConfig,
        OpenRouterConfig,
        DeepseekConfig,
        GeminiConfig,
        OllamaConfig,
    ]
    display_name: Optional[str] = None  # Human-readable name


class HostConfig(BaseSettings):
    """Main configuration for the MCP Host."""

    # Class-level cache for model lists to prevent repeated API calls
    _model_cache: Dict[str, Dict] = {}
    _cache_ttl: int = 300  # 5 minutes TTL

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

    # Anthropic configuration
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL"
    )
    anthropic_temperature: float = Field(default=0.7, alias="ANTHROPIC_TEMPERATURE")
    anthropic_max_tokens: int = Field(default=8192, alias="ANTHROPIC_MAX_TOKENS")

    # OpenAI configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4096, alias="OPENAI_MAX_TOKENS")

    # OpenRouter configuration
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="anthropic/claude-3.5-sonnet", alias="OPENROUTER_MODEL"
    )
    openrouter_temperature: float = Field(default=0.7, alias="OPENROUTER_TEMPERATURE")
    openrouter_max_tokens: int = Field(default=8192, alias="OPENROUTER_MAX_TOKENS")

    # Ollama configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama2", alias="OLLAMA_MODEL")
    ollama_temperature: float = Field(default=0.7, alias="OLLAMA_TEMPERATURE")
    ollama_max_tokens: int = Field(default=4096, alias="OLLAMA_MAX_TOKENS")
    ollama_timeout: float = Field(default=180.0, alias="OLLAMA_TIMEOUT")

    # Provider-Model selection
    default_provider_model: str = Field(
        default="", alias="DEFAULT_PROVIDER_MODEL"  # Empty default triggers intelligent selection
    )
    model_type: str = Field(
        default="deepseek", alias="MODEL_TYPE"
    )  # For backward compatibility

    # Session management
    last_session_id: Optional[str] = Field(default=None, alias="LAST_SESSION_ID")

    # Host configuration
    host_name: str = Field(default="mcp-agent", alias="HOST_NAME")
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

    # Tool result display configuration
    truncate_tool_results: bool = Field(default=True, alias="TRUNCATE_TOOL_RESULTS")
    tool_result_max_length: int = Field(default=1000, alias="TOOL_RESULT_MAX_LENGTH")

    # Background subagent configuration
    background_subagents: bool = Field(default=False, alias="BACKGROUND_SUBAGENTS")

    # Subagent permissions bypass
    subagent_permissions_bypass: bool = Field(
        default=False, alias="SUBAGENT_PERMISSIONS_BYPASS"
    )

    # Hooks system configuration
    hooks_enabled: bool = Field(default=True, alias="HOOKS_ENABLED")
    hooks_timeout: int = Field(default=30, alias="HOOKS_TIMEOUT")

    # MCP Server settings
    mcp_server_enabled: bool = Field(default=False, alias="MCP_SERVER_ENABLED")
    mcp_server_port: int = Field(default=3000, alias="MCP_SERVER_PORT")
    mcp_server_host: str = Field(default="localhost", alias="MCP_SERVER_HOST")
    mcp_server_auth_token: str = Field(default="", alias="MCP_SERVER_AUTH_TOKEN")
    mcp_server_log_level: str = Field(default="INFO", alias="MCP_SERVER_LOG_LEVEL")

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

    def get_anthropic_config(self) -> AnthropicConfig:
        """Get Anthropic configuration."""
        return AnthropicConfig(
            api_key=self.anthropic_api_key,
            model=self.anthropic_model,
            temperature=self.anthropic_temperature,
            max_tokens=self.anthropic_max_tokens,
        )

    def get_openai_config(self) -> OpenAIConfig:
        """Get OpenAI configuration."""
        return OpenAIConfig(
            api_key=self.openai_api_key,
            model=self.openai_model,
            temperature=self.openai_temperature,
            max_tokens=self.openai_max_tokens,
        )

    def get_openrouter_config(self) -> OpenRouterConfig:
        """Get OpenRouter configuration."""
        return OpenRouterConfig(
            api_key=self.openrouter_api_key,
            model=self.openrouter_model,
            temperature=self.openrouter_temperature,
            max_tokens=self.openrouter_max_tokens,
        )

    def get_ollama_config(self) -> OllamaConfig:
        """Get Ollama configuration."""
        return OllamaConfig(
            base_url=self.ollama_base_url,
            model=self.ollama_model,
            temperature=self.ollama_temperature,
            max_tokens=self.ollama_max_tokens,
            timeout=self.ollama_timeout,
        )

    def get_intelligent_default_provider_model(self) -> str:
        """Intelligently select the best available provider based on configured API keys.
        
        Selection preference: OpenAI > DeepSeek > Anthropic > OpenRouter > Ollama
        
        Returns:
            String like 'openai:gpt-4-turbo-preview' for the best available provider
        """
        # Define default models for each provider
        provider_defaults = {
            "openai": "gpt-4-turbo-preview",
            "deepseek": "deepseek-chat", 
            "anthropic": "claude-3-5-sonnet-20241022",
            "openrouter": "anthropic/claude-3.5-sonnet",
            "ollama": "llama2"  # Ollama doesn't require an API key
        }
        
        # Priority order: OpenAI > DeepSeek > Anthropic > OpenRouter > Ollama
        priority_order = ["openai", "deepseek", "anthropic", "openrouter", "ollama"]
        
        for provider in priority_order:
            # Check if API key is configured for this provider
            if provider == "openai" and self.openai_api_key:
                return f"openai:{provider_defaults['openai']}"
            elif provider == "deepseek" and self.deepseek_api_key:
                return f"deepseek:{provider_defaults['deepseek']}"
            elif provider == "anthropic" and self.anthropic_api_key:
                return f"anthropic:{provider_defaults['anthropic']}"
            elif provider == "openrouter" and self.openrouter_api_key:
                return f"openrouter:{provider_defaults['openrouter']}"
            elif provider == "ollama":
                # Check if Ollama is actually running
                if self._is_ollama_available():
                    return f"ollama:{provider_defaults['ollama']}"
        
        # No providers available - raise informative error
        self._raise_no_providers_error()

    def _raise_no_providers_error(self) -> None:
        """Raise an informative error when no providers are available."""
        error_msg = """❌ No AI providers are configured or available.

To use the MCP Agent, you need at least one of the following:

📋 Configure an API key for a cloud provider:
   export OPENAI_API_KEY=your_key_here
   export DEEPSEEK_API_KEY=your_key_here  
   export ANTHROPIC_API_KEY=your_key_here
   export GEMINI_API_KEY=your_key_here
   export OPENROUTER_API_KEY=your_key_here

🏠 Or install and start Ollama for local inference:
   # Install: https://ollama.ai
   ollama serve
   ollama pull llama2

Then restart the agent.

💡 Tip: The agent will automatically select the best available provider."""
        
        raise ValueError(error_msg)

    def _provider_has_api_key(self, provider_name: str) -> bool:
        """Check if a provider has a configured API key.
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'deepseek')
            
        Returns:
            True if the provider has a non-empty API key, False otherwise
        """
        if provider_name == "openai":
            return bool(self.openai_api_key)
        elif provider_name == "deepseek":
            return bool(self.deepseek_api_key)
        elif provider_name == "anthropic":
            return bool(self.anthropic_api_key)
        elif provider_name == "openrouter":
            return bool(self.openrouter_api_key)
        elif provider_name == "google":
            return bool(self.gemini_api_key)
        elif provider_name == "ollama":
            return self._is_ollama_available()  # Check if Ollama is actually running
        else:
            return False

    def _is_ollama_available(self) -> bool:
        """Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            import httpx
            import asyncio
            
            async def check_ollama_async():
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{self.ollama_base_url}/api/tags")
                        return response.status_code == 200
                except:
                    return False
            
            def check_ollama_sync():
                try:
                    with httpx.Client(timeout=5.0) as client:
                        response = client.get(f"{self.ollama_base_url}/api/tags")
                        return response.status_code == 200
                except:
                    return False
            
            # Run the check
            try:
                loop = asyncio.get_running_loop()
                # If we're in an event loop, use synchronous client to avoid conflicts
                return check_ollama_sync()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                return asyncio.run(check_ollama_async())
                
        except ImportError:
            # httpx not available
            return False
        except Exception:
            return False

    def parse_provider_model_string(self, provider_model: str) -> Tuple[str, str]:
        """Parse provider:model string into provider and model components.

        Args:
            provider_model: String like 'openrouter:claude-3.5-sonnet' or 'deepseek:deepseek-chat'

        Returns:
            Tuple of (provider_name, model_name)
        """
        if ":" in provider_model:
            provider, model = provider_model.split(":", 1)
            return provider.strip(), model.strip()
        else:
            # Fallback to backward compatibility
            return provider_model.strip(), ""

    def get_provider_model_config(
        self, provider_model: Optional[str] = None
    ) -> ProviderModelConfig:
        """Get provider and model configuration for the specified combination.

        Args:
            provider_model: String like 'openrouter:claude-3.5-sonnet'. If None, uses default.

        Returns:
            ProviderModelConfig with provider name, model name, and provider config
        """
        # Use intelligent selection if no provider specified or default is empty
        if not provider_model and not self.default_provider_model:
            target_provider_model = self.get_intelligent_default_provider_model()
        else:
            target_provider_model = provider_model or self.default_provider_model
            
            # If using default and it requires an API key that's not configured, use intelligent selection
            if not provider_model:  # Only for default selection, not explicit user choice
                default_provider_name, _ = self.parse_provider_model_string(target_provider_model)
                if not self._provider_has_api_key(default_provider_name):
                    target_provider_model = self.get_intelligent_default_provider_model()
            
        provider_name, model_name = self.parse_provider_model_string(
            target_provider_model
        )

        # Get appropriate provider config
        if provider_name == "anthropic":
            provider_config = self.get_anthropic_config()
            if model_name:
                provider_config.model = model_name
        elif provider_name == "openai":
            provider_config = self.get_openai_config()
            if model_name:
                provider_config.model = model_name
        elif provider_name == "openrouter":
            provider_config = self.get_openrouter_config()
            if model_name:
                provider_config.model = model_name
        elif provider_name == "deepseek":
            provider_config = self.get_deepseek_config()
            if model_name:
                provider_config.model = model_name
        elif provider_name == "google":
            provider_config = self.get_gemini_config()
            if model_name:
                provider_config.model = model_name
        elif provider_name == "ollama":
            provider_config = self.get_ollama_config()
            if model_name:
                provider_config.model = model_name
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        return ProviderModelConfig(
            provider_name=provider_name,
            model_name=provider_config.model,
            provider_config=provider_config,
            display_name=f"{provider_name}:{provider_config.model}",
        )

    def create_host_from_provider_model(
        self, provider_model: Optional[str] = None, is_subagent: bool = False
    ):
        """Create MCPHost instance from provider-model configuration.

        Args:
            provider_model: String like 'openrouter:claude-3.5-sonnet'. If None, uses default.
            is_subagent: Whether this host is for a subagent (affects tool availability)

        Returns:
            MCPHost instance configured with the specified provider and model
        """
        pm_config = self.get_provider_model_config(provider_model)

        # Import here to avoid circular imports
        from cli_agent.core.mcp_host import MCPHost
        from cli_agent.core.model_config import (
            ClaudeModel,
            DeepSeekModel,
            GeminiModel,
            GPTModel,
        )
        from cli_agent.providers.anthropic_provider import AnthropicProvider
        from cli_agent.providers.deepseek_provider import DeepSeekProvider
        from cli_agent.providers.google_provider import GoogleProvider
        from cli_agent.providers.ollama_provider import OllamaProvider
        from cli_agent.providers.openai_provider import OpenAIProvider
        from cli_agent.providers.openrouter_provider import OpenRouterProvider

        # Create provider instance
        if pm_config.provider_name == "anthropic":
            provider = AnthropicProvider(
                api_key=pm_config.provider_config.api_key,
                base_url=pm_config.provider_config.base_url,
                timeout=pm_config.provider_config.timeout,
            )
        elif pm_config.provider_name == "openai":
            provider = OpenAIProvider(
                api_key=pm_config.provider_config.api_key,
                base_url=pm_config.provider_config.base_url,
                timeout=pm_config.provider_config.timeout,
            )
        elif pm_config.provider_name == "openrouter":
            provider = OpenRouterProvider(
                api_key=pm_config.provider_config.api_key,
                base_url=pm_config.provider_config.base_url,
                timeout=pm_config.provider_config.timeout,
            )
            # Set app info if available
            if (
                hasattr(pm_config.provider_config, "app_name")
                and pm_config.provider_config.app_name
            ):
                provider.set_app_info(
                    pm_config.provider_config.app_name,
                    pm_config.provider_config.site_url
                    or "https://github.com/your-repo",
                )
        elif pm_config.provider_name == "deepseek":
            provider = DeepSeekProvider(
                api_key=pm_config.provider_config.api_key,
                base_url=pm_config.provider_config.base_url,
                timeout=600.0,  # DeepSeek can be slower
            )
        elif pm_config.provider_name == "google":
            provider = GoogleProvider(
                api_key=pm_config.provider_config.api_key, timeout=120.0
            )
        elif pm_config.provider_name == "ollama":
            provider = OllamaProvider(
                api_key="",  # Ollama doesn't require API key
                base_url=pm_config.provider_config.base_url,
                timeout=pm_config.provider_config.timeout,
            )
        else:
            raise ValueError(f"Unknown provider: {pm_config.provider_name}")

        # Create model instance based on provider first, then model name
        model_name = pm_config.model_name.lower()
        
        # Provider-specific logic for standardized APIs
        if pm_config.provider_name == "openrouter":
            # OpenRouter standardizes everything to OpenAI format regardless of underlying model
            # Use GPTModel for all OpenRouter models (claude, gpt, gemini, llama, qwen, etc.)
            model = GPTModel(variant=pm_config.model_name)
        elif pm_config.provider_name == "ollama":
            # Ollama standardizes everything to OpenAI format through its API
            # Detect underlying model family only for special content parsing and streaming support
            if "deepseek" in model_name:
                # DeepSeek models need reasoning content parsing but use OpenAI format for tools
                if "reasoner" in model_name:
                    model = DeepSeekModel(variant=pm_config.model_name)
                else:
                    model = DeepSeekModel(variant=pm_config.model_name)
            elif model_name.startswith("qwen3:"):
                # Qwen3 models need special handling (thinking content parsing)
                from cli_agent.core.models.qwen_model import QwenModel
                model = QwenModel(variant=pm_config.model_name)
            else:
                # All other Ollama models use OpenAI format (llama, claude, gemini, etc.)
                # Ollama's OpenAI-compatible API standardizes tool calling to OpenAI format
                model = GPTModel(variant=pm_config.model_name)
        # Direct provider cases - use native formats
        elif pm_config.provider_name == "anthropic":
            # Direct Anthropic API - use Anthropic format
            model = ClaudeModel(variant=pm_config.model_name)
        elif pm_config.provider_name == "google":
            # Direct Google API - use Gemini format
            if "2.5-flash" in model_name:
                model = GeminiModel(variant="gemini-2.5-flash")
            elif "1.5-pro" in model_name:
                model = GeminiModel(variant="gemini-1.5-pro")
            elif "1.5-flash" in model_name:
                model = GeminiModel(variant="gemini-1.5-flash")
            else:
                model = GeminiModel(variant="gemini-2.5-flash")  # Default
        elif pm_config.provider_name == "deepseek":
            # Direct DeepSeek API - use DeepSeek format
            if "reasoner" in model_name:
                model = DeepSeekModel(variant="deepseek-reasoner")
            else:
                model = DeepSeekModel(variant="deepseek-chat")
        elif pm_config.provider_name == "openai":
            # Direct OpenAI API - use OpenAI format
            model = GPTModel(variant=pm_config.model_name)
        else:
            # Fallback for unknown providers - try to detect by model name
            if "claude" in model_name:
                model = ClaudeModel(variant=pm_config.model_name)
            elif "gpt" in model_name or "o1" in model_name:
                model = GPTModel(variant=pm_config.model_name)
            elif "gemini" in model_name:
                model = GeminiModel(variant="gemini-2.5-flash")
            elif "deepseek" in model_name:
                model = DeepSeekModel(variant="deepseek-chat")
            else:
                # Final fallback
                model = GPTModel(variant=pm_config.model_name)

        # Override temperature and max_tokens after creation
        model.temperature = pm_config.provider_config.temperature
        if hasattr(pm_config.provider_config, "max_tokens"):
            model.max_tokens = pm_config.provider_config.max_tokens
        elif hasattr(pm_config.provider_config, "max_output_tokens"):
            model.max_tokens = pm_config.provider_config.max_output_tokens

        return MCPHost(
            provider=provider, model=model, config=self, is_subagent=is_subagent
        )

    def get_available_provider_models(self) -> Dict[str, List[str]]:
        """Get available provider-model combinations with caching.

        Returns:
            Dict mapping provider names to lists of available models
        """
        current_time = time.time()
        cache_key = "provider_models"

        # Check if we have a valid cached result
        if (
            cache_key in self._model_cache
            and current_time - self._model_cache[cache_key].get("timestamp", 0)
            < self._cache_ttl
        ):
            logger.debug("Using cached provider models")
            return self._model_cache[cache_key]["data"]

        logger.debug("Fetching fresh provider models")
        available = {}

        if self.anthropic_api_key:
            # Try to get dynamic model list from Anthropic API with caching
            anthropic_cache_key = f"anthropic_models_{hash(self.anthropic_api_key)}"

            # Check Anthropic-specific cache first
            if (
                anthropic_cache_key in self._model_cache
                and current_time
                - self._model_cache[anthropic_cache_key].get("timestamp", 0)
                < self._cache_ttl
            ):
                available["anthropic"] = self._model_cache[anthropic_cache_key]["data"]
                logger.debug("Using cached Anthropic models")
            else:
                try:
                    import asyncio

                    from cli_agent.providers.anthropic_provider import AnthropicProvider

                    # Use static method to avoid client lifecycle issues
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a new thread to run the async call
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                models_data = new_loop.run_until_complete(
                                    AnthropicProvider.fetch_available_models_static(
                                        self.anthropic_api_key
                                    )
                                )
                                return [model["id"] for model in models_data]
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            models = future.result(timeout=15)  # 15 second timeout

                    except RuntimeError:
                        # No event loop running, we can use asyncio.run
                        models_data = asyncio.run(
                            AnthropicProvider.fetch_available_models_static(
                                self.anthropic_api_key
                            )
                        )
                        models = [model["id"] for model in models_data]

                    if models:  # Only add if we got models
                        available["anthropic"] = models
                        # Cache the Anthropic models separately
                        self._model_cache[anthropic_cache_key] = {
                            "data": models,
                            "timestamp": current_time,
                        }
                        logger.info(
                            f"Successfully fetched and cached {len(models)} Anthropic models dynamically"
                        )

                except Exception as e:
                    logger.warning(f"Failed to get dynamic Anthropic models: {e}")
                    # Fallback to static list if API fails
                    fallback_models = [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                        "claude-3-opus-20240229",
                    ]
                    available["anthropic"] = fallback_models
                    logger.info(f"Using fallback Anthropic models: {fallback_models}")

        if self.openai_api_key:
            # Try to get dynamic model list from OpenAI API with caching
            openai_cache_key = f"openai_models_{hash(self.openai_api_key)}"

            # Check OpenAI-specific cache first
            if (
                openai_cache_key in self._model_cache
                and current_time
                - self._model_cache[openai_cache_key].get("timestamp", 0)
                < self._cache_ttl
            ):
                available["openai"] = self._model_cache[openai_cache_key]["data"]
                logger.debug("Using cached OpenAI models")
            else:
                try:
                    import asyncio

                    from cli_agent.providers.openai_provider import OpenAIProvider

                    # Use static method to avoid client lifecycle issues
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a new thread to run the async call
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(
                                    OpenAIProvider.fetch_available_models_static(
                                        self.openai_api_key
                                    )
                                )
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            models = future.result(timeout=15)  # 15 second timeout

                    except RuntimeError:
                        # No event loop running, we can use asyncio.run
                        models = asyncio.run(
                            OpenAIProvider.fetch_available_models_static(
                                self.openai_api_key
                            )
                        )

                    if models:  # Only add if we got models
                        available["openai"] = models
                        # Cache the OpenAI models separately
                        self._model_cache[openai_cache_key] = {
                            "data": models,
                            "timestamp": current_time,
                        }
                        logger.info(
                            f"Successfully fetched and cached {len(models)} OpenAI models dynamically"
                        )

                except Exception as e:
                    logger.warning(f"Failed to get dynamic OpenAI models: {e}")
                    # Fallback to static list if API fails
                    fallback_models = [
                        "gpt-4-turbo-preview",
                        "gpt-4",
                        "gpt-3.5-turbo",
                        "o1-preview",
                        "o1-mini",
                    ]
                    available["openai"] = fallback_models
                    logger.info(f"Using fallback OpenAI models: {fallback_models}")

        if self.openrouter_api_key:
            # Try to get dynamic model list from OpenRouter API with caching
            openrouter_cache_key = f"openrouter_models_{hash(self.openrouter_api_key)}"

            # Check OpenRouter-specific cache first
            if (
                openrouter_cache_key in self._model_cache
                and current_time
                - self._model_cache[openrouter_cache_key].get("timestamp", 0)
                < self._cache_ttl
            ):
                available["openrouter"] = self._model_cache[openrouter_cache_key][
                    "data"
                ]
                logger.debug("Using cached OpenRouter models")
            else:
                try:
                    import asyncio

                    from cli_agent.providers.openrouter_provider import (
                        OpenRouterProvider,
                    )

                    # Use static method to avoid client lifecycle issues
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a new thread to run the async call
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                models_data = new_loop.run_until_complete(
                                    OpenRouterProvider.fetch_available_models_static(
                                        self.openrouter_api_key
                                    )
                                )
                                return [model["id"] for model in models_data]
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            models = future.result(timeout=15)  # 15 second timeout

                    except RuntimeError:
                        # No event loop running, we can use asyncio.run
                        models_data = asyncio.run(
                            OpenRouterProvider.fetch_available_models_static(
                                self.openrouter_api_key
                            )
                        )
                        models = [model["id"] for model in models_data]

                    if models:  # Only add if we got models
                        available["openrouter"] = models
                        # Cache the OpenRouter models separately
                        self._model_cache[openrouter_cache_key] = {
                            "data": models,
                            "timestamp": current_time,
                        }
                        logger.info(
                            f"Successfully fetched and cached {len(models)} OpenRouter models dynamically"
                        )

                except Exception as e:
                    logger.warning(f"Failed to get dynamic OpenRouter models: {e}")
                    # Fallback to static list if API fails
                    fallback_models = [
                        "anthropic/claude-3.5-sonnet",
                        "openai/gpt-4-turbo-preview",
                        "google/gemini-pro-1.5",
                        "meta-llama/llama-3.1-405b-instruct",
                        "deepseek/deepseek-chat",
                    ]
                    available["openrouter"] = fallback_models
                    logger.info(f"Using fallback OpenRouter models: {fallback_models}")

        if self.deepseek_api_key:
            # Try to get dynamic model list from DeepSeek API with caching
            deepseek_cache_key = f"deepseek_models_{hash(self.deepseek_api_key)}"

            # Check DeepSeek-specific cache first
            if (
                deepseek_cache_key in self._model_cache
                and current_time
                - self._model_cache[deepseek_cache_key].get("timestamp", 0)
                < self._cache_ttl
            ):
                available["deepseek"] = self._model_cache[deepseek_cache_key]["data"]
                logger.debug("Using cached DeepSeek models")
            else:
                try:
                    import asyncio

                    from cli_agent.providers.deepseek_provider import DeepSeekProvider

                    # Use static method to avoid client lifecycle issues
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a new thread to run the async call
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                # Create temporary provider to use get_available_models
                                provider = DeepSeekProvider(self.deepseek_api_key)
                                models_data = new_loop.run_until_complete(
                                    provider.get_available_models()
                                )
                                return [model["id"] for model in models_data]
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            models = future.result(timeout=15)  # 15 second timeout

                    except RuntimeError:
                        # No event loop running, we can use asyncio.run
                        provider = DeepSeekProvider(self.deepseek_api_key)
                        models_data = asyncio.run(provider.get_available_models())
                        models = [model["id"] for model in models_data]

                    if models:  # Only add if we got models
                        available["deepseek"] = models
                        # Cache the DeepSeek models separately
                        self._model_cache[deepseek_cache_key] = {
                            "data": models,
                            "timestamp": current_time,
                        }
                        logger.info(
                            f"Successfully fetched and cached {len(models)} DeepSeek models dynamically"
                        )

                except Exception as e:
                    logger.warning(f"Failed to get dynamic DeepSeek models: {e}")
                    # Fallback to static list if API fails
                    fallback_models = ["deepseek-chat", "deepseek-reasoner"]
                    available["deepseek"] = fallback_models
                    logger.info(f"Using fallback DeepSeek models: {fallback_models}")

        if self.gemini_api_key:
            # Try to get dynamic model list from Google API with caching
            google_cache_key = f"google_models_{hash(self.gemini_api_key)}"

            # Check Google-specific cache first
            if (
                google_cache_key in self._model_cache
                and current_time
                - self._model_cache[google_cache_key].get("timestamp", 0)
                < self._cache_ttl
            ):
                available["google"] = self._model_cache[google_cache_key]["data"]
                logger.debug("Using cached Google models")
            else:
                try:
                    import asyncio

                    from cli_agent.providers.google_provider import GoogleProvider

                    # Use static method to avoid client lifecycle issues
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a new thread to run the async call
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                models_data = new_loop.run_until_complete(
                                    GoogleProvider.fetch_available_models_static(
                                        self.gemini_api_key
                                    )
                                )
                                return [model["id"] for model in models_data]
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            models = future.result(timeout=15)  # 15 second timeout

                    except RuntimeError:
                        # No event loop running, we can use asyncio.run
                        models_data = asyncio.run(
                            GoogleProvider.fetch_available_models_static(
                                self.gemini_api_key
                            )
                        )
                        models = [model["id"] for model in models_data]

                    if models:  # Only add if we got models
                        available["google"] = models
                        # Cache the Google models separately
                        self._model_cache[google_cache_key] = {
                            "data": models,
                            "timestamp": current_time,
                        }
                        logger.info(
                            f"Successfully fetched and cached {len(models)} Google models dynamically"
                        )

                except Exception as e:
                    logger.warning(f"Failed to get dynamic Google models: {e}")
                    # Fallback to static list if API fails
                    fallback_models = [
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                    ]
                    available["google"] = fallback_models
                    logger.info(f"Using fallback Google models: {fallback_models}")

        # Add Ollama models (mostly static as it's typically local)
        if self.ollama_base_url:  # Check if Ollama is configured
            # Try to get dynamic model list from Ollama API with caching
            ollama_cache_key = f"ollama_models_{hash(self.ollama_base_url)}"

            # Check Ollama-specific cache first
            if (
                ollama_cache_key in self._model_cache
                and current_time
                - self._model_cache[ollama_cache_key].get("timestamp", 0)
                < self._cache_ttl
            ):
                available["ollama"] = self._model_cache[ollama_cache_key]["data"]
                logger.debug("Using cached Ollama models")
            else:
                try:
                    import asyncio

                    import httpx

                    # Try to get models from Ollama API
                    async def fetch_ollama_models():
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.get(
                                f"{self.ollama_base_url}/api/tags"
                            )
                            if response.status_code == 200:
                                data = response.json()
                                return [
                                    model["name"] for model in data.get("models", [])
                                ]
                            return []

                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a new thread to run the async call
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(
                                    fetch_ollama_models()
                                )
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            models = future.result(timeout=10)  # 10 second timeout

                    except RuntimeError:
                        # No event loop running, we can use asyncio.run
                        models = asyncio.run(fetch_ollama_models())

                    if models:  # Only add if we got models
                        available["ollama"] = models
                        # Cache the Ollama models separately
                        self._model_cache[ollama_cache_key] = {
                            "data": models,
                            "timestamp": current_time,
                        }
                        logger.info(
                            f"Successfully fetched and cached {len(models)} Ollama models dynamically"
                        )
                    else:
                        logger.info("No Ollama models found - skipping Ollama provider")

                except Exception as e:
                    logger.debug(f"Failed to get dynamic Ollama models: {e}")
                    logger.debug("Ollama provider unavailable - skipping")

        # Cache the complete result
        self._model_cache[cache_key] = {"data": available, "timestamp": current_time}

        return available

    def clear_model_cache(self):
        """Clear the model cache to force fresh API calls."""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    def get_default_provider_for_model(self, model_name: str) -> Optional[str]:
        """Determine the default provider for a given model name.

        Args:
            model_name: The model name to find a provider for

        Returns:
            The default provider name, or None if not found
        """
        available = self.get_available_provider_models()

        # Check each provider to see if they have this model
        for provider, models in available.items():
            if model_name in models:
                return provider

        return None

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
        config_dir = get_config_dir()
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
        config_dir = get_config_dir()
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
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "ANTHROPIC_MODEL": self.anthropic_model,
            "ANTHROPIC_TEMPERATURE": str(self.anthropic_temperature),
            "ANTHROPIC_MAX_TOKENS": str(self.anthropic_max_tokens),
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_MODEL": self.openai_model,
            "OPENAI_TEMPERATURE": str(self.openai_temperature),
            "OPENAI_MAX_TOKENS": str(self.openai_max_tokens),
            "OPENROUTER_API_KEY": self.openrouter_api_key,
            "OPENROUTER_MODEL": self.openrouter_model,
            "OPENROUTER_TEMPERATURE": str(self.openrouter_temperature),
            "OPENROUTER_MAX_TOKENS": str(self.openrouter_max_tokens),
            "OLLAMA_BASE_URL": self.ollama_base_url,
            "OLLAMA_MODEL": self.ollama_model,
            "OLLAMA_TEMPERATURE": str(self.ollama_temperature),
            "OLLAMA_MAX_TOKENS": str(self.ollama_max_tokens),
            "OLLAMA_TIMEOUT": str(self.ollama_timeout),
            "DEFAULT_PROVIDER_MODEL": self.default_provider_model,
            "MODEL_TYPE": self.model_type,
            "HOST_NAME": self.host_name,
            "LOG_LEVEL": self.log_level,
            "TRUNCATE_TOOL_RESULTS": str(self.truncate_tool_results).lower(),
            "TOOL_RESULT_MAX_LENGTH": str(self.tool_result_max_length),
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
            anthropic_vars = [v for v in missing_vars if v.startswith("ANTHROPIC_")]
            openai_vars = [v for v in missing_vars if v.startswith("OPENAI_")]
            openrouter_vars = [v for v in missing_vars if v.startswith("OPENROUTER_")]
            ollama_vars = [v for v in missing_vars if v.startswith("OLLAMA_")]
            host_vars = [
                v
                for v in missing_vars
                if v.startswith("HOST_")
                or v in ["LOG_LEVEL", "DEFAULT_PROVIDER_MODEL", "MODEL_TYPE"]
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

            if anthropic_vars:
                updated_lines.append("# Anthropic API Configuration")
                for var in sorted(anthropic_vars):
                    updated_lines.append(f"{var}={managed_vars[var]}")
                updated_lines.append("")

            if openai_vars:
                updated_lines.append("# OpenAI API Configuration")
                for var in sorted(openai_vars):
                    updated_lines.append(f"{var}={managed_vars[var]}")
                updated_lines.append("")

            if openrouter_vars:
                updated_lines.append("# OpenRouter API Configuration")
                for var in sorted(openrouter_vars):
                    updated_lines.append(f"{var}={managed_vars[var]}")
                updated_lines.append("")

            if ollama_vars:
                updated_lines.append("# Ollama API Configuration")
                for var in sorted(ollama_vars):
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

    def save_persistent_config(self, silent: bool = False):
        """Save persistent configuration to ~/.config/agent/config.json."""
        import json
        import os
        from pathlib import Path

        # Store persistent config in ~/.config/agent/
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.json"

        # Only save model configuration that should persist
        persistent_config = {
            "deepseek_model": self.deepseek_model,
            "gemini_model": self.gemini_model,
            "anthropic_model": self.anthropic_model,
            "openai_model": self.openai_model,
            "openrouter_model": self.openrouter_model,
            "ollama_model": self.ollama_model,
            "default_provider_model": self.default_provider_model,
            "model_type": self.model_type,
            "last_session_id": getattr(self, "last_session_id", None),
            "last_updated": time.time(),
        }

        try:
            with open(config_file, "w") as f:
                json.dump(persistent_config, f, indent=2)
            # Check global stream-json mode flag
            suppress_output = silent or os.environ.get("STREAM_JSON_MODE") == "true"
            if not suppress_output:
                print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Warning: Could not save persistent configuration: {e}")

    def load_persistent_config(self):
        """Load persistent configuration from ~/.config/agent/config.json."""
        import json
        from pathlib import Path

        config_dir = get_config_dir()
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
            if "anthropic_model" in persistent_config:
                self.anthropic_model = persistent_config["anthropic_model"]
            if "openai_model" in persistent_config:
                self.openai_model = persistent_config["openai_model"]
            if "openrouter_model" in persistent_config:
                self.openrouter_model = persistent_config["openrouter_model"]
            if "ollama_model" in persistent_config:
                self.ollama_model = persistent_config["ollama_model"]
            if "default_provider_model" in persistent_config:
                self.default_provider_model = persistent_config[
                    "default_provider_model"
                ]
            if "model_type" in persistent_config:
                self.model_type = persistent_config["model_type"]
            if "last_session_id" in persistent_config:
                self.last_session_id = persistent_config["last_session_id"]

            # Migrate legacy model settings to provider-model format
            self._migrate_legacy_model_settings()

            # Config loading message suppressed - subagents emit their own message
            # The persistent configuration has been loaded but we don't print it here
        except Exception as e:
            print(f"Warning: Could not load persistent configuration: {e}")

    def _migrate_legacy_model_settings(self):
        """Migrate legacy model settings to provider-model format."""
        # Only migrate if default_provider_model is not set or is clearly wrong
        migration_needed = False

        # Fix the root cause: deepseek_model should never be "gemini"
        if self.deepseek_model == "gemini":
            # Reset deepseek_model to a proper DeepSeek model
            self.deepseek_model = "deepseek-chat"
            migration_needed = True

        # If default_provider_model is empty or invalid, set a sensible default
        if not self.default_provider_model or ":" not in self.default_provider_model:
            # Default to deepseek with the configured deepseek model
            self.default_provider_model = f"deepseek:{self.deepseek_model}"
            migration_needed = True

        # Fix any obviously wrong configurations where deepseek_model was abused
        provider, model = self.parse_provider_model_string(self.default_provider_model)
        if provider == "deepseek" and model == "gemini":
            # This is clearly wrong - fix it to use google provider with gemini model
            self.default_provider_model = f"google:{self.gemini_model}"
            migration_needed = True

        # Save updated configuration if migration occurred
        if migration_needed:
            self.save_persistent_config()
            print(f"Migrated to provider-model: {self.default_provider_model}")

    def set_last_session_id(self, session_id: str, silent: bool = False):
        """Set and persist the last session ID."""
        self.last_session_id = session_id
        self.save_persistent_config(silent=silent)
        logger.debug(f"Saved last session ID: {session_id}")

    def get_last_session_id(self) -> Optional[str]:
        """Get the last session ID."""
        return self.last_session_id

    def clear_last_session_id(self, silent: bool = False):
        """Clear the last session ID."""
        self.last_session_id = None
        self.save_persistent_config(silent=silent)
        logger.debug("Cleared last session ID")

    def load_hook_config(self):
        """Load hook configuration from standard sources if hooks are enabled."""
        if not self.hooks_enabled:
            logger.debug("Hooks disabled, skipping hook configuration loading")
            return None
            
        try:
            from cli_agent.core.hooks.hook_config import HookConfig
            
            hook_config = HookConfig.load_from_multiple_sources()
            
            if hook_config.has_hooks():
                logger.info("Loaded hooks configuration successfully")
                # Validate configuration
                validation_errors = hook_config.validate()
                if validation_errors:
                    logger.warning(f"Hook configuration validation warnings: {validation_errors}")
                return hook_config
            else:
                logger.debug("No hooks found in configuration")
                return None
                
        except ImportError as e:
            logger.warning(f"Hook system not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading hook configuration: {e}")
            return None


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

# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_TEMPERATURE=0.7
ANTHROPIC_MAX_TOKENS=8192

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4096

# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=8192

# Ollama Configuration (Local Models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=4096
OLLAMA_TIMEOUT=180.0

# Provider-Model Selection
DEFAULT_PROVIDER_MODEL=deepseek:deepseek-chat
MODEL_TYPE=deepseek

# Host Configuration
HOST_NAME=mcp-agent
LOG_LEVEL=INFO
"""

    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(sample_content)
        print("Created sample .env file. Please update with your actual API key.")
    else:
        print(".env file already exists.")
