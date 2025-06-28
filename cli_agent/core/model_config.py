"""Model configuration classes for different LLMs."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig(ABC):
    """Base configuration for model-specific behavior.

    Handles model-specific logic:
    - Tool calling formats
    - System prompt styles
    - Special features (reasoning, thinking)
    - Token limits and pricing
    - Temperature/parameter defaults
    """

    name: str  # User-friendly name (e.g., "claude-3.5-sonnet")
    provider_model_name: str  # Provider's model identifier
    context_length: int  # Maximum context window in tokens
    supports_tools: bool = True
    supports_streaming: bool = True
    temperature: float = 0.7
    max_tokens: int = 4000

    @property
    @abstractmethod
    def model_family(self) -> str:
        """Get model family identifier (claude, gpt, gemini, etc.)"""
        pass

    @abstractmethod
    def get_tool_format(self) -> str:
        """Get tool calling format: 'openai', 'anthropic', 'gemini', etc."""
        pass

    @abstractmethod
    def get_system_prompt_style(self) -> str:
        """Get system prompt style: 'message', 'parameter', 'prepend'"""
        pass

    @abstractmethod
    def format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply model-specific message formatting.

        Args:
            messages: Standard message format

        Returns:
            Messages formatted for this model's requirements
        """
        pass

    @abstractmethod
    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        """Get instructions optimized for this model.

        Args:
            is_subagent: Whether this is for a subagent

        Returns:
            Model-specific instruction text
        """
        pass

    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        """Parse universal thinking/reasoning tags from any model.

        This method provides universal parsing for common thinking tag formats:
        <thinking>, <reasoning>, <think>, <reflection>, <analysis>
        
        Subclasses can override this for model-specific behavior.

        Args:
            response_text: Raw response text from model

        Returns:
            Dict of extracted special content
        """
        import re

        special_content = {}

        # Parse various thinking/reasoning tag formats
        tag_patterns = [
            ("thinking", r"<thinking>(.*?)</thinking>"),
            ("reasoning", r"<reasoning>(.*?)</reasoning>"),
            ("think", r"<think>(.*?)</think>"),
            ("reflection", r"<reflection>(.*?)</reflection>"),
            ("analysis", r"<analysis>(.*?)</analysis>"),
        ]

        for tag_name, pattern in tag_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                # All thinking content goes into "reasoning" for consistent handling
                content = "\n".join(match.strip() for match in matches)
                if "reasoning" not in special_content:
                    special_content["reasoning"] = content
                else:
                    special_content["reasoning"] += "\n\n" + content

        return special_content

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for API requests."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def get_token_limit(self) -> int:
        """Get effective token limit for conversations."""
        # Reserve some tokens for response
        return max(self.context_length - self.max_tokens, 1000)

    def validate_parameters(self, **params) -> Dict[str, Any]:
        """Validate and normalize model parameters.

        Args:
            **params: Parameters to validate

        Returns:
            Validated parameters dict
        """
        validated = {}

        # Temperature validation
        temp = params.get("temperature", self.temperature)
        if isinstance(temp, (int, float)) and 0 <= temp <= 2:
            validated["temperature"] = float(temp)
        else:
            logger.warning(
                f"Invalid temperature {temp}, using default {self.temperature}"
            )
            validated["temperature"] = self.temperature

        # Max tokens validation
        max_tokens = params.get("max_tokens", self.max_tokens)
        if isinstance(max_tokens, int) and 1 <= max_tokens <= self.context_length:
            validated["max_tokens"] = max_tokens
        else:
            logger.warning(
                f"Invalid max_tokens {max_tokens}, using default {self.max_tokens}"
            )
            validated["max_tokens"] = self.max_tokens

        # Add other validated parameters
        for key, value in params.items():
            if key not in ["temperature", "max_tokens"]:
                validated[key] = value

        return validated

    def __str__(self) -> str:
        return f"{self.model_family}:{self.name}"


class ClaudeModel(ModelConfig):
    """Claude model configuration."""

    def __init__(self, variant: str = "claude-3.5-sonnet"):
        """Initialize Claude model configuration.

        Args:
            variant: Claude model variant (claude-3.5-sonnet, claude-3.5-haiku, etc.)
        """
        model_map = {
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3.5-haiku": "claude-3-5-haiku-20241022",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
        }

        # Context length varies by model
        context_lengths = {
            "claude-3.5-sonnet": 200000,
            "claude-3.5-haiku": 200000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
        }

        super().__init__(
            name=variant,
            provider_model_name=model_map.get(variant, variant),
            context_length=context_lengths.get(variant, 200000),
            supports_tools=True,
            supports_streaming=True,
            temperature=0.7,
            max_tokens=4000,
        )

    @property
    def model_family(self) -> str:
        return "claude"

    def get_tool_format(self) -> str:
        return "anthropic"

    def get_system_prompt_style(self) -> str:
        return "parameter"  # Claude uses system parameter, not system message

    def format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages for Claude's alternating user/assistant pattern."""
        formatted = []
        last_role = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Keep system messages for provider to extract (don't skip them)
            # The Anthropic provider will extract system messages and use them as the system parameter

            # Ensure alternating pattern
            if role == last_role:
                # Merge with previous message if same role
                if formatted and formatted[-1]["role"] == role:
                    formatted[-1]["content"] += f"\n\n{content}"
                else:
                    formatted.append({"role": role, "content": content})
            else:
                formatted.append({"role": role, "content": content})

            last_role = role

        return formatted

    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        if is_subagent:
            return """You are a focused subagent working on a specific task. 

Key behaviors:
- Use tools immediately when you need to take actions
- Be direct and task-focused
- End your response by calling `emit_result` with your findings
- Provide clear, actionable results

Remember: You cannot spawn other subagents - focus on completing your assigned task efficiently."""

        return """You are Claude, an AI assistant created by Anthropic.

Key behaviors:
- Use available tools when you need to take actions (run commands, read files, etc.)
- Think step by step and be thorough
- For complex tasks, consider delegating to subagents using the `task` tool
- Be helpful, harmless, and honest in all interactions

Tool usage: When users ask you to do something that requires action, use the appropriate tools immediately rather than just describing what you would do."""

    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude content using universal parsing plus Claude-specific blocks."""
        import re

        # Start with universal parsing
        special_content = super().parse_special_content(response_text)

        # Add Claude-specific <result> blocks for subagents
        result_pattern = r"<result>(.*?)</result>"
        result_matches = re.findall(result_pattern, response_text, re.DOTALL)
        if result_matches:
            special_content["result"] = "\n".join(
                match.strip() for match in result_matches
            )

        return special_content


class GPTModel(ModelConfig):
    """GPT model configuration."""

    def __init__(self, variant: str = "gpt-4o"):
        """Initialize GPT model configuration.

        Args:
            variant: GPT model variant (dynamically discovered from OpenAI API or known models)
        """
        # Known model mappings - use specific versions where preferred
        model_map = {
            "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
            "gpt-4o": "gpt-4o-2024-08-06",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
            "o1-preview": "o1-preview",
            "o1-mini": "o1-mini",
        }
        # For dynamically discovered models, use the model name as-is
        provider_model_name = model_map.get(variant, variant)

        # Context lengths for known models
        context_lengths = {
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-3.5-turbo": 16385,
            "o1-preview": 128000,
            "o1-mini": 128000,
        }

        # Dynamic context length determination
        if variant in context_lengths:
            context_length = context_lengths[variant]
        elif variant.startswith("gpt-4.1"):
            context_length = 1000000  # GPT-4.1 models have 1M context
        elif variant.startswith("gpt-4"):
            context_length = 128000  # Most GPT-4 models have 128K context
        elif variant.startswith("gpt-3.5"):
            context_length = 16385  # GPT-3.5 models
        elif variant.startswith("o1"):
            context_length = 128000  # o1 models
        else:
            context_length = 128000  # Default for unknown models

        # o1 models have different capabilities
        is_o1_model = variant.startswith("o1-")

        super().__init__(
            name=variant,
            provider_model_name=provider_model_name,
            context_length=context_length,
            supports_tools=not is_o1_model,  # o1 models don't support tools
            supports_streaming=not is_o1_model,  # o1 models don't support streaming
            temperature=(
                0.7 if not is_o1_model else 1.0
            ),  # o1 models don't use temperature
            max_tokens=4000,
        )

    @property
    def model_family(self) -> str:
        return "gpt"

    def get_tool_format(self) -> str:
        return "openai"

    def get_system_prompt_style(self) -> str:
        # o1 models don't support system messages and work better without explicit instructions
        if self.name.startswith("o1-"):
            return "none"
        return "message"  # Regular GPT models use system messages

    def format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """GPT messages are already in correct format."""
        return messages

    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        if is_subagent:
            return """You are a subagent focused on completing a specific task efficiently.

Key behaviors:
- Use tools immediately when you need to perform actions
- Be direct and results-oriented  
- Call `emit_result` at the end with your findings
- Provide clear, actionable information

Focus on your assigned task and complete it thoroughly."""

        return """You are GPT-4, a large language model trained by OpenAI.

Key behaviors:
- Use available tools when you need to take actions or get information
- Be helpful, accurate, and thorough in your responses
- For complex multi-step tasks, consider using the `task` tool to spawn subagents
- Think step by step and explain your reasoning when appropriate

Tool usage: When users request actions (running commands, reading files, etc.), use the appropriate tools immediately rather than just describing what you would do."""

    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT content using universal parsing."""
        # Use the universal parsing from the base class
        return super().parse_special_content(response_text)


class GeminiModel(ModelConfig):
    """Gemini model configuration."""

    def __init__(self, variant: str = "gemini-2.5-flash"):
        """Initialize Gemini model configuration.

        Args:
            variant: Gemini model variant (gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-pro, etc.)
        """
        model_map = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.0-flash": "gemini-2.0-flash-exp",
            "gemini-1.5-pro": "gemini-1.5-pro-002",
            "gemini-1.5-flash": "gemini-1.5-flash-002",
        }

        # Context length varies by model
        context_lengths = {
            "gemini-2.5-flash": 1000000,
            "gemini-2.5-pro": 2000000,
            "gemini-2.0-flash": 1000000,
            "gemini-1.5-pro": 2000000,
            "gemini-1.5-flash": 1000000,
        }

        super().__init__(
            name=variant,
            provider_model_name=model_map.get(variant, variant),
            context_length=context_lengths.get(variant, 1000000),
            supports_tools=True,
            supports_streaming=True,
            temperature=0.7,
            max_tokens=4000,
        )

    @property
    def model_family(self) -> str:
        return "gemini"

    def get_tool_format(self) -> str:
        return "gemini"

    def get_system_prompt_style(self) -> str:
        return "prepend"  # Gemini typically prepends system info to user messages

    def format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages for Gemini's requirements."""
        # Gemini expects specific format, might need conversion
        return messages

    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        if is_subagent:
            return """You are a Gemini subagent focused on a specific task.

CRITICAL REQUIREMENTS:
- Use tools immediately when you need to perform actions - don't just describe what you'll do
- Be direct and action-oriented
- End with `emit_result` containing your complete findings
- Focus only on your assigned task

Example: If asked to "run uname -a", immediately use bash_execute with that command, then call emit_result with the system information."""

        return """You are Gemini, Google's advanced AI model.

CRITICAL REQUIREMENTS:
- When asked to perform actions (run commands, read files, etc.), use tools immediately
- Don't just describe what you would do - actually DO it using the available tools
- For complex tasks involving multiple files or commands, consider delegating to subagents
- Provide thorough, helpful responses based on actual results

Tool Usage Priority: Take action FIRST, then provide analysis based on actual results."""

    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini content using universal parsing."""
        # Use the universal parsing from the base class
        return super().parse_special_content(response_text)


class DeepSeekModel(ModelConfig):
    """DeepSeek model configuration."""

    def __init__(self, variant: str = "deepseek-chat"):
        """Initialize DeepSeek model configuration.

        Args:
            variant: DeepSeek model variant (deepseek-chat, deepseek-reasoner, etc.)
        """
        model_map = {
            "deepseek-chat": "deepseek-chat",
            "deepseek-reasoner": "deepseek-reasoner",
        }

        # Context lengths
        context_lengths = {
            "deepseek-chat": 64000,
            "deepseek-reasoner": 64000,
        }

        super().__init__(
            name=variant,
            provider_model_name=model_map.get(variant, variant),
            context_length=context_lengths.get(variant, 64000),
            supports_tools=True,
            supports_streaming=True,
            temperature=0.7,
            max_tokens=4000,
        )

    @property
    def model_family(self) -> str:
        return "deepseek"

    def get_tool_format(self) -> str:
        return "openai"  # DeepSeek uses OpenAI-compatible format

    def get_system_prompt_style(self) -> str:
        return "message"  # DeepSeek uses system messages

    def format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """DeepSeek uses standard OpenAI format."""
        return messages

    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        if is_subagent:
            return """You are a DeepSeek subagent focused on completing a specific task.

Key behaviors for DeepSeek:
- Use the <reasoning> section to plan your approach before taking action
- Execute tools immediately when you need to perform actions
- End with `emit_result` containing comprehensive findings
- Be thorough and systematic in your approach

Pattern: Reason → Act → Report results via emit_result"""

        return """You are DeepSeek, an advanced reasoning AI model.

Key behaviors for DeepSeek:
- Use <reasoning> sections to think through problems systematically
- Execute tools when you need to take actions or gather information
- For complex tasks, consider delegating to subagents using the `task` tool
- Provide well-reasoned, thorough responses

Reasoning Pattern: Think through problems step-by-step in <reasoning> blocks, then take appropriate actions using available tools."""

    def parse_special_content(self, response_text: str) -> Dict[str, Any]:
        """Parse DeepSeek content using universal parsing."""
        # Use the universal parsing from the base class
        return super().parse_special_content(response_text)
