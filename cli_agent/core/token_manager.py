#!/usr/bin/env python3
"""Token management and conversation compaction for MCP Agent."""

import logging
from typing import Any, Dict, List, Optional

from cli_agent.utils.token_counting import (
    token_counter,
    count_tokens,
    count_message_tokens,
    count_conversation_tokens,
    get_effective_context_limit,
)

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages token counting, limits, and conversation compaction."""

    def __init__(self, config=None, model_name: Optional[str] = None):
        """Initialize TokenManager with optional config and model name."""
        self.config = config
        self.model_name = model_name

    def estimate_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Accurate token counting using model-specific tokenizers."""
        model_name = model_name or self.model_name or self._get_current_model_name()
        if model_name:
            return count_tokens(text, model_name)
        # Fallback to basic estimation
        return max(1, len(text) // 4)

    def count_conversation_tokens(self, messages: List[Dict[str, Any]], model_name: Optional[str] = None) -> int:
        """Count accurate tokens in the conversation using model-specific tokenizers."""
        model_name = model_name or self.model_name or self._get_current_model_name()
        if model_name:
            return count_conversation_tokens(messages, model_name)
        
        # Fallback to basic estimation
        total_tokens = 0
        for message in messages:
            if isinstance(message.get("content"), str):
                total_tokens += max(1, len(message["content"]) // 4)
            # Add small overhead for role and structure
            total_tokens += 10
        return total_tokens

    def count_message_tokens(self, message: Dict[str, Any], model_name: Optional[str] = None) -> int:
        """Count tokens in a single message using model-specific tokenizers."""
        model_name = model_name or self.model_name or self._get_current_model_name()
        if model_name:
            return count_message_tokens(message, model_name)
        
        # Fallback to basic estimation
        content = message.get("content", "")
        if isinstance(content, str):
            return max(1, len(content) // 4) + 10  # Add overhead
        return 10

    def get_token_limit(self, model_name: Optional[str] = None, context_length: Optional[int] = None, max_tokens: Optional[int] = None) -> int:
        """Get the effective token limit for the specified model."""
        model_name = model_name or self.model_name or self._get_current_model_name()
        
        if model_name and context_length:
            # Use the comprehensive token counting system to get effective limit
            return get_effective_context_limit(model_name, context_length, max_tokens)
        
        # Fallback to enhanced model limits
        model_limits = self._get_enhanced_model_token_limits()

        if model_name and model_name in model_limits:
            return model_limits[model_name]

        # Fallback: check for model patterns
        if model_name:
            for pattern, limit in model_limits.items():
                if pattern in model_name.lower():
                    return limit

        # Conservative default
        return 32000

    def has_reliable_token_info(self, model_name: Optional[str] = None, context_length: Optional[int] = None) -> bool:
        """Check if we have reliable token information for the specified model.
        
        Returns True if we have explicit model limits or context length, False if using fallback.
        """
        model_name = model_name or self.model_name or self._get_current_model_name()
        
        # If we have explicit context_length, we have reliable info
        if context_length and context_length > 0:
            return True
        
        # Check if we have the model in our known limits
        model_limits = self._get_enhanced_model_token_limits()
        
        if model_name and model_name in model_limits:
            return True
            
        # Handle provider:model format by extracting just the model name
        if model_name and ":" in model_name:
            provider, just_model = model_name.split(":", 1)
            if just_model in model_limits:
                return True
            
        # Check for exact pattern matches (not partial matches)
        if model_name:
            for pattern in model_limits.keys():
                if model_name.lower() == pattern.lower():
                    return True
        
        # If we're falling back to pattern matching or conservative default, it's unreliable
        return False

    def _get_enhanced_model_token_limits(self) -> Dict[str, int]:
        """Enhanced token limits based on comprehensive model survey."""
        return {
            # OpenAI GPT Models
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-4-1106-preview": 128000,
            "gpt-4-0613": 8192,
            "gpt-4-0125-preview": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4o-2024-05-13": 128000,
            "gpt-4o-2024-08-06": 128000,
            "gpt-4o-2024-11-20": 128000,
            "chatgpt-4o-latest": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-0125": 16385,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "o1-preview": 128000,
            "o1-mini": 128000,
            
            # Anthropic Claude Models - effective limits (reserve tokens for response)
            "claude-3-5-sonnet-20241022": 190000,
            "claude-3-5-sonnet-20240620": 190000,
            "claude-3-5-haiku-20241022": 190000,
            "claude-3-opus-20240229": 190000,
            "claude-3-sonnet-20240229": 190000,
            "claude-3-haiku-20240307": 190000,
            "claude-sonnet-4-20250514": 200000,
            "claude-opus-4-20250514": 200000,
            
            # DeepSeek Models - updated limits
            "deepseek-chat": 60000,
            "deepseek-reasoner": 125000,
            "deepseek-coder": 60000,
            
            # Google Gemini Models - effective limits
            "gemini-2.5-flash": 950000,
            "gemini-2.5-pro": 1950000,
            "gemini-1.5-pro": 1950000,
            "gemini-1.5-flash": 950000,
            "gemini-1.0-pro": 28000,
            "gemini-pro": 28000,
            "gemini-pro-vision": 14000,
            
            # Qwen Models
            "qwen-turbo": 30000,
            "qwen-plus": 30000,
            "qwen-max": 30000,
            
            # Pattern matching for unknown variants
            "gpt-4": 128000,
            "gpt-3.5": 16000,
            "claude": 190000,
            "gemini": 950000,
            "deepseek": 60000,
            "qwen": 30000,
            "llama": 4000,
            "mistral": 8000,
        }

    def _get_current_model_name(self) -> Optional[str]:
        """Get the current model name from config if available."""
        if not self.config:
            return None

        # Try common config patterns
        for attr_name in ["deepseek_config", "gemini_config", "openai_config"]:
            if hasattr(self.config, attr_name):
                config = getattr(self.config, attr_name)
                if hasattr(config, "model"):
                    return config.model

        return None

    def should_compact(
        self, messages: List[Dict[str, Any]], model_name: Optional[str] = None, context_length: Optional[int] = None
    ) -> bool:
        """Determine if conversation should be compacted using accurate token counting."""
        model_name = model_name or self.model_name or self._get_current_model_name()
        current_tokens = self.count_conversation_tokens(messages, model_name)
        limit = self.get_token_limit(model_name, context_length)
        
        # Compact when we're at 80% of the effective limit
        threshold = limit * 0.8
        
        logger.debug(f"Token check: {current_tokens}/{limit} tokens ({current_tokens/limit*100:.1f}%), threshold: {threshold}")
        return current_tokens > threshold

    async def compact_conversation(
        self, messages: List[Dict[str, Any]], generate_response_func=None
    ) -> List[Dict[str, Any]]:
        """Create a compact summary of the conversation to preserve context while reducing tokens.

        Args:
            messages: List of conversation messages
            generate_response_func: Function to generate summary response (should accept messages and tools args)

        Returns:
            Compacted list of messages
        """
        if len(messages) <= 3:  # Keep conversations that are already short
            return messages

        # Always keep the first message (system prompt) and last 2 messages
        system_message = messages[0] if messages[0].get("role") == "system" else None
        recent_messages = messages[-2:]

        # Messages to summarize (everything except system and last 2)
        start_idx = 1 if system_message else 0
        messages_to_summarize = messages[start_idx:-2]

        if not messages_to_summarize:
            return messages

        # Create summary prompt
        conversation_text = "\n".join(
            [
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in messages_to_summarize
            ]
        )

        summary_prompt = f"""Please create a concise summary of this conversation that preserves:
1. Key decisions and actions taken
2. Important file changes or tool usage
3. Current project state and context
4. Any pending tasks or next steps

Conversation to summarize:
{conversation_text}

Provide a brief but comprehensive summary that maintains continuity for ongoing work."""

        try:
            if generate_response_func:
                # Use the provided response generation function to create summary
                summary_messages = [{"role": "user", "content": summary_prompt}]
                summary_response = await generate_response_func(
                    summary_messages, tools=None
                )
            else:
                # Fallback if no response function provided
                summary_response = (
                    "Summary unavailable - compacting to recent messages only"
                )

            # Create condensed conversation
            condensed = []
            if system_message:
                condensed.append(system_message)

            # Add summary as a system message
            condensed.append(
                {
                    "role": "system",
                    "content": f"[CONVERSATION SUMMARY] {summary_response}",
                }
            )

            # Add recent messages
            condensed.extend(recent_messages)

            print(
                f"\n🗜️  Conversation compacted: {len(messages)} → {len(condensed)} messages"
            )
            return condensed

        except Exception as e:
            logger.warning(f"Failed to compact conversation: {e}")
            # Fallback: just keep system + last 5 messages
            fallback = []
            if system_message:
                fallback.append(system_message)
            fallback.extend(messages[-5:])
            return fallback
