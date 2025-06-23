#!/usr/bin/env python3
"""Token management and conversation compaction for MCP Agent."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages token counting, limits, and conversation compaction."""

    def __init__(self, config=None):
        """Initialize TokenManager with optional config."""
        self.config = config

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters for most models)."""
        return len(text) // 4

    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count estimated tokens in the conversation."""
        total_tokens = 0
        for message in messages:
            if isinstance(message.get("content"), str):
                total_tokens += self.estimate_tokens(message["content"])
            # Add small overhead for role and structure
            total_tokens += 10
        return total_tokens

    def get_token_limit(self, model_name: Optional[str] = None) -> int:
        """Get the context token limit for the specified or current model."""
        # Enhanced centralized token limit management with model configuration support
        model_limits = self._get_model_token_limits()

        # Use provided model name or try to get from config
        if not model_name:
            model_name = self._get_current_model_name()

        if model_name and model_name in model_limits:
            return model_limits[model_name]

        # Fallback: check for model patterns
        if model_name:
            for pattern, limit in model_limits.items():
                if pattern in model_name.lower():
                    return limit

        # Conservative default - subclasses can override
        return 32000

    def _get_model_token_limits(self) -> Dict[str, int]:
        """Define token limits for known models. Can be extended via config."""
        return {
            # DeepSeek models
            "deepseek-reasoner": 128000,
            "deepseek-chat": 64000,
            # Gemini models
            "gemini-pro": 128000,
            "pro": 128000,  # Pattern matching for any "pro" model
            "gemini-flash": 64000,
            "flash": 64000,  # Pattern matching for any "flash" model
            # Common defaults
            "gpt-4": 128000,
            "gpt-3.5": 16000,
            "claude": 200000,
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
        self, messages: List[Dict[str, Any]], model_name: Optional[str] = None
    ) -> bool:
        """Determine if conversation should be compacted."""
        current_tokens = self.count_conversation_tokens(messages)
        limit = self.get_token_limit(model_name)
        # Compact when we're at 80% of the limit
        return current_tokens > (limit * 0.8)

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
