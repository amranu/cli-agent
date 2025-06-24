"""Message processing framework for MCP agents."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MessageProcessor:
    """Handles message preprocessing, cleaning, and format conversion."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Centralized message preprocessing pipeline with provider hooks."""
        # Step 1: Clean messages using provider-specific logic
        cleaned_messages = self.clean_message_format(messages)

        # Step 2: Enhance messages for specific model requirements
        enhanced_messages = self.enhance_messages_for_model(cleaned_messages)

        # Step 3: Convert to provider format
        provider_messages = self.convert_to_provider_format(enhanced_messages)

        return provider_messages

    def clean_message_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Clean messages - delegates to agent's provider-specific cleaning."""
        return self.agent._clean_message_format(messages)

    def enhance_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance messages for model-specific requirements - delegates to agent."""
        return self.agent._enhance_messages_for_model(messages)

    def convert_to_provider_format(self, messages: List[Dict[str, Any]]) -> Any:
        """Convert messages to provider format - delegates to agent."""
        return self.agent._convert_to_provider_format(messages)
