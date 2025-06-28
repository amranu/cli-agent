"""Qwen model configuration for Ollama."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from cli_agent.core.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass  
class QwenModel(ModelConfig):
    """Qwen model configuration (for Ollama)."""

    def __init__(self, variant: str = "qwen3:4b"):
        """Initialize Qwen model configuration.

        Args:
            variant: Qwen model variant (qwen3:1.7b, qwen3:4b, etc.)
        """
        # Context lengths for different Qwen variants
        context_lengths = {
            "qwen3:1b": 32000,
            "qwen3:1.7b": 32000, 
            "qwen3:4b": 32000,
            "qwen3:8b": 32000,
        }

        # Streaming support - qwen3 models work with streaming
        supports_streaming = True

        super().__init__(
            name=variant,
            provider_model_name=variant,  # Ollama uses exact model name
            context_length=context_lengths.get(variant, 32000),
            supports_tools=True,
            supports_streaming=supports_streaming,
            temperature=0.7,
            max_tokens=4000,
        )

    @property
    def model_family(self) -> str:
        return "qwen"

    def get_tool_format(self) -> str:
        return "openai"  # Ollama uses OpenAI-compatible format

    def get_system_prompt_style(self) -> str:
        return "message"  # Qwen uses system messages

    def format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Qwen uses standard OpenAI format."""
        return messages
    
    def format_streaming_tool_calls(self, tool_calls_data):
        """Convert Qwen tool calls from dict format to object format for MCPHost compatibility."""
        if not tool_calls_data:
            return None
            
        # Create object classes for tool call compatibility
        class ToolCallObject:
            def __init__(self, tc_dict):
                self.id = tc_dict.get("id")
                self.index = tc_dict.get("index")
                self.type = tc_dict.get("type")
                if "function" in tc_dict:
                    self.function = FunctionObject(tc_dict["function"])

        class FunctionObject:
            def __init__(self, func_dict):
                self.name = func_dict.get("name")
                self.arguments = func_dict.get("arguments", "")
        
        # Convert tool call dicts to objects
        formatted_calls = []
        for tc_dict in tool_calls_data:
            tc_obj = ToolCallObject(tc_dict)
            formatted_calls.append(tc_obj)
        
        return formatted_calls

    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        if is_subagent:
            return """You are a Qwen subagent focused on completing a specific task.
Use available tools when needed and provide clear results.
End with your findings."""
        return """You are Qwen, a helpful AI assistant.
Use available tools when needed to help users effectively.
Be precise and thorough in your responses."""

    def parse_special_content(self, text_content: str) -> Dict[str, Any]:
        """Parse Qwen thinking content for proper formatting."""
        import re
        import logging
        
        logger = logging.getLogger(__name__)
        logger.debug(f"QwenModel.parse_special_content called with: '{text_content[:200]}...'")
        
        # Extract <think>...</think> blocks
        thinking_pattern = r'<think>(.*?)</think>'
        thinking_match = re.search(thinking_pattern, text_content, re.DOTALL)
        
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            logger.debug(f"Found thinking content: '{thinking_content[:100]}...'")
            return {"thinking_content": thinking_content}
        
        logger.debug("No thinking content found")
        return {}