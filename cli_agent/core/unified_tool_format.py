"""Unified tool format parsing that handles multiple formats simultaneously."""

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UnifiedToolFormatModel:
    """Model config that can parse tool calls in any format dynamically."""
    
    def __init__(self, variant: str, base_model_config):
        """Initialize with a base model config to inherit other properties from."""
        self.name = variant
        self.base_model = base_model_config
        
    @property
    def model_family(self) -> str:
        return "unified"
        
    def get_tool_format(self) -> str:
        return "unified"
        
    def get_system_prompt_style(self) -> str:
        return self.base_model.get_system_prompt_style()
        
    def format_messages_for_model(self, messages) -> List[Dict[str, Any]]:
        return self.base_model.format_messages_for_model(messages)
        
    def parse_special_content(self, text_content: str) -> Dict[str, Any]:
        return self.base_model.parse_special_content(text_content)
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return self.base_model.get_default_parameters()
        
    def validate_parameters(self, **params) -> Dict[str, Any]:
        return self.base_model.validate_parameters(**params)
        
    def get_token_limit(self) -> int:
        return self.base_model.get_token_limit()
        
    def get_model_specific_instructions(self, is_subagent: bool = False) -> str:
        return self.base_model.get_model_specific_instructions(is_subagent)
        
    @property
    def provider_model_name(self) -> str:
        return self.base_model.provider_model_name
        
    @property
    def context_length(self) -> int:
        return self.base_model.context_length
        
    @property
    def supports_tools(self) -> bool:
        return self.base_model.supports_tools
        
    @property
    def supports_streaming(self) -> bool:
        return self.base_model.supports_streaming
        
    def __str__(self) -> str:
        return f"UnifiedToolFormat({self.name})"


class UnifiedToolCallParser:
    """Parser that attempts all tool call formats simultaneously."""
    
    @staticmethod
    def extract_tool_calls_all_formats(response: Any) -> Tuple[List[Any], str]:
        """
        Attempt to extract tool calls using all known formats simultaneously.
        
        Returns:
            Tuple of (tool_calls, detected_format)
        """
        logger.debug("Attempting unified tool call extraction")
        
        # Try OpenAI format first (most common)
        try:
            tool_calls = UnifiedToolCallParser._extract_openai_format(response)
            if tool_calls:
                logger.debug(f"Successfully extracted {len(tool_calls)} tool calls using OpenAI format")
                return tool_calls, "openai"
        except Exception as e:
            logger.debug(f"OpenAI format extraction failed: {e}")
        
        # Try Anthropic format
        try:
            tool_calls = UnifiedToolCallParser._extract_anthropic_format(response)
            if tool_calls:
                logger.debug(f"Successfully extracted {len(tool_calls)} tool calls using Anthropic format")
                return tool_calls, "anthropic"
        except Exception as e:
            logger.debug(f"Anthropic format extraction failed: {e}")
        
        # Try Gemini format
        try:
            tool_calls = UnifiedToolCallParser._extract_gemini_format(response)
            if tool_calls:
                logger.debug(f"Successfully extracted {len(tool_calls)} tool calls using Gemini format")
                return tool_calls, "gemini"
        except Exception as e:
            logger.debug(f"Gemini format extraction failed: {e}")
        
        # Try generic object inspection
        try:
            tool_calls = UnifiedToolCallParser._extract_generic_format(response)
            if tool_calls:
                logger.debug(f"Successfully extracted {len(tool_calls)} tool calls using generic format")
                return tool_calls, "generic"
        except Exception as e:
            logger.debug(f"Generic format extraction failed: {e}")
        
        logger.debug("No tool calls found in any format")
        return [], "none"
    
    @staticmethod
    def _extract_openai_format(response: Any) -> List[Any]:
        """Extract tool calls from OpenAI format response."""
        tool_calls = []
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        call = SimpleNamespace()
                        call.name = tc.function.name
                        try:
                            call.args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        except json.JSONDecodeError:
                            call.args = {}
                        call.id = getattr(tc, 'id', None)
                        tool_calls.append(call)
        
        return tool_calls
    
    @staticmethod
    def _extract_anthropic_format(response: Any) -> List[Any]:
        """Extract tool calls from Anthropic format response."""
        tool_calls = []
        
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    call = SimpleNamespace()
                    call.name = block.name
                    call.args = block.input
                    call.id = getattr(block, 'id', None)
                    tool_calls.append(call)
        
        return tool_calls
    
    @staticmethod
    def _extract_gemini_format(response: Any) -> List[Any]:
        """Extract tool calls from Gemini format response."""
        tool_calls = []
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            call = SimpleNamespace()
                            call.name = fc.name
                            call.args = dict(fc.args) if fc.args else {}
                            call.id = getattr(fc, 'id', None)
                            tool_calls.append(call)
        
        return tool_calls
    
    @staticmethod
    def _extract_generic_format(response: Any) -> List[Any]:
        """Extract tool calls using generic object inspection."""
        tool_calls = []
        
        # Try to find tool calls in various locations
        potential_locations = [
            response,
            getattr(response, 'message', None),
            getattr(response, 'content', None),
            getattr(response, 'choices', [None])[0] if hasattr(response, 'choices') and response.choices else None,
        ]
        
        for location in potential_locations:
            if location is None:
                continue
                
            # Look for tool_calls attribute
            if hasattr(location, 'tool_calls') and location.tool_calls:
                for tc in location.tool_calls:
                    call = SimpleNamespace()
                    call.name = getattr(tc, 'name', getattr(getattr(tc, 'function', None), 'name', None))
                    call.args = getattr(tc, 'args', getattr(getattr(tc, 'function', None), 'arguments', {}))
                    if isinstance(call.args, str):
                        try:
                            call.args = json.loads(call.args)
                        except:
                            call.args = {}
                    call.id = getattr(tc, 'id', None)
                    if call.name:
                        tool_calls.append(call)
                        
            # Look for function_calls attribute
            if hasattr(location, 'function_calls') and location.function_calls:
                for fc in location.function_calls:
                    call = SimpleNamespace()
                    call.name = getattr(fc, 'name', None)
                    call.args = getattr(fc, 'args', getattr(fc, 'arguments', {}))
                    if isinstance(call.args, str):
                        try:
                            call.args = json.loads(call.args)
                        except:
                            call.args = {}
                    call.id = getattr(fc, 'id', None)
                    if call.name:
                        tool_calls.append(call)
        
        return tool_calls


class UnifiedToolConverter:
    """Tool converter that supports all formats and can adapt based on provider."""
    
    def __init__(self, provider_name: Optional[str] = None):
        # Import converters
        from cli_agent.utils.tool_conversion import (
            OpenAIStyleToolConverter,
            AnthropicToolConverter,
            GeminiToolConverter,
        )
        
        self.openai_converter = OpenAIStyleToolConverter()
        self.anthropic_converter = AnthropicToolConverter()
        self.gemini_converter = GeminiToolConverter()
        self.provider_name = provider_name
    
    def convert_tools(self, available_tools: Dict[str, Dict]) -> List[Dict]:
        """Convert tools to the optimal format based on provider."""
        # Choose converter based on provider name for best compatibility
        if self.provider_name:
            if self.provider_name.lower() in ['anthropic']:
                converter = self.anthropic_converter
                format_name = 'anthropic'
            elif self.provider_name.lower() in ['google', 'gemini']:
                converter = self.gemini_converter
                format_name = 'gemini'
            else:
                # OpenAI format for most providers:
                # - openai, openrouter, deepseek (explicitly OpenAI-compatible)
                # - ollama (uses OpenAI-compatible format for all models including Llama/Qwen)
                # - llama models (Llama 3.1+ use OpenAI-compatible format)
                # - qwen models (use OpenAI-compatible format, with Hermes-style support)
                converter = self.openai_converter
                format_name = 'openai'
        else:
            # No provider specified - use OpenAI format as most widely supported
            converter = self.openai_converter
            format_name = 'openai'
        
        tools = converter.convert_tools(available_tools)
        
        # Add format metadata to help with parsing
        for tool in tools:
            tool['_unified_format'] = True
            tool['_preferred_format'] = format_name
            tool['_supported_formats'] = ['openai', 'anthropic', 'gemini']
        
        logger.debug(f"Converted {len(tools)} tools using {format_name} format for provider {self.provider_name}")
        return tools