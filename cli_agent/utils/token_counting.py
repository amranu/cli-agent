"""
Comprehensive token counting system for all supported models.

This module provides accurate token counting for different model families using
their specific tokenizers. Falls back to reasonable estimations when exact
tokenizers aren't available.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy import flags - imports only happen when needed
TIKTOKEN_AVAILABLE = None
TRANSFORMERS_AVAILABLE = None  
ANTHROPIC_AVAILABLE = None

# Lazy import cache
_tiktoken = None
_transformers = None
_anthropic = None

def _get_tiktoken():
    """Lazy import tiktoken only when needed."""
    global _tiktoken, TIKTOKEN_AVAILABLE
    if _tiktoken is None and TIKTOKEN_AVAILABLE is not False:
        try:
            import tiktoken
            _tiktoken = tiktoken
            TIKTOKEN_AVAILABLE = True
        except ImportError:
            TIKTOKEN_AVAILABLE = False
            logger.warning("tiktoken not available - falling back to estimation for OpenAI models")
    return _tiktoken

def _get_transformers():
    """Lazy import transformers only when needed."""
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None and TRANSFORMERS_AVAILABLE is not False:
        try:
            from transformers import AutoTokenizer
            _transformers = AutoTokenizer
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
            logger.warning("transformers not available - falling back to estimation for some models")
    return _transformers

def _get_anthropic():
    """Lazy import anthropic only when needed."""
    global _anthropic, ANTHROPIC_AVAILABLE
    if _anthropic is None and ANTHROPIC_AVAILABLE is not False:
        try:
            import anthropic
            _anthropic = anthropic
            ANTHROPIC_AVAILABLE = True
        except ImportError:
            ANTHROPIC_AVAILABLE = False
            logger.warning("anthropic library not available - falling back to estimation for Claude models")
    return _anthropic


class TokenCounter:
    """Unified token counting interface for all model families."""
    
    def __init__(self):
        self._tokenizer_cache = {}
        self._model_mappings = self._build_model_mappings()
    
    def _build_model_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive mapping of model names to tokenizer info."""
        return {
            # OpenAI GPT Models
            "gpt-4": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-4-turbo": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-4-turbo-preview": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-4-turbo-2024-04-09": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-4-1106-preview": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-4-0613": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-4-0125-preview": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-4o": {"family": "openai", "encoding": "o200k_base"},
            "gpt-4o-mini": {"family": "openai", "encoding": "o200k_base"},
            "gpt-4o-2024-05-13": {"family": "openai", "encoding": "o200k_base"},
            "gpt-4o-2024-08-06": {"family": "openai", "encoding": "o200k_base"},
            "gpt-4o-2024-11-20": {"family": "openai", "encoding": "o200k_base"},
            "chatgpt-4o-latest": {"family": "openai", "encoding": "o200k_base"},
            "gpt-3.5-turbo": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-3.5-turbo-0125": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-3.5-turbo-1106": {"family": "openai", "encoding": "cl100k_base"},
            "gpt-3.5-turbo-16k": {"family": "openai", "encoding": "cl100k_base"},
            "o1-preview": {"family": "openai", "encoding": "o200k_base"},
            "o1-mini": {"family": "openai", "encoding": "o200k_base"},
            "o1-preview-2024-09-12": {"family": "openai", "encoding": "o200k_base"},
            "o1-mini-2024-09-12": {"family": "openai", "encoding": "o200k_base"},
            
            # Anthropic Claude Models
            "claude-3-5-sonnet-20241022": {"family": "anthropic", "model": "claude-3-5-sonnet-20241022"},
            "claude-3-5-sonnet-20240620": {"family": "anthropic", "model": "claude-3-5-sonnet-20240620"},
            "claude-3-5-haiku-20241022": {"family": "anthropic", "model": "claude-3-5-haiku-20241022"},
            "claude-3-opus-20240229": {"family": "anthropic", "model": "claude-3-opus-20240229"},
            "claude-3-sonnet-20240229": {"family": "anthropic", "model": "claude-3-sonnet-20240229"},
            "claude-3-haiku-20240307": {"family": "anthropic", "model": "claude-3-haiku-20240307"},
            
            # DeepSeek Models (OpenAI-compatible)
            "deepseek-chat": {"family": "openai", "encoding": "cl100k_base"},
            "deepseek-reasoner": {"family": "openai", "encoding": "cl100k_base"},
            "deepseek-coder": {"family": "openai", "encoding": "cl100k_base"},
            
            # Google Gemini Models (custom tokenization)
            "gemini-2.5-flash": {"family": "gemini", "model": "gemini-2.5-flash"},
            "gemini-2.5-pro": {"family": "gemini", "model": "gemini-2.5-pro"},
            "gemini-1.5-pro": {"family": "gemini", "model": "gemini-1.5-pro"},
            "gemini-1.5-flash": {"family": "gemini", "model": "gemini-1.5-flash"},
            "gemini-1.0-pro": {"family": "gemini", "model": "gemini-1.0-pro"},
            "gemini-pro": {"family": "gemini", "model": "gemini-pro"},
            "gemini-pro-vision": {"family": "gemini", "model": "gemini-pro-vision"},
            
            # Qwen Models
            "qwen-turbo": {"family": "qwen", "hf_model": "Qwen/Qwen2.5-7B"},
            "qwen-plus": {"family": "qwen", "hf_model": "Qwen/Qwen2.5-14B"},
            "qwen-max": {"family": "qwen", "hf_model": "Qwen/Qwen2.5-72B"},
            
            # Other common models that might appear through OpenRouter
            "llama-2-7b": {"family": "llama", "hf_model": "meta-llama/Llama-2-7b-chat-hf"},
            "llama-2-13b": {"family": "llama", "hf_model": "meta-llama/Llama-2-13b-chat-hf"},
            "llama-2-70b": {"family": "llama", "hf_model": "meta-llama/Llama-2-70b-chat-hf"},
            "llama-3-8b": {"family": "llama", "hf_model": "meta-llama/Meta-Llama-3-8B"},
            "llama-3-70b": {"family": "llama", "hf_model": "meta-llama/Meta-Llama-3-70B"},
            "mistral-7b": {"family": "mistral", "hf_model": "mistralai/Mistral-7B-v0.1"},
            "mixtral-8x7b": {"family": "mistral", "hf_model": "mistralai/Mixtral-8x7B-v0.1"},
        }
    
    @lru_cache(maxsize=128)
    def _get_tokenizer(self, model_name: str) -> Optional[Any]:
        """Get appropriate tokenizer for a model, with caching."""
        if model_name in self._tokenizer_cache:
            return self._tokenizer_cache[model_name]
        
        # Get model info
        model_info = self._get_model_info(model_name)
        family = model_info["family"]
        tokenizer = None
        
        try:
            if family == "openai":
                tiktoken = _get_tiktoken()
                if tiktoken:
                    encoding = model_info.get("encoding", "cl100k_base")
                    tokenizer = tiktoken.get_encoding(encoding)
                    logger.debug(f"Loaded tiktoken encoding {encoding} for {model_name}")
                
            elif family == "anthropic":
                anthropic = _get_anthropic()
                if anthropic:
                    # Anthropic provides a tokenizer through their library
                    tokenizer = anthropic.Anthropic().get_tokenizer()
                    logger.debug(f"Loaded Anthropic tokenizer for {model_name}")
                
            elif family in ["llama", "mistral", "qwen"]:
                AutoTokenizer = _get_transformers()
                if AutoTokenizer:
                    hf_model = model_info.get("hf_model")
                    if hf_model:
                        tokenizer = AutoTokenizer.from_pretrained(hf_model)
                        logger.debug(f"Loaded HuggingFace tokenizer {hf_model} for {model_name}")
                    
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            tokenizer = None
        
        self._tokenizer_cache[model_name] = tokenizer
        return tokenizer
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model info, with pattern matching for unknown models."""
        # Direct lookup
        if model_name in self._model_mappings:
            return self._model_mappings[model_name]
        
        # Pattern matching for variations
        model_lower = model_name.lower()
        
        # GPT models
        if any(pattern in model_lower for pattern in ["gpt-4o", "gpt4o"]):
            return {"family": "openai", "encoding": "o200k_base"}
        elif any(pattern in model_lower for pattern in ["gpt-4", "gpt4"]):
            return {"family": "openai", "encoding": "cl100k_base"}
        elif any(pattern in model_lower for pattern in ["gpt-3.5", "gpt3.5"]):
            return {"family": "openai", "encoding": "cl100k_base"}
        elif "o1" in model_lower:
            return {"family": "openai", "encoding": "o200k_base"}
        
        # Claude models
        elif any(pattern in model_lower for pattern in ["claude", "sonnet", "haiku", "opus"]):
            return {"family": "anthropic", "model": model_name}
        
        # DeepSeek models
        elif "deepseek" in model_lower:
            return {"family": "openai", "encoding": "cl100k_base"}
        
        # Gemini models
        elif any(pattern in model_lower for pattern in ["gemini", "flash", "pro"]):
            return {"family": "gemini", "model": model_name}
        
        # Qwen models
        elif "qwen" in model_lower:
            return {"family": "qwen", "hf_model": "Qwen/Qwen2.5-7B"}
        
        # Llama models
        elif "llama" in model_lower:
            return {"family": "llama", "hf_model": "meta-llama/Llama-2-7b-chat-hf"}
        
        # Mistral models
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return {"family": "mistral", "hf_model": "mistralai/Mistral-7B-v0.1"}
        
        # Default to OpenAI-compatible for unknown models
        return {"family": "openai", "encoding": "cl100k_base"}
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text for the specified model."""
        if not text:
            return 0
        
        # Try to get exact tokenizer
        tokenizer = self._get_tokenizer(model_name)
        model_info = self._get_model_info(model_name)
        family = model_info["family"]
        
        if tokenizer is not None:
            try:
                if family == "openai":
                    # tiktoken tokenizer
                    return len(tokenizer.encode(text))
                elif family == "anthropic":
                    # Anthropic tokenizer
                    return tokenizer.count_tokens(text)
                elif family in ["llama", "mistral", "qwen"]:
                    # HuggingFace tokenizer
                    return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception as e:
                logger.warning(f"Tokenizer failed for {model_name}: {e}, falling back to estimation")
        
        # Fallback to improved estimation based on model family
        return self._estimate_tokens(text, family)
    
    def _estimate_tokens(self, text: str, family: str) -> int:
        """Improved token estimation based on model family characteristics."""
        if not text:
            return 0
        
        # Different model families have different token-to-character ratios
        if family == "openai":
            # GPT models: ~0.75 tokens per word, ~4 chars per token on average
            return max(1, len(text) // 4)
        elif family == "anthropic":
            # Claude models: similar to GPT but slightly more efficient
            return max(1, len(text) // 4)
        elif family == "gemini":
            # Gemini models: more efficient tokenization
            return max(1, len(text) // 5)
        elif family in ["llama", "mistral"]:
            # LLaMA/Mistral: less efficient than GPT
            return max(1, len(text) // 3)
        elif family == "qwen":
            # Qwen: efficient for multilingual, especially Chinese
            return max(1, len(text) // 4)
        else:
            # Default conservative estimate
            return max(1, len(text) // 3)
    
    def count_message_tokens(self, message: Dict[str, Any], model_name: str) -> int:
        """Count tokens in a single message, accounting for structure."""
        if not message:
            return 0
        
        total_tokens = 0
        
        # Role overhead (varies by model)
        model_info = self._get_model_info(model_name)
        family = model_info["family"]
        
        if family == "openai":
            # OpenAI format: role + content + structure tokens
            total_tokens += 4  # Base message overhead
        elif family == "anthropic":
            # Anthropic format: less overhead
            total_tokens += 3
        else:
            # Default overhead
            total_tokens += 3
        
        # Count content tokens
        content = message.get("content", "")
        if isinstance(content, str):
            total_tokens += self.count_tokens(content, model_name)
        elif isinstance(content, list):
            # Multi-modal content
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        total_tokens += self.count_tokens(item.get("text", ""), model_name)
                    elif item.get("type") == "image":
                        # Image token estimation (very rough)
                        total_tokens += 85  # Base image token cost
                
        # Tool call tokens (if present)
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_args = tool_call.get("function", {}).get("arguments", "")
                    total_tokens += self.count_tokens(tool_name, model_name)
                    total_tokens += self.count_tokens(str(tool_args), model_name)
                    total_tokens += 10  # Tool call overhead
        
        return max(total_tokens, 1)
    
    def count_conversation_tokens(self, messages: List[Dict[str, Any]], model_name: str) -> int:
        """Count total tokens in a conversation."""
        if not messages:
            return 0
        
        total_tokens = 0
        
        # Count tokens for each message
        for message in messages:
            total_tokens += self.count_message_tokens(message, model_name)
        
        # Add conversation-level overhead
        model_info = self._get_model_info(model_name)
        family = model_info["family"]
        
        if family == "openai":
            total_tokens += 3  # Conversation formatting
        elif family == "anthropic":
            total_tokens += 5  # System prompt handling
        
        return total_tokens
    
    def estimate_response_tokens(self, model_name: str, max_tokens: Optional[int] = None) -> int:
        """Estimate tokens that will be used for response."""
        if max_tokens:
            return min(max_tokens, 4096)  # Cap at reasonable response size
        
        # Default response estimates by model family
        model_info = self._get_model_info(model_name)
        family = model_info["family"]
        
        if family == "gemini":
            return 1024  # Gemini tends to give longer responses
        else:
            return 512   # Conservative default
    
    def get_effective_context_limit(self, model_name: str, context_length: int, max_tokens: Optional[int] = None) -> int:
        """Get effective context limit after reserving space for response."""
        response_reserve = self.estimate_response_tokens(model_name, max_tokens)
        return max(context_length - response_reserve, 1000)  # Always leave at least 1000 tokens


# Global instance for easy access
token_counter = TokenCounter()

# Convenience functions
def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in text for the specified model."""
    return token_counter.count_tokens(text, model_name)

def count_message_tokens(message: Dict[str, Any], model_name: str) -> int:
    """Count tokens in a single message."""
    return token_counter.count_message_tokens(message, model_name)

def count_conversation_tokens(messages: List[Dict[str, Any]], model_name: str) -> int:
    """Count total tokens in a conversation."""
    return token_counter.count_conversation_tokens(messages, model_name)

def get_effective_context_limit(model_name: str, context_length: int, max_tokens: Optional[int] = None) -> int:
    """Get effective context limit after reserving space for response."""
    return token_counter.get_effective_context_limit(model_name, context_length, max_tokens)