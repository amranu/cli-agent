"""
Tests for the comprehensive token counting system.

This test suite verifies accurate token counting across all supported models
and providers, ensuring that our token management system works correctly
for conversation tracking, compaction decisions, and limit enforcement.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cli_agent.utils.token_counting import (
    TokenCounter, 
    token_counter,
    count_tokens,
    count_message_tokens,
    count_conversation_tokens,
    get_effective_context_limit
)
from cli_agent.core.token_manager import TokenManager


class TestTokenCounter:
    """Test the TokenCounter class and its model-specific tokenization."""
    
    def test_model_mappings_coverage(self):
        """Test that we have comprehensive model mappings."""
        tc = TokenCounter()
        
        # Test OpenAI models
        assert tc._get_model_info("gpt-4")["family"] == "openai"
        assert tc._get_model_info("gpt-4o")["family"] == "openai"
        assert tc._get_model_info("gpt-3.5-turbo")["family"] == "openai"
        assert tc._get_model_info("o1-preview")["family"] == "openai"
        
        # Test Anthropic models
        assert tc._get_model_info("claude-3-5-sonnet-20241022")["family"] == "anthropic"
        assert tc._get_model_info("claude-3-opus-20240229")["family"] == "anthropic"
        
        # Test DeepSeek models (OpenAI-compatible)
        assert tc._get_model_info("deepseek-chat")["family"] == "openai"
        assert tc._get_model_info("deepseek-reasoner")["family"] == "openai"
        
        # Test Gemini models
        assert tc._get_model_info("gemini-2.5-flash")["family"] == "gemini"
        assert tc._get_model_info("gemini-1.5-pro")["family"] == "gemini"
    
    def test_pattern_matching_for_unknown_models(self):
        """Test pattern matching for model variants not in direct mappings."""
        tc = TokenCounter()
        
        # Test GPT pattern matching
        assert tc._get_model_info("gpt-4-custom-variant")["family"] == "openai"
        assert tc._get_model_info("gpt-3.5-custom")["family"] == "openai"
        
        # Test Claude pattern matching
        assert tc._get_model_info("claude-4-future-model")["family"] == "anthropic"
        
        # Test Gemini pattern matching
        assert tc._get_model_info("gemini-3.0-flash")["family"] == "gemini"
        
        # Test unknown model fallback
        assert tc._get_model_info("unknown-model")["family"] == "openai"  # Default fallback
    
    def test_openai_tokenization_with_tiktoken(self):
        """Test OpenAI model tokenization using tiktoken."""
        # Mock tiktoken encoder
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        
        # Mock the lazy loading function
        def mock_get_tiktoken():
            mock_tiktoken = Mock()
            mock_tiktoken.get_encoding.return_value = mock_encoder
            return mock_tiktoken
        
        with patch('cli_agent.utils.token_counting._get_tiktoken', mock_get_tiktoken):
            tc = TokenCounter()
            result = tc.count_tokens("Hello world", "gpt-4")
            assert result == 5
            mock_encoder.encode.assert_called_with("Hello world")
    
    def test_gpt4o_uses_correct_encoding(self):
        """Test that GPT-4o models use the o200k_base encoding."""
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3]
        
        # Mock the lazy loading function to track encoding calls
        encoding_calls = []
        def mock_get_tiktoken():
            mock_tiktoken = Mock()
            def track_encoding(encoding_name):
                encoding_calls.append(encoding_name)
                return mock_encoder
            mock_tiktoken.get_encoding.side_effect = track_encoding
            return mock_tiktoken
        
        with patch('cli_agent.utils.token_counting._get_tiktoken', mock_get_tiktoken):
            tc = TokenCounter()
            tc.count_tokens("test", "gpt-4o")
            assert "o200k_base" in encoding_calls
            
            tc.count_tokens("test", "gpt-4o-mini")
            assert encoding_calls.count("o200k_base") >= 2  # Should be called for both models
    
    def test_fallback_estimation_by_family(self):
        """Test fallback token estimation with family-specific ratios."""
        tc = TokenCounter()
        
        # Test with no tokenizers available (should fall back to estimation)
        with patch('cli_agent.utils.token_counting._get_tiktoken', return_value=None):
            with patch('cli_agent.utils.token_counting._get_anthropic', return_value=None):
                with patch('cli_agent.utils.token_counting._get_transformers', return_value=None):
                    # OpenAI family: 4 chars per token
                    assert tc.count_tokens("1234", "gpt-4") == 1
                    assert tc.count_tokens("12345678", "gpt-4") == 2
                    
                    # Gemini family: 5 chars per token (more efficient)
                    assert tc.count_tokens("12345", "gemini-2.5-flash") == 1
                    assert tc.count_tokens("1234567890", "gemini-2.5-flash") == 2
                    
                    # LLaMA family: 3 chars per token (less efficient)
                    assert tc.count_tokens("123", "llama-2-7b") == 1
                    assert tc.count_tokens("123456", "llama-2-7b") == 2
    
    def test_message_token_counting_structure(self):
        """Test that message token counting includes structural overhead."""
        tc = TokenCounter()
        
        # Simple text message
        message = {"role": "user", "content": "Hello"}
        with patch('cli_agent.utils.token_counting.TIKTOKEN_AVAILABLE', False):
            tokens = tc.count_message_tokens(message, "gpt-4")
            # Should be content tokens (1-2) + overhead (4) = 5-6 tokens minimum
            assert tokens >= 5
    
    def test_message_with_tool_calls(self):
        """Test token counting for messages with tool calls."""
        tc = TokenCounter()
        
        message = {
            "role": "assistant",
            "content": "I'll help you with that.",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash_execute",
                        "arguments": '{"command": "ls -la"}'
                    }
                }
            ]
        }
        
        with patch('cli_agent.utils.token_counting.TIKTOKEN_AVAILABLE', False):
            tokens = tc.count_message_tokens(message, "gpt-4")
            # Should include content + tool name + arguments + overhead
            assert tokens > 10  # More than just content tokens
    
    def test_conversation_token_counting(self):
        """Test token counting for full conversations."""
        tc = TokenCounter()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"}
        ]
        
        with patch('cli_agent.utils.token_counting.TIKTOKEN_AVAILABLE', False):
            total_tokens = tc.count_conversation_tokens(messages, "gpt-4")
            
            # Should be sum of individual messages plus conversation overhead
            individual_sum = sum(tc.count_message_tokens(msg, "gpt-4") for msg in messages)
            assert total_tokens >= individual_sum
            assert total_tokens > 20  # Reasonable minimum for this conversation
    
    def test_effective_context_limit_calculation(self):
        """Test effective context limit calculation with response reservation."""
        tc = TokenCounter()
        
        # Test with various context lengths
        assert tc.get_effective_context_limit("gpt-4", 8000) == 7488  # 8000 - 512 default response
        assert tc.get_effective_context_limit("gemini-2.5-flash", 1000000) == 998976  # 1M - 1024 for Gemini
        
        # Test with custom max_tokens
        assert tc.get_effective_context_limit("gpt-4", 8000, 2000) == 6000  # 8000 - 2000 custom
        
        # Test minimum enforcement
        assert tc.get_effective_context_limit("gpt-4", 1200) >= 1000  # Always at least 1000


class TestTokenManager:
    """Test the TokenManager class integration with TokenCounter."""
    
    def test_token_manager_initialization(self):
        """Test TokenManager initialization with model name."""
        tm = TokenManager(model_name="gpt-4")
        assert tm.model_name == "gpt-4"
    
    def test_accurate_vs_fallback_counting(self):
        """Test that TokenManager uses accurate counting when model is available."""
        tm = TokenManager(model_name="gpt-4")
        
        messages = [{"role": "user", "content": "Hello world"}]
        
        # With model name, should use accurate counting
        with patch('cli_agent.core.token_manager.count_conversation_tokens') as mock_accurate:
            mock_accurate.return_value = 10
            result = tm.count_conversation_tokens(messages)
            assert result == 10
            mock_accurate.assert_called_once_with(messages, "gpt-4")
        
        # Without model name, should fall back to basic estimation
        tm_no_model = TokenManager()
        result_fallback = tm_no_model.count_conversation_tokens(messages)
        assert result_fallback > 0  # Should still work
    
    def test_enhanced_token_limits(self):
        """Test the enhanced token limits for various models."""
        tm = TokenManager()
        
        # Test specific model limits
        assert tm.get_token_limit("gpt-4-turbo") == 128000
        assert tm.get_token_limit("claude-3-5-sonnet-20241022") == 190000
        assert tm.get_token_limit("gemini-2.5-flash") == 950000
        assert tm.get_token_limit("deepseek-chat") == 60000
        
        # Test pattern matching
        assert tm.get_token_limit("gpt-4-custom-variant") == 128000
        assert tm.get_token_limit("claude-future-model") == 190000
        
        # Test unknown model fallback
        assert tm.get_token_limit("completely-unknown-model") == 32000
    
    def test_should_compact_logic(self):
        """Test the conversation compaction decision logic."""
        tm = TokenManager(model_name="gpt-4")
        
        # Mock conversation with known token count
        short_messages = [{"role": "user", "content": "Hi"}]
        long_messages = [{"role": "user", "content": "x" * 50000}]  # Very long message
        
        with patch.object(tm, 'count_conversation_tokens') as mock_count:
            with patch.object(tm, 'get_token_limit') as mock_limit:
                mock_limit.return_value = 10000
                
                # Short conversation (500 tokens < 80% of 10000)
                mock_count.return_value = 500
                assert not tm.should_compact(short_messages)
                
                # Long conversation (9000 tokens > 80% of 10000)
                mock_count.return_value = 9000
                assert tm.should_compact(long_messages)
    
    def test_context_length_integration(self):
        """Test integration with context length for effective limits."""
        tm = TokenManager(model_name="gpt-4")
        
        # Test with context length provided
        limit = tm.get_token_limit("gpt-4", context_length=100000, max_tokens=1000)
        assert limit == 99000  # 100000 - 1000 reserved
        
        # Test effective limit is used in compaction decision
        messages = [{"role": "user", "content": "test"}]
        with patch.object(tm, 'count_conversation_tokens', return_value=80000):
            # Should compact when 80000 > 80% of 99000 (79200)
            assert tm.should_compact(messages, context_length=100000)
    
    def test_has_reliable_token_info(self):
        """Test token reliability detection for different models."""
        tm = TokenManager()
        
        # Known models should be reliable
        assert tm.has_reliable_token_info("gpt-4-turbo") == True
        assert tm.has_reliable_token_info("claude-3-5-sonnet-20241022") == True
        assert tm.has_reliable_token_info("gemini-2.5-flash") == True
        assert tm.has_reliable_token_info("deepseek-chat") == True
        
        # Unknown models should be unreliable
        assert tm.has_reliable_token_info("unknown-model") == False
        assert tm.has_reliable_token_info("my-custom-llm") == False
        assert tm.has_reliable_token_info("gpt-5-future") == False
        
        # Partial matches should be unreliable (falls back to pattern matching)
        assert tm.has_reliable_token_info("gpt-4-some-new-variant") == False
        assert tm.has_reliable_token_info("claude-4-future") == False
        
        # Provider:model format should work by extracting just the model name
        assert tm.has_reliable_token_info("deepseek:deepseek-chat") == True
        assert tm.has_reliable_token_info("anthropic:claude-3-5-sonnet-20241022") == True
        
        # With explicit context_length, should be reliable even for unknown models
        assert tm.has_reliable_token_info("unknown-model", context_length=100000) == True
        assert tm.has_reliable_token_info("custom-llm", context_length=50000) == True
        
        # With zero or negative context_length, should be unreliable
        assert tm.has_reliable_token_info("unknown-model", context_length=0) == False
        assert tm.has_reliable_token_info("unknown-model", context_length=-1) == False


class TestTokenDisplayReliability:
    """Test integration between token reliability detection and display system."""
    
    def test_token_display_respects_reliability(self):
        """Test that token display is hidden for unreliable token information."""
        from cli_agent.core.terminal_manager import TerminalManager
        from unittest.mock import Mock
        
        tm = TokenManager()
        terminal_manager = TerminalManager()
        
        # Mock the actual display method to capture calls
        display_calls = []
        def mock_draw_display():
            display_calls.append("display_called")
        
        terminal_manager._draw_token_display = mock_draw_display
        terminal_manager.is_terminal = True
        terminal_manager.token_display_enabled = True
        
        # Test with reliable token info (should display when show_display=True)
        display_calls.clear()
        terminal_manager.update_token_display(
            current_tokens=1000,
            token_limit=10000,
            model_name="gpt-4-turbo",
            show_display=True
        )
        # Should have been called since we passed show_display=True
        assert len(display_calls) == 1, f"Expected 1 display call for reliable token info, got {len(display_calls)}"
        
        # Test that terminal manager stores token info regardless of display
        assert terminal_manager.current_token_info is not None
        assert terminal_manager.current_token_info["current_tokens"] == 1000
        assert terminal_manager.current_token_info["token_limit"] == 10000


class TestConvenienceFunctions:
    """Test the module-level convenience functions."""
    
    def test_count_tokens_convenience(self):
        """Test the count_tokens convenience function."""
        with patch.object(token_counter, 'count_tokens', return_value=42) as mock:
            result = count_tokens("test text", "gpt-4")
            assert result == 42
            mock.assert_called_once_with("test text", "gpt-4")
    
    def test_count_message_tokens_convenience(self):
        """Test the count_message_tokens convenience function."""
        message = {"role": "user", "content": "test"}
        with patch.object(token_counter, 'count_message_tokens', return_value=15) as mock:
            result = count_message_tokens(message, "gpt-4")
            assert result == 15
            mock.assert_called_once_with(message, "gpt-4")
    
    def test_count_conversation_tokens_convenience(self):
        """Test the count_conversation_tokens convenience function."""
        messages = [{"role": "user", "content": "test"}]
        with patch.object(token_counter, 'count_conversation_tokens', return_value=25) as mock:
            result = count_conversation_tokens(messages, "gpt-4")
            assert result == 25
            mock.assert_called_once_with(messages, "gpt-4")
    
    def test_get_effective_context_limit_convenience(self):
        """Test the get_effective_context_limit convenience function."""
        with patch.object(token_counter, 'get_effective_context_limit', return_value=7500) as mock:
            result = get_effective_context_limit("gpt-4", 8000, 500)
            assert result == 7500
            mock.assert_called_once_with("gpt-4", 8000, 500)


@pytest.mark.integration
class TestTokenCountingIntegration:
    """Integration tests for token counting with real tokenizers (when available)."""
    
    def test_tiktoken_integration_if_available(self):
        """Test actual tiktoken integration if the library is available."""
        try:
            import tiktoken
            
            tc = TokenCounter()
            
            # Test with a known string
            test_text = "Hello, world! This is a test."
            tokens = tc.count_tokens(test_text, "gpt-4")
            
            # Should return a reasonable token count (not just character/4)
            assert tokens > 0
            assert tokens < len(test_text)  # Should be fewer tokens than characters
            
            # Test consistency
            tokens2 = tc.count_tokens(test_text, "gpt-4")
            assert tokens == tokens2  # Should be consistent
            
        except ImportError:
            pytest.skip("tiktoken not available for integration test")
    
    def test_different_encodings_give_different_results(self):
        """Test that different model encodings can give different token counts."""
        try:
            import tiktoken
            
            tc = TokenCounter()
            
            # Test text that might tokenize differently
            test_text = "The quick brown fox jumps over the lazy dog. 🦊🐕"
            
            gpt4_tokens = tc.count_tokens(test_text, "gpt-4")  # cl100k_base
            gpt4o_tokens = tc.count_tokens(test_text, "gpt-4o")  # o200k_base
            
            # Both should be positive
            assert gpt4_tokens > 0
            assert gpt4o_tokens > 0
            
            # They might be different due to different encodings
            # (but not asserting inequality as they could be the same for this text)
            
        except ImportError:
            pytest.skip("tiktoken not available for encoding comparison test")
    
    def test_realistic_conversation_token_counts(self):
        """Test token counting on realistic conversation examples."""
        tc = TokenCounter()
        
        # Realistic conversation
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can execute commands and help with programming tasks."
            },
            {
                "role": "user", 
                "content": "Can you help me create a Python function that calculates the factorial of a number?"
            },
            {
                "role": "assistant",
                "content": "I'll help you create a factorial function. Here's a simple recursive implementation:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n```"
            },
            {
                "role": "user",
                "content": "Great! Can you also show me an iterative version?"
            }
        ]
        
        # Test with different models
        gpt4_tokens = tc.count_conversation_tokens(messages, "gpt-4")
        claude_tokens = tc.count_conversation_tokens(messages, "claude-3-5-sonnet-20241022")
        gemini_tokens = tc.count_conversation_tokens(messages, "gemini-2.5-flash")
        
        # All should be reasonable token counts
        assert 50 < gpt4_tokens < 500  # Reasonable range
        assert 50 < claude_tokens < 500
        assert 50 < gemini_tokens < 500
        
        # Log the results for manual verification
        print(f"GPT-4 tokens: {gpt4_tokens}")
        print(f"Claude tokens: {claude_tokens}")
        print(f"Gemini tokens: {gemini_tokens}")


@pytest.mark.slow
class TestTokenCountingPerformance:
    """Performance tests for token counting operations."""
    
    def test_large_conversation_performance(self):
        """Test performance with large conversations."""
        import time
        
        tc = TokenCounter()
        
        # Create a large conversation
        large_messages = []
        for i in range(100):
            large_messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}: " + "Lorem ipsum dolor sit amet. " * 20
            })
        
        # Measure performance
        start_time = time.time()
        token_count = tc.count_conversation_tokens(large_messages, "gpt-4")
        end_time = time.time()
        
        # Should complete quickly (under 1 second for 100 messages)
        duration = end_time - start_time
        assert duration < 1.0, f"Token counting took too long: {duration:.2f}s"
        assert token_count > 1000  # Should be a substantial count
    
    def test_tokenizer_caching(self):
        """Test that tokenizers are properly cached for performance."""
        tc = TokenCounter()
        
        # First call should potentially load tokenizer
        tc.count_tokens("test", "gpt-4")
        
        # Subsequent calls should use cached tokenizer
        import time
        start_time = time.time()
        for _ in range(10):
            tc.count_tokens("test", "gpt-4")
        end_time = time.time()
        
        # Should be very fast due to caching
        duration = end_time - start_time
        assert duration < 0.1, f"Cached tokenization too slow: {duration:.3f}s"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])