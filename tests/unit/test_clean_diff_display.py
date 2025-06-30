"""Tests for the clean diff display functionality."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from cli_agent.core.diff_display import CleanDiffDisplay, get_clean_diff_display


class TestCleanDiffDisplay:
    """Test the CleanDiffDisplay class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clean_diff = CleanDiffDisplay()

    def test_initialization(self):
        """Test CleanDiffDisplay initialization."""
        assert self.clean_diff.terminal_state is not None
        assert self.clean_diff.in_clean_mode is False

    @patch('cli_agent.core.diff_display.get_terminal_state')
    def test_can_use_clean_display(self, mock_get_terminal_state):
        """Test clean display capability detection."""
        # Mock terminal state
        mock_terminal_state = Mock()
        mock_terminal_state.can_clear_screen.return_value = True
        mock_get_terminal_state.return_value = mock_terminal_state
        
        clean_diff = CleanDiffDisplay()
        assert clean_diff.can_use_clean_display() is True
        
        # Test when clearing not supported
        mock_terminal_state.can_clear_screen.return_value = False
        assert clean_diff.can_use_clean_display() is False

    @patch('cli_agent.core.diff_display.get_terminal_state')
    def test_show_single_edit_diff_not_supported(self, mock_get_terminal_state):
        """Test single edit diff when clean display not supported."""
        # Mock terminal state that doesn't support clearing
        mock_terminal_state = Mock()
        mock_terminal_state.can_clear_screen.return_value = False
        mock_get_terminal_state.return_value = mock_terminal_state
        
        clean_diff = CleanDiffDisplay()
        
        result = clean_diff.show_single_edit_diff(
            file_path="test.py",
            old_text="old",
            new_text="new"
        )
        
        assert result is None

    @patch('cli_agent.core.diff_display.get_terminal_state')
    @patch('builtins.input')
    def test_show_single_edit_diff_user_confirms(self, mock_input, mock_get_terminal_state):
        """Test single edit diff when user confirms changes."""
        # Mock terminal state that supports clearing
        mock_terminal_state = Mock()
        mock_terminal_state.can_clear_screen.return_value = True
        mock_terminal_state.clear_screen.return_value = True
        mock_terminal_state.position_vertically_centered.return_value = "test content"
        mock_terminal_state.restore_screen.return_value = True
        mock_get_terminal_state.return_value = mock_terminal_state
        
        # Mock user input
        mock_input.return_value = "y"
        
        clean_diff = CleanDiffDisplay()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tf:
            tf.write("print('old text')")
            tf.flush()
            
            try:
                result = clean_diff.show_single_edit_diff(
                    file_path=tf.name,
                    old_text="old text",
                    new_text="new text"
                )
                
                assert result is True
                mock_terminal_state.clear_screen.assert_called_once()
                mock_terminal_state.restore_screen.assert_called_once()
                
            finally:
                os.unlink(tf.name)

    @patch('cli_agent.core.diff_display.get_terminal_state')
    @patch('builtins.input')
    def test_show_single_edit_diff_user_cancels(self, mock_input, mock_get_terminal_state):
        """Test single edit diff when user cancels changes."""
        # Mock terminal state that supports clearing
        mock_terminal_state = Mock()
        mock_terminal_state.can_clear_screen.return_value = True
        mock_terminal_state.clear_screen.return_value = True
        mock_terminal_state.position_vertically_centered.return_value = "test content"
        mock_terminal_state.restore_screen.return_value = True
        mock_get_terminal_state.return_value = mock_terminal_state
        
        # Mock user input
        mock_input.return_value = "n"
        
        clean_diff = CleanDiffDisplay()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tf:
            tf.write("print('old text')")
            tf.flush()
            
            try:
                result = clean_diff.show_single_edit_diff(
                    file_path=tf.name,
                    old_text="old text",
                    new_text="new text"
                )
                
                assert result is False
                
            finally:
                os.unlink(tf.name)

    @patch('cli_agent.core.diff_display.get_terminal_state')
    @patch('builtins.input')
    def test_show_multiedit_diff(self, mock_input, mock_get_terminal_state):
        """Test multiedit diff display."""
        # Mock terminal state that supports clearing
        mock_terminal_state = Mock()
        mock_terminal_state.can_clear_screen.return_value = True
        mock_terminal_state.clear_screen.return_value = True
        mock_terminal_state.position_vertically_centered.return_value = "test content"
        mock_terminal_state.restore_screen.return_value = True
        mock_get_terminal_state.return_value = mock_terminal_state
        
        # Mock user input
        mock_input.return_value = "yes"
        
        clean_diff = CleanDiffDisplay()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tf:
            tf.write("print('old text 1')\nprint('old text 2')")
            tf.flush()
            
            try:
                edits = [
                    {"old_string": "old text 1", "new_string": "new text 1"},
                    {"old_string": "old text 2", "new_string": "new text 2"}
                ]
                
                result = clean_diff.show_multiedit_diff(
                    file_path=tf.name,
                    edits=edits
                )
                
                assert result is True
                
            finally:
                os.unlink(tf.name)

    def test_restore_terminal_not_in_clean_mode(self):
        """Test restore terminal when not in clean mode."""
        result = self.clean_diff.restore_terminal()
        assert result is True

    @patch('cli_agent.core.diff_display.get_terminal_state')
    def test_restore_terminal_in_clean_mode(self, mock_get_terminal_state):
        """Test restore terminal when in clean mode."""
        mock_terminal_state = Mock()
        mock_terminal_state.restore_screen.return_value = True
        mock_get_terminal_state.return_value = mock_terminal_state
        
        clean_diff = CleanDiffDisplay()
        clean_diff.in_clean_mode = True
        
        result = clean_diff.restore_terminal()
        assert result is True
        assert clean_diff.in_clean_mode is False
        mock_terminal_state.restore_screen.assert_called_once()

    def test_get_user_confirmation_variations(self):
        """Test various user input responses."""
        clean_diff = CleanDiffDisplay()
        
        # Test different positive responses
        with patch('builtins.input', side_effect=['y']):
            assert clean_diff._get_user_confirmation() is True
            
        with patch('builtins.input', side_effect=['yes']):
            assert clean_diff._get_user_confirmation() is True
            
        with patch('builtins.input', side_effect=['Y']):
            assert clean_diff._get_user_confirmation() is True
            
        # Test different negative responses
        with patch('builtins.input', side_effect=['n']):
            assert clean_diff._get_user_confirmation() is False
            
        with patch('builtins.input', side_effect=['no']):
            assert clean_diff._get_user_confirmation() is False
            
        with patch('builtins.input', side_effect=['N']):
            assert clean_diff._get_user_confirmation() is False

    def test_get_user_confirmation_invalid_then_valid(self):
        """Test user confirmation with invalid input followed by valid input."""
        clean_diff = CleanDiffDisplay()
        
        # Mock print to avoid output during tests
        with patch('builtins.print'):
            with patch('builtins.input', side_effect=['invalid', 'maybe', 'y']):
                assert clean_diff._get_user_confirmation() is True

    def test_get_user_confirmation_keyboard_interrupt(self):
        """Test user confirmation with keyboard interrupt."""
        clean_diff = CleanDiffDisplay()
        
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            assert clean_diff._get_user_confirmation() is False

    def test_get_user_confirmation_eof_error(self):
        """Test user confirmation with EOF error."""
        clean_diff = CleanDiffDisplay()
        
        with patch('builtins.input', side_effect=EOFError):
            assert clean_diff._get_user_confirmation() is False

    def test_global_instance(self):
        """Test that the global instance function works."""
        instance1 = get_clean_diff_display()
        instance2 = get_clean_diff_display()
        
        # Should return the same instance
        assert instance1 is instance2
        assert isinstance(instance1, CleanDiffDisplay)

    def test_create_diff_content(self):
        """Test diff content creation."""
        clean_diff = CleanDiffDisplay()
        
        diff_output = "sample diff output"
        content = clean_diff._create_diff_content(
            diff_output, "test.py", "single edit"
        )
        
        assert "📝 File Edit Preview" in content
        assert "test.py" in content
        assert "single edit" in content
        assert "sample diff output" in content
        assert "Do you want to proceed" in content
        assert "Your choice:" in content

    @patch('cli_agent.utils.diff_display.ColoredDiffDisplay.show_replace_diff')
    def test_capture_diff_output(self, mock_show_diff):
        """Test capturing diff output."""
        mock_show_diff.return_value = True
        
        clean_diff = CleanDiffDisplay()
        
        # Since we're capturing stdout, the actual content depends on ColoredDiffDisplay
        result = clean_diff._capture_diff_output(
            "test.py", "old", "new", "file content", 3
        )
        
        # Should return some content (even if empty due to mocking)
        assert result is not None
        mock_show_diff.assert_called_once()

    @patch('cli_agent.utils.diff_display.ColoredDiffDisplay.show_replace_diff')
    def test_capture_diff_output_failure(self, mock_show_diff):
        """Test capturing diff output when diff generation fails."""
        mock_show_diff.return_value = False
        
        clean_diff = CleanDiffDisplay()
        
        result = clean_diff._capture_diff_output(
            "test.py", "old", "new", "file content", 3
        )
        
        assert result is None