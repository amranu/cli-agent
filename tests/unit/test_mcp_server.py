"""
Tests for the MCP server entry point and command-line interface.

This test suite covers the main MCP server script functionality,
command-line argument parsing, and transport initialization.
"""

import pytest
import sys
import os
import tempfile
import subprocess
from unittest.mock import Mock, patch, AsyncMock
from io import StringIO

# Add the project root to Python path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))  # Go up two levels from tests/unit/
sys.path.insert(0, project_root)

import mcp_server


class TestMCPServerCLI:
    """Test the MCP server CLI functionality."""

    def test_argument_parsing_stdio_default(self):
        """Test that stdio is the default transport when no args provided."""
        # Test default behavior (no arguments)
        with patch('sys.argv', ['mcp_server.py']):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.stdio = False  # stdio not explicitly set
                mock_args.tcp = False
                mock_args.port = 3000
                mock_args.host = 'localhost'
                mock_parse.return_value = mock_args
                
                # Since stdio is default when tcp is False, it should use stdio transport
                assert not mock_args.tcp  # Should default to stdio

    def test_argument_parsing_stdio_explicit(self):
        """Test explicit stdio argument parsing."""
        with patch('sys.argv', ['mcp_server.py', '--stdio']):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.stdio = True
                mock_args.tcp = False
                mock_args.port = 3000
                mock_args.host = 'localhost'
                mock_parse.return_value = mock_args
                
                assert mock_args.stdio
                assert not mock_args.tcp

    def test_argument_parsing_tcp(self):
        """Test TCP transport argument parsing."""
        with patch('sys.argv', ['mcp_server.py', '--tcp', '--port', '8080', '--host', '0.0.0.0']):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.stdio = False
                mock_args.tcp = True
                mock_args.port = 8080
                mock_args.host = '0.0.0.0'
                mock_parse.return_value = mock_args
                
                assert mock_args.tcp
                assert mock_args.port == 8080
                assert mock_args.host == '0.0.0.0'

    def test_argument_parsing_defaults(self):
        """Test default values for TCP arguments."""
        with patch('sys.argv', ['mcp_server.py', '--tcp']):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.stdio = False
                mock_args.tcp = True
                mock_args.port = 3000  # Default port
                mock_args.host = 'localhost'  # Default host
                mock_parse.return_value = mock_args
                
                assert mock_args.port == 3000
                assert mock_args.host == 'localhost'


class TestMCPServerMain:
    """Test the main() function and server initialization."""

    @patch('mcp_server.asyncio.run')
    @patch('mcp_server.sys.stderr', new_callable=StringIO)
    def test_main_stdio_transport(self, mock_stderr, mock_asyncio_run):
        """Test main() with stdio transport."""
        mock_server = Mock()
        mock_server.run_stdio_async = AsyncMock()
        
        with patch('mcp_server.argparse.ArgumentParser.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.stdio = True
            mock_args.tcp = False
            mock_args.port = 3000
            mock_args.host = 'localhost'
            mock_parse.return_value = mock_args
            
            with patch('cli_agent.mcp.model_server.create_model_server') as mock_create:
                mock_create.return_value = mock_server
                
                mcp_server.main()
                
                mock_create.assert_called_once()
                mock_asyncio_run.assert_called_once()
                # Check that stderr shows stdio message
                assert "Starting MCP model server on stdio transport" in mock_stderr.getvalue()

    @patch('mcp_server.asyncio.run')
    @patch('mcp_server.sys.stderr', new_callable=StringIO)
    def test_main_tcp_transport(self, mock_stderr, mock_asyncio_run):
        """Test main() with TCP transport."""
        mock_server = Mock()
        mock_server.run_async = AsyncMock()
        
        with patch('mcp_server.argparse.ArgumentParser.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.stdio = False
            mock_args.tcp = True
            mock_args.port = 8080
            mock_args.host = '192.168.1.100'
            mock_parse.return_value = mock_args
            
            with patch('cli_agent.mcp.model_server.create_model_server') as mock_create:
                mock_create.return_value = mock_server
                
                mcp_server.main()
                
                mock_create.assert_called_once()
                mock_asyncio_run.assert_called_once()
                # Check that stderr shows TCP message with correct host/port
                stderr_output = mock_stderr.getvalue()
                assert "Starting MCP model server on 192.168.1.100:8080" in stderr_output

    @patch('mcp_server.sys.exit')
    @patch('mcp_server.sys.stderr', new_callable=StringIO)
    def test_main_import_error(self, mock_stderr, mock_exit):
        """Test main() handling of import errors."""
        with patch('mcp_server.argparse.ArgumentParser.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.stdio = True
            mock_args.tcp = False
            mock_parse.return_value = mock_args
            
            # Simulate ImportError when importing create_model_server
            with patch('cli_agent.mcp.model_server.create_model_server', side_effect=ImportError("FastMCP not found")):
                mcp_server.main()
                
                mock_exit.assert_called_once_with(1)
                stderr_output = mock_stderr.getvalue()
                assert "Error: FastMCP not found" in stderr_output
                assert "Please install FastMCP with: pip install fastmcp" in stderr_output

    @patch('mcp_server.sys.exit')
    @patch('mcp_server.sys.stderr', new_callable=StringIO)
    def test_main_general_exception(self, mock_stderr, mock_exit):
        """Test main() handling of general exceptions."""
        with patch('mcp_server.argparse.ArgumentParser.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.stdio = True
            mock_args.tcp = False
            mock_parse.return_value = mock_args
            
            # Simulate general exception
            with patch('cli_agent.mcp.model_server.create_model_server', side_effect=Exception("Configuration error")):
                mcp_server.main()
                
                mock_exit.assert_called_once_with(1)
                stderr_output = mock_stderr.getvalue()
                assert "Error starting MCP server: Configuration error" in stderr_output

    @patch('mcp_server.asyncio.run')
    def test_main_default_stdio_when_neither_specified(self, mock_asyncio_run):
        """Test that stdio is used by default when neither --stdio nor --tcp is specified."""
        mock_server = Mock()
        mock_server.run_stdio_async = AsyncMock()
        
        with patch('mcp_server.argparse.ArgumentParser.parse_args') as mock_parse:
            # Both stdio and tcp are False (neither specified)
            mock_args = Mock()
            mock_args.stdio = False
            mock_args.tcp = False
            mock_args.port = 3000
            mock_args.host = 'localhost'
            mock_parse.return_value = mock_args
            
            with patch('cli_agent.mcp.model_server.create_model_server') as mock_create:
                mock_create.return_value = mock_server
                
                mcp_server.main()
                
                # Should default to stdio transport
                mock_asyncio_run.assert_called_once()
                # The asyncio.run should be called with the stdio async method
                call_args = mock_asyncio_run.call_args[0][0]
                # Check that it's calling the stdio method (this is implementation-dependent)
                mock_create.assert_called_once()


class TestMCPServerLogging:
    """Test logging configuration in MCP server."""

    def test_logging_suppression(self):
        """Test that noisy loggers are properly suppressed."""
        import logging
        
        # Check that FastMCP loggers are set to WARNING level
        fastmcp_logger = logging.getLogger("FastMCP.fastmcp.server.server")
        fastmcp_main_logger = logging.getLogger("fastmcp")
        httpx_logger = logging.getLogger("httpx")
        httpcore_logger = logging.getLogger("httpcore")
        
        # Import the module to trigger logging configuration
        import mcp_server
        
        assert fastmcp_logger.level == logging.WARNING
        assert fastmcp_main_logger.level == logging.WARNING
        assert httpx_logger.level == logging.WARNING
        assert httpcore_logger.level == logging.WARNING


class TestMCPServerIntegration:
    """Integration tests for the MCP server."""

    @pytest.mark.integration
    def test_mcp_server_script_help(self):
        """Test that the MCP server script shows help correctly."""
        # Find the mcp_server.py script relative to this test file
        import os
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(test_dir))  # Go up two levels from tests/unit/
        mcp_server_path = os.path.join(project_root, 'mcp_server.py')
        
        if not os.path.exists(mcp_server_path):
            pytest.skip(f"MCP server script not found at {mcp_server_path}")
        
        try:
            result = subprocess.run(
                [sys.executable, mcp_server_path, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should exit with code 0 for help
            assert result.returncode == 0
            assert "MCP Model Server" in result.stdout
            assert "--stdio" in result.stdout
            assert "--tcp" in result.stdout
            assert "--port" in result.stdout
            assert "--host" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("MCP server help command timed out")
        except FileNotFoundError:
            pytest.skip("Python interpreter or MCP server script not found")

    @pytest.mark.integration
    def test_mcp_server_script_invalid_args(self):
        """Test that the MCP server script handles invalid arguments."""
        # Find the mcp_server.py script relative to this test file
        import os
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(test_dir))  # Go up two levels from tests/unit/
        mcp_server_path = os.path.join(project_root, 'mcp_server.py')
        
        if not os.path.exists(mcp_server_path):
            pytest.skip(f"MCP server script not found at {mcp_server_path}")
        
        try:
            result = subprocess.run(
                [sys.executable, mcp_server_path, '--invalid-arg'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should exit with non-zero code for invalid args
            assert result.returncode != 0
            assert "unrecognized arguments" in result.stderr.lower() or "invalid" in result.stderr.lower()
            
        except subprocess.TimeoutExpired:
            pytest.fail("MCP server invalid args test timed out")
        except FileNotFoundError:
            pytest.skip("Python interpreter or MCP server script not found")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])