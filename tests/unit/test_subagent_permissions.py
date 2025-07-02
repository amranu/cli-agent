#!/usr/bin/env python3
"""Test subagent permission handling functionality."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from cli_agent.core.tool_execution_engine import ToolExecutionEngine
from cli_agent.core.tool_permissions import ToolPermissionManager, ToolPermissionConfig


class TestSubagentPermissions:
    """Test that subagents properly forward tools to parent for permission handling."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.is_subagent = True
        agent.comm_socket = Mock()
        agent.available_tools = {
            "builtin:bash_execute": {
                "server": "builtin",
                "name": "bash_execute", 
                "description": "Execute bash commands",
                "schema": {},
                "client": None
            }
        }
        return agent

    @pytest.fixture
    def permission_manager(self):
        """Create a tool permission manager."""
        config = ToolPermissionConfig()
        return ToolPermissionManager(config)

    @pytest.fixture
    def tool_engine(self, mock_agent, permission_manager):
        """Create a tool execution engine."""
        mock_agent.permission_manager = permission_manager
        return ToolExecutionEngine(mock_agent)

    @pytest.mark.asyncio
    async def test_subagent_forwards_tools_to_parent(self, tool_engine, mock_agent):
        """Test that subagents forward tool execution to parent instead of checking permissions locally."""
        
        # Mock the forward method to return a successful result
        async def mock_forward(tool_key, tool_name, args):
            return "Tool executed on parent successfully"
            
        mock_agent._forward_tool_to_parent = AsyncMock(side_effect=mock_forward)
        
        # Execute a tool on the subagent
        result = await tool_engine.execute_mcp_tool("builtin:bash_execute", {"command": "ls -l"})
        
        # Verify the tool was forwarded to parent
        mock_agent._forward_tool_to_parent.assert_called_once_with(
            "builtin:bash_execute", 
            "bash_execute",
            {"command": "ls -l"}
        )
        
        # Verify the result came from the parent
        assert result == "Tool executed on parent successfully"

    @pytest.mark.asyncio
    async def test_subagent_excludes_management_tools(self, tool_engine, mock_agent):
        """Test that subagent management tools are not forwarded to parent."""
        
        # Mock the execute method for subagent management tools
        async def mock_execute(tool_name, args):
            return f"Executed {tool_name} locally"
            
        mock_agent._execute_builtin_tool = AsyncMock(side_effect=mock_execute)
        
        # Test task tool (should not be forwarded)
        tool_engine.agent.available_tools["builtin:task"] = {
            "server": "builtin", "name": "task", "description": "Spawn task", "schema": {}, "client": None
        }
        
        result = await tool_engine.execute_mcp_tool("builtin:task", {"description": "test task"})
        
        # Verify it was executed locally, not forwarded (tool name has prefix removed)
        mock_agent._execute_builtin_tool.assert_called_once_with("task", {"description": "test task"})
        assert "Executed task locally" in result

    @pytest.mark.asyncio
    async def test_main_agent_checks_permissions_locally(self):
        """Test that main agents check permissions locally without forwarding."""
        
        # Create a main agent (not subagent)
        main_agent = Mock()
        main_agent.is_subagent = False
        main_agent.comm_socket = None
        
        # Mock permission manager to always allow
        mock_permission_manager = Mock()
        from cli_agent.core.tool_permissions import ToolPermissionResult
        mock_permission_manager.check_tool_permission = AsyncMock(
            return_value=ToolPermissionResult(allowed=True, reason="Test approval")
        )
        main_agent.permission_manager = mock_permission_manager
        
        main_agent.available_tools = {
            "builtin:bash_execute": {
                "server": "builtin",
                "name": "bash_execute",
                "description": "Execute bash commands", 
                "schema": {},
                "client": None
            }
        }
        
        async def mock_execute(tool_name, args):
            return f"Executed {tool_name} on main agent"
            
        main_agent._execute_builtin_tool = AsyncMock(side_effect=mock_execute)
        
        # Create tool engine for main agent
        main_tool_engine = ToolExecutionEngine(main_agent)
        
        # Execute a tool
        result = await main_tool_engine.execute_mcp_tool("builtin:bash_execute", {"command": "ls -l"})
        
        # Verify permission was checked
        mock_permission_manager.check_tool_permission.assert_called_once()
        
        # Verify it was executed locally (tool name has prefix removed)
        main_agent._execute_builtin_tool.assert_called_once_with("bash_execute", {"command": "ls -l"})
        assert "Executed bash_execute on main agent" in result

    @pytest.mark.asyncio
    async def test_subagent_without_comm_socket_fallback(self, tool_engine, mock_agent):
        """Test that subagent without communication socket falls back to local execution."""
        
        # Remove communication socket
        mock_agent.comm_socket = None
        
        async def mock_execute(tool_name, args):
            return f"Executed {tool_name} locally as fallback"
            
        mock_agent._execute_builtin_tool = AsyncMock(side_effect=mock_execute)
        
        # Execute a tool
        result = await tool_engine.execute_mcp_tool("builtin:bash_execute", {"command": "ls -l"})
        
        # Verify it was executed locally as fallback (tool name has prefix removed)
        mock_agent._execute_builtin_tool.assert_called_once_with("bash_execute", {"command": "ls -l"})
        assert "Executed bash_execute locally as fallback" in result

    @pytest.mark.asyncio
    async def test_permission_flow_integration(self, tool_engine, mock_agent):
        """Test the complete permission flow for subagents."""
        
        # Mock permission manager to simulate permission prompt
        mock_permission_manager = Mock()
        
        # Mock forward method that includes permission handling
        async def mock_forward_with_permission(tool_key, tool_name, args):
            # Simulate parent handling permission prompt and user approval
            return f"Permission granted by parent, executed {tool_name}: result"
            
        mock_agent._forward_tool_to_parent = AsyncMock(side_effect=mock_forward_with_permission)
        mock_agent.permission_manager = mock_permission_manager
        
        # Execute tool that requires permission
        result = await tool_engine.execute_mcp_tool("builtin:bash_execute", {"command": "rm important_file.txt"})
        
        # Verify forwarding occurred (permission handled by parent)
        mock_agent._forward_tool_to_parent.assert_called_once()
        assert "Permission granted by parent" in result

    def test_tool_forwarding_order(self, tool_engine, mock_agent):
        """Test that tool forwarding happens before permission checks in subagents."""
        
        # This test verifies the architectural fix where we moved tool forwarding
        # before permission checks in the execute_mcp_tool method
        
        # Check that the tool engine has the agent marked as subagent
        assert tool_engine.agent.is_subagent is True
        
        # Check that the agent has a communication socket (required for forwarding)
        assert tool_engine.agent.comm_socket is not None
        
        # Verify the tool is available and would normally require permission
        assert "builtin:bash_execute" in tool_engine.agent.available_tools
        
        # The actual forwarding logic verification happens in the other tests
        # This test just verifies the preconditions are correct