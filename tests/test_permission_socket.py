#!/usr/bin/env python3
"""
Test the socket-based permission system for subagent communication.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from cli_agent.core.permission_socket import (
    PermissionRequest,
    PermissionResponse,
    PermissionSocketServer,
    PermissionSocketClient
)

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestPermissionSocket(unittest.TestCase):
    """Test cases for the socket-based permission system."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()
        
    def test_permission_request_serialization(self):
        """Test PermissionRequest serialization/deserialization."""
        request = PermissionRequest(
            request_id="test-123",
            task_id="task-456",
            tool_name="bash_execute",
            arguments={"command": "ls -la"},
            description="List directory contents"
        )
        
        # Test to_dict
        data = request.to_dict()
        self.assertEqual(data["request_id"], "test-123")
        self.assertEqual(data["task_id"], "task-456")
        self.assertEqual(data["tool_name"], "bash_execute")
        self.assertEqual(data["arguments"], {"command": "ls -la"})
        self.assertEqual(data["description"], "List directory contents")
        self.assertIn("timestamp", data)
        
        # Test from_dict
        restored = PermissionRequest.from_dict(data)
        self.assertEqual(restored.request_id, request.request_id)
        self.assertEqual(restored.task_id, request.task_id)
        self.assertEqual(restored.tool_name, request.tool_name)
        self.assertEqual(restored.arguments, request.arguments)
        self.assertEqual(restored.description, request.description)
        
    def test_permission_response_serialization(self):
        """Test PermissionResponse serialization/deserialization."""
        response = PermissionResponse(
            request_id="test-123",
            response="y",
            error=None
        )
        
        # Test to_dict
        data = response.to_dict()
        self.assertEqual(data["request_id"], "test-123")
        self.assertEqual(data["response"], "y")
        self.assertIsNone(data["error"])
        self.assertIn("timestamp", data)
        
        # Test from_dict
        restored = PermissionResponse.from_dict(data)
        self.assertEqual(restored.request_id, response.request_id)
        self.assertEqual(restored.response, response.response)
        self.assertEqual(restored.error, response.error)
        
    async def test_server_client_communication(self):
        """Test basic server-client communication."""
        # Mock permission handler that always approves
        async def mock_permission_handler(request: PermissionRequest) -> str:
            logger.info(f"Mock handler received: {request.tool_name}")
            return "y"
        
        # Create server
        server = PermissionSocketServer(mock_permission_handler)
        
        try:
            # Start server
            await server.start()
            self.assertTrue(server._running)
            self.assertIsNotNone(server.socket_path)
            self.assertTrue(os.path.exists(server.socket_path))
            
            # Create client
            client = PermissionSocketClient("test-task-123")
            
            try:
                # Connect client
                connected = await client.connect()
                self.assertTrue(connected)
                self.assertTrue(client._connected)
                
                # Send permission request
                response = await client.request_permission(
                    tool_name="bash_execute",
                    arguments={"command": "echo hello"},
                    description="Execute echo command",
                    timeout=5.0
                )
                
                # Verify response
                self.assertEqual(response, "y")
                
            finally:
                await client.disconnect()
                
        finally:
            await server.stop()
            self.assertFalse(server._running)
            self.assertFalse(os.path.exists(server.socket_path))
    
    async def test_multiple_clients(self):
        """Test multiple clients connecting simultaneously."""
        responses = []
        
        async def mock_permission_handler(request: PermissionRequest) -> str:
            logger.info(f"Mock handler for {request.task_id}: {request.tool_name}")
            responses.append(f"{request.task_id}:{request.tool_name}")
            return "y"
        
        # Create server
        server = PermissionSocketServer(mock_permission_handler)
        
        try:
            await server.start()
            
            # Create multiple clients
            async def client_task(task_id: str, tool_name: str):
                client = PermissionSocketClient(task_id)
                try:
                    await client.connect()
                    response = await client.request_permission(
                        tool_name=tool_name,
                        arguments={},
                        description=f"Test tool {tool_name}",
                        timeout=5.0
                    )
                    return response
                finally:
                    await client.disconnect()
            
            # Run multiple clients concurrently
            tasks = [
                client_task("task-1", "read_file"),
                client_task("task-2", "write_file"),
                client_task("task-3", "bash_execute"),
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all requests were handled
            self.assertEqual(len(results), 3)
            self.assertTrue(all(r == "y" for r in results))
            self.assertEqual(len(responses), 3)
            
            # Check that all requests were received
            response_set = set(responses)
            expected = {"task-1:read_file", "task-2:write_file", "task-3:bash_execute"}
            self.assertEqual(response_set, expected)
            
        finally:
            await server.stop()
    
    async def test_permission_denial(self):
        """Test permission denial flow."""
        async def mock_permission_handler(request: PermissionRequest) -> str:
            # Deny bash_execute, approve others
            if request.tool_name == "bash_execute":
                return "n"
            return "y"
        
        server = PermissionSocketServer(mock_permission_handler)
        
        try:
            await server.start()
            
            client = PermissionSocketClient("test-task")
            try:
                await client.connect()
                
                # Test approval
                response1 = await client.request_permission(
                    tool_name="read_file",
                    arguments={"file_path": "/etc/passwd"},
                    description="Read file"
                )
                self.assertEqual(response1, "y")
                
                # Test denial
                response2 = await client.request_permission(
                    tool_name="bash_execute",
                    arguments={"command": "rm -rf /"},
                    description="Dangerous command"
                )
                self.assertEqual(response2, "n")
                
            finally:
                await client.disconnect()
                
        finally:
            await server.stop()
    
    async def test_client_timeout(self):
        """Test client timeout when server doesn't respond."""
        async def slow_permission_handler(request: PermissionRequest) -> str:
            # Simulate slow response
            await asyncio.sleep(10)
            return "y"
        
        server = PermissionSocketServer(slow_permission_handler)
        
        try:
            await server.start()
            
            client = PermissionSocketClient("test-task")
            try:
                await client.connect()
                
                # Request with short timeout
                start_time = time.time()
                response = await client.request_permission(
                    tool_name="bash_execute",
                    arguments={"command": "sleep 1"},
                    description="Sleep command",
                    timeout=2.0  # 2 second timeout
                )
                end_time = time.time()
                
                # Should timeout and return "n"
                self.assertEqual(response, "n")
                self.assertLess(end_time - start_time, 5.0)  # Should be much less than 10s
                
            finally:
                await client.disconnect()
                
        finally:
            await server.stop()
    
    def test_socket_path_uniqueness(self):
        """Test that different server instances get unique socket paths."""
        server1 = PermissionSocketServer(AsyncMock())
        server2 = PermissionSocketServer(AsyncMock())
        
        # Socket paths should be the same for same process (based on PID)
        self.assertEqual(server1.socket_path, server2.socket_path)
        
        # But they should include the process ID
        self.assertIn(str(os.getpid()), server1.socket_path)


def run_async_test(test_func):
    """Helper to run async test functions."""
    def wrapper(self):
        return self.loop.run_until_complete(test_func(self))
    return wrapper


# Add async test methods
TestPermissionSocket.test_server_client_communication = run_async_test(TestPermissionSocket.test_server_client_communication)
TestPermissionSocket.test_multiple_clients = run_async_test(TestPermissionSocket.test_multiple_clients)
TestPermissionSocket.test_permission_denial = run_async_test(TestPermissionSocket.test_permission_denial)
TestPermissionSocket.test_client_timeout = run_async_test(TestPermissionSocket.test_client_timeout)


if __name__ == "__main__":
    unittest.main()