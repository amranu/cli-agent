"""
Socket-based permission request system for subagent communication.

This module provides a reliable communication channel between the main agent
and subagents for handling tool permission requests, replacing the unreliable
file-based system.
"""

import asyncio
import json
import logging
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, Callable

logger = logging.getLogger(__name__)


class PermissionRequest:
    """Represents a permission request from a subagent."""
    
    def __init__(self, request_id: str, task_id: str, tool_name: str, 
                 arguments: Dict[str, Any], description: str):
        self.request_id = request_id
        self.task_id = task_id
        self.tool_name = tool_name
        self.arguments = arguments
        self.description = description
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "description": self.description,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionRequest":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            task_id=data["task_id"],
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            description=data["description"]
        )


class PermissionResponse:
    """Represents a response to a permission request."""
    
    def __init__(self, request_id: str, response: str, error: Optional[str] = None):
        self.request_id = request_id
        self.response = response  # "y", "n", "a", "A", "d"
        self.error = error
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "response": self.response,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionResponse":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            response=data["response"],
            error=data.get("error")
        )


class PermissionSocketServer:
    """Socket server running in the main agent to handle permission requests."""
    
    def __init__(self, permission_handler: Callable[[PermissionRequest], str]):
        """
        Initialize the permission socket server.
        
        Args:
            permission_handler: Async function that takes a PermissionRequest 
                              and returns the user's choice as a string
        """
        self.permission_handler = permission_handler
        self.socket_path = self._get_socket_path()
        self.server = None
        self.active_connections: Dict[str, asyncio.StreamWriter] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._running = False
        
    def _get_socket_path(self) -> str:
        """Get the Unix domain socket path for this process."""
        # Use a unique socket path per main agent process to avoid conflicts
        temp_dir = tempfile.gettempdir()
        agent_id = os.getpid()  # Use PID as unique identifier
        socket_path = os.path.join(temp_dir, f"mcp_agent_permissions_{agent_id}.sock")
        return socket_path
    
    async def start(self):
        """Start the socket server."""
        if self._running:
            return
            
        # Clean up any existing socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
            
        logger.info(f"Starting permission socket server at {self.socket_path}")
        
        try:
            self.server = await asyncio.start_unix_server(
                self._handle_client_connection,
                path=self.socket_path
            )
            self._running = True
            logger.info(f"Permission socket server started successfully")
            
            # Store the socket path in environment for subagents to find
            os.environ["MCP_AGENT_PERMISSION_SOCKET"] = self.socket_path
            
        except Exception as e:
            logger.error(f"Failed to start permission socket server: {e}")
            raise
    
    async def stop(self):
        """Stop the socket server."""
        if not self._running:
            return
            
        logger.info("Stopping permission socket server")
        self._running = False
        
        # Close all active connections - make a copy to avoid "dictionary changed size during iteration"
        active_connections_copy = dict(self.active_connections)
        for conn_id, writer in active_connections_copy.items():
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing connection {conn_id}: {e}")
        
        self.active_connections.clear()
        
        # Cancel pending requests - make a copy to avoid "dictionary changed size during iteration"
        pending_requests_copy = dict(self.pending_requests)
        for request_id, future in pending_requests_copy.items():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()
        
        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Clean up socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        
        # Clean up environment
        if "MCP_AGENT_PERMISSION_SOCKET" in os.environ:
            del os.environ["MCP_AGENT_PERMISSION_SOCKET"]
        
        logger.info("Permission socket server stopped")
    
    async def _handle_client_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new client connection from a subagent."""
        conn_id = str(uuid.uuid4())
        client_addr = writer.get_extra_info('sockname', 'unknown')
        logger.info(f"New permission client connected: {conn_id} from {client_addr}")
        
        self.active_connections[conn_id] = writer
        
        try:
            while self._running:
                # Read message length header (4 bytes)
                length_data = await reader.readexactly(4)
                if not length_data:
                    break
                    
                message_length = int.from_bytes(length_data, byteorder='big')
                
                # Read the actual message
                message_data = await reader.readexactly(message_length)
                if not message_data:
                    break
                
                try:
                    # Parse JSON message
                    message = json.loads(message_data.decode('utf-8'))
                    await self._handle_permission_request(message, writer)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client {conn_id}: {e}")
                    await self._send_error_response(writer, "unknown", "Invalid JSON")
                    
                except Exception as e:
                    logger.error(f"Error handling request from client {conn_id}: {e}")
                    await self._send_error_response(writer, "unknown", str(e))
                    
        except asyncio.IncompleteReadError:
            logger.info(f"Client {conn_id} disconnected")
        except Exception as e:
            logger.error(f"Error with client {conn_id}: {e}")
        finally:
            # Clean up connection
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(f"Client {conn_id} connection closed")
    
    async def _handle_permission_request(self, message: Dict[str, Any], writer: asyncio.StreamWriter):
        """Handle a permission request from a subagent."""
        try:
            # Parse the permission request
            request = PermissionRequest.from_dict(message)
            logger.info(f"Received permission request {request.request_id} for tool {request.tool_name} from task {request.task_id}")
            
            # Call the permission handler (this will prompt the user)
            try:
                response_str = await self.permission_handler(request)
                logger.info(f"Permission handler returned: {response_str} for request {request.request_id}")
            except Exception as e:
                logger.error(f"Permission handler error for request {request.request_id}: {e}")
                response_str = "n"  # Default to deny on error
            
            # Send response back to subagent
            response = PermissionResponse(request.request_id, response_str)
            await self._send_response(writer, response)
            
        except Exception as e:
            logger.error(f"Error processing permission request: {e}")
            request_id = message.get("request_id", "unknown")
            await self._send_error_response(writer, request_id, str(e))
    
    async def _send_response(self, writer: asyncio.StreamWriter, response: PermissionResponse):
        """Send a response back to the subagent."""
        try:
            response_data = json.dumps(response.to_dict()).encode('utf-8')
            
            # Send length header followed by data
            length_header = len(response_data).to_bytes(4, byteorder='big')
            writer.write(length_header)
            writer.write(response_data)
            await writer.drain()
            
            logger.debug(f"Sent response for request {response.request_id}: {response.response}")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}")
    
    async def _send_error_response(self, writer: asyncio.StreamWriter, request_id: str, error_msg: str):
        """Send an error response back to the subagent."""
        response = PermissionResponse(request_id, "n", error=error_msg)
        await self._send_response(writer, response)


class PermissionSocketClient:
    """Socket client used by subagents to send permission requests."""
    
    def __init__(self, task_id: str):
        """Initialize the permission socket client."""
        self.task_id = task_id
        self.socket_path = os.environ.get("MCP_AGENT_PERMISSION_SOCKET")
        self.reader = None
        self.writer = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the permission socket server."""
        if self._connected:
            return True
            
        if not self.socket_path:
            logger.error("No permission socket path found in environment")
            return False
            
        if not os.path.exists(self.socket_path):
            logger.error(f"Permission socket does not exist: {self.socket_path}")
            return False
        
        try:
            logger.info(f"Connecting to permission socket: {self.socket_path}")
            self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)
            self._connected = True
            logger.info("Connected to permission socket server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to permission socket: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the permission socket server."""
        if not self._connected:
            return
            
        logger.info("Disconnecting from permission socket")
        self._connected = False
        
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing socket connection: {e}")
        
        self.reader = None
        self.writer = None
    
    async def request_permission(self, tool_name: str, arguments: Dict[str, Any], 
                               description: str, timeout: float = 60.0) -> str:
        """
        Request permission for a tool execution.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            description: Human-readable description of the action
            timeout: Timeout in seconds
            
        Returns:
            User's response: "y", "n", "a", "A", "d"
        """
        if not self._connected:
            if not await self.connect():
                logger.error("Cannot request permission - not connected to server")
                return "n"  # Default to deny
        
        # Create request
        request_id = str(uuid.uuid4())
        request = PermissionRequest(request_id, self.task_id, tool_name, arguments, description)
        
        logger.info(f"Requesting permission for {tool_name} (request {request_id})")
        
        try:
            # Send request
            request_data = json.dumps(request.to_dict()).encode('utf-8')
            length_header = len(request_data).to_bytes(4, byteorder='big')
            
            self.writer.write(length_header)
            self.writer.write(request_data)
            await self.writer.drain()
            
            # Wait for response with timeout
            response_data = await asyncio.wait_for(self._read_response(), timeout=timeout)
            
            if response_data is None:
                logger.error(f"No response received for request {request_id}")
                return "n"
            
            response = PermissionResponse.from_dict(response_data)
            
            if response.error:
                logger.error(f"Permission request error: {response.error}")
                return "n"
            
            logger.info(f"Permission response for {tool_name}: {response.response}")
            return response.response
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for permission response (request {request_id})")
            return "n"
        except Exception as e:
            logger.error(f"Error requesting permission: {e}")
            return "n"
    
    async def _read_response(self) -> Optional[Dict[str, Any]]:
        """Read a response from the socket server."""
        try:
            # Read length header
            length_data = await self.reader.readexactly(4)
            if not length_data:
                return None
                
            message_length = int.from_bytes(length_data, byteorder='big')
            
            # Read message data
            message_data = await self.reader.readexactly(message_length)
            if not message_data:
                return None
            
            # Parse JSON
            return json.loads(message_data.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error reading response: {e}")
            return None