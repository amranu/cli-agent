"""MCP (Model Context Protocol) server implementation.

This module provides MCP server functionality to expose all available AI models
as standardized MCP tools. This allows any MCP-compatible client to access our
diverse model collection through a standardized interface.
"""

from .model_server import create_model_server

__all__ = ["create_model_server"]
