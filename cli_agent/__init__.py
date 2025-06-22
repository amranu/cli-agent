"""
CLI Agent - Modular MCP Agent implementation.

This package provides a modular architecture for MCP (Model Context Protocol) agents
with support for multiple LLM backends, tool integration, and interactive chat.
"""

from .core.base_agent import BaseMCPAgent
from .core.input_handler import InterruptibleInput
from .core.slash_commands import SlashCommandManager
from .tools.builtin_tools import get_all_builtin_tools

__version__ = "1.0.0"

__all__ = [
    "BaseMCPAgent",
    "InterruptibleInput",
    "SlashCommandManager",
    "get_all_builtin_tools",
]
