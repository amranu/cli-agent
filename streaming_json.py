#!/usr/bin/env python3
"""
Streaming JSON support for MCP Agent to be compatible with Claude Code format.

Based on analysis of Claude Code streaming format from logs.
"""

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class StreamMessage:
    """Base class for streaming JSON messages."""

    def __init__(self, type: str, session_id: str):
        self.type = type
        self.session_id = session_id

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.__dict__)


class SystemInitMessage(StreamMessage):
    """System initialization message."""

    def __init__(
        self,
        session_id: str,
        cwd: str = "",
        tools: List[str] = None,
        mcp_servers: List[str] = None,
        model: str = "",
    ):
        super().__init__("system", session_id)
        self.subtype = "init"
        self.cwd = cwd
        self.tools = tools or []
        self.mcp_servers = mcp_servers or []
        self.model = model
        self.permissionMode = "default"
        self.apiKeySource = "none"


class AssistantMessage(StreamMessage):
    """Assistant message with Claude API format."""

    def __init__(
        self,
        session_id: str,
        message: Dict[str, Any] = None,
        parent_tool_use_id: Optional[str] = None,
    ):
        super().__init__("assistant", session_id)
        self.message = message or {}
        self.parent_tool_use_id = parent_tool_use_id


class UserMessage(StreamMessage):
    """User message with tool results."""

    def __init__(
        self,
        session_id: str,
        message: Dict[str, Any] = None,
        parent_tool_use_id: Optional[str] = None,
    ):
        super().__init__("user", session_id)
        self.message = message or {}
        self.parent_tool_use_id = parent_tool_use_id


class StreamingJSONHandler:
    """Handler for streaming JSON input/output compatible with Claude Code."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.output_format = "stream-json"
        self.input_format = "stream-json"

    def send_system_init(
        self, cwd: str, tools: List[str], mcp_servers: List[str], model: str
    ):
        """Send system initialization message."""
        msg = SystemInitMessage(
            session_id=self.session_id,
            cwd=cwd,
            tools=tools,
            mcp_servers=mcp_servers,
            model=model,
        )
        self._output_json(msg)

    def send_assistant_text(self, text: str, message_id: Optional[str] = None):
        """Send assistant text response."""
        msg_id = message_id or f"msg_{uuid.uuid4().hex[:20]}"

        message = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": self._get_model(),
            "content": [{"type": "text", "text": text}],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": self._get_usage_stats(),
        }

        msg = AssistantMessage(
            session_id=self.session_id, message=message, parent_tool_use_id=None
        )
        self._output_json(msg)

    def send_assistant_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ):
        """Send assistant tool use."""
        msg_id = message_id or f"msg_{uuid.uuid4().hex[:20]}"
        tool_id = tool_use_id or f"toolu_{uuid.uuid4().hex[:20]}"

        message = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": self._get_model(),
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            ],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": self._get_usage_stats(),
        }

        msg = AssistantMessage(
            session_id=self.session_id, message=message, parent_tool_use_id=None
        )
        self._output_json(msg)
        return tool_id

    def send_assistant_combined(
        self,
        text: str = None,
        tool_calls: List[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ):
        """Send assistant message with combined text and tool calls."""
        msg_id = message_id or f"msg_{uuid.uuid4().hex[:20]}"

        content = []

        # Add text content if provided
        if text and text.strip():
            content.append({"type": "text", "text": text})

        # Add tool calls if provided
        tool_ids = []
        if tool_calls:
            for tool_call in tool_calls:
                tool_id = tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}"
                tool_ids.append(tool_id)
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_call["name"],
                        "input": tool_call["input"],
                    }
                )

        message = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": self._get_model(),
            "content": content,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": self._get_usage_stats(),
        }

        msg = AssistantMessage(
            session_id=self.session_id, message=message, parent_tool_use_id=None
        )
        self._output_json(msg)
        return tool_ids

    def send_tool_result(self, tool_use_id: str, content: str, is_error: bool = False):
        """Send tool execution result."""
        message = {
            "role": "user",
            "content": [
                {
                    "tool_use_id": tool_use_id,
                    "type": "tool_result",
                    "content": content,
                    "is_error": is_error,
                }
            ],
        }

        msg = UserMessage(
            session_id=self.session_id, message=message, parent_tool_use_id=None
        )
        self._output_json(msg)

    def read_input_json(self) -> Optional[Dict[str, Any]]:
        """Read streaming JSON input from stdin."""
        try:
            line = sys.stdin.readline().strip()
            if line:
                return json.loads(line)
            return None
        except json.JSONDecodeError:
            return None
        except EOFError:
            return None

    def _output_json(self, msg: StreamMessage):
        """Output JSON message to stdout."""
        json_str = msg.to_json()
        print(json_str, flush=True)

    def _get_model(self) -> str:
        """Get current model name."""
        # This should be set from the actual config
        return "deepseek-chat"  # Default

    def _get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return {
            "input_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 1,
            "service_tier": "standard",
        }

    def send_tool_execution_updates(
        self, tool_calls: List[Dict[str, Any]], tool_results: List[str]
    ):
        """Send tool execution updates in Claude Code format."""
        # Send assistant message with tool use
        for i, tool_call in enumerate(tool_calls):
            tool_id = self.send_assistant_tool_use(
                tool_name=tool_call["function"]["name"],
                tool_input=json.loads(tool_call["function"]["arguments"]),
                tool_use_id=tool_call.get("id"),
            )

            # Send tool result
            if i < len(tool_results):
                self.send_tool_result(tool_id, tool_results[i], is_error=False)

    async def process_streaming_response(self, response_generator):
        """Process streaming response and handle tool calls."""
        full_response = ""
        tool_calls = []

        async for chunk in response_generator:
            if isinstance(chunk, str):
                full_response += chunk
                # Check if this chunk contains text to send
                if not any(
                    marker in chunk
                    for marker in ["<tool_call>", "</tool_call>", "tool_calls"]
                ):
                    self.send_assistant_text(chunk)

            # Handle tool calls if present
            elif hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls"):
                    if choice.delta.tool_calls:
                        tool_calls.extend(choice.delta.tool_calls)

        return full_response, tool_calls


def main():
    """Test the streaming JSON handler."""
    import os

    handler = StreamingJSONHandler()

    # Send system init
    handler.send_system_init(
        cwd=os.getcwd(),
        tools=["Bash", "Read", "Write", "todo_read", "todo_write"],
        mcp_servers=[],
        model="deepseek-chat",
    )

    # Send assistant text
    handler.send_assistant_text("I'll help you with your request.")

    # Send tool use
    tool_id = handler.send_assistant_tool_use(
        tool_name="Bash",
        tool_input={"command": "ls -la", "description": "List current directory"},
    )

    # Send tool result
    handler.send_tool_result(
        tool_id, "total 10\ndrwxr-xr-x 2 user user 4096 Jan 1 12:00 ."
    )


if __name__ == "__main__":
    main()
