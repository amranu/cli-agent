#!/usr/bin/env python3
"""
Simple Subagent Management System

Event-driven architecture with JSON-over-stdout communication.
No complex inheritance or socket communication.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SubagentMessage:
    """Structured message from subagent."""

    def __init__(self, msg_type: str, content: str, **kwargs):
        self.type = msg_type
        self.content = content
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
        self.data = kwargs

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type,
                "content": self.content,
                "timestamp": self.timestamp,
                "id": self.id,
                **self.data,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SubagentMessage":
        data = json.loads(json_str)
        msg_type = data.pop("type")
        content = data.pop("content", "")
        data.pop("timestamp", None)  # Remove timestamp from kwargs
        data.pop("id", None)  # Remove id from kwargs
        return cls(msg_type, content, **data)


class SubagentProcess:
    """Manages a single subagent subprocess."""

    def __init__(self, task_id: str, description: str, prompt: str, model: str = None):
        self.task_id = task_id
        self.description = description
        self.prompt = prompt
        self.model = model  # Store model preference for this subagent
        self.process: Optional[subprocess.Popen] = None
        self.start_time = time.time()
        self.completed = False
        self.result = None

    async def start(self, config, agent=None) -> bool:
        """Start the subagent subprocess."""
        try:
            # Create task file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                task_data = {
                    "task_id": self.task_id,
                    "description": self.description,
                    "prompt": self.prompt,
                    "timestamp": self.start_time,
                    "model": self.model,  # Include model preference in task data
                }
                json.dump(task_data, f)
                task_file = f.name

            # Start subprocess
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            runner_path = os.path.join(current_dir, "subagent_runner.py")
            cmd = ["python", runner_path, task_file]
            
            # Stream-json mode is inherited via environment variables
            # Pass current environment including STREAM_JSON_MODE to subprocess
            import os
            env = os.environ.copy()
            self.process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
            )

            logger.info(f"Started subagent {self.task_id} with PID {self.process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start subagent {self.task_id}: {e}")
            return False

    async def read_messages(self) -> AsyncGenerator[SubagentMessage, None]:
        """Read messages from subagent stdout."""
        if not self.process or not self.process.stdout:
            return

        buffer = ""
        while True:
            try:
                # Check if process is still running
                if self.process.returncode is not None:
                    break

                # Read available data
                try:
                    chunk = await asyncio.wait_for(
                        self.process.stdout.read(1024), timeout=0.1
                    )
                    if not chunk:
                        break

                    # Decode bytes to string
                    chunk_str = chunk.decode("utf-8", errors="ignore")
                    buffer += chunk_str

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if line.startswith("SUBAGENT_MSG:"):
                            # Parse structured message
                            try:
                                msg_json = line[13:]  # Remove "SUBAGENT_MSG:" prefix
                                msg = SubagentMessage.from_json(msg_json)
                                yield msg
                            except Exception as e:
                                logger.error(f"Failed to parse subagent message: {e}")
                        elif line:
                            # Treat as regular output
                            yield SubagentMessage("output", line)

                except asyncio.TimeoutError:
                    # No data available, continue
                    continue

            except Exception as e:
                logger.error(f"Error reading from subagent {self.task_id}: {e}")
                break

        # Mark as completed
        self.completed = True

    async def terminate(self):
        """Terminate the subagent process."""
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()


class SubagentManager:
    """Manages multiple subagent processes with event-driven messaging."""

    def __init__(self, config, agent=None):
        self.config = config
        self.agent = agent
        self.subagents: Dict[str, SubagentProcess] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_callbacks: List[Callable[[SubagentMessage], None]] = []
        self._running = True

    async def spawn_subagent(
        self, description: str, prompt: str, model: str = None
    ) -> str:
        """Spawn a new subagent and return its task_id."""
        # Generate unique task_id using timestamp + microseconds + counter to avoid collisions
        import uuid

        timestamp = time.time()
        # Use UUID to ensure absolute uniqueness
        task_id = f"task_{int(timestamp)}_{uuid.uuid4().hex[:8]}"

        # Ensure task_id is unique (shouldn't be needed with UUID, but safety check)
        while task_id in self.subagents:
            task_id = f"task_{int(timestamp)}_{uuid.uuid4().hex[:8]}"

        logger.info(f"NEW SUBAGENT SYSTEM: spawn_subagent called for {task_id}")
        logger.info(
            f"NEW SUBAGENT SYSTEM: description={description}, prompt={prompt[:50]}..., model={model}"
        )

        subagent = SubagentProcess(task_id, description, prompt, model=model)
        success = await subagent.start(self.config, agent=self.agent)

        if success:
            self.subagents[task_id] = subagent
            # Start monitoring messages
            asyncio.create_task(self._monitor_subagent(subagent))
            logger.info(f"Started monitoring subagent {task_id}")
            return task_id
        else:
            raise Exception(f"Failed to start subagent for: {description}")

    async def _monitor_subagent(self, subagent: SubagentProcess):
        """Monitor a subagent for messages and trigger events."""
        logger.info(f"Starting to monitor subagent {subagent.task_id}")
        try:
            async for message in subagent.read_messages():
                # ONLY store results that were explicitly emitted via emit_result (type="result")
                # Do NOT automatically capture tool outputs as results
                if message.type == "result":
                    subagent.result = message.content
                    logger.info(
                        f"Stored EXPLICIT result for {subagent.task_id}: {message.content[:100]}..."
                    )

                # Add to queue for polling fallback
                await self.message_queue.put(message)

                # Trigger immediate callbacks for event-driven handling
                for callback in self.message_callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")

                logger.info(
                    f"Received message from {subagent.task_id}: {message.type} - {message.content[:50]}"
                )
        except Exception as e:
            logger.error(f"Error monitoring subagent {subagent.task_id}: {e}")
        logger.info(f"Finished monitoring subagent {subagent.task_id}")

    def add_message_callback(self, callback: Callable[[SubagentMessage], None]):
        """Add a callback that gets called immediately when a message is received."""
        self.message_callbacks.append(callback)

    def remove_message_callback(self, callback: Callable[[SubagentMessage], None]):
        """Remove a message callback."""
        if callback in self.message_callbacks:
            self.message_callbacks.remove(callback)

    async def get_pending_messages(self) -> List[SubagentMessage]:
        """Get all pending messages from the async queue."""
        messages = []
        try:
            while True:
                message = self.message_queue.get_nowait()
                messages.append(message)
        except asyncio.QueueEmpty:
            pass

        if messages:
            logger.info(f"Retrieved {len(messages)} pending subagent messages")
        return messages

    async def terminate_all(self):
        """Terminate all running subagents."""
        if not self.subagents:
            return

        logger.info(f"Terminating {len(self.subagents)} subagents")
        for task_id, subagent in self.subagents.items():
            logger.info(f"Terminating subagent {task_id}")
            await subagent.terminate()
        self.subagents.clear()
        logger.info("All subagents terminated")

    async def terminate_subagent(self, task_id: str) -> bool:
        """Terminate a specific subagent by task_id."""
        if task_id not in self.subagents:
            return False

        logger.info(f"Terminating subagent {task_id}")
        await self.subagents[task_id].terminate()
        del self.subagents[task_id]
        return True

    def get_active_count(self) -> int:
        """Get number of active subagents."""
        return len([s for s in self.subagents.values() if not s.completed])

    def get_active_task_ids(self) -> List[str]:
        """Get list of active subagent task IDs."""
        return [task_id for task_id, s in self.subagents.items() if not s.completed]


# Helper functions for subagent communication
def emit_message(msg_type: str, content: str, **kwargs):
    """Emit a structured message from within a subagent."""
    msg = SubagentMessage(msg_type, content, **kwargs)
    print(f"SUBAGENT_MSG:{msg.to_json()}", flush=True)


def emit_output(text: str):
    """Emit regular output from subagent."""
    emit_message("output", text)


def emit_tool_request(tool_name: str, arguments: dict, request_id: str = None):
    """Emit a tool execution request."""
    if not request_id:
        request_id = str(uuid.uuid4())
    emit_message(
        "tool_request",
        f"Requesting tool: {tool_name}",
        tool_name=tool_name,
        arguments=arguments,
        request_id=request_id,
    )
    return request_id


def emit_tool_result(result: str, request_id: str, is_error: bool = False):
    """Emit a tool execution result."""
    emit_message(
        "tool_result",
        result,
        request_id=request_id,
        is_error=is_error,
    )


def emit_status(status: str, details: str = ""):
    """Emit status update."""
    emit_message("status", f"Status: {status}", status=status, details=details)


def emit_result(result: str):
    """Emit final result."""
    emit_message("result", result)


def emit_error(error: str, details: str = ""):
    """Emit error message."""
    emit_message("error", error, details=details)
