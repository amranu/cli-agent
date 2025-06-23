"""Built-in tool execution implementations for BaseMCPAgent."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import requests


class BuiltinToolExecutor:
    """Handles execution of built-in tools for BaseMCPAgent."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    def bash_execute(self, args: Dict[str, Any]) -> str:
        """Execute bash commands and return output."""
        command = args.get("command", "")
        timeout = args.get("timeout", 120)

        if not command:
            return "Error: No command provided"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\nReturn code: {result.returncode}"

            return output or "Command completed with no output"

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def read_file(self, args: Dict[str, Any]) -> str:
        """Read file contents with optional offset and limit."""
        file_path = args.get("file_path", "")
        offset = args.get("offset", 0)
        limit = args.get("limit", None)

        if not file_path:
            return "Error: No file path provided"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Apply offset and limit
            if offset > 0:
                lines = lines[offset:]
            if limit is not None:
                lines = lines[:limit]

            # Add line numbers starting from offset + 1
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = offset + i + 1
                numbered_lines.append(f"{line_num:5d}→{line.rstrip()}")

            return "\n".join(numbered_lines)

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except PermissionError:
            return f"Error: Permission denied reading file: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, args: Dict[str, Any]) -> str:
        """Write content to a file."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            return "Error: No file path provided"

        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully wrote {len(content)} characters to {file_path}"

        except PermissionError:
            return f"Error: Permission denied writing to file: {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def list_directory(self, args: Dict[str, Any]) -> str:
        """List directory contents."""
        directory_path = args.get("directory_path", ".")

        try:
            path = Path(directory_path)
            if not path.exists():
                return f"Error: Directory does not exist: {directory_path}"
            if not path.is_dir():
                return f"Error: Path is not a directory: {directory_path}"

            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")

            if not items:
                return f"Directory is empty: {directory_path}"

            return f"Contents of {directory_path}:\n" + "\n".join(items)

        except PermissionError:
            return f"Error: Permission denied accessing directory: {directory_path}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def get_current_directory(self, args: Dict[str, Any]) -> str:
        """Get the current working directory."""
        try:
            return f"Current directory: {os.getcwd()}"
        except Exception as e:
            return f"Error getting current directory: {str(e)}"

    def todo_read(self, args: Dict[str, Any]) -> str:
        """Read the current todo list."""
        try:
            if hasattr(self.agent, "todo_manager"):
                return self.agent.todo_manager.read_todos()
            else:
                return "Todo manager not available"
        except Exception as e:
            return f"Error reading todos: {str(e)}"

    def todo_write(self, args: Dict[str, Any]) -> str:
        """Write/update the todo list."""
        todos = args.get("todos", [])

        try:
            if hasattr(self.agent, "todo_manager"):
                return self.agent.todo_manager.write_todos(todos)
            else:
                return "Todo manager not available"
        except Exception as e:
            return f"Error writing todos: {str(e)}"

    def replace_in_file(self, args: Dict[str, Any]) -> str:
        """Replace text in a file."""
        file_path = args.get("file_path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")

        if not file_path:
            return "Error: No file path provided"
        if not old_text:
            return "Error: No old text to replace provided"

        try:
            # Read the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if old text exists
            if old_text not in content:
                return f"Error: Text to replace not found in {file_path}"

            # Replace the text
            new_content = content.replace(old_text, new_text)

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"Successfully replaced text in {file_path}"

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error replacing text in file: {str(e)}"

    def webfetch(self, args: Dict[str, Any]) -> str:
        """Fetch content from a webpage."""
        url = args.get("url", "")
        limit = args.get("limit", 1000)

        if not url:
            return "Error: No URL provided"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            content = response.text
            lines = content.split("\n")

            if limit and len(lines) > limit:
                lines = lines[:limit]
                content = "\n".join(lines) + f"\n\n[Content truncated at {limit} lines]"
            else:
                content = "\n".join(lines)

            return f"Content from {url}:\n{content}"

        except requests.exceptions.RequestException as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def task(self, args: Dict[str, Any]) -> str:
        """Spawn a subagent task."""
        description = args.get("description", "")
        prompt = args.get("prompt", "")
        context = args.get("context", "")
        model = args.get("model", None)

        if not description:
            return "Error: No task description provided"
        if not prompt:
            return "Error: No task prompt provided"

        try:
            if not self.agent.subagent_manager:
                return "Subagent management not available"

            # Add context to prompt if provided
            full_prompt = prompt
            if context:
                full_prompt = f"{prompt}\n\nAdditional Context:\n{context}"

            # If no model specified, detect and use the current running model
            if model is None:
                model = self.agent._get_current_runtime_model()

            # Track active count before and after spawning
            initial_count = self.agent.subagent_manager.get_active_count()
            task_id = await self.agent.subagent_manager.spawn_subagent(
                description, full_prompt, model=model
            )
            final_count = self.agent.subagent_manager.get_active_count()

            # Reset timeout timer when spawning new subagents
            import time

            self.agent.last_subagent_message_time = time.time()

            # Display spawn confirmation with active count and model info
            active_info = (
                f" (Now {final_count} active subagents)" if final_count > 1 else ""
            )
            model_info = (
                f" using {model}" if model else " (inheriting main agent model)"
            )
            return f"Spawned subagent task: {task_id}\nDescription: {description}\nModel: {model_info}{active_info}\nTask is running in the background - output will appear in the chat as it becomes available."
        except Exception as e:
            return f"Error spawning subagent: {e}"

    def task_status(self, args: Dict[str, Any]) -> str:
        """Check the status of running subagent tasks."""
        if not self.agent.subagent_manager:
            return "Subagent management not available"

        task_id = args.get("task_id", None)

        if task_id:
            # Check specific task
            if task_id in self.agent.subagent_manager.subagents:
                subagent = self.agent.subagent_manager.subagents[task_id]
                status = "completed" if subagent.completed else "running"
                elapsed_time = time.time() - subagent.start_time
                return f"Task {task_id}: {status} (elapsed: {elapsed_time:.1f}s)"
            else:
                return f"Task {task_id} not found"
        else:
            # Show all tasks
            active_count = self.agent.subagent_manager.get_active_count()
            if active_count == 0:
                return "No active subagent tasks"

            status_lines = [f"Active subagent tasks: {active_count}"]
            for task_id, subagent in self.agent.subagent_manager.subagents.items():
                status = "completed" if subagent.completed else "running"
                elapsed_time = time.time() - subagent.start_time
                status_lines.append(
                    f"  {task_id}: {status} - {subagent.description} (elapsed: {elapsed_time:.1f}s)"
                )

            return "\n".join(status_lines)

    def task_results(self, args: Dict[str, Any]) -> str:
        """Retrieve results from completed subagent tasks."""
        if not self.agent.subagent_manager:
            return "Subagent management not available"

        include_running = args.get("include_running", False)
        clear_after_retrieval = args.get("clear_after_retrieval", True)

        results = []
        tasks_to_remove = []

        for task_id, subagent in self.agent.subagent_manager.subagents.items():
            if subagent.completed or include_running:
                result_info = {
                    "task_id": task_id,
                    "description": subagent.description,
                    "status": "completed" if subagent.completed else "running",
                    "result": subagent.result if subagent.result else "No result yet",
                }
                results.append(result_info)

                if clear_after_retrieval and subagent.completed:
                    tasks_to_remove.append(task_id)

        # Remove completed tasks if requested
        if clear_after_retrieval:
            for task_id in tasks_to_remove:
                del self.agent.subagent_manager.subagents[task_id]

        if not results:
            return "No task results available"

        # Format results
        result_lines = [f"Retrieved {len(results)} task result(s):"]
        for result in results:
            result_lines.append(f"\n--- Task: {result['task_id']} ---")
            result_lines.append(f"Description: {result['description']}")
            result_lines.append(f"Status: {result['status']}")
            result_lines.append(f"Result: {result['result']}")

        if clear_after_retrieval and tasks_to_remove:
            result_lines.append(f"\nCleared {len(tasks_to_remove)} completed task(s)")

        return "\n".join(result_lines)

    def emit_result(self, args: Dict[str, Any]) -> str:
        """Emit the final result of a subagent task and terminate the subagent."""
        if not self.agent.is_subagent:
            return "Error: emit_result can only be called by subagents"

        result = args.get("result", "")
        summary = args.get("summary", "")

        if not result:
            return "Error: result parameter is required"

        # Import here to avoid circular imports
        from subagent import emit_result

        # Emit the result through the subagent communication system
        emit_result(result)

        # If summary is provided, also emit it
        if summary:
            from subagent import emit_message

            emit_message("result", f"Summary: {summary}")

        # Terminate the subagent process
        import sys

        sys.exit(0)
