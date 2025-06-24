"""
Built-in tool definitions for the CLI Agent.

This module contains all the built-in tool schemas and definitions that were
extracted from the BaseMCPAgent._add_builtin_tools method in agent.py.
"""

from typing import Any, Dict


def get_bash_execute_tool() -> Dict[str, Any]:
    """Return the bash_execute tool definition."""
    return {
        "server": "builtin",
        "name": "bash_execute",
        "description": "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures. Always quote file paths that contain spaces with double quotes. It is very helpful if you write a clear, concise description of what this command does in 5-10 words.",
        "schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "default": 120,
                    "description": "Timeout in seconds",
                },
            },
            "required": ["command"],
        },
        "client": None,
    }


def get_read_file_tool() -> Dict[str, Any]:
    """Return the read_file tool definition."""
    return {
        "server": "builtin",
        "name": "read_file",
        "description": "Reads a file from the local filesystem. You can access any file directly by using this tool. The file_path parameter must be an absolute path, not a relative path. By default, it reads up to 2000 lines starting from the beginning of the file. You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters.",
        "schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start from",
                },
                "limit": {"type": "integer", "description": "Number of lines to read"},
            },
            "required": ["file_path"],
        },
        "client": None,
    }


def get_write_file_tool() -> Dict[str, Any]:
    """Return the write_file tool definition."""
    return {
        "server": "builtin",
        "name": "write_file",
        "description": "Writes a file to the local filesystem. This tool will overwrite the existing file if there is one at the provided path. If this is an existing file, you MUST use the read_file tool first to read the file's contents. ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.",
        "schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
        "client": None,
    }


def get_list_directory_tool() -> Dict[str, Any]:
    """Return the list_directory tool definition."""
    return {
        "server": "builtin",
        "name": "list_directory",
        "description": "Lists files and directories in a given path. The path parameter must be an absolute path, not a relative path. You can optionally provide an array of glob patterns to ignore. You should generally prefer the glob and grep tools, if you know which directories to search.",
        "schema": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to the directory to list",
                }
            },
            "required": ["directory_path"],
        },
        "client": None,
    }


def get_current_directory_tool() -> Dict[str, Any]:
    """Return the get_current_directory tool definition."""
    return {
        "server": "builtin",
        "name": "get_current_directory",
        "description": "Get the current working directory path. This tool takes no parameters and returns the absolute path of the current working directory.",
        "schema": {"type": "object", "properties": {}, "required": []},
        "client": None,
    }


def get_todo_read_tool() -> Dict[str, Any]:
    """Return the todo_read tool definition."""
    return {
        "server": "builtin",
        "name": "todo_read",
        "description": "Use this tool to read the current to-do list for the session. This tool should be used proactively and frequently to ensure that you are aware of the status of the current task list. You should make use of this tool as often as possible, especially at the beginning of conversations, before starting new tasks, when uncertain about what to do next, after completing tasks, and after every few messages to ensure you're on track.",
        "schema": {"type": "object", "properties": {}, "required": []},
        "client": None,
    }


def get_todo_write_tool() -> Dict[str, Any]:
    """Return the todo_write tool definition."""
    return {
        "server": "builtin",
        "name": "todo_write",
        "description": "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user. Use this tool proactively for complex multi-step tasks, when user provides multiple tasks, after receiving new instructions, when starting work on a task (mark as in_progress BEFORE beginning), and after completing tasks.",
        "schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "required": ["id", "content", "status", "priority"],
                    },
                }
            },
            "required": ["todos"],
        },
        "client": None,
    }


def get_replace_in_file_tool() -> Dict[str, Any]:
    """Return the replace_in_file tool definition."""
    return {
        "server": "builtin",
        "name": "replace_in_file",
        "description": "Performs exact string replacements in files. You must use your read_file tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. CRITICAL: When editing text from read_file tool output, you MUST preserve the EXACT indentation, spacing, and whitespace characters (spaces, tabs, newlines) as they appear in the original file. Copy the text exactly including all leading/trailing whitespace. ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.",
        "schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "old_text": {"type": "string", "description": "Exact text to replace - MUST preserve all whitespace, indentation, tabs, and spacing exactly as shown in read_file output"},
                "new_text": {
                    "type": "string", 
                    "description": "New text to replace with - MUST preserve exact indentation and spacing to match surrounding code",
                },
            },
            "required": ["file_path", "old_text", "new_text"],
        },
        "client": None,
    }


def get_web_fetch_tool() -> Dict[str, Any]:
    """Return the webfetch tool definition."""
    return {
        "server": "builtin",
        "name": "webfetch",
        "description": "Fetches content from a specified URL and processes it using an AI model. Takes a URL and a prompt as input, fetches the URL content, and returns the content. Use this tool when you need to retrieve and analyze web content. Prefer MCP commands over this if available.",
        "schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "limit": {
                    "type": "integer",
                    "description": "Optional limit to truncate the HTML response by this number of lines (default: 1000)",
                },
            },
            "required": ["url"],
        },
        "client": None,
    }


def get_task_tool() -> Dict[str, Any]:
    """Return the task tool definition."""
    return {
        "server": "builtin",
        "name": "task",
        "description": "Spawn a subagent to investigate a specific task and return a comprehensive summary. IMPORTANT: To spawn multiple subagents simultaneously, make multiple tool calls to 'builtin_task' in the same response - do not wait for results between calls. The main agent will automatically pause after spawning subagents, wait for all to complete, then restart with their combined results.",
        "schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A brief description of the task (3-5 words)",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed instructions for what the subagent should investigate or accomplish",
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context or files the subagent should consider",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use for subagent. Options: 'deepseek-chat' (fast), 'deepseek-reasoner' (complex reasoning), 'gemini-2.5-flash' (fast), 'gemini-2.5-pro' (advanced). If not specified, inherits from main agent.",
                    "enum": [
                        "deepseek-chat",
                        "deepseek-reasoner",
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                    ],
                },
            },
            "required": ["description", "prompt"],
        },
        "client": None,
    }


def get_task_status_tool() -> Dict[str, Any]:
    """Return the task_status tool definition."""
    return {
        "server": "builtin",
        "name": "task_status",
        "description": "Check the status of running subagent tasks",
        "schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Optional specific task ID to check. If not provided, shows all tasks",
                }
            },
        },
        "client": None,
    }


def get_task_results_tool() -> Dict[str, Any]:
    """Return the task_results tool definition."""
    return {
        "server": "builtin",
        "name": "task_results",
        "description": "Retrieve the results and summaries from completed subagent tasks",
        "schema": {
            "type": "object",
            "properties": {
                "include_running": {
                    "type": "boolean",
                    "description": "Whether to include running tasks (default: false, only completed)",
                },
                "clear_after_retrieval": {
                    "type": "boolean",
                    "description": "Whether to clear tasks after retrieving results (default: true)",
                },
            },
        },
        "client": None,
    }


def get_emit_result_tool() -> Dict[str, Any]:
    """Return the emit_result tool definition (subagents only)."""
    return {
        "server": "builtin",
        "name": "emit_result",
        "description": "Emit the final result of a subagent task and terminate the subagent (subagents only)",
        "schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "The final result to emit"},
                "summary": {
                    "type": "string",
                    "description": "Optional brief summary of what was accomplished",
                },
            },
            "required": ["result"],
        },
        "client": None,
    }


def get_all_builtin_tools() -> Dict[str, Dict[str, Any]]:
    """
    Return a dictionary of all built-in tools with their full qualified names as keys.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping tool names to their definitions
    """
    return {
        "builtin:bash_execute": get_bash_execute_tool(),
        "builtin:read_file": get_read_file_tool(),
        "builtin:write_file": get_write_file_tool(),
        "builtin:list_directory": get_list_directory_tool(),
        "builtin:get_current_directory": get_current_directory_tool(),
        "builtin:todo_read": get_todo_read_tool(),
        "builtin:todo_write": get_todo_write_tool(),
        "builtin:replace_in_file": get_replace_in_file_tool(),
        "builtin:webfetch": get_web_fetch_tool(),
        "builtin:task": get_task_tool(),
        "builtin:task_status": get_task_status_tool(),
        "builtin:task_results": get_task_results_tool(),
        "builtin:emit_result": get_emit_result_tool(),
    }


def get_builtin_tool_by_name(tool_name: str) -> Dict[str, Any]:
    """
    Get a specific built-in tool definition by name.

    Args:
        tool_name (str): The name of the tool (with or without 'builtin:' prefix)

    Returns:
        Dict[str, Any]: The tool definition

    Raises:
        KeyError: If the tool name is not found
    """
    # Normalize tool name to include builtin: prefix if not present
    if not tool_name.startswith("builtin:"):
        tool_name = f"builtin:{tool_name}"

    all_tools = get_all_builtin_tools()
    if tool_name not in all_tools:
        raise KeyError(f"Built-in tool '{tool_name}' not found")

    return all_tools[tool_name]


# List of all built-in tool names (without prefix) for easy reference
BUILTIN_TOOL_NAMES = [
    "bash_execute",
    "read_file",
    "write_file",
    "list_directory",
    "get_current_directory",
    "todo_read",
    "todo_write",
    "replace_in_file",
    "webfetch",
    "task",
    "task_status",
    "task_results",
    "emit_result",
]
