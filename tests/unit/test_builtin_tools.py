"""Unit tests for builtin tools functionality."""

import pytest

from cli_agent.core.builtin_tool_executor import BuiltinToolExecutor
from cli_agent.tools.builtin_tools import (
    get_all_builtin_tools,
    get_bash_execute_tool,
    get_current_directory_tool,
    get_emit_result_tool,
    get_list_directory_tool,
    get_read_file_tool,
    get_replace_in_file_tool,
    get_task_results_tool,
    get_task_status_tool,
    get_task_tool,
    get_todo_read_tool,
    get_todo_write_tool,
    get_web_fetch_tool,
    get_write_file_tool,
)


@pytest.mark.unit
class TestBuiltinTools:
    """Test builtin tools definitions and structure."""

    def test_get_all_builtin_tools(self):
        """Test getting all builtin tools."""
        tools = get_all_builtin_tools()

        assert isinstance(tools, dict)
        assert len(tools) > 0

        # Check essential tools are present
        essential_tools = [
            "builtin:bash_execute",
            "builtin:read_file",
            "builtin:write_file",
            "builtin:list_directory",
            "builtin:get_current_directory",
            "builtin:todo_read",
            "builtin:todo_write",
            "builtin:replace_in_file",
            "builtin:webfetch",
            "builtin:task",
            "builtin:task_status",
            "builtin:task_results",
            "builtin:emit_result",
        ]

        for tool in essential_tools:
            assert tool in tools

    def test_bash_execute_tool(self):
        """Test bash_execute tool definition."""
        tool = get_bash_execute_tool()

        assert tool["server"] == "builtin"
        assert tool["name"] == "bash_execute"
        assert "description" in tool
        assert "schema" in tool

        # Check schema structure
        schema = tool["schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "command" in schema["properties"]
        assert "required" in schema
        assert "command" in schema["required"]

    def test_read_file_tool(self):
        """Test read_file tool definition."""
        tool = get_read_file_tool()

        assert tool["name"] == "read_file"
        assert "file_path" in tool["schema"]["properties"]
        assert "file_path" in tool["schema"]["required"]

    def test_write_file_tool(self):
        """Test write_file tool definition."""
        tool = get_write_file_tool()

        assert tool["name"] == "write_file"
        schema = tool["schema"]

        assert "file_path" in schema["properties"]
        assert "content" in schema["properties"]
        assert "file_path" in schema["required"]
        assert "content" in schema["required"]

    def test_list_directory_tool(self):
        """Test list_directory tool definition."""
        tool = get_list_directory_tool()

        assert tool["name"] == "list_directory"
        assert "directory_path" in tool["schema"]["properties"]
        # directory_path is required
        assert "directory_path" in tool["schema"]["required"]

    def test_get_current_directory_tool(self):
        """Test get_current_directory tool definition."""
        tool = get_current_directory_tool()

        assert tool["name"] == "get_current_directory"
        # Should not require any parameters
        assert len(tool["schema"].get("required", [])) == 0

    def test_todo_tools(self):
        """Test todo management tools."""
        read_tool = get_todo_read_tool()
        write_tool = get_todo_write_tool()

        assert read_tool["name"] == "todo_read"
        assert write_tool["name"] == "todo_write"

        # todo_write should require todos parameter
        assert "todos" in write_tool["schema"]["properties"]
        assert "todos" in write_tool["schema"]["required"]

    def test_replace_in_file_tool(self):
        """Test replace_in_file tool definition."""
        tool = get_replace_in_file_tool()

        assert tool["name"] == "replace_in_file"
        schema = tool["schema"]

        required_props = ["file_path", "old_text", "new_text"]
        for prop in required_props:
            assert prop in schema["properties"]
            assert prop in schema["required"]

    def test_webfetch_tool(self):
        """Test webfetch tool definition."""
        tool = get_web_fetch_tool()

        assert tool["name"] == "webfetch"
        schema = tool["schema"]

        assert "url" in schema["properties"]
        assert "url" in schema["required"]
        # limit is optional
        if "limit" in schema["properties"]:
            assert "limit" not in schema.get("required", [])

    def test_task_management_tools(self):
        """Test task management tools."""
        task_tool = get_task_tool()
        status_tool = get_task_status_tool()
        results_tool = get_task_results_tool()

        # Task tool
        assert task_tool["name"] == "task"
        assert "description" in task_tool["schema"]["properties"]
        assert "prompt" in task_tool["schema"]["properties"]

        # Status tool
        assert status_tool["name"] == "task_status"
        # task_id should be optional
        assert "task_id" in status_tool["schema"]["properties"]

        # Results tool
        assert results_tool["name"] == "task_results"

    def test_emit_result_tool(self):
        """Test emit_result tool (subagents only)."""
        tool = get_emit_result_tool()

        assert tool["name"] == "emit_result"
        schema = tool["schema"]

        assert "result" in schema["properties"]
        assert "result" in schema["required"]

        # summary should be optional
        assert "summary" in schema["properties"]
        assert "summary" not in schema.get("required", [])

    def test_tool_structure_consistency(self):
        """Test that all tools have consistent structure."""
        tools = get_all_builtin_tools()

        for tool_key, tool_def in tools.items():
            # Each tool should have required fields
            assert "server" in tool_def
            assert "name" in tool_def
            assert "description" in tool_def
            assert "schema" in tool_def

            # Server should be "builtin"
            assert tool_def["server"] == "builtin"

            # Schema should be valid JSON schema structure
            schema = tool_def["schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

            # Description should not be empty
            assert len(tool_def["description"]) > 0


class MockAgent:
    """Mock agent for testing builtin tool executor."""

    def __init__(self):
        self.is_subagent = False
        self.subagent_manager = None


@pytest.mark.unit
class TestBuiltinToolExecutor:
    """Test builtin tool executor functionality."""

    def test_brace_expansion(self):
        """Test brace expansion functionality in glob patterns."""
        agent = MockAgent()
        executor = BuiltinToolExecutor(agent)

        # Test simple brace expansion
        expanded = executor._expand_braces("*.{py,md}")
        assert expanded == ["*.py", "*.md"]

        # Test complex brace expansion
        expanded = executor._expand_braces("**/*.{js,ts,py}")
        assert expanded == ["**/*.js", "**/*.ts", "**/*.py"]

        # Test no braces (should return original)
        expanded = executor._expand_braces("**/*.py")
        assert expanded == ["**/*.py"]

        # Test empty braces (edge case)
        expanded = executor._expand_braces("*.{}")
        assert expanded == ["*."]

        # Test single option in braces
        expanded = executor._expand_braces("*.{py}")
        assert expanded == ["*.py"]

        # Test nested path with braces
        expanded = executor._expand_braces("src/**/*.{js,ts}")
        assert expanded == ["src/**/*.js", "src/**/*.ts"]

    def test_tool_naming_convention(self):
        """Test tool naming conventions."""
        tools = get_all_builtin_tools()

        for tool_key, tool_def in tools.items():
            # Tool key should match "builtin:name" format
            assert tool_key.startswith("builtin:")
            expected_name = tool_key.replace("builtin:", "")
            assert tool_def["name"] == expected_name

    def test_schema_validation(self):
        """Test that tool schemas are valid."""
        tools = get_all_builtin_tools()

        for tool_key, tool_def in tools.items():
            schema = tool_def["schema"]

            # Basic JSON schema validation
            assert isinstance(schema, dict)
            assert schema.get("type") == "object"

            if "properties" in schema:
                assert isinstance(schema["properties"], dict)

            if "required" in schema:
                assert isinstance(schema["required"], list)
                # All required fields should exist in properties
                for req_field in schema["required"]:
                    assert req_field in schema["properties"]

    def test_tool_categories(self):
        """Test that tools are properly categorized."""
        tools = get_all_builtin_tools()

        # File operations
        file_tools = [
            "read_file",
            "write_file",
            "list_directory",
            "get_current_directory",
            "replace_in_file",
        ]
        for tool in file_tools:
            assert f"builtin:{tool}" in tools

        # System operations
        system_tools = ["bash_execute"]
        for tool in system_tools:
            assert f"builtin:{tool}" in tools

        # Task management
        task_tools = ["task", "task_status", "task_results"]
        for tool in task_tools:
            assert f"builtin:{tool}" in tools

        # Utility tools
        utility_tools = ["todo_read", "todo_write", "webfetch", "emit_result"]
        for tool in utility_tools:
            assert f"builtin:{tool}" in tools

    def test_tool_descriptions_quality(self):
        """Test that tool descriptions are informative."""
        tools = get_all_builtin_tools()

        for tool_key, tool_def in tools.items():
            description = tool_def["description"]

            # Should be reasonably descriptive
            assert len(description) > 10

            # Should not just be the tool name
            tool_name = tool_def["name"]
            assert description.lower() != tool_name.lower()

            # Should contain some action words
            action_words = [
                "execute",
                "read",
                "write",
                "list",
                "get",
                "manage",
                "fetch",
                "emit",
                "spawn",
                "check",
                "replace",
                "retrieve",
                "search",
            ]
            assert any(
                word in description.lower() for word in action_words
            ), f"Tool {tool_key} description '{description}' should contain an action word"

    def test_parameter_types(self):
        """Test that parameter types are properly defined."""
        tools = get_all_builtin_tools()

        for tool_key, tool_def in tools.items():
            schema = tool_def["schema"]

            if "properties" in schema:
                for prop_name, prop_def in schema["properties"].items():
                    # Each property should have a type
                    assert "type" in prop_def

                    # Type should be valid JSON schema type
                    valid_types = [
                        "string",
                        "number",
                        "integer",
                        "boolean",
                        "array",
                        "object",
                    ]
                    assert prop_def["type"] in valid_types

    def test_glob_tool(self):
        """Test the glob tool definition and basic functionality."""
        tools = get_all_builtin_tools()

        # Check tool exists
        assert "builtin:glob" in tools

        glob_tool = tools["builtin:glob"]

        # Check basic structure
        assert glob_tool["name"] == "glob"
        assert glob_tool["server"] == "builtin"
        assert "description" in glob_tool
        assert "schema" in glob_tool

        # Check schema structure
        schema = glob_tool["schema"]
        assert "pattern" in schema["properties"]
        assert schema["properties"]["pattern"]["type"] == "string"
        assert "path" in schema["properties"]
        assert schema["properties"]["path"]["type"] == "string"
        assert schema["required"] == ["pattern"]

        # Test basic execution
        from cli_agent.core.builtin_tool_executor import BuiltinToolExecutor

        class MockAgent:
            is_subagent = False

        executor = BuiltinToolExecutor(MockAgent())

        # Test with a simple pattern that should find some files
        result = executor.glob({"pattern": "*.py"})
        assert isinstance(result, str)
        assert "Found" in result or "No files" in result or "Error" in result

    def test_grep_tool(self):
        """Test the grep tool definition and basic functionality."""
        tools = get_all_builtin_tools()

        # Check tool exists
        assert "builtin:grep" in tools

        grep_tool = tools["builtin:grep"]

        # Check basic structure
        assert grep_tool["name"] == "grep"
        assert grep_tool["server"] == "builtin"
        assert "description" in grep_tool
        assert "schema" in grep_tool

        # Check schema structure
        schema = grep_tool["schema"]
        assert "pattern" in schema["properties"]
        assert schema["properties"]["pattern"]["type"] == "string"
        assert "path" in schema["properties"]
        assert schema["properties"]["path"]["type"] == "string"
        assert "include" in schema["properties"]
        assert schema["properties"]["include"]["type"] == "string"
        assert schema["required"] == ["pattern"]

        # Test basic execution
        from cli_agent.core.builtin_tool_executor import BuiltinToolExecutor

        class MockAgent:
            is_subagent = False

        executor = BuiltinToolExecutor(MockAgent())

        # Test with a simple pattern
        result = executor.grep({"pattern": "import"})
        assert isinstance(result, str)
        assert "Found" in result or "No files" in result or "Error" in result

    def test_glob_grep_tool_integration(self):
        """Test that glob and grep tools are properly integrated."""
        tools = get_all_builtin_tools()

        # Both tools should be in the registry
        assert "builtin:glob" in tools
        assert "builtin:grep" in tools

        # Check they're in the tool names list
        from cli_agent.tools.builtin_tools import BUILTIN_TOOL_NAMES

        assert "glob" in BUILTIN_TOOL_NAMES
        assert "grep" in BUILTIN_TOOL_NAMES

        # Test executor has the methods
        from cli_agent.core.builtin_tool_executor import BuiltinToolExecutor

        class MockAgent:
            is_subagent = False

        executor = BuiltinToolExecutor(MockAgent())
        assert hasattr(executor, "glob")
        assert hasattr(executor, "grep")
        assert callable(getattr(executor, "glob"))
        assert callable(getattr(executor, "grep"))

    def test_multiedit_tool(self):
        """Test the multiedit tool definition and basic functionality."""
        tools = get_all_builtin_tools()

        # Check tool exists
        assert "builtin:multiedit" in tools

        multiedit_tool = tools["builtin:multiedit"]

        # Check basic structure
        assert multiedit_tool["name"] == "multiedit"
        assert multiedit_tool["server"] == "builtin"
        assert "description" in multiedit_tool
        assert "schema" in multiedit_tool

        # Check schema structure
        schema = multiedit_tool["schema"]
        assert "file_path" in schema["properties"]
        assert schema["properties"]["file_path"]["type"] == "string"
        assert "edits" in schema["properties"]
        assert schema["properties"]["edits"]["type"] == "array"
        assert schema["required"] == ["file_path", "edits"]

        # Check edits array schema
        edits_schema = schema["properties"]["edits"]
        assert "items" in edits_schema
        edit_item = edits_schema["items"]
        assert "old_string" in edit_item["properties"]
        assert "new_string" in edit_item["properties"]
        assert "replace_all" in edit_item["properties"]
        assert edit_item["required"] == ["old_string", "new_string"]

        # Test executor has the method
        from cli_agent.core.builtin_tool_executor import BuiltinToolExecutor

        class MockAgent:
            is_subagent = False

        executor = BuiltinToolExecutor(MockAgent())
        assert hasattr(executor, "multiedit")
        assert callable(getattr(executor, "multiedit"))

    def test_multiedit_tool_integration(self):
        """Test that multiedit tool is properly integrated."""
        tools = get_all_builtin_tools()

        # Tool should be in the registry
        assert "builtin:multiedit" in tools

        # Check it's in the tool names list
        from cli_agent.tools.builtin_tools import BUILTIN_TOOL_NAMES

        assert "multiedit" in BUILTIN_TOOL_NAMES
