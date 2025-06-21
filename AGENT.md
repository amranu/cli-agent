# AGENT.md

This document provides a technical overview of the MCP Agent codebase, designed to help a coding agent understand the system's architecture, components, and core functionalities.

## 1. Overview

The MCP Agent is an extensible framework for creating language model-powered agents that can interact with a variety of tools. It provides a core set of functionalities, including a command-line interface (CLI), interactive chat, tool management, and conversation handling. The agent is designed to be model-agnostic, with specific implementations for different language models like Deepseek and Google Gemini.

The key features of the agent are:

- **Extensible Tool Integration:** The agent can use both built-in tools and external tools through the **Multi-Capability Peripheral (MCP) protocol**.
- **Model Agnosticism:** The agent's core logic is separated from the language model implementation, allowing for easy integration of new models.
- **Interactive and Non-Interactive Modes:** The agent can be used in an interactive chat mode or for single-turn "ask" requests.
- **Advanced Chat Features:** The interactive chat includes features like multiline input, command history, and the ability to interrupt ongoing operations.
- **Conversation Management:** The agent can automatically compact long conversations to stay within the language model's context window.

## 2. Core Components

The codebase is structured around a few key components:

### `agent.py`: The Core Agent Framework

This file contains the foundational `BaseMCPAgent` class, which is an abstract base class that defines the core agent functionality.

- **`BaseMCPAgent` class:**
    - **Initialization:** The constructor (`__init__`) initializes the agent's configuration, loads built-in tools, and sets up the MCP client management system.
    - **Tool Management:**
        - `_add_builtin_tools()`: Registers a set of built-in tools, such as `bash_execute`, `read_file`, `write_file`, and more.
        - `_execute_builtin_tool()`: Executes the built-in tools.
        - `start_mcp_server()`: Starts and connects to external MCP tool servers.
        - `_execute_mcp_tool()`: Executes both built-in and external tools.
    - **Conversation Management:**
        - `conversation_history`: Stores the conversation as a list of messages.
        - `compact_conversation()`: Summarizes the conversation to reduce token usage.
    - **Abstract Methods:**
        - `generate_response()`: Must be implemented by subclasses to generate a response from a specific language model.
        - `convert_tools_to_llm_format()`: Must be implemented by subclasses to format tools for the specific language model's API.
        - `parse_tool_calls()`: Must be implemented by subclasses to parse tool calls from the model's response.
- **`SlashCommandManager` class:**
  - Manages slash commands like `/help`, `/clear`, and `/model` to control the agent during a chat session.
- **`InterruptibleInput` class:**
  - Provides a robust input handler for the interactive chat, with support for multiline input and interruption.
- **CLI:**
  - The file also defines the command-line interface using the `click` library, with commands like `chat`, `ask`, and `mcp` for managing servers.

### `mcp_deepseek_host.py`: Deepseek Model Implementation

This file provides the `MCPDeepseekHost` class, which is a concrete implementation of `BaseMCPAgent` for the Deepseek language model.

- **`MCPDeepseekHost` class:**
    - **Deepseek API Integration:** It uses the `openai` library to communicate with the Deepseek API.
    - **Tool Formatting:**
        - `_convert_tools_to_deepseek_format()`: Implements the `convert_tools_to_llm_format` method to format tools as expected by the Deepseek API.
    - **Tool Call Parsing:**
        - `_parse_deepseek_tool_calls()`: Implements the `parse_tool_calls` method to parse the tool-calling syntax used by Deepseek.
    - **System Prompt:**
        - `_create_system_prompt()`: Customizes the system prompt with instructions specific to the Deepseek model.
    - **Streaming and Tool Use:**
        - `chat_completion()`: Manages the interaction with the Deepseek API, including handling streaming responses and iterative tool calls.

### `mcp_gemini_host.py`: Gemini Model Implementation

This file provides the `MCPGeminiHost` class, which is a concrete implementation of `BaseMCPAgent` for the Google Gemini language model.

- **`MCPGeminiHost` class:**
    - **Gemini API Integration:** It uses the `google-generativeai` library to communicate with the Gemini API.
    - **Tool Formatting:**
        - `_convert_tools_to_gemini_format()`: Implements the `convert_tools_to_llm_format` method to format tools for the Gemini API.
    - **Tool Call Parsing:**
        - `_parse_response_content()`: Implements the `parse_tool_calls` method to parse tool calls from the Gemini API's response. It is designed to handle multiple response formats, including standard function calls, Python-style calls, and XML-style calls.
    - **System Prompt:**
        - `_create_system_prompt()`: Customizes the system prompt for the Gemini model.
    - **Streaming and Tool Use:**
        - `chat_completion()`: Manages the interaction with the Gemini API, including handling streaming responses and iterative tool calls. It also includes retry logic for API requests.

## 3. Execution Flow

The agent's execution flow can be summarized as follows:

1.  **Initialization:**
    - The `main` function in `agent.py` is the entry point.
    - The appropriate host class (`MCPDeepseekHost` or `MCPGeminiHost`) is instantiated based on the configuration.
    - MCP servers are started, and the available tools are registered.
2.  **User Input:**
    - In interactive mode, the `interactive_chat` function captures user input.
    - In non-interactive mode, the `ask` command takes a single message as input.
3.  **Message Processing:**
    - The user's message is added to the conversation history.
    - The `chat_completion` method of the host class is called.
4.  **Model-Specific Processing:**
    - The host class formats the tools for the specific language model.
    - It sends the conversation history and tools to the language model's API.
5.  **Response Handling:**
    - The host class receives the response from the model.
    - If the response contains tool calls, the `parse_tool_calls` method is used to extract them.
    - The agent executes the requested tools and captures their output.
    - The tool output is added to the conversation history, and the process is repeated until the model generates a final response without tool calls.
6.  **Output:**
    - The final response is streamed to the user in interactive mode or printed to the console in non-interactive mode.

## 4. Tool Integration

The agent's tool integration is a core feature and is based on the following principles:

- **Built-in Tools:** A set of essential tools is defined directly in `agent.py`. These tools are available to all agent implementations.
- **External Tools (MCP):** The agent can connect to external tool servers using the MCP protocol. This allows for the integration of tools written in any language.
- **Tool Abstraction:** The `BaseMCPAgent` class provides a unified interface for executing both built-in and external tools, making the tool execution process transparent to the agent's core logic.
- **Model-Specific Formatting:** Each model implementation is responsible for formatting the tool specifications in the way that its API expects.

## 5. Key Design Patterns

The codebase employs several key design patterns:

- **Abstract Base Classes:** The use of `BaseMCPAgent` as an abstract base class allows for a clear separation of concerns between the core agent logic and the model-specific implementations.
- **Strategy Pattern:** The different model implementations (`MCPDeepseekHost`, `MCPGeminiHost`) can be seen as different strategies for generating responses. The main application can switch between these strategies based on the configuration.
- **Command Pattern:** The CLI is implemented using the `click` library, which follows the command pattern to encapsulate actions as objects.
- **Asynchronous Programming:** The agent uses `asyncio` to handle I/O-bound operations, such as making API requests and communicating with MCP servers, in a non-blocking way.

This document should provide a solid foundation for any coding agent to understand and work with this codebase.
