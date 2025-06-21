#!/usr/bin/env python3
"""Simplified Deepseek host with direct tool integration."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
from openai import OpenAI

from config import HostConfig, load_config, create_sample_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDeepseekHost:
    """Simplified Deepseek host with direct tool integration."""
    
    def __init__(self, config: HostConfig):
        self.config = config
        self.deepseek_config = config.get_deepseek_config()
        
        # Initialize Deepseek client
        self.deepseek_client = OpenAI(
            api_key=self.deepseek_config.api_key,
            base_url=self.deepseek_config.base_url
        )
        
        logger.info(f"Initialized Simple Deepseek Host with model: {self.deepseek_config.model}")
    
    def get_available_tools(self) -> List[Dict]:
        """Get list of available tools in Deepseek format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash_execute",
                    "description": "Execute a bash command and return the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The bash command to execute"},
                            "timeout": {"type": "integer", "default": 120, "description": "Timeout in seconds"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file with line numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file to read"},
                            "offset": {"type": "integer", "description": "Line number to start from"},
                            "limit": {"type": "integer", "description": "Number of lines to read"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file to write"},
                            "content": {"type": "string", "description": "Content to write to the file"}
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories in a directory path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to list (use current directory if not specified)"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_directory",
                    "description": "Get the current working directory",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        try:
            if tool_name == "bash_execute":
                return self._bash_execute(arguments)
            elif tool_name == "read_file":
                return self._read_file(arguments)
            elif tool_name == "write_file":
                return self._write_file(arguments)
            elif tool_name == "list_directory":
                return self._list_directory(arguments)
            elif tool_name == "get_current_directory":
                return self._get_current_directory()
            else:
                return f"Error: Unknown tool {tool_name}"
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def _bash_execute(self, args: Dict[str, Any]) -> str:
        """Execute a bash command."""
        command = args.get("command", "")
        timeout = args.get("timeout", 120)
        
        logger.info(f"Executing bash command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            output += f"Return code: {result.returncode}"
            
            return output
            
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _read_file(self, args: Dict[str, Any]) -> str:
        """Read a file."""
        file_path = Path(args["file_path"]).resolve()
        offset = args.get("offset")
        limit = args.get("limit")
        
        try:
            if not file_path.exists():
                return f"Error: File does not exist: {file_path}"
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            start = (offset - 1) if offset else 0
            end = start + limit if limit else len(lines)
            selected_lines = lines[start:end]
            
            result = ""
            for i, line in enumerate(selected_lines, start=start + 1):
                if len(line) > 2000:
                    line = line[:2000] + "...[truncated]"
                result += f"{i:6d}→{line}"
            
            return result
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _write_file(self, args: Dict[str, Any]) -> str:
        """Write to a file."""
        file_path = Path(args["file_path"]).resolve()
        content = args["content"]
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {file_path}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _list_directory(self, args: Dict[str, Any]) -> str:
        """List directory contents."""
        path = Path(args["path"]).resolve()
        
        try:
            if not path.exists():
                return f"Error: Directory does not exist: {path}"
            
            if not path.is_dir():
                return f"Error: Path is not a directory: {path}"
            
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
            
            return "\n".join(items) if items else "Directory is empty"
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def _get_current_directory(self) -> str:
        """Get current directory."""
        return str(Path.cwd().resolve())
    
    async def chat_completion(self, messages: List[Dict[str, str]], stream: bool = None) -> Union[str, Any]:
        """Handle chat completion using Deepseek with tool support."""
        if stream is None:
            stream = self.deepseek_config.stream
        
        tools = self.get_available_tools()
        
        try:
            # Make request to Deepseek
            response = self.deepseek_client.chat.completions.create(
                model=self.deepseek_config.model,
                messages=messages,
                temperature=self.deepseek_config.temperature,
                max_tokens=self.deepseek_config.max_tokens,
                stream=stream,
                tools=tools
            )
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return await self._handle_complete_response(response, messages)
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return f"Error: {str(e)}"
    
    async def _handle_complete_response(self, response, original_messages: List[Dict[str, str]]) -> str:
        """Handle non-streaming response from Deepseek."""
        choice = response.choices[0]
        message = choice.message
        
        # Check if the model wants to call tools
        if message.tool_calls:
            logger.info(f"Deepseek wants to call {len(message.tool_calls)} tool(s)")
            
            # Execute tool calls
            tool_messages = original_messages.copy()
            tool_messages.append({
                "role": "assistant", 
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                
                # Execute the tool
                tool_result = self.execute_tool(tool_name, arguments)
                
                # Add tool result to messages
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # Make another request with tool results
            logger.info("Making final request to Deepseek with tool results...")
            final_response = self.deepseek_client.chat.completions.create(
                model=self.deepseek_config.model,
                messages=tool_messages,
                temperature=self.deepseek_config.temperature,
                max_tokens=self.deepseek_config.max_tokens,
                stream=False
            )
            
            final_content = final_response.choices[0].message.content
            logger.info(f"Final response: {final_content}")
            return final_content
        else:
            return message.content
    
    def _handle_streaming_response(self, response):
        """Handle streaming response from Deepseek."""
        def stream_generator():
            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
        
        return stream_generator()


async def interactive_chat(host: SimpleDeepseekHost):
    """Run an interactive chat session."""
    print(f"Simple Deepseek Host - Interactive Chat")
    print(f"Model: {host.deepseek_config.model}")
    print(f"Available tools: {len(host.get_available_tools())}")
    print("Type 'quit' to exit, 'tools' to list available tools")
    print("-" * 50)
    
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'tools':
                tools = host.get_available_tools()
                print("\nAvailable tools:")
                for tool in tools:
                    func = tool['function']
                    print(f"  - {func['name']}: {func['description']}")
                continue
            elif not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response from Deepseek
            print("\nAssistant: ", end="", flush=True)
            
            response = await host.chat_completion(messages, stream=True)
            
            if hasattr(response, '__iter__'):
                # Streaming response
                full_response = ""
                for chunk in response:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # New line after streaming
                
                # Add assistant response to messages
                messages.append({"role": "assistant", "content": full_response})
            else:
                # Non-streaming response
                print(response)
                messages.append({"role": "assistant", "content": response})
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


@click.group()
def cli():
    """Simple Deepseek Host - AI with local tools."""
    pass


@cli.command()
def init():
    """Initialize configuration file."""
    create_sample_env()


@cli.command()
def chat():
    """Start interactive chat session."""
    async def run_chat():
        try:
            config = load_config()
            
            if not config.deepseek_api_key:
                click.echo("Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            host = SimpleDeepseekHost(config)
            await interactive_chat(host)
            
        except KeyboardInterrupt:
            pass
    
    asyncio.run(run_chat())


@cli.command()
@click.argument('message')
def ask(message):
    """Ask a single question."""
    async def run_ask():
        try:
            config = load_config()
            
            if not config.deepseek_api_key:
                click.echo("Error: DEEPSEEK_API_KEY not set. Run 'init' command first and update .env file.")
                return
            
            host = SimpleDeepseekHost(config)
            
            messages = [{"role": "user", "content": message}]
            response = await host.chat_completion(messages, stream=False)
            
            if response:
                click.echo(response)
            else:
                click.echo("No response received")
            
        except Exception as e:
            click.echo(f"Error: {e}")
            import traceback
            click.echo(traceback.format_exc())
    
    asyncio.run(run_ask())


if __name__ == "__main__":
    cli()