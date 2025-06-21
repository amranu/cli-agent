"""Modern interactive prompt management using Rich and Prompt Toolkit."""

import asyncio
import sys
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text


class InteractivePromptManager:
    """Modern interactive prompt manager with Rich formatting and Prompt Toolkit input."""
    
    def __init__(self):
        self.console = Console()
        self.session = None
        self.history = FileHistory(".agent_history")
        self.interrupted = False
        self._setup_session()
    
    def _setup_session(self):
        """Setup the prompt session with key bindings and features."""
        # Key bindings for interruption
        bindings = KeyBindings()
        
        @bindings.add('escape')
        def _(event):
            """Handle escape key to interrupt operations."""
            self.interrupted = True
            get_app().exit()
        
        @bindings.add('c-c')
        def _(event):
            """Handle Ctrl+C to interrupt operations."""
            self.interrupted = True
            get_app().exit()
        
        # Setup command completion
        commands = [
            '/help', '/clear', '/model', '/review', '/tokens', '/compact',
            '/switch-chat', '/switch-reason', '/switch-gemini', '/switch-gemini-pro',
            'quit', 'exit', 'tools'
        ]
        completer = WordCompleter(commands, ignore_case=True)
        
        # Create the session
        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            key_bindings=bindings,
            multiline=False,
            wrap_lines=True,
            mouse_support=True,
            complete_style='multi-column'
        )
    
    def print_welcome(self, model_info: str, tool_count: int):
        """Print a nice welcome message."""
        welcome_panel = Panel.fit(
            f"[bold blue]MCP Agent - Interactive Chat[/bold blue]\n\n"
            f"[green]Model:[/green] {model_info}\n"
            f"[green]Available tools:[/green] {tool_count}\n\n"
            f"[yellow]Commands:[/yellow] 'quit' to exit, 'tools' to list tools, 'ESC' to interrupt\n"
            f"[yellow]Model switching:[/yellow] '/switch-chat', '/switch-reason', '/switch-gemini', '/switch-gemini-pro'\n"
            f"[yellow]Utility:[/yellow] '/compact' to compact conversation, '/tokens' to show token count\n"
            f"[yellow]Input:[/yellow] Enter to send, automatic multiline detection, arrow keys for navigation",
            title="🤖 Welcome",
            border_style="blue"
        )
        self.console.print(welcome_panel)
        self.console.print()
    
    def get_input(self, prompt: str = "You: ") -> Optional[str]:
        """Get user input with modern prompt handling."""
        self.interrupted = False
        try:
            # Create a formatted prompt
            formatted_prompt = FormattedText([
                ('class:prompt', prompt),
            ])
            
            result = self.session.prompt(formatted_prompt)
            return result
        except (EOFError, KeyboardInterrupt):
            self.interrupted = True
            return None
        except Exception as e:
            self.console.print(f"[red]Input error: {e}[/red]")
            return None
    
    def get_multiline_input(self, prompt: str = "You: ") -> Optional[str]:
        """Get potentially multiline input with smart detection."""
        first_line = self.get_input(prompt)
        if first_line is None:
            return None
        
        # Check if this looks like multiline content
        is_likely_multiline = (
            len(first_line) > 300 or
            first_line.startswith(('def ', 'class ', 'import ', 'from ')) or
            '```' in first_line or
            (first_line.endswith(':') and len(first_line) > 80) or
            first_line.endswith('{') or
            first_line.endswith('\\')
        )
        
        if is_likely_multiline:
            self.console.print("[dim]Detected multiline content. Continue typing (empty line to finish):[/dim]")
            lines = [first_line]
            
            while True:
                line = self.get_input("... ")
                if line is None:  # Interrupted
                    return None
                if line.strip() == "":  # Empty line ends multiline input
                    break
                lines.append(line)
            
            return '\n'.join(lines)
        
        return first_line
    
    def print_thinking(self) -> Live:
        """Show a thinking spinner."""
        spinner = Spinner("dots", text="[yellow]Thinking... (press ESC to interrupt)[/yellow]")
        return Live(spinner, console=self.console, refresh_per_second=10)
    
    def print_assistant_response(self, content: str):
        """Print assistant response with nice formatting."""
        # Try to detect if it's markdown
        if any(marker in content for marker in ['```', '**', '*', '#', '-', '1.']):
            try:
                md = Markdown(content)
                panel = Panel(md, title="🤖 Assistant", border_style="green", padding=(1, 2))
                self.console.print(panel)
                return
            except:
                pass  # Fall back to plain text
        
        # Plain text response
        panel = Panel(
            Text(content, style="white"),
            title="🤖 Assistant",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def print_streaming_start(self):
        """Indicate start of streaming response."""
        self.console.print("\n[green]🤖 Assistant[/green] [dim](streaming - press ESC to interrupt)[/dim]")
    
    def print_streaming_chunk(self, chunk: str):
        """Print a chunk of streaming response."""
        self.console.print(chunk, end="", style="white")
    
    def print_streaming_end(self):
        """Clean up after streaming."""
        self.console.print("\n")
    
    def print_error(self, error: str):
        """Print an error message."""
        self.console.print(f"[red]❌ Error: {error}[/red]")
    
    def print_info(self, message: str):
        """Print an info message."""
        self.console.print(f"[blue]ℹ️  {message}[/blue]")
    
    def print_success(self, message: str):
        """Print a success message."""
        self.console.print(f"[green]✅ {message}[/green]")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")
    
    def print_tools_list(self, tools: dict):
        """Print available tools in a nice format."""
        if not tools:
            self.console.print("[yellow]No tools available[/yellow]")
            return
        
        tool_text = "\n".join([f"• [cyan]{name}[/cyan]: {info['description']}" 
                              for name, info in tools.items()])
        
        panel = Panel(
            tool_text,
            title="🛠️ Available Tools",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def print_interrupted(self):
        """Print interruption message."""
        self.console.print("\n[red]🛑 Operation cancelled by user[/red]")
    
    def clear_screen(self):
        """Clear the screen."""
        self.console.clear()
    
    async def monitor_for_interruption(self, task):
        """Monitor for escape key while a task is running."""
        # This is a simplified version - in practice, prompt_toolkit handles this
        # through the key bindings we set up in the session
        try:
            return await task
        except asyncio.CancelledError:
            self.interrupted = True
            self.print_interrupted()
            return None


# Example usage and integration
async def example_interactive_chat(agent):
    """Example of how to use the new prompt manager."""
    prompt_manager = InteractivePromptManager()
    
    # Get model info for welcome message
    if hasattr(agent, 'deepseek_config'):
        model_info = agent.deepseek_config.model
    elif hasattr(agent, 'gemini_config'):
        model_info = agent.gemini_config.model
    else:
        model_info = "Unknown"
    
    prompt_manager.print_welcome(model_info, len(agent.available_tools))
    
    messages = []
    
    while True:
        try:
            # Get user input
            user_input = prompt_manager.get_multiline_input()
            
            if user_input is None:  # Interrupted
                continue
            
            if user_input.lower().strip() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower().strip() == 'tools':
                prompt_manager.print_tools_list(agent.available_tools)
                continue
            
            # Handle slash commands
            if user_input.strip().startswith('/'):
                try:
                    if hasattr(agent, 'slash_commands'):
                        slash_response = await agent.slash_commands.handle_slash_command(user_input.strip())
                        if slash_response:
                            prompt_manager.print_assistant_response(slash_response)
                    continue
                except Exception as e:
                    prompt_manager.print_error(f"Slash command error: {e}")
                    continue
            
            if not user_input.strip():
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Show thinking indicator and get response
            with prompt_manager.print_thinking():
                # Check for interruption during thinking
                if prompt_manager.interrupted:
                    continue
                
                response = await agent.generate_response(messages)
            
            if hasattr(response, '__aiter__'):
                # Streaming response
                prompt_manager.print_streaming_start()
                full_response = ""
                
                try:
                    async for chunk in response:
                        if prompt_manager.interrupted:
                            break
                        
                        chunk_text = str(chunk)
                        prompt_manager.print_streaming_chunk(chunk_text)
                        full_response += chunk_text
                    
                    prompt_manager.print_streaming_end()
                    
                    if full_response and not prompt_manager.interrupted:
                        messages.append({"role": "assistant", "content": full_response})
                        
                except Exception as e:
                    prompt_manager.print_error(f"Streaming error: {e}")
            else:
                # Non-streaming response
                prompt_manager.print_assistant_response(str(response))
                messages.append({"role": "assistant", "content": str(response)})
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            prompt_manager.print_error(f"Chat error: {e}")
    
    prompt_manager.print_info("Chat session ended.")