"""Built-in tool execution implementations for BaseMCPAgent."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import requests
from config import get_config_dir


class BuiltinToolExecutor:
    """Handles execution of built-in tools for BaseMCPAgent."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    async def bash_execute(self, args: Dict[str, Any]) -> str:
        """Execute bash commands and return output with interrupt support."""
        command = args.get("command", "")
        timeout = args.get("timeout", 120)

        if not command:
            return "Error: No command provided"

        try:
            from cli_agent.core.interrupt_aware_streaming import (
                InterruptAwareSubprocess,
            )

            # Use interrupt-aware subprocess execution
            result = await InterruptAwareSubprocess.run_with_interrupt_checking(
                command, timeout=timeout, operation_name=f"bash: {command[:50]}..."
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\nReturn code: {result.returncode}"

            return output or "Command completed with no output"

        except KeyboardInterrupt:
            return "Error: Command execution interrupted by user"
        except asyncio.TimeoutError:
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

    def _get_todo_file_path(self) -> str:
        """Get the session-specific todo file path."""
        # Create .config/agent/todos directory if it doesn't exist
        config_dir = get_config_dir()
        todos_dir = config_dir / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)

        # Get session ID from agent if available, otherwise use 'default'
        session_id = getattr(self.agent, "_session_id", "default")
        if not session_id or session_id == "None":
            session_id = "default"

        # Debug logging to track session ID usage
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Todo file path: session_id='{session_id}', file=todos/{session_id}.json"
        )

        return str(todos_dir / f"{session_id}.json")

    def todo_read(self, args: Dict[str, Any]) -> str:
        """Read the current session's todo list."""
        # Debug info to help track down the session issue
        session_id = getattr(self.agent, "_session_id", "NOT_SET")
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"todo_read called: agent._session_id = '{session_id}'")

        todo_file = self._get_todo_file_path()

        try:
            if not os.path.exists(todo_file):
                logger.info(
                    f"Todo file doesn't exist: {todo_file}, returning empty list"
                )
                return "[]"  # Empty todo list

            with open(todo_file, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(
                    f"Read todo file: {todo_file}, content length: {len(content)}"
                )
                return content

        except Exception as e:
            return f"Error reading todo list: {str(e)}"

    def todo_write(self, args: Dict[str, Any]) -> str:
        """Write/update the current session's todo list."""
        todos = args.get("todos", [])
        todo_file = self._get_todo_file_path()

        try:
            with open(todo_file, "w", encoding="utf-8") as f:
                json.dump(todos, f, indent=2)

            # Return the actual todo list data to the LLM for proper feedback
            session_id = getattr(self.agent, "_session_id", "default")
            session_info = f" (session: {session_id})"
            return f"Successfully updated todo list with {len(todos)} items{session_info}. Current todo list:\n{json.dumps(todos, indent=2)}"

        except Exception as e:
            return f"Error writing todo list: {str(e)}"

    def replace_in_file(self, args: Dict[str, Any]) -> str:
        """Replace text in a file."""
        file_path = args.get("file_path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")

        if not file_path:
            return "Error: No file path provided"
        if not old_text:
            return "Error: No old text to replace provided"

        # Validate spacing patterns to help detect model issues
        warnings = []

        # Check for leading/trailing whitespace that might be lost
        if old_text != old_text.strip():
            if old_text.startswith(" ") or old_text.startswith("\t"):
                warnings.append("old_text has leading whitespace")
            if old_text.endswith(" ") or old_text.endswith("\t"):
                warnings.append("old_text has trailing whitespace")

        if new_text != new_text.strip():
            if new_text.startswith(" ") or new_text.startswith("\t"):
                warnings.append("new_text has leading whitespace")
            if new_text.endswith(" ") or new_text.endswith("\t"):
                warnings.append("new_text has trailing whitespace")

        # Log warnings for debugging
        if warnings:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"replace_in_file whitespace check: {', '.join(warnings)}")

        try:
            # Read the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if old text exists
            if old_text not in content:
                # Provide helpful debugging for spacing issues
                lines = content.split("\n")
                old_lines = old_text.split("\n")

                # Check if text exists but with different whitespace
                stripped_old = old_text.strip()
                if stripped_old and stripped_old in content:
                    return f"Error: Text found but whitespace doesn't match in {file_path}. Check for exact indentation, tabs vs spaces, and trailing whitespace. Use read_file to see exact formatting."

                # Check if first line of old_text exists (might be partial match)
                if old_lines and old_lines[0].strip():
                    first_line_stripped = old_lines[0].strip()
                    matching_lines = [
                        i for i, line in enumerate(lines) if first_line_stripped in line
                    ]
                    if matching_lines:
                        hint_lines = matching_lines[:3]  # Show first 3 matches
                        return f"Error: Text to replace not found in {file_path}. Found similar text on line(s) {hint_lines} - check exact whitespace and indentation."

                return f"Error: Text to replace not found in {file_path}. Ensure you copy the exact text including all whitespace from read_file output."

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

    def multiedit(self, args: Dict[str, Any]) -> str:
        """Perform multiple edits to a single file in one operation."""
        file_path = args.get("file_path", "")
        edits = args.get("edits", [])

        if not file_path:
            return "Error: No file path provided"
        if not edits:
            return "Error: No edits provided"

        try:
            # Read the file once
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            edit_results = []
            current_content = content

            # Apply each edit in sequence
            for i, edit in enumerate(edits):
                old_string = edit.get("old_string", "")
                new_string = edit.get("new_string", "")
                replace_all = edit.get("replace_all", False)

                if not old_string:
                    return f"Error: Edit {i+1} has no old_string"

                # Check if old_string exists in current content
                if old_string not in current_content:
                    # Provide helpful debugging
                    stripped_old = old_string.strip()
                    if stripped_old and stripped_old in current_content:
                        return f"Error: Edit {i+1} - text found but whitespace doesn't match in {file_path}. Check for exact indentation, tabs vs spaces, and trailing whitespace."
                    return f"Error: Edit {i+1} - text to replace not found in {file_path}. Current edits may have changed the text. Use read_file to check current state."

                # Perform the replacement
                if replace_all:
                    # Replace all occurrences
                    if old_string == new_string:
                        return f"Error: Edit {i+1} - old_string and new_string are identical"

                    count = current_content.count(old_string)
                    current_content = current_content.replace(old_string, new_string)
                    edit_results.append(f"Edit {i+1}: Replaced {count} occurrence(s)")
                else:
                    # Replace first occurrence only
                    if old_string == new_string:
                        return f"Error: Edit {i+1} - old_string and new_string are identical"

                    if current_content.count(old_string) > 1:
                        # Find-and-replace for first occurrence
                        current_content = current_content.replace(
                            old_string, new_string, 1
                        )
                        edit_results.append(
                            f"Edit {i+1}: Replaced first occurrence (found {current_content.count(old_string) + 1} total)"
                        )
                    else:
                        current_content = current_content.replace(
                            old_string, new_string
                        )
                        edit_results.append(f"Edit {i+1}: Replaced 1 occurrence")

            # Validate that something actually changed
            if current_content == original_content:
                return f"Warning: No changes made to {file_path}. All edits resulted in no changes."

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(current_content)

            result = (
                f"Successfully applied {len(edits)} edits to {file_path}:\n"
                + "\n".join(edit_results)
            )
            return result

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error performing multi-edit: {str(e)}"

    def webfetch(self, args: Dict[str, Any]) -> str:
        """Fetch content from a webpage and convert HTML to markdown."""
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

            # Convert HTML to markdown
            markdown_content = self._html_to_markdown(response.text)
            
            # Split into lines and apply limit
            lines = markdown_content.split("\n")
            
            if limit and len(lines) > limit:
                lines = lines[:limit]
                content = "\n".join(lines) + f"\n\n[Content truncated at {limit} lines]"
            else:
                content = "\n".join(lines)

            return f"Content from {url}:\n\n{content}"

        except requests.exceptions.RequestException as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _html_to_markdown(self, html_content: str) -> str:
        """Convert HTML content to clean markdown format."""
        import re
        from html import unescape
        
        # Remove script and style elements completely
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove comments
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        
        # Convert common HTML elements to markdown
        conversions = [
            # Headers
            (r'<h1[^>]*>(.*?)</h1>', r'# \1'),
            (r'<h2[^>]*>(.*?)</h2>', r'## \1'),
            (r'<h3[^>]*>(.*?)</h3>', r'### \1'),
            (r'<h4[^>]*>(.*?)</h4>', r'#### \1'),
            (r'<h5[^>]*>(.*?)</h5>', r'##### \1'),
            (r'<h6[^>]*>(.*?)</h6>', r'###### \1'),
            
            # Bold and italic
            (r'<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>', r'**\1**'),
            (r'<(?:i|em)[^>]*>(.*?)</(?:i|em)>', r'*\1*'),
            
            # Links
            (r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)'),
            
            # Images
            (r'<img[^>]*src=["\']([^"\']*)["\'][^>]*alt=["\']([^"\']*)["\'][^>]*/?>', r'![\2](\1)'),
            (r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*src=["\']([^"\']*)["\'][^>]*/?>', r'![\1](\2)'),
            (r'<img[^>]*src=["\']([^"\']*)["\'][^>]*/?>', r'![](\1)'),
            
            # Lists
            (r'<ul[^>]*>', '\n'),
            (r'</ul>', '\n'),
            (r'<ol[^>]*>', '\n'),
            (r'</ol>', '\n'),
            (r'<li[^>]*>(.*?)</li>', r'- \1'),
            
            # Paragraphs and line breaks
            (r'<p[^>]*>(.*?)</p>', r'\1\n\n'),
            (r'<br[^>]*/?>', '\n'),
            (r'<hr[^>]*/?>', '\n---\n'),
            
            # Divs and spans (preserve content, remove tags)
            (r'<div[^>]*>(.*?)</div>', r'\1\n'),
            (r'<span[^>]*>(.*?)</span>', r'\1'),
            
            # Tables (simplified)
            (r'<table[^>]*>', '\n'),
            (r'</table>', '\n'),
            (r'<tr[^>]*>', ''),
            (r'</tr>', '\n'),
            (r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', r'\1 | '),
            
            # Code
            (r'<code[^>]*>(.*?)</code>', r'`\1`'),
            (r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```'),
            
            # Blockquotes
            (r'<blockquote[^>]*>(.*?)</blockquote>', r'> \1'),
        ]
        
        # Apply conversions
        for pattern, replacement in conversions:
            html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)
        
        # Decode HTML entities
        html_content = unescape(html_content)
        
        # Clean up whitespace
        # Replace multiple consecutive newlines with double newlines
        html_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', html_content)
        
        # Replace multiple spaces with single space
        html_content = re.sub(r' +', ' ', html_content)
        
        # Trim leading/trailing whitespace from each line
        lines = [line.strip() for line in html_content.split('\n')]
        
        # Remove empty lines at the beginning and end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        
        return '\n'.join(lines)

    def websearch(self, args: Dict[str, Any]) -> str:
        """Search the web using DuckDuckGo direct search."""
        query = args.get("query", "")
        allowed_domains = args.get("allowed_domains", [])
        blocked_domains = args.get("blocked_domains", [])

        if not query:
            return "Error: No search query provided"

        try:
            import re
            from urllib.parse import quote_plus
            
            # Prepare the search query
            encoded_query = quote_plus(query)
            
            # Apply domain filtering if specified
            if allowed_domains:
                site_restriction = " OR ".join([f"site:{domain}" for domain in allowed_domains])
                encoded_query = quote_plus(f"{query} ({site_restriction})")
            
            # Use DuckDuckGo HTML search (no API key required)
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(search_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML response
            html_content = response.text
            results = []
            
            # Extract search results using regex patterns
            # DuckDuckGo HTML structure: results are in divs with class "result"
            result_pattern = r'<div class="result".*?</div>(?=\s*(?:<div class="result"|<div id="links_wrapper"|$))'
            title_pattern = r'<a class="result__title"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
            
            result_matches = re.findall(result_pattern, html_content, re.DOTALL | re.IGNORECASE)
            
            if result_matches:
                results.append(f"Found {len(result_matches)} search results for '{query}'")
                results.append("")
                
                for i, result_html in enumerate(result_matches[:10], 1):  # Limit to 10 results
                    # Extract title and URL
                    title_match = re.search(title_pattern, result_html, re.DOTALL | re.IGNORECASE)
                    snippet_match = re.search(snippet_pattern, result_html, re.DOTALL | re.IGNORECASE)
                    
                    if title_match:
                        url = title_match.group(1)
                        title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
                        
                        # Apply blocked domain filtering
                        if blocked_domains:
                            if any(domain in url for domain in blocked_domains):
                                continue
                        
                        # Clean up snippet
                        snippet = "No description available"
                        if snippet_match:
                            snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
                            # Decode HTML entities
                            snippet = snippet.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                        
                        # Clean up title and URL
                        title = title.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                        
                        results.append(f"**{i}. {title}**")
                        results.append(f"URL: {url}")
                        results.append(f"Description: {snippet}")
                        results.append("")
            else:
                # Fallback: try simpler pattern extraction
                # Look for any links that might be search results
                link_pattern = r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
                links = re.findall(link_pattern, html_content)
                
                if links:
                    results.append(f"Found search results for '{query}' (simplified extraction):")
                    results.append("")
                    
                    # Filter out DuckDuckGo internal links
                    filtered_links = [(url, title) for url, title in links 
                                    if not any(domain in url for domain in ['duckduckgo.com', 'duck.co']) 
                                    and url.startswith(('http://', 'https://'))]
                    
                    for i, (url, title) in enumerate(filtered_links[:10], 1):
                        # Apply blocked domain filtering
                        if blocked_domains:
                            if any(domain in url for domain in blocked_domains):
                                continue
                                
                        results.append(f"**{i}. {title.strip()}**")
                        results.append(f"URL: {url}")
                        results.append("")
                else:
                    results.append("No search results found for this query.")
                    results.append("DuckDuckGo may have changed their HTML structure or the query returned no results.")
                    results.append("Try rephrasing your search terms or using the webfetch tool with specific URLs.")
            
            return f"Web search results:\n\n" + "\n".join(results)
            
        except requests.exceptions.RequestException as e:
            return f"Error performing web search: {str(e)}"
        except Exception as e:
            return f"Error parsing search results: {str(e)}"

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
                # Only include results that were explicitly provided via emit_result
                result_info = {
                    "task_id": task_id,
                    "description": subagent.description,
                    "status": "completed" if subagent.completed else "running",
                    "result": (
                        subagent.result
                        if subagent.result
                        else "No explicit result provided - subagent must call emit_result"
                    ),
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

    def _expand_braces(self, pattern: str) -> list:
        """Expand brace patterns like {a,b,c} into multiple patterns."""
        import re

        # Find brace patterns
        brace_pattern = r"\{([^}]*)\}"
        match = re.search(brace_pattern, pattern)

        if not match:
            return [pattern]

        # Extract the options from the braces
        options_str = match.group(1)
        if not options_str.strip():
            # Empty braces - remove the braces entirely
            expanded_pattern = pattern[: match.start()] + pattern[match.end() :]
            return self._expand_braces(expanded_pattern)

        options = options_str.split(",")
        expanded_patterns = []

        for option in options:
            # Replace the brace pattern with each option
            expanded_pattern = (
                pattern[: match.start()] + option.strip() + pattern[match.end() :]
            )
            # Recursively expand if there are more braces
            expanded_patterns.extend(self._expand_braces(expanded_pattern))

        return expanded_patterns

    def glob(self, args: Dict[str, Any]) -> str:
        """Execute glob pattern matching and return matching file paths."""
        import glob as glob_module
        import os.path

        pattern = args.get("pattern", "")
        path = args.get("path", None)

        if not pattern:
            return "Error: pattern parameter is required"

        # Warn about potentially problematic patterns
        if pattern == "**/*" or pattern == "**":
            return "Error: Pattern '**/*' or '**' is too broad and may cause performance issues. Please use a more specific pattern like '**/*.py' or limit the search scope."

        try:
            # Change to the specified directory if provided
            original_cwd = os.getcwd()

            if path:
                if not os.path.isdir(path):
                    return f"Error: Directory '{path}' does not exist"
                os.chdir(path)

            # Expand brace patterns into multiple glob patterns
            expanded_patterns = self._expand_braces(pattern)
            all_matches = set()  # Use set to avoid duplicates

            # Limit the number of results to prevent overwhelming output
            MAX_RESULTS = 10000

            # Execute glob pattern matching for each expanded pattern
            for expanded_pattern in expanded_patterns:
                try:
                    # Use a safer approach for recursive patterns
                    if "**" in expanded_pattern:
                        # For recursive patterns, implement our own walking with depth limit
                        matches = self._safe_recursive_glob(
                            expanded_pattern, max_depth=10
                        )
                    else:
                        # For non-recursive patterns, use standard glob
                        matches = glob_module.glob(expanded_pattern, recursive=False)

                    all_matches.update(matches)

                    # Stop if we have too many results
                    if len(all_matches) > MAX_RESULTS:
                        break

                except (OSError, RecursionError) as e:
                    # Skip patterns that cause issues
                    continue

            # Convert back to list and restore original directory
            matches = list(all_matches)
            os.chdir(original_cwd)

            if not matches:
                return f"No files found matching pattern: {pattern}"

            # Limit results for display
            if len(matches) > MAX_RESULTS:
                matches = matches[:MAX_RESULTS]
                truncated_msg = (
                    f" (showing first {MAX_RESULTS} of {len(all_matches)} results)"
                )
            else:
                truncated_msg = ""

            # Sort by modification time (newest first)
            try:
                if path:
                    # Convert to full paths for sorting
                    full_matches = [os.path.join(path, match) for match in matches]
                    full_matches.sort(
                        key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0,
                        reverse=True,
                    )
                    # Convert back to relative paths for display
                    matches = [os.path.relpath(match, path) for match in full_matches]
                else:
                    matches.sort(
                        key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0,
                        reverse=True,
                    )
            except (OSError, ValueError):
                # If sorting fails, just return unsorted results
                pass

            return f"Found {len(matches)} files{truncated_msg}:\n" + "\n".join(matches)

        except Exception as e:
            # Restore original directory on error
            try:
                os.chdir(original_cwd)
            except:
                pass
            return f"Error executing glob pattern: {str(e)}"

    def _safe_recursive_glob(self, pattern: str, max_depth: int = 10) -> list:
        """Safely perform recursive glob with depth limiting."""
        import fnmatch

        matches = []

        # Split pattern into directory part and file part
        if "**" in pattern:
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix = parts[0].rstrip("/")
                suffix = parts[1].lstrip("/")

                # Start from current directory if no prefix
                start_dir = prefix if prefix else "."

                # Walk directories with depth limit
                try:
                    for root, dirs, files in os.walk(start_dir, followlinks=False):
                        # Calculate depth
                        depth = root.replace(start_dir, "").count(os.sep)
                        if depth >= max_depth:
                            dirs[:] = []  # Don't recurse deeper
                            continue

                        # Match files in current directory
                        for file in files:
                            full_path = os.path.join(root, file)
                            if suffix:
                                if fnmatch.fnmatch(file, suffix):
                                    matches.append(full_path)
                            else:
                                matches.append(full_path)

                        # Also match directories if no suffix specified
                        if not suffix:
                            for dir_name in dirs:
                                full_path = os.path.join(root, dir_name)
                                matches.append(full_path)

                except (OSError, PermissionError):
                    # Skip directories we can't access
                    pass

        return matches

    def grep(self, args: Dict[str, Any]) -> str:
        """Execute grep pattern search and return matching file paths."""
        import glob as glob_module
        import re

        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        include = args.get("include", None)

        if not pattern:
            return "Error: pattern parameter is required"

        try:
            # Compile the regex pattern
            try:
                regex = re.compile(pattern, re.MULTILINE)
            except re.error as e:
                return f"Error: Invalid regex pattern '{pattern}': {str(e)}"

            # Determine files to search
            if include:
                # Use glob pattern for file filtering
                search_pattern = os.path.join(path, "**", include)
                files_to_search = glob_module.glob(search_pattern, recursive=True)
            else:
                # Search all files recursively
                files_to_search = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        files_to_search.append(os.path.join(root, file))

            matching_files = []

            # Search each file for the pattern
            for file_path in files_to_search:
                try:
                    # Skip binary files and directories
                    if os.path.isdir(file_path):
                        continue

                    # Try to read as text file
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if regex.search(content):
                            matching_files.append(file_path)

                except (IOError, OSError, UnicodeDecodeError):
                    # Skip files that can't be read
                    continue

            if not matching_files:
                include_msg = f" (filtered by {include})" if include else ""
                return f"No files found containing pattern: {pattern}{include_msg}"

            # Sort by modification time (newest first)
            try:
                matching_files.sort(
                    key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0,
                    reverse=True,
                )
            except (OSError, ValueError):
                # If sorting fails, just return unsorted results
                pass

            include_msg = f" (filtered by {include})" if include else ""
            return (
                f"Found {len(matching_files)} files containing pattern{include_msg}:\n"
                + "\n".join(matching_files)
            )

        except Exception as e:
            return f"Error executing grep search: {str(e)}"

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
