"""
Hook executor for running shell commands with proper error handling and security.

This module handles the actual execution of hook commands with template
variable substitution, timeout handling, and result parsing.
"""

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .hook_config import HookDefinition

logger = logging.getLogger(__name__)


@dataclass
class HookResult:
    """Result of executing a hook command."""
    
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    json_output: Optional[Dict[str, Any]] = None
    should_continue: bool = True
    decision: str = "approve"
    reason: Optional[str] = None
    
    def __post_init__(self):
        """Parse JSON output and set control flags after initialization."""
        if self.success and self.stdout.strip():
            self.json_output = self._parse_json_output(self.stdout)
            
        if self.json_output:
            self.should_continue = self.json_output.get("continue", True)
            self.decision = self.json_output.get("decision", "approve")
            self.reason = self.json_output.get("reason")
        else:
            # Use exit code to determine behavior
            if self.exit_code == 2:
                self.should_continue = False
                self.decision = "block"
                self.reason = "Hook returned blocking exit code (2)"
            elif self.exit_code != 0:
                self.reason = f"Hook failed with exit code {self.exit_code}"
    
    def _parse_json_output(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Try to parse JSON output from hook stdout."""
        try:
            # Look for JSON output (may be mixed with other text)
            lines = stdout.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    return json.loads(line)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Hook output is not valid JSON: {e}")
            
        return None


class HookExecutor:
    """Executes hook commands with proper error handling and security."""
    
    def __init__(self, default_timeout: int = 30):
        """Initialize hook executor with default timeout."""
        self.default_timeout = default_timeout
        
    async def execute_hook(
        self,
        hook: HookDefinition,
        context: Dict[str, Any]
    ) -> HookResult:
        """Execute a single hook command with template substitution."""
        start_time = time.time()
        
        try:
            # Substitute template variables in command
            command = self.substitute_template_variables(hook.command, context)
            logger.debug(f"Executing hook command: {command}")
            
            # Set up execution environment
            env = os.environ.copy()
            if hook.env:
                env.update(hook.env)
                
            working_dir = hook.working_directory
            if working_dir:
                working_dir = Path(working_dir).expanduser()
                if not working_dir.exists():
                    logger.warning(f"Hook working directory does not exist: {working_dir}")
                    working_dir = None
            
            timeout = hook.timeout or self.default_timeout
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=working_dir
            )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                stdout = stdout_bytes.decode('utf-8', errors='replace')
                stderr = stderr_bytes.decode('utf-8', errors='replace')
                exit_code = process.returncode
                
                execution_time = time.time() - start_time
                success = (exit_code in [0, 2])  # 0 = success, 2 = blocking but handled
                
                logger.debug(
                    f"Hook completed: exit_code={exit_code}, "
                    f"execution_time={execution_time:.3f}s"
                )
                
                return HookResult(
                    success=success,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time
                )
                
            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass  # Process already terminated
                    
                execution_time = time.time() - start_time
                logger.warning(f"Hook command timed out after {timeout}s: {command}")
                
                return HookResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Hook command timed out after {timeout} seconds",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing hook command: {e}")
            
            return HookResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Hook execution error: {str(e)}",
                execution_time=execution_time
            )
    
    def substitute_template_variables(
        self, 
        command: str, 
        context: Dict[str, Any]
    ) -> str:
        """Replace template variables in hook command with context values."""
        result = command
        
        # Define available template variables and their extractors
        template_vars = {
            "tool_name": lambda ctx: ctx.get("tool_name", ""),
            "tool_args": lambda ctx: json.dumps(ctx.get("tool_args", {})),
            "result": lambda ctx: str(ctx.get("result", "")),
            "error": lambda ctx: str(ctx.get("error", "")),
            "message": lambda ctx: str(ctx.get("message", "")),
            "timestamp": lambda ctx: ctx.get("timestamp", ""),
            "session_id": lambda ctx: ctx.get("session_id", ""),
            "user_input": lambda ctx: str(ctx.get("user_input", "")),
            "execution_time": lambda ctx: str(ctx.get("execution_time", "")),
        }
        
        # Substitute each template variable
        for var_name, extractor in template_vars.items():
            placeholder = f"{{{{{var_name}}}}}"
            if placeholder in result:
                try:
                    value = extractor(context)
                    # Escape shell-dangerous characters if needed
                    safe_value = self._escape_shell_value(str(value))
                    result = result.replace(placeholder, safe_value)
                except Exception as e:
                    logger.warning(f"Error substituting template variable {var_name}: {e}")
                    result = result.replace(placeholder, "")
        
        return result
    
    def _escape_shell_value(self, value: str) -> str:
        """Escape shell-dangerous characters in template values."""
        if not value:
            return '""'
            
        # Use shlex.quote for proper shell escaping
        return shlex.quote(value)
    
    def validate_command_safety(self, command: str) -> tuple[bool, str]:
        """Basic validation of command safety (advisory only)."""
        dangerous_patterns = [
            'rm -rf /',
            'sudo rm',
            'mkfs',
            'dd if=',
            'format',
            ':(){:|:&};:',  # Fork bomb
            'curl.*|.*sh',  # Pipe to shell
            'wget.*|.*sh',  # Pipe to shell
        ]
        
        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return False, f"Command contains potentially dangerous pattern: {pattern}"
                
        return True, "Command appears safe"