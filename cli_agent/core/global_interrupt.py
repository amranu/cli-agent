"""Global interrupt management system for MCP agents.

This module provides a centralized interrupt handling system that allows
any part of the application to check for and respond to user interrupts
(Ctrl+C, ESC, etc.) in a consistent manner.
"""

import logging
import signal
import threading
import time
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class GlobalInterruptManager:
    """Global interrupt manager for handling user interrupts across the application.

    This singleton class provides:
    - Centralized interrupt state management
    - Signal handler registration
    - Callback system for interrupt notifications
    - Thread-safe interrupt checking
    - Graceful interrupt handling with cleanup
    """

    _instance: Optional["GlobalInterruptManager"] = None
    _lock = threading.Lock()
    _current_input_handler: Optional[Any] = None  # Reference to current input handler

    def __new__(cls) -> "GlobalInterruptManager":
        """Ensure singleton pattern for global interrupt manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the global interrupt manager."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._interrupted = False
        self._exit_requested = False  # Flag for graceful exit
        self._interrupt_count = 0
        self._last_interrupt_time = 0
        self._interrupt_timeout = 2.0  # Reset count after 2 seconds
        self._interrupt_lock = (
            threading.RLock()
        )  # Use reentrant lock to prevent deadlock
        self._callbacks: List[Callable[[], None]] = []
        self._original_handlers = {}
        self._setup_signal_handlers()
        self._initialized = True
        logger.debug("Global interrupt manager initialized")

    def _setup_signal_handlers(self):
        """Set up signal handlers for global interrupt detection."""

        def interrupt_handler(signum, frame):
            """Handle interrupt signals globally."""
            import time

            current_time = time.time()

            with self._interrupt_lock:
                # Reset count if too much time has passed since last interrupt
                if current_time - self._last_interrupt_time > self._interrupt_timeout:
                    self._interrupt_count = 0

                self._interrupt_count += 1
                self._last_interrupt_time = current_time

                logger.info(
                    f"Global interrupt signal received: {signum} (count: {self._interrupt_count})"
                )

                # Always interrupt operations and continue - don't exit from signal handler
                # The prompt_toolkit key binding will handle empty prompt exits
                self.set_interrupted(True)
                print(
                    "\n🛑 Operation interrupted.",
                    flush=True,
                )

            # Call all registered callbacks
            for callback in self._callbacks[
                :
            ]:  # Copy list to avoid modification during iteration
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in interrupt callback: {e}")

        # Store original handlers for restoration
        try:
            self._original_handlers[signal.SIGINT] = signal.signal(
                signal.SIGINT, interrupt_handler
            )
            self._original_handlers[signal.SIGTERM] = signal.signal(
                signal.SIGTERM, interrupt_handler
            )
            logger.debug("Global signal handlers registered")
        except Exception as e:
            logger.warning(f"Could not set up signal handlers: {e}")

    def set_interrupted(self, state: bool = True):
        """Set the global interrupt state.

        Args:
            state: True to mark as interrupted, False to clear
        """
        with self._interrupt_lock:
            if state and not self._interrupted:
                logger.info("Global interrupt state set to True")
            elif not state and self._interrupted:
                logger.info("Global interrupt state cleared")
            self._interrupted = state

    def is_interrupted(self) -> bool:
        """Check if a global interrupt has been received.

        Returns:
            True if interrupted, False otherwise
        """
        with self._interrupt_lock:
            return self._interrupted

    def clear_interrupt(self):
        """Clear the global interrupt state."""
        self.set_interrupted(False)

    def reset_interrupt_count(self):
        """Reset the interrupt count (call this after successful operations)."""
        with self._interrupt_lock:
            self._interrupt_count = 0
            self._last_interrupt_time = 0

    def set_current_input_handler(self, input_handler):
        """Set the current input handler for empty prompt detection."""
        GlobalInterruptManager._current_input_handler = input_handler

    def get_current_input_handler(self):
        """Get the current input handler."""
        return GlobalInterruptManager._current_input_handler

    def _should_exit_immediately(self) -> bool:
        """Check if we should exit immediately on Ctrl+C.
        
        Returns:
            True if prompt is empty and no operations are running,
            False if there's content in prompt or active operations.
        """
        try:
            # Check if there's content in the current prompt
            try:
                from prompt_toolkit.application import get_app
                app = get_app()
                current_text = app.current_buffer.text
                if current_text and current_text.strip():
                    # There's content in the prompt - don't exit
                    return False
            except Exception:
                # If we can't access prompt content, be conservative
                return False
            
            # Check if we have any active operations
            if self._interrupted:
                return False
            
            # Check the call stack for active operations
            import inspect
            current_frame = inspect.currentframe()
            try:
                frame = current_frame
                
                while frame:
                    frame_info = inspect.getframeinfo(frame)
                    filename = frame_info.filename
                    function_name = frame_info.function
                    
                    # Skip signal handler functions
                    if function_name in ['interrupt_handler', '_should_exit_immediately']:
                        frame = frame.f_back
                        continue
                    
                    # Check for active LLM operations
                    if any(name in filename for name in [
                        'base_llm_provider', 'interrupt_aware_streaming', 
                        'response_handler', 'builtin_tool_executor'
                    ]):
                        if any(func in function_name for func in [
                            'generate_response', '_make_api_request', 'run_with_interrupt_monitoring',
                            '_generate_completion', 'bash_execute', 'execute_tool_call',
                            'handle_complete_response', '_process_streaming_chunks'
                        ]):
                            # Active operation detected - don't exit
                            return False
                    
                    frame = frame.f_back
                    
            finally:
                del current_frame
            
            # If we get here: empty prompt, no active operations
            # Only exit if we're in interactive mode (TTY)
            import sys
            return sys.stdin.isatty()
            
        except Exception as e:
            logger.debug(f"Error checking exit conditions: {e}")
            # Be conservative on errors - don't exit
            return False

    def get_interrupt_count(self) -> int:
        """Get the current interrupt count.

        Returns:
            int: The number of interrupts received within the timeout window
        """
        with self._interrupt_lock:
            return self._interrupt_count

    def check_and_raise_if_interrupted(self, message: str = "Operation interrupted"):
        """Check for interrupt and raise KeyboardInterrupt if found.

        Args:
            message: Custom message for the exception

        Raises:
            KeyboardInterrupt: If an interrupt has been received
        """
        if self.is_interrupted():
            # Always raise KeyboardInterrupt to aggressively cancel operations
            raise KeyboardInterrupt(message)

    def add_callback(self, callback: Callable[[], None]):
        """Add a callback to be called when an interrupt is received.

        Args:
            callback: Function to call on interrupt (should be fast and safe)
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Added interrupt callback: {callback.__name__}")

    def remove_callback(self, callback: Callable[[], None]):
        """Remove an interrupt callback.

        Args:
            callback: Function to remove from callbacks
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Removed interrupt callback: {callback.__name__}")

    def with_interrupt_check(self, operation: Callable, check_interval: float = 0.1):
        """Execute an operation with periodic interrupt checking.

        Args:
            operation: Function to execute
            check_interval: How often to check for interrupts (seconds)

        Returns:
            Result of the operation

        Raises:
            KeyboardInterrupt: If interrupted during execution
        """
        import asyncio

        if asyncio.iscoroutinefunction(operation):
            return self._with_interrupt_check_async(operation, check_interval)
        else:
            return self._with_interrupt_check_sync(operation, check_interval)

    def _with_interrupt_check_sync(self, operation: Callable, check_interval: float):
        """Execute synchronous operation with interrupt checking."""
        import threading
        import time

        result = None
        exception = None
        finished = threading.Event()

        def run_operation():
            nonlocal result, exception
            try:
                result = operation()
            except Exception as e:
                exception = e
            finally:
                finished.set()

        # Start operation in separate thread
        thread = threading.Thread(target=run_operation)
        thread.daemon = True
        thread.start()

        # Check for interrupts while waiting
        while not finished.is_set():
            if self.is_interrupted():
                # Check interrupt count to decide whether to raise KeyboardInterrupt or FirstInterruptException
                from cli_agent.core.interrupt_aware_streaming import FirstInterruptException
                if self._interrupt_count >= 2:
                    raise KeyboardInterrupt("Operation interrupted - multiple interrupts")
                else:
                    raise FirstInterruptException("Operation interrupted - first interrupt")
            finished.wait(check_interval)

        # Re-raise any exception from the operation
        if exception:
            raise exception

        return result

    async def _with_interrupt_check_async(
        self, operation: Callable, check_interval: float
    ):
        """Execute asynchronous operation with interrupt checking."""
        import asyncio

        # Create task for the operation
        task = asyncio.create_task(operation())

        try:
            while not task.done():
                if self.is_interrupted():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    # Check interrupt count to decide whether to raise KeyboardInterrupt or FirstInterruptException
                    from cli_agent.core.interrupt_aware_streaming import FirstInterruptException
                    if self._interrupt_count >= 2:
                        raise KeyboardInterrupt("Operation interrupted - multiple interrupts")
                    else:
                        raise FirstInterruptException("Operation interrupted - first interrupt")

                # Wait a bit before checking again
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=check_interval)
                except asyncio.TimeoutError:
                    continue

            return await task
        except asyncio.CancelledError:
            # Check interrupt count to decide whether to raise KeyboardInterrupt or FirstInterruptException
            from cli_agent.core.interrupt_aware_streaming import FirstInterruptException
            if self._interrupt_count >= 2:
                raise KeyboardInterrupt("Operation interrupted - multiple interrupts")
            else:
                raise FirstInterruptException("Operation interrupted - first interrupt")

    def restore_handlers(self):
        """Restore original signal handlers."""
        for signum, handler in self._original_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception as e:
                logger.warning(f"Could not restore signal handler for {signum}: {e}")
        logger.debug("Original signal handlers restored")

    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.restore_handlers()
        except:
            pass


# Global instance
_global_interrupt_manager = None


def get_global_interrupt_manager() -> GlobalInterruptManager:
    """Get the global interrupt manager instance."""
    global _global_interrupt_manager
    if _global_interrupt_manager is None:
        _global_interrupt_manager = GlobalInterruptManager()
    return _global_interrupt_manager


def is_interrupted() -> bool:
    """Quick check if globally interrupted."""
    return get_global_interrupt_manager().is_interrupted()


def clear_interrupt():
    """Quick clear of global interrupt."""
    get_global_interrupt_manager().clear_interrupt()


def check_interrupt(message: str = "Operation interrupted"):
    """Quick check and raise if interrupted."""
    get_global_interrupt_manager().check_and_raise_if_interrupted(message)


def reset_interrupt_count():
    """Quick reset of interrupt count."""
    get_global_interrupt_manager().reset_interrupt_count()
