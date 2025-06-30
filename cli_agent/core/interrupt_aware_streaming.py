"""Centralized interrupt-aware streaming utilities."""

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)


class FirstInterruptException(Exception):
    """Exception raised for first interrupt that should be handled gracefully."""
    pass


class InterruptAwareStream:
    """Wrapper that adds interrupt checking to async streaming responses."""

    def __init__(
        self,
        stream: AsyncIterator[Any],
        operation_name: str = "Streaming operation",
        interrupt_callback: Optional[Callable] = None,
    ):
        """
        Initialize interrupt-aware stream wrapper.

        Args:
            stream: The original async iterator/stream
            operation_name: Description for interrupt logging
            interrupt_callback: Optional callback to run on interrupt
        """
        self.stream = stream
        self.operation_name = operation_name
        self.interrupt_callback = interrupt_callback
        self._interrupted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        from cli_agent.core.global_interrupt import check_interrupt

        # Check for interrupt before processing next chunk
        try:
            check_interrupt(f"{self.operation_name} interrupted")
        except KeyboardInterrupt:
            logger.info(f"{self.operation_name} interrupted by user")
            self._interrupted = True

            # Run interrupt callback if provided
            if self.interrupt_callback:
                try:
                    (
                        await self.interrupt_callback()
                        if callable(self.interrupt_callback)
                        else None
                    )
                except Exception as e:
                    logger.warning(f"Interrupt callback failed: {e}")

            raise StopAsyncIteration

        # Get next chunk from original stream
        try:
            return await self.stream.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as e:
            logger.debug(f"Stream error in {self.operation_name}: {e}")
            raise

    @property
    def was_interrupted(self) -> bool:
        """Check if stream was interrupted."""
        return self._interrupted


def make_interruptible(
    stream: AsyncIterator[Any],
    operation_name: str = "Stream",
    interrupt_callback: Optional[Callable] = None,
) -> InterruptAwareStream:
    """
    Convenience function to wrap any async stream with interrupt checking.

    Args:
        stream: The async iterator to wrap
        operation_name: Description for logging
        interrupt_callback: Optional cleanup callback on interrupt

    Returns:
        InterruptAwareStream wrapper

    Example:
        async for chunk in make_interruptible(api_response, "LLM Response"):
            process_chunk(chunk)
    """
    return InterruptAwareStream(stream, operation_name, interrupt_callback)


def make_sync_interruptible(
    stream,
    operation_name: str = "Sync Stream",
    interrupt_callback: Optional[Callable] = None,
):
    """
    Wrap a synchronous iterator with interrupt checking.

    Args:
        stream: The synchronous iterator to wrap
        operation_name: Description for logging
        interrupt_callback: Optional cleanup callback on interrupt

    Yields:
        Items from the original iterator, with interrupt checking

    Example:
        for chunk in make_sync_interruptible(api_response, "LLM Response"):
            process_chunk(chunk)
    """
    from cli_agent.core.global_interrupt import check_interrupt

    interrupted = False

    try:
        for item in stream:
            # Check for interrupt before yielding next item
            try:
                check_interrupt(f"{operation_name} interrupted")
            except KeyboardInterrupt:
                logger.info(f"{operation_name} interrupted by user")
                interrupted = True

                # Run interrupt callback if provided
                if interrupt_callback:
                    try:
                        interrupt_callback() if callable(interrupt_callback) else None
                    except Exception as e:
                        logger.warning(f"Interrupt callback failed: {e}")

                raise

            yield item

    except Exception as e:
        if not interrupted:  # Don't log if we caused the exception
            logger.debug(f"Stream error in {operation_name}: {e}")
        raise


class InterruptAwareSubprocess:
    """Wrapper for subprocess operations with interrupt handling."""

    @staticmethod
    async def run_with_interrupt_checking(
        cmd: str,
        timeout: Optional[float] = None,
        check_interval: float = 0.1,
        operation_name: str = "Subprocess",
    ):
        """
        Run subprocess with periodic interrupt checking.

        Args:
            cmd: Command to run
            timeout: Maximum execution time
            check_interval: How often to check for interrupts (seconds)
            operation_name: Description for logging

        Returns:
            subprocess.CompletedProcess result
        """
        import asyncio
        import time

        from cli_agent.core.global_interrupt import check_interrupt

        logger.debug(f"Starting interrupt-aware subprocess: {cmd}")

        # Start subprocess
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        start_time = time.time()

        try:
            while True:
                # Check for interrupt
                try:
                    check_interrupt(f"{operation_name} interrupted")
                except KeyboardInterrupt:
                    logger.info(f"{operation_name} interrupted by user")
                    # Terminate subprocess
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    raise

                # Check if process completed
                if process.returncode is not None:
                    break

                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"{operation_name} timed out after {timeout}s")
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    raise asyncio.TimeoutError(f"Command timed out: {cmd}")

                # Wait before next check
                await asyncio.sleep(check_interval)

            # Get results
            stdout, stderr = await process.communicate()

            # Create result object similar to subprocess.run
            from types import SimpleNamespace

            result = SimpleNamespace()
            result.returncode = process.returncode
            result.stdout = stdout.decode() if stdout else ""
            result.stderr = stderr.decode() if stderr else ""
            result.args = cmd

            return result

        except Exception as e:
            # Ensure subprocess is cleaned up
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            raise


class InterruptAwareMCPClient:
    """Wrapper for MCP client operations with interrupt handling."""

    @staticmethod
    async def execute_with_interrupt_checking(
        client_method,
        *args,
        operation_name: str = "MCP Operation",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Execute MCP client method with interrupt checking.

        Args:
            client_method: The MCP client method to call
            *args: Arguments for the method
            operation_name: Description for logging
            timeout: Maximum execution time
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the client method call
        """
        import asyncio

        from cli_agent.core.global_interrupt import check_interrupt

        logger.debug(f"Starting interrupt-aware MCP operation: {operation_name}")

        # Create the task
        task = asyncio.create_task(client_method(*args, **kwargs))

        check_interval = 0.1  # Check every 100ms

        try:
            while not task.done():
                # Check for interrupt
                try:
                    check_interrupt(f"{operation_name} interrupted")
                except KeyboardInterrupt:
                    logger.info(f"{operation_name} interrupted by user")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise

                # Wait a bit before next check
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=check_interval)
                    break  # Task completed
                except asyncio.TimeoutError:
                    continue  # Keep checking

            return await task

        except Exception as e:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            raise


async def run_with_interrupt_monitoring(
    async_operation,
    operation_name: str = "Async operation",
    check_interval: float = 0.01,  # Check every 10ms for very aggressive interrupt handling
):
    """
    Run an async operation with aggressive interrupt monitoring.

    This creates a background task that polls for interrupts every 10ms
    and cancels the main operation if an interrupt is detected.

    Args:
        async_operation: The async function/coroutine to run
        operation_name: Name for logging
        check_interval: How often to check for interrupts (seconds)

    Returns:
        Result of the async operation

    Raises:
        KeyboardInterrupt: If user interrupts the operation
    """
    from cli_agent.core.global_interrupt import check_interrupt

    # Create the main operation task
    if asyncio.iscoroutine(async_operation):
        main_task = asyncio.create_task(async_operation)
    elif callable(async_operation):
        main_task = asyncio.create_task(async_operation())
    else:
        raise ValueError("async_operation must be a coroutine or callable")

    async def interrupt_monitor():
        """Background task that monitors for interrupts."""
        while not main_task.done():
            try:
                check_interrupt(f"{operation_name} interrupted")
            except KeyboardInterrupt:
                logger.info(f"{operation_name} cancelled due to interrupt")
                main_task.cancel()
                # Don't re-raise KeyboardInterrupt - let the main logic handle it
                return

            await asyncio.sleep(check_interval)

    # Start interrupt monitoring task
    monitor_task = asyncio.create_task(interrupt_monitor())

    try:
        # Wait for either the main operation or interrupt monitor to complete
        done, pending = await asyncio.wait(
            [main_task, monitor_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check if main task completed successfully
        if main_task in done:
            return main_task.result()
        else:
            # Monitor task completed first (interrupt detected)
            # Always raise FirstInterruptException to allow graceful handling
            raise FirstInterruptException("Operation interrupted by user")

    except asyncio.CancelledError:
        logger.info(f"{operation_name} was cancelled")
        # Always raise FirstInterruptException for cancelled operations to allow graceful handling
        raise FirstInterruptException("Operation cancelled - first interrupt")
    except Exception as e:
        # Ensure all tasks are cancelled on any error
        for task in [main_task, monitor_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        raise
