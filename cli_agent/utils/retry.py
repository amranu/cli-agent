"""
Retry utilities with exponential backoff for handling API failures.

This module provides standardized retry logic that can be used across
different LLM implementations to handle transient failures gracefully.
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


class RetryHandler:
    """Handles retry logic with exponential backoff."""

    # Default retryable error indicators
    DEFAULT_RETRYABLE_ERRORS = [
        "timeout",
        "network",
        "connection",
        "rate limit",
        "rate_limit",
        "429",
        "502",
        "503",
        "504",
        "500",
        "retryerror",
        "temporary",
    ]

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[List[str]] = None,
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay between retries
            backoff_factor: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_errors: List of error indicators that should trigger retries
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_errors = retryable_errors or self.DEFAULT_RETRYABLE_ERRORS

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error should trigger a retry."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check error message and type against retryable patterns
        for pattern in self.retryable_errors:
            if pattern.lower() in error_str or pattern.lower() in error_type:
                return True

        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter up to 25% of delay
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay

    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with retry logic.

        Args:
            func: Async function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Function result if successful

        Raises:
            RetryError: If all retry attempts are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} for {func.__name__}"
                )
                result = await func(*args, **kwargs)

                if attempt > 0:
                    logger.info(
                        f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                last_exception = e

                # Don't retry on the last attempt
                if attempt == self.max_retries:
                    break

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.info(f"Non-retryable error in {func.__name__}: {e}")
                    break

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)

        # All attempts exhausted
        error_msg = (
            f"All {self.max_retries + 1} attempts failed for {func.__name__}. "
            f"Last error: {last_exception}"
        )
        logger.error(error_msg)
        raise RetryError(error_msg, last_exception, self.max_retries + 1)

    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute sync function with retry logic.

        Args:
            func: Sync function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Function result if successful

        Raises:
            RetryError: If all retry attempts are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} for {func.__name__}"
                )
                result = func(*args, **kwargs)

                if attempt > 0:
                    logger.info(
                        f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                last_exception = e

                # Don't retry on the last attempt
                if attempt == self.max_retries:
                    break

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.info(f"Non-retryable error in {func.__name__}: {e}")
                    break

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)

        # All attempts exhausted
        error_msg = (
            f"All {self.max_retries + 1} attempts failed for {func.__name__}. "
            f"Last error: {last_exception}"
        )
        logger.error(error_msg)
        raise RetryError(error_msg, last_exception, self.max_retries + 1)


# Convenience functions and decorators


def retry_async_call(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_errors: Optional[List[str]] = None,
):
    """
    Decorator for async functions to add retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for exponential backoff
        retryable_errors: List of error indicators that should trigger retries
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = RetryHandler(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                retryable_errors=retryable_errors,
            )
            return await handler.retry_async(func, *args, **kwargs)

        return wrapper

    return decorator


def retry_sync_call(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_errors: Optional[List[str]] = None,
):
    """
    Decorator for sync functions to add retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for exponential backoff
        retryable_errors: List of error indicators that should trigger retries
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = RetryHandler(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                retryable_errors=retryable_errors,
            )
            return handler.retry_sync(func, *args, **kwargs)

        return wrapper

    return decorator


# Global retry handler instance for convenience
default_retry_handler = RetryHandler()


async def retry_with_backoff(
    request_func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_errors: Optional[List[str]] = None,
) -> Any:
    """
    Convenience function for retrying operations with exponential backoff.

    Args:
        request_func: Function to retry (can be sync or async)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        retryable_errors: List of error indicators that should trigger retries

    Returns:
        Function result if successful

    Raises:
        RetryError: If all retry attempts are exhausted
    """
    handler = RetryHandler(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_errors=retryable_errors,
    )

    if asyncio.iscoroutinefunction(request_func):
        return await handler.retry_async(request_func)
    else:
        return handler.retry_sync(request_func)
