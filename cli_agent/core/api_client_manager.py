"""API client management and retry logic for MCP agents."""

import asyncio
import logging
import random
from typing import Any, Callable

logger = logging.getLogger(__name__)


class APIClientManager:
    """Handles API requests, retry logic, and error management."""

    def __init__(self, agent):
        """Initialize with reference to the parent agent."""
        self.agent = agent

    async def make_api_request_with_retry(
        self, request_func: Callable, max_retries: int = 3, base_delay: float = 1.0
    ):
        """Generic API request with exponential backoff retry logic."""
        for attempt in range(max_retries + 1):
            try:
                # Execute the request function
                if asyncio.iscoroutinefunction(request_func):
                    return await request_func()
                else:
                    return request_func()

            except Exception as e:
                error_str = str(e)
                logger.error(
                    f"API request failed (attempt {attempt+1}/{max_retries+1}): {error_str}"
                )

                # Check if this error is retryable (provider-specific)
                if attempt == max_retries or not self.is_retryable_error(e):
                    # Last attempt or non-retryable error
                    raise e

                # Calculate delay with exponential backoff and jitter
                delay = self.calculate_retry_delay(attempt, base_delay)
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                logger.warning(f"Retrying in {delay:.2f} seconds...")

                await asyncio.sleep(delay)

    def is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable - delegates to agent for provider specifics."""
        return self.agent._is_retryable_error(error)

    def calculate_retry_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate retry delay - delegates to agent or uses default strategy."""
        if hasattr(self.agent, "_calculate_retry_delay"):
            return self.agent._calculate_retry_delay(attempt, base_delay)
        else:
            # Default implementation
            return base_delay * (2**attempt) + random.uniform(0, 1)
