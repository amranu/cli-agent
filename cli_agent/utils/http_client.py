"""
HTTP client utilities for creating standardized HTTP clients across different LLM integrations.

This module provides factory methods and utilities for creating HTTP clients with
consistent timeout, connection pooling, and cleanup handling.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HTTPClientFactory:
    """Factory for creating configured HTTP clients."""

    @staticmethod
    def create_async_client(
        timeout_seconds: float = 120.0,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        extra_headers: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
    ):
        """
        Create HTTP client with common configuration.

        Args:
            timeout_seconds: Timeout for requests in seconds
            max_connections: Maximum number of connections in pool
            max_keepalive_connections: Maximum keepalive connections
            extra_headers: Additional headers to include in requests
            base_url: Base URL for all requests

        Returns:
            Configured httpx.AsyncClient instance
        """
        try:
            import httpx

            client_config = {
                "timeout": httpx.Timeout(timeout_seconds),
                "limits": httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive_connections,
                ),
                "headers": extra_headers or {},
            }

            if base_url:
                client_config["base_url"] = base_url

            return httpx.AsyncClient(**client_config)

        except ImportError:
            logger.error("httpx not available for async HTTP client creation")
            raise
        except Exception as e:
            logger.error(f"Failed to create HTTP client: {e}")
            raise

    @staticmethod
    def create_openai_client(
        api_key: str, base_url: str, timeout: float = 600, max_retries: int = 3
    ):
        """
        Create OpenAI-style client (for DeepSeek, OpenAI, etc.).

        Args:
            api_key: API key for authentication
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries

        Returns:
            Configured OpenAI client instance
        """
        try:
            from openai import OpenAI

            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        except ImportError:
            logger.error("openai package not available")
            raise
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            raise

    @staticmethod
    def create_gemini_client(
        api_key: str,
        timeout_seconds: float = 120.0,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
    ):
        """
        Create Gemini client with custom HTTP configuration.

        Args:
            api_key: Gemini API key
            timeout_seconds: Timeout for requests
            max_connections: Maximum connections in pool
            max_keepalive_connections: Maximum keepalive connections

        Returns:
            Tuple of (gemini_client, http_client) for cleanup management
        """
        try:
            from google import genai

            # Create custom HTTP client for better control
            http_client = HTTPClientFactory.create_async_client(
                timeout_seconds=timeout_seconds,
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            )

            # Create Gemini client with custom HTTP client
            gemini_client = genai.Client(api_key=api_key, http_client=http_client)

            return gemini_client, http_client

        except ImportError:
            logger.error("google.genai not available")
            raise
        except Exception as e:
            logger.warning(f"Failed to create custom Gemini client: {e}")
            # Fallback to default client
            try:
                from google import genai

                gemini_client = genai.Client(api_key=api_key)
                return gemini_client, None
            except Exception as fallback_error:
                logger.error(
                    f"Failed to create even default Gemini client: {fallback_error}"
                )
                raise


class HTTPClientManager:
    """Manager for HTTP client lifecycle and cleanup."""

    def __init__(self):
        self._clients: Dict[str, Any] = {}
        self._cleanup_tasks: Dict[str, Any] = {}

    def register_client(
        self, name: str, client: Any, cleanup_func: Optional[callable] = None
    ):
        """
        Register an HTTP client for lifecycle management.

        Args:
            name: Unique name for the client
            client: HTTP client instance
            cleanup_func: Optional cleanup function to call on shutdown
        """
        self._clients[name] = client
        if cleanup_func:
            self._cleanup_tasks[name] = cleanup_func

        logger.debug(f"Registered HTTP client: {name}")

    async def cleanup_client(self, name: str):
        """Clean up a specific HTTP client."""
        if name in self._clients:
            client = self._clients[name]

            # Call custom cleanup if registered
            if name in self._cleanup_tasks:
                try:
                    await self._cleanup_tasks[name](client)
                except Exception as e:
                    logger.warning(f"Custom cleanup failed for {name}: {e}")

            # Default cleanup for common client types
            try:
                if hasattr(client, "aclose"):
                    await client.aclose()
                elif hasattr(client, "close"):
                    close_method = getattr(client, "close")
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
            except RuntimeError as e:
                # Ignore "Event loop is closed" errors - this is expected during shutdown
                if "Event loop is closed" in str(e):
                    logger.debug(
                        f"Ignoring event loop closed error for {name} during shutdown"
                    )
                else:
                    logger.warning(f"Runtime error during cleanup for {name}: {e}")
            except Exception as e:
                logger.warning(f"Default cleanup failed for {name}: {e}")

            # Remove from tracking
            del self._clients[name]
            if name in self._cleanup_tasks:
                del self._cleanup_tasks[name]

            logger.debug(f"Cleaned up HTTP client: {name}")

    async def cleanup_all(self):
        """Clean up all registered HTTP clients."""
        cleanup_tasks = []

        for name in list(self._clients.keys()):
            cleanup_tasks.append(self.cleanup_client(name))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error during bulk cleanup: {e}")

        logger.info("HTTP client cleanup completed")

    def get_client(self, name: str) -> Optional[Any]:
        """Get a registered HTTP client by name."""
        return self._clients.get(name)

    def list_clients(self) -> Dict[str, Any]:
        """Get all registered clients."""
        return self._clients.copy()


# Global HTTP client manager instance
http_client_manager = HTTPClientManager()


# Convenience functions


async def create_llm_http_clients(
    llm_type: str, api_key: str, base_url: Optional[str] = None, timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Create appropriate HTTP clients for specific LLM type.

    Args:
        llm_type: Type of LLM (deepseek, gemini, openai)
        api_key: API key for authentication
        base_url: Base URL for API (required for some LLMs)
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing created clients
    """
    clients = {}

    if llm_type.lower() == "deepseek":
        if not base_url:
            raise ValueError("base_url required for DeepSeek client")

        client = HTTPClientFactory.create_openai_client(
            api_key=api_key, base_url=base_url, timeout=timeout
        )
        clients["openai"] = client

    elif llm_type.lower() == "gemini":
        gemini_client, http_client = HTTPClientFactory.create_gemini_client(
            api_key=api_key, timeout_seconds=timeout
        )
        clients["gemini"] = gemini_client
        if http_client:
            clients["http"] = http_client

    elif llm_type.lower() == "openai":
        if not base_url:
            base_url = "https://api.openai.com/v1"

        client = HTTPClientFactory.create_openai_client(
            api_key=api_key, base_url=base_url, timeout=timeout
        )
        clients["openai"] = client

    else:
        logger.warning(f"Unknown LLM type: {llm_type}, creating generic HTTP client")
        client = HTTPClientFactory.create_async_client(
            timeout_seconds=timeout, base_url=base_url
        )
        clients["http"] = client

    return clients


async def cleanup_llm_clients(clients: Dict[str, Any]):
    """Clean up LLM HTTP clients."""
    cleanup_tasks = []

    for name, client in clients.items():
        try:
            if hasattr(client, "aclose"):
                cleanup_tasks.append(client.aclose())
            elif hasattr(client, "close"):
                cleanup_tasks.append(
                    asyncio.create_task(asyncio.to_thread(client.close))
                )
        except Exception as e:
            logger.warning(f"Error preparing cleanup for {name}: {e}")

    if cleanup_tasks:
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")

    logger.debug("LLM client cleanup completed")
