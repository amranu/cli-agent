#!/usr/bin/env python3
"""Standalone MCP server for AI models.

This script provides a standalone MCP server that exposes all available AI models
as MCP tools. It can be run directly or used as a subprocess by MCP clients.

Usage:
    python mcp_server.py --stdio          # Use stdio transport (default)
    python mcp_server.py --tcp --port 3000 # Use TCP transport
"""

import argparse
import asyncio
import logging
import sys

# Suppress noisy logging from FastMCP and other libraries
logging.getLogger("FastMCP.fastmcp.server.server").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    """Run the MCP model server."""
    parser = argparse.ArgumentParser(description="MCP Model Server")
    parser.add_argument(
        "--stdio", action="store_true", help="Use stdio transport (default)"
    )
    parser.add_argument(
        "--tcp", action="store_true", help="Use TCP transport instead of stdio"
    )
    parser.add_argument(
        "--port", type=int, default=3000, help="Port for TCP transport (default: 3000)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for TCP transport (default: localhost)",
    )

    args = parser.parse_args()

    try:
        from cli_agent.mcp.model_server import create_model_server

        server = create_model_server()

        if args.tcp:
            print(
                f"Starting MCP model server on {args.host}:{args.port}", file=sys.stderr
            )
            asyncio.run(server.run_async(host=args.host, port=args.port))
        else:
            # Default to stdio for MCP compatibility
            print("Starting MCP model server on stdio transport", file=sys.stderr)
            asyncio.run(server.run_stdio_async())

    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please install FastMCP with: pip install fastmcp", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
