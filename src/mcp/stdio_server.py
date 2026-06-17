"""Stdio entry point for MCP server (for Claude Code)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.mcp.mcp_server import server
from mcp.server.stdio import stdio_server
import anyio


async def main():
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
