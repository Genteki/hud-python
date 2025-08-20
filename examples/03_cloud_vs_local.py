#!/usr/bin/env python3
"""
Cloud vs Local Deployment Example

This example demonstrates two ways to deploy HUD environments:
1. Cloud deployment using HUD's infrastructure
2. Local deployment using Docker

Both approaches use the same agent code, showing the flexibility
of the MCP protocol.
"""

import asyncio
import argparse
import os
import hud
from hud.agents import ClaudeAgent
from hud.clients import MCPClient
from hud.settings import settings
from hud.datasets import Task

import logging

logging.basicConfig(level=logging.INFO)


async def run_cloud_example():
    """Run agent in HUD's cloud environment."""
    print("\n☁️  Cloud Deployment Example")
    print("-" * 40)

    with hud.trace("Cloud Browser Demo") as run_id:
        # Cloud configuration - uses HUD's servers
        task = {
            "prompt": "Open Sent mail, find the Series B pitch deck email, forward it to billgates@microsoft.com, and mark the original message as important.",
            "mcp_config": {
                "hud": {
                    "url": settings.hud_mcp_url,
                    "headers": {
                        "Authorization": f"Bearer {settings.api_key}",
                        "Mcp-Image": "hudpython/hud-gmail-clone:v1.5",
                        "Mcp-Expose-Ports": "9222,6080",
                        "Run-Id": run_id,
                    },
                }
            },
            "setup_tool": {
                "name": "setup",
                "arguments": {
                    "problem_id": "forward-series-b-deck-to-billgates",
                },
            },
            "evaluate_tool": {
                "name": "evaluate",
                "arguments": {
                    "problem_id": "forward-series-b-deck-to-billgates",
                },
            },
            "metadata": {"id": "forward-series-b-deck-to-billgates"},
        }

        task = Task(**task)
        print(task)

        client = MCPClient(mcp_config=task.mcp_config)
        await client.initialize()

        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-sonnet-4-20250514",
            allowed_tools=["anthropic_computer"],
        )
        try:
            # Phase 1: Initialize agent with task context
            print("🔧 Initializing agent...")
            await agent.initialize(task)

            # Phase 2: Run setup tool
            print("📋 Running setup...")
            setup_result = await agent.call_tool(task.setup_tool)

            # just wait for 500 seconds
            await asyncio.sleep(2500)


            print("🚀 Launching cloud browser...")
            result = await agent.run(task)
            print(f"✅ Cloud task completed: {result}")

        finally:
            await client.close()


async def run_local_example():
    """Run agent with local Docker environment."""
    print("\n🐳 Local Docker Deployment Example")
    print("-" * 40)

    with hud.trace("Local Browser Demo"):
        # Local configuration - runs Docker container
        mcp_config = {
            "browser": {
                "command": "docker",
                "args": [
                    "run",
                    "--rm",
                    "-i",
                    "-p",
                    "9222:9222",  # VNC port for viewing
                    "-p",
                    "6080:6080",  # VNC port for viewing
                    "hud-gmail-clone:v1.0",
                ],
            }
        }

        client = MCPClient(mcp_config=mcp_config)
        agent = ClaudeAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
        )

        try:
            print("🚀 Starting local Docker container...")
            print("   View browser at: http://localhost:8080/vnc.html")

            result = await agent.run("Navigate to example.com and take a screenshot", max_steps=3)
            print(f"✅ Local task completed: {result.reward}")

        finally:
            await client.close()


async def main():
    """Run both examples to show the differences."""
    print("🔍 Comparing Cloud vs Local Deployments")
    print("=" * 50)

    # Parse args --cloud or --local
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", action="store_true", help="Run cloud example")
    parser.add_argument("--local", action="store_true", help="Run local example")
    args = parser.parse_args()

    if args.cloud:
        await run_cloud_example()
    elif args.local:
        await run_local_example()
    else:
        # Check prerequisites
        if not os.getenv("HUD_API_KEY"):
            print("\n⚠️  Note: HUD_API_KEY not set, cloud example will fail")
            print("   Get your key at: https://app.hud.so")

        # Run cloud example
        try:
            await run_cloud_example()
        except Exception as e:
            print(f"❌ Cloud example failed: {e}")

        # Run local example (requires Docker)
        print("\n" + "=" * 50)

        try:
            # Check if Docker is available
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            if proc.returncode == 0:
                await run_local_example()
            else:
                print("❌ Docker not available, skipping local example")

        except FileNotFoundError:
            print("❌ Docker not found, skipping local example")
            print("   Install Docker to run local environments")

        print("\n📚 Key Differences:")
        print("   Cloud: No setup required, scalable, requires API key")
        print("   Local: Full control, works offline, requires Docker")


if __name__ == "__main__":
    asyncio.run(main())
