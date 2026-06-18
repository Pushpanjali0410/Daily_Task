"""
Simple Claude Agent SDK example.

Install:
    pip install claude-agent-sdk   (requires Python 3.10+)

Set your API key:
    export ANTHROPIC_API_KEY="Your API key here"
Run:
    python simple_agent.py "List the files in this directory and summarize what each one does"
"""

import asyncio
import sys

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
)


async def run_agent(task: str) -> None:
    # `query()` is the entry point for the agent loop. It returns an async
    # iterator that yields messages as Claude thinks, calls tools, and
    # observes results — you just consume the stream.
    options = ClaudeAgentOptions(
        # Tools the agent is allowed to use. Read-only here is intentional;
        # add "Edit" or "Bash" once you trust the agent with writes/commands.
        allowed_tools=["Read", "Glob", "Grep", "WebSearch"],
        # Auto-approves the allowed tools without an interactive prompt.
        # Use "default" + a canUseTool callback if you want a human-in-the-loop.
        permission_mode="acceptEdits",
        system_prompt="You are a concise, careful coding assistant.",
    )

    async for message in query(prompt=task, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
                elif hasattr(block, "name"):
                    print(f"[tool call] {block.name}")
        elif isinstance(message, ResultMessage):
            print(f"\n--- done: {message.subtype} ---")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python simple_agent.py "<task for the agent>"')
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    asyncio.run(run_agent(task))


if __name__ == "__main__":
    main()
