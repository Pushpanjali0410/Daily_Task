# Simple Claude Agent SDK Example

A minimal, read-only agent built with the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview). Give it a task in plain English, and it autonomously reads files, searches your project, and searches the web to complete it — streaming its reasoning and tool calls live to your terminal.

## What it does

This isn't a single API call — it's an agent loop. You give it a task, and Claude decides on its own which files to open, what to search for, and in what order, based on what it finds at each step. You watch it work in real time: its reasoning prints as plain text, and every tool it calls (`Read`, `Glob`, `Grep`, `WebSearch`) prints a `[tool call]` line so you can see exactly what it's doing.

This version is intentionally **read-only** — it can look around and report back, but it can't edit files or run shell commands. That's a safe default to start from before granting more access.

## Requirements

- Python 3.10+
- An Anthropic API key, or an active Claude Pro/Max/Team/Enterprise subscription

## Installation

```bash
pip install claude-agent-sdk
```

## Setup

Set your API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

Get a key from [platform.claude.com](https://platform.claude.com). New accounts include a small amount of free credits, enough to test this script before deciding on a paid plan.

## Usage

```bash
python simple_agent.py "List the files in this directory and summarize what each one does"
```

Any natural-language task works:

```bash
python simple_agent.py "Find every function in this repo that doesn't validate its inputs"
python simple_agent.py "Search the web for the latest version of FastAPI and check if we're using it"
```

## Configuration

The agent's behavior is controlled by `ClaudeAgentOptions` in `simple_agent.py`:

| Option | What it controls |
|---|---|
| `allowed_tools` | Which tools Claude can use. Currently `Read`, `Glob`, `Grep`, `WebSearch`. Add `"Edit"` or `"Bash"` to let it write files or run commands. |
| `permission_mode` | Whether tool use needs approval. `"acceptEdits"` auto-approves allowed tools; use `"default"` with a `canUseTool` callback for a human-in-the-loop. |
| `system_prompt` | The agent's persona/instructions, applied to every run. |

## ⚠️ Granting write or execute access

Adding `"Edit"` or `"Bash"` to `allowed_tools` lets the agent modify files or run shell commands on your machine without confirmation under `acceptEdits` mode. Test in a sandboxed directory or version-controlled repo first, and consider `permission_mode="default"` with a custom approval callback for anything beyond local experiments.

## License

MIT (or replace with your project's license)
