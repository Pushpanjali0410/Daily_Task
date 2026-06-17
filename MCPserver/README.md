# Personal Assistant MCP Server

A working MCP server that tracks tasks and notes, ready to run and connect to Claude.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Test it on its own (no AI client needed)

```bash
npx @modelcontextprotocol/inspector python server.py
```

This opens a browser UI where you can call `add_task`, `complete_task`, `add_note`,
and read the `tasks://all` / `notes://all` resources directly, so you can see it
working before plugging it into anything else.

## 3. Connect it to Claude Desktop

Open this file (create it if it doesn't exist):

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add an entry, using the **full absolute path** to `server.py` on your machine:

```json
{
  "mcpServers": {
    "personal-assistant": {
      "command": "python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

Restart Claude Desktop. The server's tools, resources, and prompts will now be
available in any conversation.

## 4. Connect it to Claude Code instead

```bash
claude mcp add personal-assistant -- python /absolute/path/to/server.py
```

## 5. Try it

Once connected, just talk to Claude normally:

- "Add a task: finish the MCP guide, high priority"
- "What are my open tasks?"
- "Mark task 1 as done"

Claude will call the right tool or resource on its own -- you never name them.
