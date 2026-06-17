"""
A personal notes & task-tracking MCP server.

This exposes three kinds of capabilities to any MCP client
(Claude Desktop, Claude Code, or a client you write yourself):

    Tools     -> things the AI can DO     (add_task, complete_task, add_note)
    Resources -> things the AI can READ   (tasks://all, notes://all)
    Prompts   -> templates a user triggers (daily_standup)

Run it directly:
    python server.py

Test it without any AI client at all, using the official inspector:
    npx @modelcontextprotocol/inspector python server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("personal-assistant")

# In-memory storage. Swap this for a file or a real database if you want
# data to survive a restart -- everything here resets when the script stops.
TASKS: list[dict] = []
NOTES: list[str] = []


# ---------------------------------------------------------------------------
# TOOLS -- the model calls these to take an action
# ---------------------------------------------------------------------------

@mcp.tool()
def add_task(title: str, priority: str = "medium") -> str:
    """Add a new task.

    Args:
        title: what the task is
        priority: low, medium, or high
    """
    task = {"id": len(TASKS) + 1, "title": title, "priority": priority, "done": False}
    TASKS.append(task)
    return f"Added task #{task['id']}: {title} ({priority})"


@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a task as done.

    Args:
        task_id: the numeric id of the task
    """
    for t in TASKS:
        if t["id"] == task_id:
            t["done"] = True
            return f"Task #{task_id} marked complete."
    return f"No task with id {task_id}."


@mcp.tool()
def add_note(content: str) -> str:
    """Save a free-text note.

    Args:
        content: the text to save
    """
    NOTES.append(content)
    return f"Saved note #{len(NOTES)}"


# ---------------------------------------------------------------------------
# RESOURCES -- the model reads these for context, no side effects
# ---------------------------------------------------------------------------

@mcp.resource("tasks://all")
def get_all_tasks() -> str:
    """The current task list."""
    if not TASKS:
        return "No tasks yet."
    return "\n".join(
        f"[{'x' if t['done'] else ' '}] #{t['id']} {t['title']} ({t['priority']})"
        for t in TASKS
    )


@mcp.resource("notes://all")
def get_all_notes() -> str:
    """All saved notes."""
    return "\n---\n".join(NOTES) if NOTES else "No notes yet."


# ---------------------------------------------------------------------------
# PROMPTS -- reusable templates a user can trigger directly
# ---------------------------------------------------------------------------

@mcp.prompt()
def daily_standup() -> str:
    """A standup-style summary prompt."""
    return (
        "Read my current tasks (tasks://all) and give me a 3-line standup: "
        "what's done, what's in progress, what's blocked."
    )


if __name__ == "__main__":
    mcp.run()
