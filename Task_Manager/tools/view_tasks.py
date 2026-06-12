"""View Tasks Tool implementation."""

from langchain.tools import tool
from task_manager import TaskManager

task_manager = TaskManager()


@tool
def view_tasks_tool() -> str:
    """Display all tasks in the task list.

    Returns:
        Formatted list of all tasks
    """
    return task_manager.view_tasks()


# Tool metadata for LangChain
view_tasks_tool.name = "view_tasks"
view_tasks_tool.description = "Display all tasks in the task list. Takes no input. Returns a numbered list of all current tasks."
