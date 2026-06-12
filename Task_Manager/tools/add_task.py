"""Add Task Tool implementation."""

from langchain.tools import tool
from task_manager import TaskManager

task_manager = TaskManager()


@tool
def add_task_tool(task_name: str) -> str:
    """Add a new task to the task list.

    Args:
        task_name: The name of the task to add

    Returns:
        Success or failure message
    """
    return task_manager.add_task(task_name)


# Tool metadata for LangChain
add_task_tool.name = "add_task"
add_task_tool.description = "Add a new task to the task list. Input: task_name (string). Returns success confirmation."
