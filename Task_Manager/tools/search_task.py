"""Search Task Tool implementation."""

from langchain.tools import tool
from task_manager import TaskManager

task_manager = TaskManager()


@tool
def search_task_tool(keyword: str) -> str:
    """Search for tasks matching a given keyword.

    Args:
        keyword: The keyword to search for in tasks

    Returns:
        Formatted list of matching tasks
    """
    return task_manager.search_task(keyword)


# Tool metadata for LangChain
search_task_tool.name = "search_task"
search_task_tool.description = "Search for tasks containing a specific keyword. Input: keyword (string). Returns a numbered list of matching tasks."
