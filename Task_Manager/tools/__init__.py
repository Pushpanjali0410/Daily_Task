"""Task Manager tools package with advanced features."""

from langchain_community.tools import Tool
from task_manager import TaskManager

# Initialize task manager
task_manager = TaskManager()

# Add Task Tool
def add_task_func(input_str: str) -> str:
    """Add a new task."""
    try:
        if "|" in input_str:
            parts = input_str.split("|")
            task_name = parts[0].strip()
            priority = parts[1].strip() if len(parts) > 1 else "medium"
            due_date = parts[2].strip() if len(parts) > 2 else None
        else:
            task_name = input_str.strip()
            priority = "medium"
            due_date = None
        
        return task_manager.add_task(task_name, priority, due_date)
    except Exception as e:
        return f"Error adding task: {str(e)}"

add_task_tool = Tool(
    name="add_task",
    func=add_task_func,
    description="Add a new task. Input: task_name (or task_name|priority|due_date)"
)

# View Tasks Tool
def view_tasks_func(filter_type: str = "all") -> str:
    """View all tasks or filtered."""
    try:
        return task_manager.view_tasks(filter_type)
    except Exception as e:
        return f"Error viewing tasks: {str(e)}"

view_tasks_tool = Tool(
    name="view_tasks",
    func=view_tasks_func,
    description="Display all tasks or filtered (completed, pending, high, medium, low)"
)

# Search Task Tool
def search_task_func(keyword: str) -> str:
    """Search for tasks by keyword."""
    try:
        return task_manager.search_task(keyword)
    except Exception as e:
        return f"Error searching tasks: {str(e)}"

search_task_tool = Tool(
    name="search_task",
    func=search_task_func,
    description="Search for tasks matching a keyword"
)

# Complete Task Tool
def complete_task_func(task_name: str) -> str:
    """Mark a task as completed."""
    try:
        return task_manager.complete_task(task_name)
    except Exception as e:
        return f"Error completing task: {str(e)}"

complete_task_tool = Tool(
    name="complete_task",
    func=complete_task_func,
    description="Mark a task as completed"
)

# Delete Task Tool
def delete_task_func(task_name: str) -> str:
    """Delete a task."""
    try:
        return task_manager.delete_task(task_name)
    except Exception as e:
        return f"Error deleting task: {str(e)}"

delete_task_tool = Tool(
    name="delete_task",
    func=delete_task_func,
    description="Delete a task by name"
)

__all__ = ["add_task_tool", "view_tasks_tool", "search_task_tool", "complete_task_tool", "delete_task_tool"]
