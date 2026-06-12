"""Advanced Task storage and management with complete features."""

import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from config import TASK_FILE, DATABASE_PATH


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(Enum):
    """Task status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class Task:
    """Represents a single task with advanced features."""
    
    def __init__(
        self,
        name: str,
        priority: str = "medium",
        due_date: Optional[str] = None,
        category: str = "general",
        description: str = "",
        subtasks: Optional[List[str]] = None,
        status: str = "pending"
    ):
        self.id = self._generate_id()
        self.name = name
        self.priority = priority.lower() if priority in ["low", "medium", "high", "urgent"] else "medium"
        self.due_date = due_date
        self.category = category
        self.description = description
        self.subtasks = subtasks or []
        self.status = status
        self.completed = status == "completed"
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.completed_at: Optional[str] = None
        self.tags: List[str] = []
    
    @staticmethod
    def _generate_id() -> str:
        """Generate a unique task ID."""
        return f"task_{datetime.now().timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "due_date": self.due_date,
            "category": self.category,
            "description": self.description,
            "subtasks": self.subtasks,
            "status": self.status,
            "completed": self.completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "tags": self.tags
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        task = Task(
            name=data.get("name", "Unnamed"),
            priority=data.get("priority", "medium"),
            due_date=data.get("due_date"),
            category=data.get("category", "general"),
            description=data.get("description", ""),
            subtasks=data.get("subtasks", []),
            status=data.get("status", "pending")
        )
        task.id = data.get("id", task.id)
        task.created_at = data.get("created_at", task.created_at)
        task.updated_at = data.get("updated_at", task.updated_at)
        task.completed_at = data.get("completed_at")
        task.tags = data.get("tags", [])
        task.completed = data.get("completed", False)
        return task
    
    def __str__(self) -> str:
        """String representation of task."""
        status_icon = {"completed": "✓", "pending": "○", "in_progress": "→", "on_hold": "⏸", "cancelled": "✗"}.get(self.status, "○")
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "urgent": "🔥"}.get(self.priority, "○")
        due = f" | Due: {self.due_date}" if self.due_date else ""
        category = f" | {self.category}" if self.category != "general" else ""
        return f"{status_icon} {priority_emoji} {self.name}{due}{category}"


class TaskManager:
    """Advanced task management with full features."""

    def __init__(self, task_file: str = TASK_FILE):
        self.task_file = task_file
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create task file if it doesn't exist."""
        if not os.path.exists(self.task_file):
            os.makedirs(os.path.dirname(self.task_file) or ".", exist_ok=True)
            with open(self.task_file, "w") as f:
                json.dump({"tasks": [], "version": "2.0"}, f, indent=2)

    def _load_tasks(self) -> List[Task]:
        """Load tasks from JSON file."""
        try:
            with open(self.task_file, "r") as f:
                data = json.load(f)
                return [Task.from_dict(t) for t in data.get("tasks", [])]
        except Exception as e:
            print(f"Error loading tasks: {e}")
            return []

    def _save_tasks(self, tasks: List[Task]) -> None:
        """Save tasks to JSON file."""
        try:
            with open(self.task_file, "w") as f:
                json.dump(
                    {"tasks": [t.to_dict() for t in tasks], "version": "2.0"},
                    f,
                    indent=2
                )
        except Exception as e:
            print(f"Error saving tasks: {e}")

    def add_task(self, task_name: str, priority: str = "medium", due_date: Optional[str] = None, category: str = "general", description: str = "") -> str:
        """Add a new task."""
        if not task_name or task_name.strip() == "":
            return "❌ Task name cannot be empty"
        
        task_name = task_name.strip()
        priority = priority.lower() if priority in ["low", "medium", "high", "urgent"] else "medium"
        
        tasks = self._load_tasks()
        
        # Check for duplicates
        for t in tasks:
            if t.name.lower() == task_name.lower() and t.status != "completed":
                return f"⚠️ Task already exists: {task_name}"
        
        new_task = Task(task_name, priority, due_date, category, description)
        tasks.append(new_task)
        self._save_tasks(tasks)
        
        return f"✅ Task added: '{task_name}' (Priority: {priority}, Category: {category})"

    def view_tasks(self, filter_type: str = "all", sort_by: str = "priority") -> str:
        """View tasks with filtering and sorting."""
        tasks = self._load_tasks()
        
        if not tasks:
            return "📭 No tasks found. Add one with: 'add task [name]'"
        
        # Apply filters
        if filter_type == "completed":
            tasks = [t for t in tasks if t.completed]
        elif filter_type == "pending":
            tasks = [t for t in tasks if not t.completed]
        elif filter_type in ["high", "medium", "low", "urgent"]:
            tasks = [t for t in tasks if t.priority == filter_type]
        elif filter_type == "overdue":
            from datetime import date
            today = date.today().isoformat()
            tasks = [t for t in tasks if t.due_date and t.due_date < today and not t.completed]
        elif filter_type == "today":
            from datetime import date
            today = date.today().isoformat()
            tasks = [t for t in tasks if t.due_date and t.due_date == today]
        
        if not tasks:
            return f"📭 No tasks found with filter: {filter_type}"
        
        # Sort tasks
        if sort_by == "priority":
            priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
            tasks = sorted(tasks, key=lambda t: priority_order.get(t.priority, 4))
        elif sort_by == "due_date":
            tasks = sorted(tasks, key=lambda t: t.due_date or "9999-12-31")
        elif sort_by == "created":
            tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)
        
        task_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        return f"📋 Your tasks ({len(tasks)}):\n{task_list}"

    def search_task(self, keyword: str) -> str:
        """Search for tasks by keyword."""
        if not keyword or keyword.strip() == "":
            return "❌ Please provide a search keyword"
        
        keyword = keyword.strip().lower()
        tasks = self._load_tasks()
        matching_tasks = [t for t in tasks if keyword in t.name.lower() or keyword in t.description.lower()]
        
        if not matching_tasks:
            return f"🔍 No tasks found matching '{keyword}'"
        
        task_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(matching_tasks)])
        return f"🔍 Found {len(matching_tasks)} task(s):\n{task_list}"

    def complete_task(self, task_name: str) -> str:
        """Mark a task as completed."""
        if not task_name or task_name.strip() == "":
            return "❌ Please provide a task name"
        
        task_name = task_name.strip()
        tasks = self._load_tasks()
        
        for task in tasks:
            if task.name.lower() == task_name.lower():
                if task.completed:
                    return f"ℹ️ Task already completed: {task_name}"
                task.completed = True
                task.status = "completed"
                task.completed_at = datetime.now().isoformat()
                task.updated_at = datetime.now().isoformat()
                self._save_tasks(tasks)
                return f"✅ Task completed: '{task_name}'"
        
        return f"❌ Task not found: '{task_name}'"

    def delete_task(self, task_name: str) -> str:
        """Delete a task."""
        if not task_name or task_name.strip() == "":
            return "❌ Please provide a task name"
        
        task_name = task_name.strip()
        tasks = self._load_tasks()
        original_count = len(tasks)
        
        tasks = [t for t in tasks if t.name.lower() != task_name.lower()]
        
        if len(tasks) == original_count:
            return f"❌ Task not found: '{task_name}'"
        
        self._save_tasks(tasks)
        return f"✅ Task deleted: '{task_name}'"

    def update_task(self, task_name: str, **kwargs) -> str:
        """Update a task."""
        if not task_name or task_name.strip() == "":
            return "❌ Please provide a task name"
        
        task_name = task_name.strip()
        tasks = self._load_tasks()
        
        for task in tasks:
            if task.name.lower() == task_name.lower():
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                task.updated_at = datetime.now().isoformat()
                self._save_tasks(tasks)
                return f"✅ Task updated: '{task_name}'"
        
        return f"❌ Task not found: '{task_name}'"

    def get_task_stats(self) -> str:
        """Get task statistics."""
        tasks = self._load_tasks()
        
        if not tasks:
            return "📊 No tasks to analyze"
        
        total = len(tasks)
        completed = len([t for t in tasks if t.completed])
        pending = total - completed
        high_priority = len([t for t in tasks if t.priority == "high" and not t.completed])
        urgent = len([t for t in tasks if t.priority == "urgent" and not t.completed])
        
        stats = f"""📊 Task Statistics:
  Total Tasks: {total}
  Completed: {completed} ({100*completed//total if total > 0 else 0}%)
  Pending: {pending}
  High Priority: {high_priority}
  Urgent: {urgent}"""
        
        return stats

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks as a list."""
        return self._load_tasks()
