"""Configuration settings for the ReAct Task Manager Agent."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Task Storage
TASK_FILE = "tasks.json"
DATABASE_PATH = "./data"

# Agent Configuration
AGENT_NAME = "ReAct Task Manager Agent"
AGENT_DESCRIPTION = """An intelligent task management agent powered by LangChain ReAct Framework.
I help you manage tasks using natural language understanding and advanced reasoning.
I can add tasks, view your task list, search for tasks, mark them complete, and delete tasks.
I understand priorities and provide smart task filtering and organization."""

# Tool Names
ADD_TASK_TOOL = "add_task"
VIEW_TASKS_TOOL = "view_tasks"
SEARCH_TASK_TOOL = "search_task"
COMPLETE_TASK_TOOL = "complete_task"
DELETE_TASK_TOOL = "delete_task"
UPDATE_TASK_TOOL = "update_task"
GET_TASK_STATS_TOOL = "get_task_stats"

# Validation
if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set when using Groq provider")

# Agent Settings
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "15"))
EARLY_STOPPING_METHOD = os.getenv("EARLY_STOPPING_METHOD", "force")
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"

# Feature Flags
ENABLE_PRIORITIES = True
ENABLE_DUE_DATES = True
ENABLE_SUBTASKS = True
ENABLE_CATEGORIES = True
ENABLE_REMINDERS = True
