# Task Manager Agent - LangChain ReAct Framework

An intelligent Task Manager Agent built with LangChain that uses the ReAct (Reasoning + Acting) framework to automatically manage tasks through tool-based interactions.

## 📋 Project Overview

This project implements a task management system powered by an LLM agent that can:
- **Add tasks** to a task list
- **View all tasks** currently stored
- **Search tasks** by keyword
- Make intelligent decisions about which tool to use based on user input

## ✨ Key Features

✅ Natural language task interaction
✅ Automatic tool selection using ReAct framework
✅ Persistent task storage (JSON-based)
✅ Semantic task search capabilities
✅ Comprehensive logging and feedback
✅ **Uses free LLM (Ollama/Groq) - No paid API required**

## 🔧 Tools Implemented

### 1. Add Task Tool
- **Purpose**: Add new tasks to the task list
- **Input**: Task name (string)
- **Output**: Success confirmation
- **Example**: "Add task 'Prepare AI presentation'"

### 2. View Tasks Tool
- **Purpose**: Display all tasks
- **Input**: None
- **Output**: Numbered list of all tasks
- **Example**: "Show my tasks"

### 3. Search Task Tool
- **Purpose**: Find tasks by keyword
- **Input**: Keyword (string)
- **Output**: Filtered list of matching tasks
- **Example**: "Search task homework"

## 🚀 Getting Started

### Prerequisites

**Option 1: Using Ollama (Local, Completely Free)**
```bash
# Install Ollama from https://ollama.ai
# Then run:
ollama pull llama2
ollama serve
```

**Option 2: Using Groq (Free API with rate limits)**
```bash
# Sign up for free at https://console.groq.com
# Get your API key
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pushrajesand-coder/DailyTaskManager.git
cd DailyTaskManager
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

#### For Ollama (Local - Recommended):
```bash
# .env file
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

#### For Groq (Free Cloud):
```bash
# .env file
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768
```

### Usage

Run the task manager agent:
```bash
python main.py
```

Example interactions:
```
User: Add task "Finish LangChain homework"
Agent: Task added successfully: Finish LangChain homework

User: Show my tasks
Agent: 
1. Prepare AI presentation
2. Finish Python assignment
3. Finish LangChain homework

User: Search task homework
Agent:
1. Finish LangChain homework
```

## 📁 Project Structure

```
DailyTaskManager/
├── main.py                 # Entry point for the application
├── agent.py               # ReAct Agent implementation
├── llm_config.py         # LLM provider configuration (Ollama/Groq)
├── tools/
│   ├── __init__.py
│   ├── add_task.py       # Add Task tool
│   ├── view_tasks.py     # View Tasks tool
│   └── search_task.py    # Search Task tool
├── task_manager.py        # Task storage and management
├── config.py             # Configuration and settings
├── requirements.txt      # Project dependencies
├── .env.example          # Environment variables template
├── tasks.json           # Task storage file (auto-created)
└── README.md            # Project documentation
```

## 🔄 Agent Workflow

1. **User Input**: Accept natural language task-related commands
2. **Reasoning**: Agent analyzes the input and determines the appropriate tool
3. **Tool Selection**: Based on reasoning, select one of three tools:
   - Add Task
   - View Tasks
   - Search Task
4. **Acting**: Execute the selected tool
5. **Response**: Return meaningful feedback to the user
6. **Loop**: Continue accepting new inputs

## 🎓 Technologies Used

- **Python**: Core programming language
- **LangChain**: Framework for building LLM-powered agents
- **ReAct Framework**: Reasoning + Acting paradigm
- **Ollama**: Free local LLM (Alternative)
- **Groq**: Free cloud LLM API (Alternative)
- **JSON**: Task storage

## 📝 Example Outputs

### Example 1: Add Task
```
Input: "Add task 'Prepare AI presentation'"
Thinking: The user wants to add a new task
Action: Add Task
Result: Task added successfully
```

### Example 2: View Tasks
```
Input: "Show my tasks"
Thinking: The user wants to see all existing tasks
Action: View Tasks
Result:
1. Prepare AI presentation
2. Finish Python assignment
```

### Example 3: Search Tasks
```
Input: "Search task homework"
Thinking: The user wants to search for specific tasks
Action: Search Task
Result:
1. Finish LangChain homework
```

## Free LLM Options Comparison

| Provider | Cost | Setup | Speed | Quality |
|----------|------|-------|-------|----------|
| **Ollama (Llama2)** | Free | Local Installation | Medium | Good |
| **Groq** | Free | API Key | Very Fast | Excellent |
| **Hugging Face** | Free | API Key | Medium | Good |

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is part of an educational assignment.
