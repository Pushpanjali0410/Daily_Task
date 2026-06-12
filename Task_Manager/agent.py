"""Advanced LangChain ReAct Agent for Task Management."""

from typing import List, Dict, Any, Optional
from langchain_community.tools import Tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import AGENT_NAME, AGENT_DESCRIPTION, MAX_ITERATIONS, EARLY_STOPPING_METHOD, VERBOSE
from llm_config import get_llm
from task_manager import TaskManager


class ReactTaskAgent:
    """LangChain ReAct Agent for Advanced Task Management."""

    def __init__(self):
        """Initialize the ReAct Agent with all tools."""
        self.llm = get_llm()
        self.task_manager = TaskManager()
        self.tools = self._create_advanced_tools()
        self.agent = self._create_react_agent()
        self.executor = self._create_executor()
        self.interaction_history: List[Dict[str, str]] = []

    def _create_advanced_tools(self) -> List[Tool]:
        """Create advanced tools for the agent."""
        
        # Tool 1: Add Task
        def add_task_func(task_info: str) -> str:
            """Add a new task with extended attributes.
            Input format: task_name|priority|due_date|category|description
            """
            if not task_info or task_info.strip() == "":
                return "❌ Task information required"
            
            parts = task_info.split("|")
            task_name = parts[0].strip() if parts else ""
            priority = parts[1].strip().lower() if len(parts) > 1 else "medium"
            due_date = parts[2].strip() if len(parts) > 2 else None
            category = parts[3].strip() if len(parts) > 3 else "general"
            description = parts[4].strip() if len(parts) > 4 else ""
            
            return self.task_manager.add_task(task_name, priority, due_date, category, description)
        
        add_task_tool = Tool(
            name="add_task",
            func=add_task_func,
            description="""Add a new task to the task list.
            Input: task_name (required) | priority (optional: low/medium/high/urgent) | due_date (optional: YYYY-MM-DD) | category (optional) | description (optional)
            Example: 'Prepare AI presentation|high|2024-12-25|work|Create slides and speech'
            Output: Confirmation message with task details"""
        )
        
        # Tool 2: View Tasks
        def view_tasks_func(filter_spec: str = "") -> str:
            """Display tasks with filtering and sorting.
            Filter options: all, pending, completed, high, medium, low, urgent, overdue, today
            """
            filter_type = filter_spec.strip().lower() if filter_spec else "pending"
            return self.task_manager.view_tasks(filter_type)
        
        view_tasks_tool = Tool(
            name="view_tasks",
            func=view_tasks_func,
            description="""Display tasks with filtering options.
            Input: filter_type (all, pending, completed, high, medium, low, urgent, overdue, today)
            Example: 'pending' or 'high' or 'today'
            Output: Formatted list of tasks matching the filter"""
        )
        
        # Tool 3: Search Task
        def search_task_func(keyword: str) -> str:
            """Search for tasks by keyword in name or description."""
            if not keyword or keyword.strip() == "":
                return "❌ Please provide a search keyword"
            return self.task_manager.search_task(keyword)
        
        search_task_tool = Tool(
            name="search_task",
            func=search_task_func,
            description="""Search for tasks matching a keyword.
            Input: keyword (string to search for in task names and descriptions)
            Example: 'presentation' or 'homework' or 'python'
            Output: List of matching tasks or 'No tasks found'"""
        )
        
        # Tool 4: Complete Task
        def complete_task_func(task_name: str) -> str:
            """Mark a task as completed."""
            if not task_name or task_name.strip() == "":
                return "❌ Please provide a task name"
            return self.task_manager.complete_task(task_name)
        
        complete_task_tool = Tool(
            name="complete_task",
            func=complete_task_func,
            description="""Mark a task as completed.
            Input: task_name (exact or partial name of the task)
            Example: 'Finish LangChain homework' or 'homework'
            Output: Confirmation that task is marked complete"""
        )
        
        # Tool 5: Delete Task
        def delete_task_func(task_name: str) -> str:
            """Delete a task from the list."""
            if not task_name or task_name.strip() == "":
                return "❌ Please provide a task name"
            return self.task_manager.delete_task(task_name)
        
        delete_task_tool = Tool(
            name="delete_task",
            func=delete_task_func,
            description="""Delete a task from the task list.
            Input: task_name (exact or partial name of the task to delete)
            Example: 'Buy groceries' or 'groceries'
            Output: Confirmation that task has been deleted"""
        )
        
        # Tool 6: Update Task
        def update_task_func(update_info: str) -> str:
            """Update task properties.
            Input format: task_name|property=value|property=value
            """
            if not update_info or "|" not in update_info:
                return "❌ Invalid update format. Use: task_name|priority=high|status=completed"
            
            parts = update_info.split("|")
            task_name = parts[0].strip()
            updates = {}
            
            for update in parts[1:]:
                if "=" in update:
                    key, value = update.split("=", 1)
                    updates[key.strip()] = value.strip()
            
            return self.task_manager.update_task(task_name, **updates)
        
        update_task_tool = Tool(
            name="update_task",
            func=update_task_func,
            description="""Update task properties like priority, status, due date.
            Input: task_name|property=value|property=value
            Example: 'Homework|priority=high|status=in_progress'
            Output: Confirmation that task has been updated"""
        )
        
        # Tool 7: Get Task Statistics
        def get_stats_func(input_str: str = "") -> str:
            """Get task statistics and summary."""
            return self.task_manager.get_task_stats()
        
        get_stats_tool = Tool(
            name="get_task_stats",
            func=get_stats_func,
            description="""Get statistics about tasks.
            Input: (no input needed)
            Output: Summary of total tasks, completed, pending, high priority, urgent tasks"""
        )
        
        return [add_task_tool, view_tasks_tool, search_task_tool, complete_task_tool, delete_task_tool, update_task_tool, get_stats_tool]

    def _create_react_agent(self):
        """Create the ReAct agent with comprehensive prompt."""
        prompt_template = """{agent_description}

You are expert at understanding user intent and selecting the right tool.
Always reason through the user's request step by step.

Available Tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "agent_description": f"You are {AGENT_NAME}.\n{AGENT_DESCRIPTION}",
                "tools": self._format_tools_description(),
                "tool_names": ", ".join([t.name for t in self.tools])
            }
        )
        
        try:
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            return agent
        except Exception as e:
            print(f"⚠️ Error creating ReAct agent: {e}")
            return None

    def _create_executor(self) -> Optional[AgentExecutor]:
        """Create the agent executor."""
        if self.agent is None:
            return None
        
        try:
            executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=VERBOSE,
                max_iterations=MAX_ITERATIONS,
                early_stopping_method=EARLY_STOPPING_METHOD,
                handle_parsing_errors=True
            )
            return executor
        except Exception as e:
            print(f"⚠️ Error creating executor: {e}")
            return None

    def _format_tools_description(self) -> str:
        """Format tools description for the prompt."""
        return "\n".join([f"{i+1}. {tool.name}: {tool.description}" for i, tool in enumerate(self.tools)])

    def run(self, user_input: str) -> str:
        """Run the ReAct agent with user input."""
        try:
            if not user_input or user_input.strip() == "":
                return "⚠️ Please provide a task command or query"
            
            # Store in history
            self.interaction_history.append({"user": user_input})
            
            if self.executor is None:
                return "❌ Agent executor not initialized"
            
            # Run the agent
            result = self.executor.invoke({"input": user_input})
            
            # Extract output
            response = ""
            if isinstance(result, dict):
                response = result.get("output", "No response generated")
            else:
                response = str(result)
            
            # Store in history
            self.interaction_history.append({"agent": response})
            
            return response
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.interaction_history.append({"error": error_msg})
            return error_msg

    def process_user_request(self, user_input: str) -> None:
        """Process and display user request with formatting."""
        print(f"\n{'='*75}")
        print(f"User: {user_input}")
        print(f"{'='*75}")
        
        response = self.run(user_input)
        print(f"\nAgent Response:\n{response}")
        print(f"{'='*75}\n")

    def get_interaction_history(self) -> List[Dict[str, str]]:
        """Get the interaction history."""
        return self.interaction_history

    def clear_history(self) -> None:
        """Clear the interaction history."""
        self.interaction_history = []
