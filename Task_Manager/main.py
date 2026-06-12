#!/usr/bin/env python3
"""Main entry point for the Advanced LangChain ReAct Task Manager Agent."""

import sys
from typing import List
from agent import ReactTaskAgent
from llm_config import validate_llm_connection


def print_banner():
    """Print welcome banner."""
    banner = """                                                  
          🤖 LangChain ReAct Task Manager Agent                    
          Advanced Task Management with AI Reasoning                
    """
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
📚 AGENT CAPABILITIES:

1. ADD TASK
   • 'add task buy milk'
   • 'add task finish homework|high|2024-12-25|work|Complete Python assignment'
   • Supports: name, priority, due date, category, description

2. VIEW TASKS
   • 'show my tasks' or 'view all'
   • 'view pending' - incomplete tasks
   • 'view completed' - finished tasks
   • 'view high' or 'view urgent' - by priority
   • 'view overdue' - past due dates
   • 'view today' - due today

3. SEARCH TASKS
   • 'search homework'
   • 'search python'
   • 'find presentation'

4. COMPLETE TASK
   • 'done homework'
   • 'mark presentation complete'
   • 'complete python assignment'

5. DELETE TASK
   • 'delete buy milk'
   • 'remove homework'

6. UPDATE TASK
   • 'update homework|priority=high'
   • 'update task|status=in_progress|priority=urgent'

7. GET STATISTICS
   • 'show stats'
   • 'task statistics'

💡 NATURAL LANGUAGE EXAMPLES:
   • "I need to prepare a presentation for tomorrow, make it high priority"
   • "Show me all my urgent tasks"
   • "Mark the homework as done"
   • "What tasks are overdue?"
    """
    print(help_text)


def run_examples(agent: ReactTaskAgent) -> None:
    """Run example interactions."""
    print("\n" + "="*75)
    print("📋 EXAMPLE INTERACTIONS DEMONSTRATING THE ReAct AGENT")
    print("="*75)
    
    examples: List[str] = [
        'Add task "Finish LangChain homework" with high priority',
        'Add task "Prepare AI presentation" due 2024-12-25 in work category',
        'Add task "Finish Python assignment" with medium priority',
        'Show all my pending tasks',
        'Search for tasks related to homework',
        'Mark the LangChain homework as complete',
        'View my completed tasks',
        'Show all high priority tasks',
        'Delete the task about buying groceries',
        'Get task statistics and summary',
    ]
    
    for user_input in examples:
        agent.process_user_request(user_input)


def interactive_mode(agent: ReactTaskAgent) -> None:
    """Run interactive mode."""
    print("\n" + "="*75)
    print("💬 INTERACTIVE MODE - Chat with the ReAct Agent")
    print("="*75)
    print("\nYou can now interact with the agent!")
    print("Type 'help' for commands, 'history' for interaction history, or 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit' or user_input.lower() == 'quit':
                print("\n✅ Thank you for using the ReAct Task Manager Agent!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
            
            elif user_input.lower() == 'history':
                history = agent.get_interaction_history()
                if history:
                    print("\n📜 Interaction History:")
                    for i, interaction in enumerate(history, 1):
                        for key, value in interaction.items():
                            print(f"{i}. {key}: {value}")
                else:
                    print("No interactions yet")
            
            elif user_input.lower() == 'clear':
                agent.clear_history()
                print("History cleared")
            
            else:
                agent.process_user_request(user_input)
        
        except KeyboardInterrupt:
            print("\n\n✅ Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def main():
    """Main function."""
    print_banner()
    
    print("\n🔧 Initializing ReAct Task Manager Agent...\n")
    
    # Validate LLM connection
    try:
        if not validate_llm_connection():
            print("⚠️ LLM validation warning - agent may not work optimally")
    except Exception as e:
        print(f"⚠️ LLM connection error: {e}")
    
    # Initialize the agent
    try:
        agent = ReactTaskAgent()
        print("\n✅ ReAct Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        sys.exit(1)
    
    # Run example interactions
    run_examples(agent)
    
    # Run interactive mode
    interactive_mode(agent)


if __name__ == "__main__":
    main()
