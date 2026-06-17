"""
Simple multi-agent system: CrewAI agents orchestrated by a LangGraph workflow,
powered by Groq's LLM API.

Flow:
    START -> research_node (CrewAI Researcher) -> write_node (CrewAI Writer) -> END

Setup:
    pip install crewai langgraph langchain-groq python-dotenv
    Create a .env file in the same folder with:
        GROQ_API_KEY=your_groq_api_key_here

Run:
    python multi_agent_system.py "your topic here"
"""

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. Setup: environment + LLM
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit(
        "ERROR: GROQ_API_KEY is not set.\n"
        "Create a .env file next to this script with:\n"
        "    GROQ_API_KEY=your_groq_api_key_here\n"
        "Get a free key at https://console.groq.com/keys"
    )

# CrewAI's LLM class talks to Groq when the model id is prefixed with "groq/".
# Model id confirmed current on Groq as of mid-2026.
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.5,
)


# ---------------------------------------------------------------------------
# 2. CrewAI agents
# ---------------------------------------------------------------------------

researcher = Agent(
    role="Senior Research Analyst",
    goal="Gather accurate, well-organized key facts and points about {topic}",
    backstory=(
        "You are a meticulous analyst who distills complex topics into "
        "clear, well-structured bullet points with no fluff."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Turn raw research notes into a clear, engaging short article",
    backstory=(
        "You are a skilled writer who turns research notes into a concise, "
        "well-structured article that is easy for a general audience to read."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# 3. LangGraph state + nodes
#    Each node packages one CrewAI agent + task into a single-step Crew run.
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    topic: str
    research: str
    article: str


def research_node(state: GraphState) -> dict:
    """Runs the Researcher agent and stores its findings in state['research']."""
    task = Task(
        description=(
            "Research the topic: '{topic}'. Produce 5-8 concise bullet points "
            "covering the most important and current facts. Avoid speculation."
        ),
        expected_output="A bullet-point list of key facts about the topic.",
        agent=researcher,
    )
    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)
    result = crew.kickoff(inputs={"topic": state["topic"]})
    return {"research": str(result)}


def write_node(state: GraphState) -> dict:
    """Runs the Writer agent on the research notes and stores the final article."""
    task = Task(
        description=(
            "Using these research notes:\n\n{research}\n\n"
            "Write a short, engaging article (3-4 paragraphs) about '{topic}' "
            "for a general audience."
        ),
        expected_output="A polished short article in plain text.",
        agent=writer,
    )
    crew = Crew(agents=[writer], tasks=[task], process=Process.sequential)
    result = crew.kickoff(
        inputs={"topic": state["topic"], "research": state["research"]}
    )
    return {"article": str(result)}


# ---------------------------------------------------------------------------
# 4. Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("research", research_node)
    workflow.add_node("write", write_node)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------

def main():
    topic = " ".join(sys.argv[1:]).strip() or "the impact of AI on small businesses"

    graph = build_graph()

    try:
        final_state = graph.invoke({"topic": topic, "research": "", "article": ""})
    except Exception as exc:  # surface a clean error instead of a raw traceback
        sys.exit(f"ERROR while running the agent workflow: {exc}")

    print("\n" + "=" * 70)
    print("RESEARCH NOTES")
    print("=" * 70)
    print(final_state["research"])

    print("\n" + "=" * 70)
    print("FINAL ARTICLE")
    print("=" * 70)
    print(final_state["article"])


if __name__ == "__main__":
    main()
