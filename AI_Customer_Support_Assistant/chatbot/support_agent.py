"""
chatbot/support_agent.py
LangChain agent with Groq LLM, RAG retrieval, tools, and conversation memory.
"""

import logging
from typing import Dict

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool

from knowledge.rag_pipeline import retrieve_context
from support_tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Per-session memory store
_session_memories: Dict[str, ConversationBufferWindowMemory] = {}


def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in _session_memories:
        _session_memories[session_id] = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
        )
        logger.info(f"Created new memory for session: {session_id}")
    return _session_memories[session_id]


def clear_memory(session_id: str):
    if session_id in _session_memories:
        del _session_memories[session_id]
        logger.info(f"Cleared memory for session: {session_id}")


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the company knowledge base for information about policies, FAQs, shipping, and returns.
    Use this for questions about return policy, shipping timelines, payment methods,
    cancellations, or any general company policy question.
    Input: the customer's question or topic to search.
    """
    logger.info(f"[TOOL] search_knowledge_base called: '{query[:80]}'")
    context = retrieve_context(query, k=3)
    if not context:
        return "No relevant information found in the knowledge base."
    return context


ALL_AGENT_TOOLS = [search_knowledge_base] + ALL_TOOLS

SYSTEM_PROMPT = """You are a helpful and professional customer support assistant for an e-commerce platform.

Your responsibilities:
1. Answer questions about company policies (returns, shipping, payments, FAQs) using the `search_knowledge_base` tool.
2. Check order status using the `check_order_status` tool when customers ask about their orders.
3. Create support tickets using the `create_support_ticket` tool when customers report problems.
4. For complex queries (e.g. "I want to return my order ORD123"), use MULTIPLE tools in sequence.

Guidelines:
- Always be polite, empathetic, and concise.
- Always use `check_order_status` for order queries — never guess order details.
- If a customer reports an issue (damaged item, missing package), create a support ticket.
- Ground your responses in retrieved information; do not make up policies.
- Remember context from earlier in the conversation.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

_agent_executor = None


def get_agent(groq_api_key: str) -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024,
        )
        agent = create_tool_calling_agent(llm=llm, tools=ALL_AGENT_TOOLS, prompt=PROMPT)
        _agent_executor = AgentExecutor(
            agent=agent,
            tools=ALL_AGENT_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            return_intermediate_steps=True,
        )
        logger.info("AgentExecutor built with Groq LLM")
    return _agent_executor


def chat(message: str, session_id: str, groq_api_key: str) -> dict:
    logger.info(f"[CHAT] session={session_id} | user='{message[:100]}'")

    memory = get_memory(session_id)
    executor = get_agent(groq_api_key)
    history = memory.chat_memory.messages

    result = executor.invoke({
        "input": message,
        "chat_history": history,
    })

    answer = result.get("output", "I'm sorry, I couldn't process your request.")

    memory.chat_memory.add_user_message(message)
    memory.chat_memory.add_ai_message(answer)

    steps = result.get("intermediate_steps", [])
    tools_used = [step[0].tool for step in steps if step] if steps else []
    logger.info(f"[CHAT] tools_used={tools_used} | answer='{answer[:120]}'")

    return {
        "response": answer,
        "session_id": session_id,
        "tools_used": tools_used,
    }