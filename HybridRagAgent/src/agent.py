"""
RAG Agent
=========
Wraps the HybridRetriever with an Ollama LLM to answer questions.
Supports multi-turn conversation with memory.
"""

from __future__ import annotations

import json
from typing import List, Optional

import httpx

from src.rag_engine import HybridRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# Ollama LLM
# ---------------------------------------------------------------------------

class OllamaLLM:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[dict]) -> str:
        try:
            r = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            return f"[LLM Error: {e}]"

    def generate(self, prompt: str) -> str:
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "options": {"temperature": self.temperature}},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["response"]
        except Exception as e:
            return f"[LLM Error: {e}]"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful document assistant powered by a Hybrid RAG system.
You answer questions strictly based on the retrieved context provided.

Rules:
1. Only use information from the provided context.
2. If the context does not contain enough information, say so clearly.
3. Cite the source document and page/chunk when possible.
4. Be concise but complete.
5. Never hallucinate facts not present in the context.
"""


def build_rag_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        src = chunk.doc.metadata.get("source", "unknown")
        page = chunk.doc.metadata.get("page", "")
        page_info = f", page {page}" if page else ""
        context_parts.append(
            f"[Context {i}] (Source: {src}{page_info}, Score: {chunk.score:.4f})\n{chunk.doc.text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return f"""Based on the following retrieved context, answer the question.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


# ---------------------------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------------------------

class RAGAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OllamaLLM,
        max_history: int = 6,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_history = max_history
        self._history: List[dict] = []
        self._last_chunks: List[RetrievedChunk] = []

    def ask(self, query: str) -> dict:
        """
        Returns:
          answer: str
          sources: list of source metadata dicts
          chunks: retrieved chunks (for evaluation)
        """
        chunks = self.retriever.retrieve(query)
        self._last_chunks = chunks

        if not chunks:
            answer = "I couldn't find relevant information in the uploaded documents."
            return {"answer": answer, "sources": [], "chunks": []}

        rag_prompt = build_rag_prompt(query, chunks)

        # Build message history for multi-turn
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._history[-self.max_history:])
        messages.append({"role": "user", "content": rag_prompt})

        answer = self.llm.chat(messages)

        # Update history with the clean query (not the full RAG prompt)
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": answer})

        sources = []
        seen = set()
        for chunk in chunks:
            src = chunk.doc.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append({
                    "source": src,
                    "page": chunk.doc.metadata.get("page"),
                    "score": round(chunk.score, 4),
                    "fusion": chunk.source,
                })

        return {"answer": answer, "sources": sources, "chunks": chunks}

    def reset_history(self):
        self._history.clear()

    @property
    def last_chunks(self) -> List[RetrievedChunk]:
        return self._last_chunks
