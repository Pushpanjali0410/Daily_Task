"""
api/app.py
FastAPI application with /chat and /upload endpoints.
"""

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel

from chatbot.support_agent import chat, clear_memory
from knowledge.rag_pipeline import build_vectorstore, add_document_to_store

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI(
    title="AI Customer Support Assistant",
    description="AI-powered customer support using RAG, LangChain Agents, Groq LLM, and FastAPI.",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"[API] {request.method} {request.url.path} — incoming")
    response = await call_next(request)
    logger.info(f"[API] {request.method} {request.url.path} — {response.status_code}")
    return response


@app.on_event("startup")
async def startup_event():
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not set! Chat endpoint will fail.")
    logger.info("Initialising RAG vector store on startup...")
    try:
        build_vectorstore(DOCS_DIR, save=True)
        logger.info("Vector store ready.")
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

    model_config = {"json_schema_extra": {"example": {"message": "Where is my order ORD123?"}}}


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: list[str] = []


class ClearMemoryRequest(BaseModel):
    session_id: str


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "AI Customer Support Assistant is running."}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(req: ChatRequest):
    """
    Send a message to the AI assistant.
    Answers policy questions via RAG, checks orders, and creates support tickets.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"[/chat] session={session_id} | message='{req.message[:100]}'")

    try:
        result = chat(message=req.message, session_id=session_id, groq_api_key=GROQ_API_KEY)
        return ChatResponse(**result)
    except Exception as e:
        logger.exception(f"[/chat] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a new .txt document to expand the RAG knowledge base."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    save_path = DOCS_DIR / file.filename
    logger.info(f"[/upload] Saving: {save_path}")

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        chunks_added = add_document_to_store(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update vector store: {e}")

    return {
        "message": f"Document '{file.filename}' uploaded and indexed successfully.",
        "chunks_added": chunks_added,
        "filename": file.filename,
    }


@app.post("/clear-memory", tags=["Chat"])
async def clear_session_memory(req: ClearMemoryRequest):
    """Clear the conversation history for a session."""
    clear_memory(req.session_id)
    return {"message": f"Memory cleared for session: {req.session_id}"}


@app.get("/sessions/{session_id}/history", tags=["Chat"])
async def get_session_history(session_id: str):
    """Return conversation history for a session."""
    from chatbot import get_memory
    try:
        memory = get_memory(session_id)
        messages = [
            {"role": "human" if i % 2 == 0 else "ai", "content": m.content}
            for i, m in enumerate(memory.chat_memory.messages)
        ]
        return {"session_id": session_id, "history": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))