# AI Customer Support Assistant

An AI-powered customer support chatbot for an e-commerce platform built with:

- **FastAPI** — REST API server
- **LangChain** — Agent orchestration and memory
- **Groq API** — LLM inference (`llama-3.3-70b-versatile`)
- **FAISS** — Local vector database for RAG
- **HuggingFace Sentence Transformers** — Local embeddings (no API key needed)

---

## Project Structure

```
project/
├── data/
│   ├── orders.json          # Order records
│   └── tickets.json         # Support tickets
├── docs/
│   ├── faq.txt              # Company FAQs
│   ├── return_policy.txt    # Return & refund policy
│   └── shipping_info.txt    # Shipping timelines & charges
├── rag/
│   └── rag_pipeline.py      # FAISS vector store + retrieval
├── agent/
│   └── support_agent.py     # LangChain agent + memory
├── tools/
│   └── support_tools.py     # check_order_status, create_support_ticket
├── api/
│   └── app.py               # FastAPI routes (/chat, /upload)
├── logs/
│   └── app.log              # Rotating log file (auto-created)
├── faiss_index/             # Persisted FAISS index (auto-created)
├── main.py                  # Entry point
├── logger_config.py         # Logging setup
├── requirements.txt
└── .env                     # API keys (you create this)
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com/)

### 2. Clone / download the project

```bash
cd project
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run. This is a one-time download.

### 5. Configure environment variables

Edit the `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free key at: https://console.groq.com/

### 6. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Server starts at: **http://localhost:8000**

Interactive API docs: **http://localhost:8000/docs**

---

## API Endpoints

### `POST /chat`

Send a message to the assistant.

**Request:**
```json
{
  "message": "Where is my order ORD123?",
  "session_id": "user-session-abc"   // optional; auto-generated if omitted
}
```

**Response:**
```json
{
  "response": "Order Details for ORD123:\n  Customer: Ravi Kumar\n  Status: Shipped\n  Expected Delivery: 2026-03-20",
  "session_id": "user-session-abc",
  "tools_used": ["check_order_status"]
}
```

---

### `POST /upload`

Upload a new `.txt` document to expand the knowledge base.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_new_policy.txt"
```

**Response:**
```json
{
  "message": "Document 'my_new_policy.txt' uploaded and indexed successfully.",
  "chunks_added": 12,
  "filename": "my_new_policy.txt"
}
```

---

### `POST /clear-memory`

Reset conversation history for a session.

```json
{ "session_id": "user-session-abc" }
```

---

### `GET /sessions/{session_id}/history`

View conversation history for debugging.

---

## Sample Test Cases

### 1. RAG Query — Return Policy

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the return policy?"}'
```

**Expected:** Retrieves from `return_policy.txt` — 7-day return window, unused product, refund in 5-7 days.

---

### 2. Tool Query — Order Status

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order ORD123?", "session_id": "s1"}'
```

**Expected:** Uses `check_order_status` tool → Ravi Kumar, Shipped, 2026-03-20.

---

### 3. Mixed Query — Return an Order

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to return my order ORD123.", "session_id": "s2"}'
```

**Expected:** Retrieves return policy via RAG + checks order status via tool → combined response.

---

### 4. Multi-turn Conversation (Memory)

```bash
# Turn 1
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My order is delayed.", "session_id": "s3"}'

# Turn 2
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "It is ORD456.", "session_id": "s3"}'
```

**Expected:** Assistant remembers context from Turn 1 and checks ORD456 status.

---

### 5. Create Support Ticket

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My package has not arrived and it has been 10 days."}'
```

**Expected:** Uses `create_support_ticket` tool → Ticket ID assigned.

---

## Architecture

```
User Request
    │
    ▼
FastAPI (/chat)
    │
    ▼
LangChain Agent (Groq LLM - llama-3.3-70b-versatile)
    │
    ├── search_knowledge_base ──► FAISS Vector Store ──► docs/*.txt
    ├── check_order_status    ──► data/orders.json
    └── create_support_ticket ──► data/tickets.json
    │
    ▼
Conversation Memory (per session_id, sliding window k=10)
    │
    ▼
Response → User
```

---

## Logging

All activity is logged to:
- **Console** — real-time output
- **logs/app.log** — rotating file (5 MB per file, 3 backups)

Logged events:
- User queries and responses
- Retrieved RAG context
- Tool calls and results
- Agent decisions
- API request/response
- Errors and exceptions

---

## Technologies

| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Agent | LangChain tool-calling agent |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | FAISS (local) |
| Memory | LangChain `ConversationBufferWindowMemory` |
| Data | JSON files |
| Logging | Python `logging` with `RotatingFileHandler` |
