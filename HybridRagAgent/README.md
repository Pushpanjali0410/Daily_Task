# 🔀 Hybrid RAG Agent with DeepEval Testing

A production-grade Hybrid Retrieval-Augmented Generation (RAG) system combining **BM25 lexical search** and **FAISS semantic search** with **RRF/RSF fusion**, powered by **Ollama** and evaluated with **10 DeepEval metrics** and a **7/10 deployment gate**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       USER QUERY                            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────┐             ┌─────────────────────┐
│  BM25 Lexical  │             │  FAISS Semantic      │
│  Search        │             │  (Ollama embeddings) │
│  (exact match) │             │  (meaning match)     │
└────────┬───────┘             └──────────┬──────────┘
         │ ranked list                    │ ranked list
         └────────────┬──────────────────┘
                      │
              ┌───────▼────────┐
              │  FUSION LAYER  │
              │  RRF or RSF    │
              └───────┬────────┘
                      │ top-K chunks
                      ▼
              ┌───────────────┐
              │  Ollama LLM   │
              │  (llama3.2)   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  FINAL ANSWER │
              │  + SOURCES    │
              └───────────────┘
```

## 📂 Project Structure

```
hybrid-rag/
├── main.py                  # CLI entrypoint
├── requirements.txt
├── src/
│   ├── rag_engine.py        # Core: FAISS + BM25 + RRF/RSF fusion
│   ├── agent.py             # RAG agent with Ollama LLM
│   ├── ingestion.py         # PDF / DOCX / TXT document loader
│   └── evaluator.py         # DeepEval 10-metric suite + deployment gate
├── uploads/                 # Drop your documents here
├── indexes/                 # FAISS + BM25 persisted indexes
└── eval_report.json         # Generated after running eval
```

---

## ⚡ Quickstart

### 1. Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2           # LLM
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest documents

```bash
# Single file
python main.py ingest uploads/my_report.pdf
python main.py ingest uploads/research.docx
python main.py ingest uploads/notes.txt

# Entire folder
python main.py ingest uploads/
```

### 4. Chat with your documents

```bash
python main.py chat
```

Commands inside chat:
- Type any question → get an answer
- `sources` → see retrieved document sources
- `reset` → clear conversation history
- `quit` → exit

### 5. Run DeepEval tests + deployment check

```bash
python main.py eval

# With custom test queries
python main.py eval --queries test_queries.json
```

### 6. Full end-to-end demo

```bash
python main.py demo
```

---

## 🔀 Fusion Algorithms

### Reciprocal Rank Fusion (RRF) — Default

```
score(d) = Σ  weight_i  ×  1 / (k + rank_i(d))
```

- `k=60` is the smoothing constant (prevents rank-1 from dominating)
- Documents appearing highly in **both** lists get a multiplicative boost
- Robust to score scale differences between BM25 and cosine similarity
- **Best for:** general-purpose, unknown query types

### Relative Score Fusion (RSF)

```
score(d) = α × sem_norm(d)  +  (1−α) × lex_norm(d)
```

- Normalizes BM25 and cosine scores independently to [0, 1]
- Default: **α = 0.6** (60% semantic, 40% lexical)
- **Best for:** when you know the query distribution

Switch modes:
```bash
export FUSION_MODE=rsf
export SEMANTIC_WEIGHT=0.7
python main.py chat
```

---

## 🧪 DeepEval — 10 Metrics

| # | Metric | Threshold | What it measures |
|---|--------|-----------|-----------------|
| 1 | **Answer Relevancy** | ≥ 0.70 | Does the answer directly address the question? |
| 2 | **Faithfulness** | ≥ 0.70 | Is the answer grounded in retrieved context? |
| 3 | **Contextual Precision** | ≥ 0.60 | Are retrieved chunks relevant to the query? |
| 4 | **Contextual Recall** | ≥ 0.60 | Does context contain enough info to answer? |
| 5 | **Contextual Relevancy** | ≥ 0.50 | Fraction of chunks that are semantically on-topic |
| 6 | **Hallucination** | ≤ 0.30 | Claims not supported by context (inverse) |
| 7 | **Answer Completeness** | ≥ 0.60 | Does the answer cover all aspects of the question? |
| 8 | **Toxicity** | ≤ 0.10 | Harmful or offensive content (inverse) |
| 9 | **Bias** | ≤ 0.20 | Unfair bias in the answer (inverse) |
| 10 | **Answer Consistency** | ≥ 0.70 | Consistent with prior conversation turns |

### 🚀 Deployment Gate

```
If passed_metrics >= 7 out of 10  →  ✅ DEPLOYED
If passed_metrics <  7 out of 10  →  🔧 NOT DEPLOYED — fix listed faults
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `LLM_MODEL` | `llama3.2` | LLM for answering |
| `INDEX_DIR` | `./indexes` | Where to persist FAISS + BM25 |
| `UPLOAD_DIR` | `./uploads` | Document upload folder |
| `FUSION_MODE` | `rrf` | Fusion: `rrf` or `rsf` |
| `SEMANTIC_WEIGHT` | `0.6` | Semantic weight for RSF (0–1) |
| `TOP_K` | `5` | Number of chunks to retrieve |

---

## 📋 Custom Test Queries Format

```json
[
  {"query": "What is the revenue for Q3?"},
  {"query": "Who are the key stakeholders mentioned?"},
  {"query": "What are the main risks identified?"}
]
```

Save as `test_queries.json` and run:
```bash
python main.py eval --queries test_queries.json
```

---

## 🔧 Troubleshooting

**Ollama not running:**
```
[LLM Error: Connection refused]
```
→ Start Ollama: `ollama serve`

**Model not found:**
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

**No documents indexed:**
```
⚠️ No documents indexed yet.
```
→ Run `python main.py ingest <your_file>`

**Deployment gate fails:**
The evaluator will list each failing metric with its score and threshold:
```
FAULTS TO FIX:
  ✗ Answer Relevancy: score=0.45 (threshold=0.7) — ...
  ✗ Faithfulness: score=0.52 (threshold=0.7) — ...
```
Common fixes:
- Increase `TOP_K` for better recall
- Switch fusion mode (`rrf` ↔ `rsf`)
- Adjust `SEMANTIC_WEIGHT` (higher for conceptual queries, lower for keyword-heavy)
- Improve chunk size in `src/rag_engine.py` → `split_text(chunk_size=...)`
