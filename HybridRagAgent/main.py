"""
Hybrid RAG CLI
==============
Usage:
  python main.py ingest  <file_or_folder>     # Index documents
  python main.py chat                          # Interactive Q&A
  python main.py eval   [--queries FILE]       # Run DeepEval tests
  python main.py demo                          # Full demo (ingest + chat + eval)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_engine import HybridRetriever, OllamaEmbedder
from src.agent import OllamaLLM, RAGAgent
from src.ingestion import ingest_file, ingest_folder
from src.evaluator import RAGEvaluator, EvalCase


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
INDEX_DIR = os.getenv("INDEX_DIR", "./indexes")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
FUSION_MODE = os.getenv("FUSION_MODE", "rrf")   # "rrf" | "rsf"
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
TOP_K = int(os.getenv("TOP_K", "5"))


def build_components():
    embedder = OllamaEmbedder(base_url=OLLAMA_URL, model=EMBED_MODEL)
    retriever = HybridRetriever(
        embedder=embedder,
        index_dir=INDEX_DIR,
        fusion=FUSION_MODE,
        semantic_weight=SEMANTIC_WEIGHT,
        top_k=TOP_K,
    )
    llm = OllamaLLM(base_url=OLLAMA_URL, model=LLM_MODEL)
    agent = RAGAgent(retriever=retriever, llm=llm)
    return embedder, retriever, llm, agent


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ingest(path: str):
    print(f"\n📂 Ingesting: {path}")
    embedder, retriever, llm, agent = build_components()
    # Load existing index first
    retriever.load()

    p = Path(path)
    if p.is_dir():
        docs = ingest_folder(path)
    else:
        docs = ingest_file(path)

    print(f"  → {len(docs)} chunks extracted")
    retriever.add_documents(docs)
    retriever.save()
    print(f"  ✅ Index saved to {INDEX_DIR}/\n")


def cmd_chat():
    print("\n" + "="*60)
    print("  🤖 Hybrid RAG Agent — Interactive Chat")
    print(f"  Fusion: {FUSION_MODE.upper()} | Semantic weight: {SEMANTIC_WEIGHT}")
    print("  Type 'quit' to exit, 'reset' to clear history, 'sources' to see last sources")
    print("="*60 + "\n")

    embedder, retriever, llm, agent = build_components()
    retriever.load()

    if not retriever.documents:
        print("⚠️  No documents indexed yet. Run: python main.py ingest <file>\n")

    last_sources = []
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "reset":
            agent.reset_history()
            print("🔄 History cleared.\n")
            continue
        if query.lower() == "sources":
            if last_sources:
                print("\n📚 Last retrieved sources:")
                for s in last_sources:
                    print(f"  • {s['source']} | page: {s.get('page','?')} | score: {s['score']} | fusion: {s['fusion']}")
            else:
                print("  (No sources yet)")
            print()
            continue

        result = agent.ask(query)
        print(f"\n🤖 Agent: {result['answer']}\n")
        last_sources = result["sources"]

        if last_sources:
            print("📚 Sources:", ", ".join(
                f"{s['source']}(p.{s.get('page','?')})" for s in last_sources
            ))
        print()


def cmd_eval(queries_file: str = None):
    print("\n" + "="*60)
    print("  🧪 DeepEval — 10-Metric RAG Evaluation")
    print("="*60)

    embedder, retriever, llm, agent = build_components()
    retriever.load()
    evaluator = RAGEvaluator(llm)

    # Load or use default test cases
    if queries_file and Path(queries_file).exists():
        with open(queries_file) as f:
            test_data = json.load(f)
        queries = [td["query"] for td in test_data]
        print(f"  Loaded {len(queries)} test queries from {queries_file}")
    else:
        # Default eval queries
        queries = [
            "What is the main topic of the document?",
            "Summarize the key points mentioned in the document.",
            "What conclusions or recommendations are provided?",
        ]
        print("  Using default evaluation queries.")

    print(f"\n  Running {len(queries)} evaluation case(s)...\n")

    eval_cases = []
    for query in queries:
        result = agent.ask(query)
        ctx_texts = [c.doc.text for c in result["chunks"]]
        eval_cases.append(
            EvalCase(
                query=query,
                answer=result["answer"],
                context_texts=ctx_texts,
                chunks=result["chunks"],
            )
        )

    reports, overall_deployed = evaluator.evaluate_batch(eval_cases)
    evaluator.print_full_report(reports, overall_deployed)

    # Save report
    report_path = "./eval_report.json"
    report_data = []
    for r in reports:
        report_data.append({
            "case_id": r.case_id,
            "query": r.query,
            "passed": r.passed_count,
            "total": r.total_count,
            "deployed": r.deployed,
            "metrics": [
                {"name": m.name, "score": m.score, "passed": m.passed, "reason": m.reason}
                for m in r.metrics
            ],
        })
    with open(report_path, "w") as f:
        json.dump({"reports": report_data, "overall_deployed": overall_deployed}, f, indent=2)
    print(f"  📄 Full report saved to {report_path}\n")

    return overall_deployed


def cmd_demo():
    """Full end-to-end demo: create a sample doc, ingest, chat, eval."""
    print("\n" + "█"*60)
    print("  HYBRID RAG — FULL DEMO")
    print("█"*60)

    # Create sample document
    sample_path = "./uploads/sample_demo.txt"
    Path("./uploads").mkdir(exist_ok=True)
    Path(sample_path).write_text(textwrap.dedent("""
        Introduction to Hybrid RAG Systems

        Hybrid Retrieval-Augmented Generation (RAG) combines two complementary
        search strategies: lexical search (BM25) and semantic search (vector/FAISS).
        Lexical search excels at finding exact keyword matches. Semantic search
        finds conceptually similar content even without exact keyword overlap.

        Fusion Algorithms
        The Reciprocal Rank Fusion (RRF) algorithm combines ranked results by
        computing 1/(k + rank) for each document across both lists, where k=60
        is the smoothing constant. Documents appearing highly in both lists
        receive the greatest boost.

        Relative Score Fusion (RSF) normalizes raw BM25 and cosine similarity
        scores into a 0-1 range and computes a weighted average. A typical
        configuration uses 60% semantic weight and 40% lexical weight.

        Evaluation with DeepEval
        DeepEval provides LLM-as-judge metrics for RAG systems including:
        answer relevancy, faithfulness, contextual precision, contextual recall,
        hallucination detection, toxicity, bias, and completeness.
        A deployment gate of 7/10 metrics ensures production quality.

        FAISS Index
        Facebook AI Similarity Search (FAISS) enables efficient vector similarity
        search over millions of embeddings. The IndexFlatIP variant uses inner
        product (cosine similarity for normalized vectors) and is suitable for
        CPU deployments up to ~100k chunks.

        Ollama Integration
        Ollama provides local LLM inference. The system uses nomic-embed-text
        for generating embeddings and llama3.2 for answer generation. Both
        models run entirely on-device, ensuring data privacy.
    """).strip())

    print(f"\n📝 Sample document created: {sample_path}")

    # Ingest
    cmd_ingest(sample_path)

    # Ask a few questions and show answers
    embedder, retriever, llm, agent = build_components()
    retriever.load()

    demo_queries = [
        "What is Hybrid RAG and how does it work?",
        "Explain RRF fusion algorithm.",
        "How does DeepEval measure RAG quality?",
    ]

    print("\n" + "="*60)
    print("  💬 DEMO CHAT")
    print("="*60)
    for q in demo_queries:
        print(f"\nQ: {q}")
        result = agent.ask(q)
        print(f"A: {result['answer'][:300]}...")
        if result["sources"]:
            print(f"   Sources: {', '.join(s['source'] for s in result['sources'])}")

    # Eval
    print("\n" + "="*60)
    print("  🧪 RUNNING DEEPEVAL")
    print("="*60)
    evaluator = RAGEvaluator(llm)

    eval_cases = []
    for q in demo_queries:
        result = agent.ask(q)
        eval_cases.append(EvalCase(
            query=q,
            answer=result["answer"],
            context_texts=[c.doc.text for c in result["chunks"]],
            chunks=result["chunks"],
        ))

    reports, overall_deployed = evaluator.evaluate_batch(eval_cases)
    evaluator.print_full_report(reports, overall_deployed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG Agent with DeepEval Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest ./uploads/my_report.pdf
  python main.py ingest ./uploads/
  python main.py chat
  python main.py eval
  python main.py eval --queries test_queries.json
  python main.py demo
        """,
    )
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Index a file or folder")
    p_ingest.add_argument("path", help="File or directory to ingest")

    sub.add_parser("chat", help="Interactive chat with RAG agent")

    p_eval = sub.add_parser("eval", help="Run DeepEval tests")
    p_eval.add_argument("--queries", help="JSON file with test queries", default=None)

    sub.add_parser("demo", help="Full end-to-end demo")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args.path)
    elif args.command == "chat":
        cmd_chat()
    elif args.command == "eval":
        deployed = cmd_eval(getattr(args, "queries", None))
        sys.exit(0 if deployed else 1)
    elif args.command == "demo":
        cmd_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
