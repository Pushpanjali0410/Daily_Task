"""
knowledge/rag_pipeline.py
RAG pipeline using FAISS + HuggingFace embeddings + LangChain
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs"
FAISS_INDEX_PATH = Path(__file__).parent.parent / "faiss_index"

_vectorstore = None


def get_embeddings():
    logger.info("Loading embedding model: all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: Path = DOCS_DIR):
    logger.info(f"Loading documents from: {docs_dir}")
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(docs_dir: Path = DOCS_DIR, save: bool = True):
    global _vectorstore
    docs = load_documents(docs_dir)
    chunks = split_documents(docs)
    embeddings = get_embeddings()

    logger.info("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if save:
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index saved to: {FAISS_INDEX_PATH}")

    _vectorstore = vectorstore
    return vectorstore


def load_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()

    if FAISS_INDEX_PATH.exists():
        logger.info("Loading existing FAISS index...")
        _vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info("No existing index found. Building from documents...")
        _vectorstore = build_vectorstore()

    return _vectorstore


def retrieve_context(query: str, k: int = 3) -> str:
    vs = load_vectorstore()
    results = vs.similarity_search(query, k=k)
    context_parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        filename = Path(source).name
        context_parts.append(f"[Source: {filename}]\n{doc.page_content.strip()}")
    context = "\n\n---\n\n".join(context_parts)
    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}'")
    return context


def add_document_to_store(file_path: str):
    global _vectorstore
    logger.info(f"Adding new document: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    chunks = split_documents(docs)

    vs = load_vectorstore()
    vs.add_documents(chunks)

    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(FAISS_INDEX_PATH))
    _vectorstore = vs

    logger.info(f"Document added. Chunks contributed: {len(chunks)}")
    return len(chunks)