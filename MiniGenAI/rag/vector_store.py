"""Vector store management using FAISS for document retrieval."""

from typing import List, Any
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


def create_vector_store(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """
    Create a FAISS vector store from documents and embeddings.
    
    Args:
        documents (List[Document]): List of documents to embed.
        embeddings (HuggingFaceEmbeddings): Embeddings model to use.
    
    Returns:
        FAISS: Vector store for document retrieval.
    
    Raises:
        ValueError: If documents list is empty or embeddings is invalid.
        RuntimeError: If vector store creation fails.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")
    
    if not isinstance(documents, list):
        raise ValueError("documents must be a list")
    
    if embeddings is None:
        raise ValueError("embeddings cannot be None")
    
    try:
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        db = FAISS.from_documents(documents, embeddings)
        
        logger.info(f"Vector store created successfully with {len(documents)} documents")
        return db
    
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise RuntimeError(f"Failed to create vector store: {str(e)}")
