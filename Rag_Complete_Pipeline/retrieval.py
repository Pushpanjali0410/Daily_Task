import logging
from typing import List, Dict, Any
from vector_store import VectorStore
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retrieval:
    """Document retrieval component for RAG pipeline"""
    
    def __init__(self, vector_store: VectorStore = None, top_k: int = None):
        """
        Initialize Retrieval
        
        Args:
            vector_store: Vector store instance
            top_k: Number of top results to retrieve
        """
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k or Config.TOP_K_RETRIEVAL
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant documents
        """
        top_k = top_k or self.top_k
        logger.info(f"Retrieving top {top_k} documents for query: {query}")
        
        results = self.vector_store.search(query, top_k=top_k)
        
        logger.info(f"Retrieved {len(results)} relevant documents")
        return results
    
    def format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for LLM context
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted string of documents
        """
        if not documents:
            return "No relevant documents found."
        
        formatted = ""
        for idx, doc in enumerate(documents, 1):
            formatted += f"Document {idx}:\n"
            formatted += f"Content: {doc['content']}\n"
            formatted += f"Source: {doc['metadata'].get('source', 'Unknown')}\n"
            formatted += f"Relevance Score: {doc['score']:.4f}\n\n"
        
        return formatted
    
    def get_context(self, query: str, top_k: int = None) -> str:
        """
        Get formatted context for RAG query
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        documents = self.retrieve(query, top_k=top_k)
        return self.format_documents(documents)
