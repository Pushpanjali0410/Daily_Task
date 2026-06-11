import logging
import pickle
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from embeddings import EmbeddingService
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based Vector Store for managing document embeddings"""
    
    def __init__(self):
        """
        Initialize FAISS Vector Store
        """
        try:
            import faiss
            logger.info("Initializing FAISS Vector Store...")
            
            self.embedding_service = EmbeddingService()
            self.documents_metadata = []  # Store document metadata
            
            # Create FAISS index (IndexFlatL2 for L2 distance)
            self.index = faiss.IndexFlatL2(Config.EMBEDDING_DIMENSION)
            self.embeddings_list = []  # Store embeddings locally
            
            # Try to load existing index
            self._load_faiss_index()
            
            logger.info("FAISS Vector Store initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to FAISS vector store
        
        Args:
            documents: List of documents with 'content' and 'metadata'
        """
        try:
            # Extract texts and generate embeddings
            texts = [doc['content'] for doc in documents]
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Convert to float32 as required by FAISS
            embeddings = embeddings.astype(np.float32)
            
            # Add embeddings to FAISS index
            self.index.add(embeddings)
            self.embeddings_list.extend(embeddings)
            
            # Store metadata
            for idx, doc in enumerate(documents):
                self.documents_metadata.append({
                    'id': len(self.documents_metadata),
                    'content': doc['content'],
                    'metadata': doc['metadata']
                })
            
            # Save FAISS index and metadata
            self._save_faiss_index()
            
            logger.info(f"Successfully added {len(documents)} documents to FAISS")
        
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using FAISS
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            if len(self.documents_metadata) == 0:
                logger.warning("No documents in vector store. Please add documents first.")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents_metadata)))
            
            search_results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents_metadata):
                    doc = self.documents_metadata[idx]
                    search_results.append({
                        'id': doc['id'],
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': 1 / (1 + distance)  # Convert distance to similarity score
                    })
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def _save_faiss_index(self) -> None:
        """
        Save FAISS index to disk
        """
        try:
            import faiss
            
            index_path = Config.FAISS_INDEX_DIR / 'index.faiss'
            metadata_path = Config.FAISS_INDEX_DIR / 'metadata.pkl'
            
            faiss.write_index(self.index, str(index_path))
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.documents_metadata, f)
            
            logger.info(f"FAISS index saved to {index_path}")
        
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
    
    def _load_faiss_index(self) -> None:
        """
        Load FAISS index from disk if it exists
        """
        try:
            import faiss
            
            index_path = Config.FAISS_INDEX_DIR / 'index.faiss'
            metadata_path = Config.FAISS_INDEX_DIR / 'metadata.pkl'
            
            if index_path.exists() and metadata_path.exists():
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, 'rb') as f:
                    self.documents_metadata = pickle.load(f)
                
                logger.info(f"Loaded existing FAISS index with {len(self.documents_metadata)} documents")
        
        except Exception as e:
            logger.debug(f"No existing FAISS index found or error loading: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_documents': len(self.documents_metadata),
            'vector_dimension': Config.EMBEDDING_DIMENSION,
            'index_type': 'FAISS IndexFlatL2',
            'faiss_ntotal': self.index.ntotal
        }
