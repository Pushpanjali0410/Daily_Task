import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using Sentence Transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize Embedding Service
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding
            
        Returns:
            Matrix of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embedding vectors
        
        Returns:
            Dimension of embeddings
        """
        sample_embedding = self.embed_text("sample")
        return len(sample_embedding)
