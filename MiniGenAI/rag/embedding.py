"""Embedding generation module using Hugging Face models."""

from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> HuggingFaceEmbeddings:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    
    try:
        logger.info(f"Loading embeddings model: {model_name}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )
        
        logger.info(f"Embeddings model loaded successfully: {model_name}")
        return embeddings
    
    except Exception as e:
        logger.error(f"Error loading embeddings model: {e}")
        raise RuntimeError(f"Failed to load embeddings model {model_name}: {str(e)}")
