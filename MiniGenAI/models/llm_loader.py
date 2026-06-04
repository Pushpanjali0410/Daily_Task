"""LLM loader module for initializing language models."""

from typing import Any
from transformers import pipeline, Pipeline
from langchain_huggingface import HuggingFacePipeline
import logging

logger = logging.getLogger(__name__)


def load_llm(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_tokens: int = 256,
    temperature: float = 0.7,
    device: int = -1
) -> HuggingFacePipeline:
    """
    Load and initialize a Hugging Face language model.
    
    Args:
        model_name (str): Name of the model to load. Default is TinyLlama.
        max_tokens (int): Maximum number of tokens to generate. Default is 256.
        temperature (float): Temperature for generation (0.0-1.0). Default is 0.7.
        device (int): Device to use (-1 for CPU, 0+ for GPU). Default is -1.
    
    Returns:
        HuggingFacePipeline: Initialized language model pipeline.
    
    Raises:
        ValueError: If invalid parameters are provided.
        RuntimeError: If model loading fails.
    """
    # Validate parameters
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")
    
    if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 1.0):
        raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature}")
    
    try:
        logger.info(f"Loading LLM model: {model_name}")
        
        pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=max_tokens,
            temperature=temperature,
            device=device
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info(f"LLM model loaded successfully: {model_name}")
        
        return llm
    
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
