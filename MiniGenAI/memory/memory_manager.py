"""Memory management for conversation context."""

from typing import Any
from langchain.memory import ConversationBufferWindowMemory
import logging

logger = logging.getLogger(__name__)


def get_memory(window_size: int = 5) -> ConversationBufferWindowMemory:
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got {window_size}")
    
    try:
        memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=False
        )
        logger.info(f"Memory initialized with window size: {window_size}")
        return memory
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        raise
