"""Chat history saving and management utilities."""

import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def save_chat(
    messages: List[Dict[str, str]],
    output_file: str | Path = "chats/history.json"
) -> None:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    
    # Validate message format
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"Each message must be a dict, got {type(msg).__name__}")
        if "role" not in msg or "content" not in msg:
            raise ValueError("Each message must have 'role' and 'content' keys")
    
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Chat history saved successfully to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
        raise IOError(f"Failed to save chat history to {output_file}: {str(e)}")


def load_chat(input_file: str | Path = "chats/history.json") -> List[Dict[str, str]]:
    input_path = Path(input_file)
    
    if not input_path.exists():
        logger.warning(f"Chat history file not found: {input_path}")
        return []
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        
        logger.info(f"Chat history loaded successfully from {input_path}")
        return messages
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chat history file: {e}")
        raise IOError(f"Failed to parse chat history from {input_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        raise IOError(f"Failed to load chat history from {input_file}: {str(e)}")
