"""Utility functions for the chatbot application.

This module contains helper functions for validation, error handling,
and other common operations.
"""

import re
import os
from typing import Optional


def validate_api_key(api_key: Optional[str]) -> bool:
    """Validate if an API key is provided and has a reasonable format.
    
    Args:
        api_key: The API key to validate.
        
    Returns:
        bool: True if API key is valid, False otherwise.
    """
    if not api_key:
        return False
    
    # Check if API key has minimum length and valid characters
    if len(api_key.strip()) < 10:
        return False
    
    # OpenAI API keys typically start with 'sk-'
    if not api_key.startswith("sk-"):
        return False
    
    return True


def format_error_message(error: Exception) -> str:
    """Format an error message for user display.
    
    Args:
        error: The exception to format.
        
    Returns:
        str: A formatted error message.
    """
    error_str = str(error).lower()
    
    if "api_key" in error_str or "authentication" in error_str:
        return (
            "Authentication error: Your API key is invalid or expired. "
            "Please check your .env file or re-enter your API key."
        )
    elif "rate_limit" in error_str or "429" in error_str:
        return (
            "Rate limit exceeded: You've made too many requests. "
            "Please wait a moment and try again."
        )
    elif "timeout" in error_str or "connection" in error_str:
        return (
            "Connection error: Unable to reach OpenAI API. "
            "Please check your internet connection and try again."
        )
    elif "invalid_request" in error_str:
        return (
            "Invalid request: There was an issue with your request. "
            "Please try again with a different question."
        )
    else:
        return f"An unexpected error occurred: {str(error)}"


def sanitize_input(user_input: str) -> str:
    """Sanitize user input to remove excessive whitespace.
    
    Args:
        user_input: The user input to sanitize.
        
    Returns:
        str: The sanitized input.
    """
    # Remove leading/trailing whitespace
    sanitized = user_input.strip()
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized


def is_question(text: str) -> bool:
    """Check if the text appears to be a question.
    
    Args:
        text: The text to check.
        
    Returns:
        bool: True if text appears to be a question.
    """
    text = text.strip()
    return text.endswith("?") or text.startswith(("what", "how", "why", "when", "where", "who"))


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: The text to truncate.
        max_length: Maximum length of the text.
        
    Returns:
        str: The truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def log_message(role: str, message: str, level: str = "info") -> None:
    """Log a message to console (for debugging purposes).
    
    Args:
        role: The role (user or assistant).
        message: The message content.
        level: The log level (info, warning, error).
    """
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    truncated_msg = truncate_text(message, 50)
    print(f"[{timestamp}] [{level.upper()}] {role.upper()}: {truncated_msg}")
