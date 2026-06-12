"""LLM Configuration and initialization."""

import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE, MAX_TOKENS
)


def get_llm():
    """Get the configured LLM."""
    if LLM_PROVIDER == "groq":
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    elif LLM_PROVIDER == "ollama":
        return OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE
        )
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def validate_llm_connection():
    """Validate LLM connection."""
    try:
        llm = get_llm()
        response = llm.invoke("Say 'Connection successful' in one word")
        
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        print(f"✓ Using {LLM_PROVIDER.upper()} LLM")
        print(f"  Model: {GROQ_MODEL if LLM_PROVIDER == 'groq' else OLLAMA_MODEL}")
        print(f"\n✓ LLM connection validated successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to validate LLM connection")
        print(f"Error: Error code: {e}")
        return False
