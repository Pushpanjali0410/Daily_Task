"""Configuration settings for MiniGenAI application."""

import os
from pathlib import Path

# Application Paths
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
CHATS_DIR = BASE_DIR / "chats"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
CHATS_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.7
LLM_DEVICE = -1  # -1 for CPU, 0+ for GPU

# Embedding Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Memory Configuration
MEMORY_WINDOW_SIZE = 5  # Number of previous messages to remember

# Chat History Configuration
CHAT_HISTORY_FILE = CHATS_DIR / "history.json"
CHAT_EXPORT_FILE = CHATS_DIR / "chat_history.pdf"

# PDF Configuration
PDF_ALLOWED_TYPES = ["pdf"]

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
