import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG Pipeline (FAISS-based)"""
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
    
    # Vector Database Configuration (FAISS only)
    VECTOR_DB_TYPE = 'faiss'  # FAISS is free and local
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    
    # File Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    DOCUMENTS_DIR = DATA_DIR / 'documents'
    FAISS_INDEX_DIR = DATA_DIR / 'faiss_index'
    
    # RAG Configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 5
    TEMPERATURE = 0.7
    MAX_TOKENS = 2048
    
    # Create necessary directories
    @staticmethod
    def create_directories():
        Config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        Config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories
Config.create_directories()
