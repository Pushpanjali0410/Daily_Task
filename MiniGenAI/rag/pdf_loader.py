"""PDF loading and document processing module."""

from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str) -> List[Document]:
    # Validate path
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == ".pdf":
        raise ValueError(f"File must be a PDF, got: {pdf_file.suffix}")
    
    try:
        logger.info(f"Loading PDF: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        if not docs:
            logger.warning(f"No documents found in PDF: {pdf_path}")
            return []
        
        logger.info(f"Successfully loaded {len(docs)} pages from PDF: {pdf_path}")
        return docs
    
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {e}")
        raise RuntimeError(f"Failed to load PDF {pdf_path}: {str(e)}")
