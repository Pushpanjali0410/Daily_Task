import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles data ingestion from multiple sources"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize DataIngestion
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def ingest_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            documents = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        documents.append({
                            'content': text,
                            'metadata': {
                                'source': file_path,
                                'page': page_num,
                                'file_type': 'pdf'
                            }
                        })
            logger.info(f"Successfully extracted {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return []
    
    def ingest_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load text from text file
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            documents = [{
                'content': text,
                'metadata': {
                    'source': file_path,
                    'file_type': 'text'
                }
            }]
            logger.info(f"Successfully loaded text file: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return []
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents with content and metadata
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            for chunk_idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    'content': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': chunk_idx
                    }
                })
        
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def ingest_from_directory(self, directory_path: str, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Ingest all documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            file_types: List of file extensions to process (e.g., ['.pdf', '.txt'])
            
        Returns:
            List of processed document chunks
        """
        if file_types is None:
            file_types = ['.pdf', '.txt']
        
        all_documents = []
        directory = Path(directory_path)
        
        for file_path in directory.iterdir():
            if file_path.suffix.lower() in file_types:
                logger.info(f"Processing file: {file_path.name}")
                
                if file_path.suffix.lower() == '.pdf':
                    docs = self.ingest_pdf(str(file_path))
                elif file_path.suffix.lower() == '.txt':
                    docs = self.ingest_text_file(str(file_path))
                else:
                    continue
                
                all_documents.extend(docs)
        
        # Chunk all documents
        chunked_documents = self.chunk_documents(all_documents)
        logger.info(f"Total documents ingested and chunked: {len(chunked_documents)}")
        
        return chunked_documents
    
    def ingest_from_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrape and ingest content from URL
        
        Args:
            url: URL to scrape
            
        Returns:
            List of document chunks with metadata
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            
            documents = [{
                'content': text,
                'metadata': {
                    'source': url,
                    'file_type': 'web'
                }
            }]
            
            logger.info(f"Successfully scraped content from {url}")
            return self.chunk_documents(documents)
        
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return []
