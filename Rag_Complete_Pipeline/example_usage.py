#!/usr/bin/env python
"""
Example usage of the RAG Pipeline with FAISS and Groq
"""

import logging
from data_ingestion import DataIngestion
from vector_store import VectorStore
from rag_chain import RAGChain
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_query():
    """
    Example 1: Basic query with document ingestion
    """
    print("\n" + "="*60)
    print("Example 1: Basic Query with FAISS and Groq")
    print("="*60)
    
    # Initialize
    data_ingestion = DataIngestion()
    vector_store = VectorStore()
    rag_chain = RAGChain(vector_store=vector_store)
    
    # Create sample documents
    sample_documents = [
        {
            'content': """
            Python is a high-level, interpreted programming language known for its simplicity and readability.
            It was created by Guido van Rossum and first released in 1991.
            Python emphasizes code readability and allows developers to express concepts in fewer lines of code.
            It is widely used in web development, data analysis, artificial intelligence, and automation.
            """,
            'metadata': {'source': 'python_basics', 'topic': 'Python'}
        },
        {
            'content': """
            Machine Learning is a field of artificial intelligence that enables computers to learn from data
            without being explicitly programmed. Common algorithms include supervised learning, unsupervised learning,
            and reinforcement learning. Applications include image recognition, recommendation systems, and natural language processing.
            """,
            'metadata': {'source': 'ml_overview', 'topic': 'Machine Learning'}
        }
    ]
    
    # Chunk and add documents
    chunked_docs = data_ingestion.chunk_documents(sample_documents)
    vector_store.add_documents(chunked_docs)
    
    # Query
    result = rag_chain.query("What is Intellectual Property?")
    if 'error' not in result:
        print(f"\nQuery: {result['query']}")
        print(f"\nAnswer: {result['answer']}")
    else:
        print(f"Error: {result['error']}")

def example_2_multi_turn_conversation():
    """
    Example 2: Multi-turn conversation with context awareness
    """
    print("\n" + "="*60)
    print("Example 2: Multi-turn Conversation")
    print("="*60)
    
    vector_store = VectorStore()
    rag_chain = RAGChain(vector_store=vector_store)
    
    # Add sample data
    sample_docs = [
        {
            'content': """
            The Transformer architecture introduced in 2017 revolutionized Natural Language Processing.
            It uses self-attention mechanisms instead of recurrence, enabling parallel processing and better
            handling of long-range dependencies. Transformers form the foundation of modern language models
            like BERT, GPT, and T5. They have been adapted for computer vision tasks as well.
            """,
            'metadata': {'source': 'transformers_intro'}
        }
    ]
    
    data_ingestion = DataIngestion()
    chunked_docs = data_ingestion.chunk_documents(sample_docs)
    vector_store.add_documents(chunked_docs)
    
    # Reset conversation
    rag_chain.reset_conversation()
    
    # Multi-turn conversation
    queries = [
        "What is a Intellectual Property?",
        "When was it introduced?",
        "What advantages does it have?"
    ]
    
    for idx, query in enumerate(queries, 1):
        result = rag_chain.multi_turn_conversation(query)
        if 'error' not in result:
            print(f"\nQuery {idx}: {query}")
            print(f"Answer: {result['answer'][:300]}..." if len(result['answer']) > 300 else f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")

def example_3_custom_configuration():
    """
    Example 3: Using custom configuration
    """
    print("\n" + "="*60)
    print("Example 3: Custom Configuration")
    print("="*60)
    
    # Show configuration
    print(f"\nCurrent Configuration:")
    print(f"  - Vector DB Type: {Config.VECTOR_DB_TYPE}")
    print(f"  - Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"  - Chunk Size: {Config.CHUNK_SIZE}")
    print(f"  - Top-K Retrieval: {Config.TOP_K_RETRIEVAL}")
    print(f"  - Temperature: {Config.TEMPERATURE}")
    print(f"  - Max Tokens: {Config.MAX_TOKENS}")
    print(f"  - Groq Model: {Config.GROQ_MODEL}")

def example_4_retrieve_and_format():
    """
    Example 4: Document retrieval and formatting
    """
    print("\n" + "="*60)
    print("Example 4: Document Retrieval and Formatting")
    print("="*60)
    
    from retrieval import Retrieval
    
    # Setup
    vector_store = VectorStore()
    retrieval = Retrieval(vector_store=vector_store)
    
    # Add sample documents
    sample_docs = [
        {
            'content': 'Deep Learning uses artificial neural networks with multiple layers to process data.',
            'metadata': {'source': 'dl_doc', 'page': 1}
        },
        {
            'content': 'Neural Networks are inspired by biological neurons in the brain and can learn complex patterns.',
            'metadata': {'source': 'nn_doc', 'page': 1}
        },
        {
            'content': 'Convolutional Neural Networks (CNNs) are specially designed for processing grid-like data such as images.',
            'metadata': {'source': 'cnn_doc', 'page': 1}
        }
    ]
    
    data_ingestion = DataIngestion()
    chunked_docs = data_ingestion.chunk_documents(sample_docs)
    vector_store.add_documents(chunked_docs)
    
    # Retrieve
    query = "Tell me about neural networks and deep learning"
    documents = retrieval.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(documents)} documents:")
    
    for idx, doc in enumerate(documents, 1):
        print(f"\n  Document {idx}:")
        print(f"  - Content: {doc['content'][:80]}...")
        print(f"  - Relevance Score: {doc['score']:.4f}")
        print(f"  - Source: {doc['metadata'].get('source', 'Unknown')}")

def example_5_faiss_statistics():
    """
    Example 5: FAISS Index Statistics
    """
    print("\n" + "="*60)
    print("Example 5: FAISS Index Statistics")
    print("="*60)
    
    vector_store = VectorStore()
    
    # Add sample documents
    sample_docs = [
        {'content': f'Sample document {i}', 'metadata': {'source': f'doc_{i}'}}
        for i in range(5)
    ]
    
    data_ingestion = DataIngestion()
    chunked_docs = data_ingestion.chunk_documents(sample_docs)
    vector_store.add_documents(chunked_docs)
    
    # Get statistics
    stats = vector_store.get_index_stats()
    
    print(f"\nFAISS Index Statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# RAG Pipeline with FAISS and Groq - Examples")
    print("#"*60)
    
    try:
        example_1_basic_query()
        example_2_multi_turn_conversation()
        example_3_custom_configuration()
        example_4_retrieve_and_format()
        example_5_faiss_statistics()
        
        print("\n" + "#"*60)
        print("# All examples completed successfully!")
        print("#"*60 + "\n")
    
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
