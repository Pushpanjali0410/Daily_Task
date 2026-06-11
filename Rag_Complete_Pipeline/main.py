import logging
from pathlib import Path
from data_ingestion import DataIngestion
from vector_store import VectorStore
from rag_chain import RAGChain
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function demonstrating RAG pipeline with FAISS and Groq
    """
    logger.info("Starting RAG Pipeline with FAISS and Groq...")
    
    # Initialize components
    data_ingestion = DataIngestion(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    vector_store = VectorStore()
    rag_chain = RAGChain(vector_store=vector_store)
    
    # Example 1: Ingest documents from directory
    logger.info("\n=== Phase 1: Data Ingestion ===")
    sample_docs_dir = Config.DOCUMENTS_DIR
    
    if list(sample_docs_dir.glob('*')):
        documents = data_ingestion.ingest_from_directory(str(sample_docs_dir))
        
        if documents:
            logger.info("\n=== Phase 2: Building FAISS Index ===")
            vector_store.add_documents(documents)
    else:
        logger.info(f"No documents found in {sample_docs_dir}")
        logger.info("Please add PDF or TXT files to the data/documents directory")
        
        # Create sample documents for demonstration
        logger.info("Creating sample documents for demonstration...")
        sample_documents = [
            {
                'content': """
                Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
                especially computer systems. These processes include learning, reasoning, and self-correction.
                AI has applications in various domains including healthcare, finance, education, and entertainment.
                """,
                'metadata': {
                    'source': 'sample_ai',
                    'file_type': 'text',
                    'chunk_id': 0
                }
            },
            {
                'content': """
                Machine Learning is a subset of AI that enables systems to learn and improve from experience
                without being explicitly programmed. Deep learning is a subset of machine learning that uses
                neural networks with multiple layers. Common ML algorithms include decision trees, random forests,
                support vector machines, and neural networks.
                """,
                'metadata': {
                    'source': 'sample_ml',
                    'file_type': 'text',
                    'chunk_id': 0
                }
            },
            {
                'content': """
                Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret,
                and generate human language in a meaningful and useful way. NLP techniques include tokenization,
                sentiment analysis, named entity recognition, and machine translation.
                """,
                'metadata': {
                    'source': 'sample_nlp',
                    'file_type': 'text',
                    'chunk_id': 0
                }
            }
        ]
        
        logger.info("\n=== Phase 2: Building FAISS Index ===")
        vector_store.add_documents(sample_documents)
    
    # Show index statistics
    stats = rag_chain.get_index_stats()
    logger.info(f"Index Statistics: {stats}")
    
    # Example 2: Query the RAG pipeline
    logger.info("\n=== Phase 3: Querying RAG Pipeline ===")
    
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What is NLP and its applications?"
    ]
    
    for idx, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Query {idx} ---")
        print(f"\nQuery: {query}")
        
        result = rag_chain.query(query, top_k=Config.TOP_K_RETRIEVAL)
        
        if 'error' not in result:
            print(f"\nAnswer: {result['answer']}")
            print(f"\nModel: {result['model']}")
            print(f"Finish Reason: {result['stop_reason']}")
        else:
            print(f"Error: {result['error']}")
    
    # Example 3: Multi-turn conversation
    logger.info("\n=== Phase 4: Multi-turn Conversation ===")
    print("\n--- Multi-turn Conversation Example ---")
    
    rag_chain.reset_conversation()
    
    conversation_queries = [
        "What are the main types of machine learning?",
        "Can you give me more details about deep learning?",
        "How does it relate to natural language processing?"
    ]
    
    for idx, query in enumerate(conversation_queries, 1):
        logger.info(f"\nMulti-turn Query {idx}: {query}")
        print(f"\nQuery {idx}: {query}")
        
        result = rag_chain.multi_turn_conversation(query)
        
        if 'error' not in result:
            print(f"Answer: {result['answer'][:500]}..." if len(result['answer']) > 500 else f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")
    
    logger.info("\n=== RAG Pipeline Execution Completed ===")

if __name__ == "__main__":
    main()
