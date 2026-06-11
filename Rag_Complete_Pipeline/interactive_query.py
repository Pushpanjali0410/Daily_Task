from data_ingestion import DataIngestion
from vector_store import VectorStore
from rag_chain import RAGChain
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*70)
    print("🔍 RAG Pipeline - Universal Query System")
    print("="*70)
    print("\nSupported Domains:")
    print("  • Intellectual Property (IP)")
    print("  • Machine Learning (ML)")
    print("  • Solar Technology")
    print("  • Medical Electronics")
    print("  • Any other domain you upload")
    
    # Initialize
    logger.info("Loading documents from data/documents/...")
    data_ingestion = DataIngestion()
    vector_store = VectorStore()
    rag_chain = RAGChain(vector_store=vector_store)
    
    # Load documents
    documents = data_ingestion.ingest_from_directory(str(Config.DOCUMENTS_DIR))
    
    if documents:
        logger.info(f"✓ Loaded {len(documents)} document chunks")
        vector_store.add_documents(documents)
        stats = rag_chain.get_index_stats()
        print(f"\n✓ RAG Index Ready!")
        print(f"  - Total document chunks: {stats['total_documents']}")
        print(f"  - Vector dimension: {stats['vector_dimension']}")
    else:
        print("\n⚠ No documents found in data/documents/")
        print("Please add your documents (PDF, TXT) and run again.")
        return
    
    # Interactive Query Loop
    print("\n" + "="*70)
    print("💬 Enter Your Questions (type 'exit' or 'quit' to end)")
    print("="*70)
    print("\nExamples of questions you can ask:")
    print("  • 'What is intellectual property?'")
    print("  • 'Explain machine learning concepts'")
    print("  • 'How do solar panels work?'")
    print("  • 'What are medical electronics?'")
    print("  • Or ANY question about your uploaded documents!")
    
    rag_chain.reset_conversation()
    
    while True:
        print("\n" + "-"*70)
        user_input = input("\n📝 Your Question: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
            print("\n👋 Goodbye! Thank you for using RAG Pipeline.\n")
            break
        
        if not user_input:
            print("⚠ Please enter a question.")
            continue
        
        print("\n🤔 Processing your question...\n")
        
        result = rag_chain.multi_turn_conversation(user_input)
        
        if 'error' not in result:
            print("="*70)
            print(f"🤖 Answer:")
            print("="*70)
            print(result['answer'])
            print("\n" + "="*70)
            print(f"📊 Model: {result['model']}")
            print("="*70)
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()