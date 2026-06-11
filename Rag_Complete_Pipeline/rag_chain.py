import logging
from typing import Dict, Any, Optional
from groq import Groq
from retrieval import Retrieval
from vector_store import VectorStore
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChain:
    """Main RAG Chain combining FAISS retrieval and Groq generation"""
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize RAG Chain
        
        Args:
            vector_store: Vector store instance (FAISS)
        """
        logger.info("Initializing RAG Chain...")
        
        # Validate Groq API key
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")
        
        # Initialize Groq client
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.GROQ_MODEL
        
        # Initialize retrieval
        self.vector_store = vector_store or VectorStore()
        self.retrieval = Retrieval(vector_store=self.vector_store)
        
        # Conversation history for multi-turn conversations
        self.conversation_history = []
        
        logger.info(f"RAG Chain initialized with model: {self.model}")
    
    def _prepare_system_prompt(self) -> str:
        """
        Prepare system prompt for the LLM
        
        Returns:
            System prompt string
        """
        return """
        You are a helpful AI assistant powered by Groq. You will be provided with context from documents.
        Use the provided context to answer questions accurately and comprehensively.
        If the context doesn't contain information to answer the question, say so clearly.
        Always cite your sources from the provided documents when possible.
        Be concise but informative in your responses.
        """.strip()
    
    def _prepare_context_prompt(self, query: str, context: str) -> str:
        """
        Prepare the full prompt with context
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Full prompt string
        """
        prompt = f"""
        Context Information:
        {context}
        
        Question: {query}
        
        Please answer the question based on the context provided above. If the context is insufficient, indicate that.
        """.strip()
        
        return prompt
    
    def query(self, query: str, top_k: int = None, include_history: bool = False) -> Dict[str, Any]:
        """
        Execute a RAG query using FAISS retrieval and Groq generation
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            include_history: Whether to include conversation history
            
        Returns:
            Dictionary with query, context, and response
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Retrieve relevant documents from FAISS
            context = self.retrieval.get_context(query, top_k=top_k)
            
            # Prepare messages
            messages = []
            
            # Add conversation history if enabled
            if include_history and self.conversation_history:
                messages.extend(self.conversation_history)
            
            # Prepare the user message with context
            user_message = self._prepare_context_prompt(query, context)
            messages.append({"role": "user", "content": user_message})
            
            # Call Groq API
            logger.info(f"Calling Groq API ({self.model}) for generation...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            logger.info("Query processed successfully")
            
            return {
                'query': query,
                'context': context,
                'answer': answer,
                'model': self.model,
                'stop_reason': response.choices[0].finish_reason
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': query,
                'error': str(e)
            }
    
    def multi_turn_conversation(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process query with multi-turn conversation enabled
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Response with conversation context
        """
        return self.query(query, top_k=top_k, include_history=True)
    
    def reset_conversation(self) -> None:
        """
        Reset conversation history
        """
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_conversation_history(self) -> list:
        """
        Get current conversation history
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index
        
        Returns:
            Dictionary with index statistics
        """
        return self.vector_store.get_index_stats()
