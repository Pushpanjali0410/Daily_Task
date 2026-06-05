"""Chatbot module for managing AI conversations.

This module provides the main chatbot class that handles interactions
with the OpenAI API through LangChain.
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import SYSTEM_PROMPT
from utils import validate_api_key


class AIChatbot:
    """AI Chatbot powered by OpenAI through LangChain.
    
    Attributes:
        llm: The language model instance.
        prompt: The prompt template for the chatbot.
        chain: The LangChain chain combining prompt and LLM.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the chatbot.
        
        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env variable.
            model: Model name to use (default: gpt-4o-mini).
            
        Raises:
            ValueError: If API key is not provided or invalid.
        """
        # Set API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Validate API key
        api_key_to_validate = api_key or os.getenv("OPENAI_API_KEY")
        if not validate_api_key(api_key_to_validate):
            raise ValueError(
                "Invalid or missing OpenAI API key. "
                "Please provide a valid API key via .env or input field."
            )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{user_input}")
        ])
        
        # Create chain using pipe operator (latest LangChain syntax)
        self.chain = self.prompt | self.llm
    
    def chat(self, user_input: str) -> str:
        """Generate a response to user input.
        
        Args:
            user_input: The user's message.
            
        Returns:
            The chatbot's response.
            
        Raises:
            Exception: If API call fails.
        """
        try:
            # Invoke chain with user input
            response = self.chain.invoke({"user_input": user_input})
            # Extract text from response
            return response.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
