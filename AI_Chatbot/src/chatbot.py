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
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        # Set API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Validate API key
        api_key_to_validate = api_key or os.getenv("OPENAI_API_KEY")
        if not validate_api_key(api_key_to_validate):
            raise ValueError(
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
        try:
            # Invoke chain with user input
            response = self.chain.invoke({"user_input": user_input})
            # Extract text from response
            return response.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")


def create_chatbot(api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> AIChatbot:
    return AIChatbot(api_key=api_key, model=model)
