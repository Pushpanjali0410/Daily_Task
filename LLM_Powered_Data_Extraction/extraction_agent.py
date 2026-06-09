from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic_models import CustomerSupportData


class StructuredExtractionAgent:
    """
    LangChain-based agent for extracting structured information from unstructured text.
    Uses free Groq API for LLM operations.
    """
    
    def __init__(self):
        """Initialize the extraction agent with Groq LLM"""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq LLM (free tier available)
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        # Initialize output parser for structured JSON
        self.parser = JsonOutputParser(pydantic_object=CustomerSupportData)
    
    def extract_information(self, unstructured_text: str) -> CustomerSupportData:
        """
        Extract structured information from unstructured text.
        
        Args:
            unstructured_text (str): Long unstructured text containing customer information
            
        Returns:
            CustomerSupportData: Validated structured data
        """
        
        # Create extraction prompt
        extraction_prompt = PromptTemplate(
            template="""You are an expert data extraction agent. 
            
Extract the following information from the provided text:
- name: Full name of the person
- email: Email address (must be valid format)
- order_id: Order identification number
- issue_type: Type of issue or problem reported

Text to process:
{input_text}

Respond with ONLY a valid JSON object with keys: name, email, order_id, issue_type.
Do not include any explanation or additional text.
Ensure email is in proper format.
If any field is not found, use reasonable inference or mark as "Not Found".""",
            input_variables=["input_text"]
        )
        
        # Create the extraction chain
        chain = extraction_prompt | self.llm | self.parser
        
        # Execute extraction
        result = chain.invoke({"input_text": unstructured_text})
        
        # Validate using Pydantic model
        validated_data = CustomerSupportData(**result)
        
        return validated_data
    
    def process_multiple_texts(self, texts: list) -> list:
        """
        Process multiple unstructured texts and extract information from each.
        
        Args:
            texts (list): List of unstructured text strings
            
        Returns:
            list: List of CustomerSupportData objects
        """
        results = []
        for text in texts:
            try:
                extracted = self.extract_information(text)
                results.append(extracted)
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                continue
        
        return results