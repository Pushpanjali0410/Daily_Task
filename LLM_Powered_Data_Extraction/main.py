"""
Day 6 - Task 2: LLM-Powered Structured Data Extraction Agent
Main execution script with test cases
"""
import os
from dotenv import load_dotenv

# This explicitly loads the .env file variables into os.environ
load_dotenv()

import json
from extraction_agent import StructuredExtractionAgent
from pydantic_models import CustomerSupportData

# Test input
TEST_INPUT = """
Hi team,

My name is Rahul Sharma. I placed an order yesterday but the payment failed.
Order ID is ORD-45678.
My email is rahul.sharma@gmail.com.

Please help resolve this issue as soon as possible.

Thanks
Rahul
"""

def print_extraction_result(result: CustomerSupportData, test_name: str):
    """Pretty print extraction results"""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Name:       {result.name}")
    print(f"Email:      {result.email}")
    print(f"Order ID:   {result.order_id}")
    print(f"Issue Type: {result.issue_type}")
    print(f"{'='*60}\n")

def save_result_to_json(result: CustomerSupportData, filename: str):
    """Save extraction result to JSON file"""
    with open(filename, 'w') as f:
        json.dump(json.loads(result.model_dump_json()), f, indent=2)
    print(f"Result saved to {filename}")

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("LLM-Powered Structured Data Extraction Agent")
    print("="*60)
    
    try:
        # Initialize the extraction agent
        print("\n[1] Initializing extraction agent...")
        agent = StructuredExtractionAgent()
        print("✓ Agent initialized successfully")
        
        # Process test input
        print("\n[2] Processing test input...")
        print(f"Input Text:\n{TEST_INPUT}")
        
        result = agent.extract_information(TEST_INPUT)
        print("✓ Extraction completed successfully")
        
        # Display results
        print_extraction_result(result, "Test Case 1 - Customer Support Message")
        
        # Validate extracted data
        print("[3] Data Validation:")
        print(f"✓ Name validation: {len(result.name) > 0}")
        print(f"✓ Email validation: {result.email}")
        print(f"✓ Order ID validation: {len(result.order_id) > 0}")
        print(f"✓ Issue Type validation: {len(result.issue_type) > 0}")
        
        # Save to JSON
        print("\n[4] Saving results...")
        save_result_to_json(result, "extraction_output.json")
        
        # Display JSON output
        print("\n[5] JSON Output:")
        print(result.model_dump_json(indent=2))
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Please ensure GROQ_API_KEY is set in .env file")

if __name__ == "__main__":
    main()
