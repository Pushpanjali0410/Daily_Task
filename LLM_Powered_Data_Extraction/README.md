# Day 6 - Task 2: LLM-Powered Structured Data Extraction Agents

## Objective
Build an LLM-powered agent using LangChain that processes long unstructured text and extracts structured key-value information using Pydantic models.

## Project Structure

```
Day6_Task2/
├── pydantic_models.py      # Pydantic schema definitions
├── extraction_agent.py     # LangChain extraction agent logic
├── main.py                 # Main execution script with test cases
├── requirements.txt        # Project dependencies
├── .env.example           # Environment variables template
├── extraction_output.json  # Sample output file
└── README.md              # This file
```

## Features

✅ **LangChain-based Agent**: Uses LangChain for orchestration and chain management
✅ **Pydantic Validation**: Strict data validation with Pydantic models
✅ **Structured Extraction**: Converts unstructured text to JSON key-value format
✅ **Free API Integration**: Uses Groq's free-tier API (100 requests/day)
✅ **Error Handling**: Robust error handling and validation
✅ **JSON Output**: Exports results in structured JSON format

## Technologies Used

- **Python**: Core programming language
- **LangChain**: LLM orchestration and chain management
- **Pydantic**: Data validation and serialization
- **Groq API**: Free LLM inference (Mixtral-8x7b model)
- **python-dotenv**: Environment variable management

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Free Groq API Key

1. Visit: https://console.groq.com/keys
2. Sign up for free account (no credit card required)
3. Copy your API key

### 3. Configure Environment

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

### 4. Run the Agent

```bash
python main.py
```

## Implementation Details

### Pydantic Model (pydantic_models.py)

```python
class CustomerSupportData(BaseModel):
    name: str              # Customer's full name
    email: EmailStr        # Validated email address
    order_id: str          # Order identification
    issue_type: str        # Type of issue
```

**Features:**
- Email validation using `EmailStr`
- Type hints for clarity
- JSON schema generation
- Built-in serialization

### Extraction Agent (extraction_agent.py)

**Main Components:**

1. **Groq LLM Integration**: Uses free Mixtral-8x7b model
2. **PromptTemplate**: Structured extraction instructions
3. **JsonOutputParser**: Converts LLM output to JSON
4. **Chain Pipeline**: Orchestrates the extraction workflow

**Process Flow:**
```
Unstructured Text 
    ↓
PromptTemplate (instruction formatting)
    ↓
Groq LLM (text analysis)
    ↓
JsonOutputParser (JSON extraction)
    ↓
Pydantic Validation (data verification)
    ↓
StructuredData (output)
```

### Main Script (main.py)

**Test Case 1: Customer Support Message**

**Input:**
```
Hi team,
My name is Rahul Sharma. I placed an order yesterday but the payment failed.
Order ID is ORD-45678.
My email is rahul.sharma@gmail.com.
Please help resolve this issue as soon as possible.
Thanks
Rahul
```

**Expected Output:**
```json
{
  "name": "Rahul Sharma",
  "email": "rahul.sharma@gmail.com",
  "order_id": "ORD-45678",
  "issue_type": "Payment Failed"
}
```

## Usage Examples

### Basic Extraction

```python
from extraction_agent import StructuredExtractionAgent

agent = StructuredExtractionAgent()
result = agent.extract_information("Your unstructured text here")

print(result.name)
print(result.email)
print(result.order_id)
print(result.issue_type)
```

### Batch Processing

```python
texts = [text1, text2, text3]
results = agent.process_multiple_texts(texts)

for result in results:
    print(result.model_dump_json())
```

### Export to JSON

```python
import json

result = agent.extract_information(text)
with open('output.json', 'w') as f:
    json.dump(json.loads(result.model_dump_json()), f, indent=2)
```

## Output Format

**Console Output:**
```
============================================================
Test: Test Case 1 - Customer Support Message
============================================================
Name:       Rahul Sharma
Email:      rahul.sharma@gmail.com
Order ID:   ORD-45678
Issue Type: Payment Failed
============================================================
```

**JSON Output (extraction_output.json):**
```json
{
  "name": "Rahul Sharma",
  "email": "rahul.sharma@gmail.com",
  "order_id": "ORD-45678",
  "issue_type": "Payment Failed"
}
```

## Extraction Logic

The agent uses the following approach:

1. **Text Analysis**: Groq LLM analyzes the unstructured input
2. **Pattern Recognition**: Identifies key information patterns (names, emails, IDs)
3. **Field Extraction**: Maps extracted data to defined fields
4. **Validation**: Pydantic validates data types and formats
5. **Structured Output**: Returns validated, structured data

## API Limits

**Groq Free Tier:**
- 100 requests/day
- Model: Mixtral-8x7b-32768
- No credit card required
- Perfect for development and testing

## Error Handling

- Email validation errors caught by Pydantic
- Missing fields handled gracefully
- API errors logged with context
- Batch processing continues on individual failures

## Deliverables Checklist

✅ LangChain Agent implementation
✅ Pydantic schema/model definition
✅ Structured extraction workflow
✅ Input text processing pipeline
✅ Validated JSON/key-value output
✅ Successful test case execution
✅ Complete documentation

## Troubleshooting

**Issue**: `GROQ_API_KEY not found`
- **Solution**: Ensure .env file is created and contains valid API key

**Issue**: `Email validation failed`
- **Solution**: Ensure input text contains valid email format

**Issue**: `API rate limit exceeded`
- **Solution**: Wait for 24 hours or get upgraded Groq plan

## Future Enhancements

- Support for multiple entity types (products, dates, amounts)
- Custom field extraction based on user requirements
- Confidence scoring for extracted fields
- Support for multiple languages
- Integration with vector databases for semantic search
- Caching extracted results

## License

MIT License - Feel free to use and modify

## Author

Created as part of Day 6 - Task 2 of Daily Task Challenge
