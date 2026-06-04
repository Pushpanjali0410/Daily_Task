# MiniGenAI Assistant 🤖

MiniGenAI Assistant is a Generative AI project developed using Streamlit, LangChain, Hugging Face models, and FAISS. The goal of this project is to build an intelligent chatbot that can answer general questions, remember previous conversations, and also answer questions from uploaded PDF documents.

This project helped me understand important GenAI concepts such as Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), embeddings, vector databases, and conversational memory.

---

## Features

- Interactive chatbot interface using Streamlit
- Conversation memory for maintaining context
- Upload PDF documents and ask questions from them
- Retrieval-Augmented Generation (RAG)
- FAISS vector database for document retrieval
- Chat history storage
- Hugging Face model integration
- Simple and user-friendly interface

---

## Project Structure

```text
MiniGenAI/
│
├── app.py
├── models/
├── memory/
├── rag/
├── utils/
├── uploads/
├── chats/
├── requirements.txt
└── README.md
```

---

## How It Works

1. The user enters a question in the chatbot.
2. The chatbot processes the query using a Hugging Face language model.
3. Previous conversation history is used to provide better context-aware responses.
4. If a PDF is uploaded:
   - The document is converted into text.
   - Text is transformed into embeddings.
   - Embeddings are stored in a FAISS vector database.
   - Relevant information is retrieved whenever the user asks questions related to the document.
5. The final response is displayed in the chat interface.

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PyPDF
- PyTorch

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pushpanjali0410/Daily_Task/MiniGenAI.git
cd MiniGenAI
```

Create a virtual environment:

```bash
python -m venv genai_env
```

Activate the environment:

Windows:

```bash
genai_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## What I Learned

Through this project, I gained hands-on experience with:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Embeddings
- Conversational AI
- LangChain Framework
- Streamlit Application Development
- Hugging Face Ecosystem

---

## Future Improvements

Some features I would like to add in the future:

- Voice-based interaction
- Multiple PDF support
- Database-based memory
- User authentication
- Chat export functionality
- Support for more advanced LLMs
- FastAPI backend deployment
- Cloud deployment

---

## Author

**Pushpanjali Siva Prasad**


Interested in Artificial Intelligence, Machine Learning, Deep Learning, NLP, and Generative AI.

---

## Note

This project was developed for learning and exploring modern Generative AI technologies and understanding how LLM-powered applications are built.
