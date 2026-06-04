import os
import streamlit as st

from models.llm_loader import load_llm
from memory.memory_manager import get_memory

from rag.pdf_loader import load_pdf
from rag.embedding import get_embeddings
from rag.vector_store import create_vector_store

from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA

from utils.save_chat import save_chat

# Page Config

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Assistant")

# Create folders

os.makedirs("uploads", exist_ok=True)
os.makedirs("chats", exist_ok=True)


# Load LLM

@st.cache_resource
def get_llm():
    return load_llm()

llm = get_llm()

# Sidebar PDF Upload

st.sidebar.header("📄 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# Process PDF

if uploaded_file:

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    st.sidebar.success(
        f"{uploaded_file.name} uploaded"
    )

    with st.spinner(
        "Processing PDF..."
    ):

        documents = load_pdf(
            pdf_path
        )

        embeddings = get_embeddings()

        vector_db = create_vector_store(
            documents,
            embeddings
        )

        st.session_state.vector_db = vector_db

    st.sidebar.success(
        "PDF indexed successfully"
    )

# Memory

if "memory" not in st.session_state:

    st.session_state.memory = get_memory()

# Conversation Chain

if "conversation" not in st.session_state:

    st.session_state.conversation = (
        ConversationChain(
            llm=llm,
            memory=st.session_state.memory,
            verbose=False
        )
    )

# Chat History

if "messages" not in st.session_state:

    st.session_state.messages = []

# Display Previous Messages

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

# User Input

prompt = st.chat_input(
    "Ask me anything..."
)

# Handle Query

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # If PDF uploaded -> RAG Mode

    if "vector_db" in st.session_state:

        retriever = (
            st.session_state
            .vector_db
            .as_retriever()
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        response = qa_chain.run(
            prompt
        )

    # Normal Chat Mode

    else:

        response = (
            st.session_state
            .conversation
            .predict(
                input=prompt
            )
        )

    # Display Assistant Response

    with st.chat_message(
        "assistant"
    ):
        st.markdown(
            response
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    save_chat(
        st.session_state.messages
    )

# Sidebar Controls

st.sidebar.markdown("---")

if st.sidebar.button(
    "Clear Chat"
):

    st.session_state.messages = []

    st.session_state.memory = get_memory()

    st.rerun()

st.sidebar.markdown(
    "### Features"
)

st.sidebar.markdown(
"""
✅ Chat Memory

✅ PDF Question Answering

✅ RAG using FAISS

✅ Hugging Face LLM

✅ Streamlit UI

✅ Chat History Saving
"""
)