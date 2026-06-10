"""Main Streamlit application for MiniGenAI Assistant."""

import logging
import streamlit as st
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
from config import (
    UPLOADS_DIR, CHATS_DIR, LLM_MODEL_NAME, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_DEVICE, EMBEDDING_MODEL_NAME, MEMORY_WINDOW_SIZE
)

# Import modules
from models.llm_loader import load_llm
from memory.memory_manager import get_memory
from rag.pdf_loader import load_pdf
from rag.embedding import get_embeddings
from rag.vector_store import create_vector_store
from langchain.chains import ConversationChain, RetrievalQA
from utils.save_chat import save_chat, load_chat
from utils.export_pdf import export_chat_to_pdf


# PAGE CONFIGURATION

st.set_page_config(
    page_title="MiniGenAI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 MiniGenAI Assistant")
st.markdown(
    "An intelligent chatbot with PDF Q&A and conversation memory powered by LangChain and Hugging Face"
)

# INITIALIZE SESSION STATE

if "llm" not in st.session_state:
    try:
        logger.info("Loading LLM...")
        st.session_state.llm = load_llm(
            model_name=LLM_MODEL_NAME,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            device=LLM_DEVICE
        )
        logger.info("LLM loaded successfully")
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        logger.error(f"LLM loading error: {e}")
        st.stop()

if "memory" not in st.session_state:
    try:
        st.session_state.memory = get_memory(window_size=MEMORY_WINDOW_SIZE)
    except Exception as e:
        st.error(f"Error initializing memory: {e}")
        logger.error(f"Memory initialization error: {e}")
        st.stop()

if "conversation" not in st.session_state:
    try:
        st.session_state.conversation = ConversationChain(
            llm=st.session_state.llm,
            memory=st.session_state.memory,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        logger.error(f"Conversation chain creation error: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
    
# SIDEBAR - DOCUMENT UPLOAD

with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF file to enable RAG-based question answering"
    )
    
    if uploaded_file:
        try:
            pdf_path = UPLOADS_DIR / uploaded_file.name
            
            # Save uploaded file
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ {uploaded_file.name} uploaded successfully")
            
            # Process PDF
            with st.spinner("🔄 Processing PDF..."):
                logger.info(f"Processing PDF: {pdf_path}")
                
                try:
                    documents = load_pdf(str(pdf_path))
                    
                    if not st.session_state.embeddings:
                        st.session_state.embeddings = get_embeddings(EMBEDDING_MODEL_NAME)
                    
                    st.session_state.vector_db = create_vector_store(
                        documents,
                        st.session_state.embeddings
                    )
                    
                    st.success(f"✅ PDF indexed successfully ({len(documents)} pages)")
                    logger.info(f"PDF processed successfully: {len(documents)} pages")
                
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
                    logger.error(f"PDF processing error: {e}")
    
    # Display current document status
    if st.session_state.vector_db:
        st.info("📌 **RAG Mode Active** - Questions will be answered based on the uploaded PDF")
    else:
        st.info("💬 **Chat Mode** - General conversation mode (no PDF loaded)")

# MAIN CHAT INTERFACE

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input(
    "Ask me anything...",
    key="chat_input"
)

if prompt:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    try:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # RAG Mode: Use uploaded PDF
                if st.session_state.vector_db:
                    logger.info("Using RAG mode for response generation")
                    retriever = st.session_state.vector_db.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type="stuff",
                        retriever=retriever,
                        verbose=False
                    )
                    response = qa_chain.run(prompt)
                
                # Chat Mode: Regular conversation
                else:
                    logger.info("Using conversation mode for response generation")
                    response = st.session_state.conversation.predict(input=prompt)
            
            st.markdown(response)
    
    except Exception as e:
        st.error(f"❌ Error generating response: {e}")
        logger.error(f"Response generation error: {e}")
        response = f"Sorry, I encountered an error: {str(e)}"
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    
    # Save chat history
    try:
        save_chat(st.session_state.messages)
    except Exception as e:
        logger.error(f"Error saving chat: {e}")

# SIDEBAR - CONTROLS

with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = get_memory(window_size=MEMORY_WINDOW_SIZE)
            st.session_state.conversation = ConversationChain(
                llm=st.session_state.llm,
                memory=st.session_state.memory,
                verbose=False
            )
            logger.info("Chat history cleared")
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat", use_container_width=True):
            try:
                save_chat(st.session_state.messages)
                st.success("✅ Chat saved!")
                logger.info("Chat manually saved")
            except Exception as e:
                st.error(f"Error saving chat: {e}")
                logger.error(f"Chat save error: {e}")
    
    with col3:
        if st.button("📥 Export PDF", use_container_width=True):
            try:
                export_chat_to_pdf(st.session_state.messages)
                st.success("✅ Exported to PDF!")
                logger.info("Chat exported to PDF")
            except Exception as e:
                st.error(f"Error exporting to PDF: {e}")
                logger.error(f"PDF export error: {e}")
    
    # Features section
    st.markdown("---")
    st.markdown("### ✨ Features")
    features = [
        "✅ Interactive Chat Interface",
        "✅ Conversation Memory (5 messages)",
        "✅ PDF Question Answering (RAG)",
        "✅ FAISS Vector Store",
        "✅ Hugging Face LLM Integration",
        "✅ Chat History Saving",
        "✅ PDF Export",
        "✅ Error Handling & Logging",
    ]
    for feature in features:
        st.markdown(feature)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 12px; color: gray;'>"
        "<p>MiniGenAI Assistant v1.0</p>"
        "<p>Built with Streamlit & LangChain</p>"
        "</div>",
        unsafe_allow_html=True
    )
