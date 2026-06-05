
import streamlit as st
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from chatbot import create_chatbot
from config import SYSTEM_PROMPT


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False


def display_sidebar():
    """Display sidebar with application information."""
    with st.sidebar:
        st.title("ℹ️ App Information")
        st.divider()
        
        st.subheader("Model Details")
        st.write(f"**Model Name:** GPT-4o Mini")
        st.write(f"**Provider:** OpenAI")
        
        st.divider()
        
        st.subheader("Environment")
        st.write(f"**Python Version:** {sys.version.split()[0]}")
        st.write(f"**Session Active:** Yes")
        
        st.divider()
        
        st.subheader("Chat History")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        st.divider()
        
        st.subheader("Instructions")
        st.info(
            "1. Enter your OpenAI API key (if not set in .env)\n"
            "2. Type your question in the input box\n"
            "3. Click 'Send Message' to get a response\n"
            "4. Use 'Clear Chat' to reset conversation"
        )


def display_chat_history():
    """Display chat history in the main area."""
    if st.session_state.messages:
        st.subheader("📜 Chat History")
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])


def handle_user_input(user_input: str, chatbot) -> None:
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
        return
    
    try:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        with st.spinner("🤖 Generating response..."):
            response = chatbot.chat(user_input)
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun to display updated chat
        st.rerun()
        
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        st.info("Please check your API key and try again.")


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    st.title("🤖 AI Chatbot Assistant")
    st.markdown(
        "Welcome to the AI Chatbot! Ask me anything and I'll provide helpful, "
        "accurate responses powered by OpenAI's GPT-4o Mini model."
    )
    st.divider()
    
    # API Key input (if not in environment)
    api_key = st.text_input(
        "🔑 Enter your OpenAI API Key (if not set in .env):",
        type="password",
        help="Your API key is not stored or shared."
    )
    
    # Initialize chatbot
    if api_key or st.session_state.chatbot:
        try:
            if not st.session_state.chatbot:
                if api_key:
                    st.session_state.chatbot = create_chatbot(api_key=api_key)
                else:
                    st.session_state.chatbot = create_chatbot()
                st.session_state.api_key_valid = True
                st.success("✅ Chatbot initialized successfully!")
        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.chatbot = None
            st.session_state.api_key_valid = False
            st.stop()
    
    # Display chat history
    display_chat_history()
    
    st.divider()
    
    # User input section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "💬 Your Question:",
            placeholder="Ask me anything...",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button(
            "Send Message",
            use_container_width=True,
            type="primary"
        )
    
    # Handle message submission
    if submit_button:
        if st.session_state.chatbot:
            handle_user_input(user_input, st.session_state.chatbot)
        else:
            st.error("❌ Chatbot not initialized. Please provide a valid API key.")
    
    # Clear chat button
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot = None
            st.session_state.api_key_valid = False
            st.success("Chat cleared!")
            st.rerun()


if __name__ == "__main__":
    main()
