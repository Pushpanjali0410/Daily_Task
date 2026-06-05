from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant.\n"
    "Provide accurate, concise, and well-structured responses.\n"
    "If information is uncertain, clearly mention it.\n"
    "Always be respectful and professional in your interactions."
)

CHATBOT_TEMPERATURE = 0.7
CHATBOT_MAX_TOKENS = 1024

APP_NAME = "AI Chatbot Assistant"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "A production-ready AI chatbot powered by OpenAI and LangChain"
