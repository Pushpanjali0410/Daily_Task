# 🤖 AI Chatbot Assistant

A production-ready AI chatbot web application built with **Streamlit**, **LangChain**, and **OpenAI**. This project provides a clean, user-friendly interface for interacting with GPT-4o Mini, an advanced language model from OpenAI.

---

## ✨ Features

- 🏄 **Simple & Intuitive UI**: Clean Streamlit interface for easy interaction
- 🔐 **Secure API Key Handling**: Support for environment variables with no hardcoded secrets
- 💬 **Conversation History**: Maintain chat history within a session
- ⚡ **Fast Responses**: Powered by OpenAI's GPT-4o Mini model
- 🛡️ **Robust Error Handling**: Comprehensive error management with user-friendly messages
- 🔄 **Session Management**: Persistent chat history using Streamlit session state
- 📊 **Sidebar Information**: Display model info, Python version, and chat statistics
- 🎨 **Professional Design**: Modern UI with clear sections and helpful icons
- 🧹 **Clear Chat Feature**: Reset conversation history at any time

---

## 📁 Project Structure

```
AI_Chatbot/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── .env.example             # Example environment variables
│
├── src/
│   ├── chatbot.py           # Main chatbot class
│   ├── config.py            # Configuration constants
│   ├── prompts.py           # Reusable prompt templates
│   └── utils.py             # Utility functions
│
└── assets/                   # Placeholder for future assets
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- An OpenAI API key (get one from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys))

### Step 1: Navigate to Project Directory

```bash
cd AI_Chatbot
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Configuration

### Option 1: Using .env File (Recommended)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. The application will automatically load this file on startup.

### Option 2: Using the UI Input Field

1. Leave the `.env` file empty or without the API key.
2. When you run the app, enter your API key in the text input field on the UI.
3. The key will be used for that session only and won't be stored.

### Option 3: Using Environment Variables

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
streamlit run app.py
```

**On macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
streamlit run app.py
```

---

## ▶️ Running the Application

1. Ensure your virtual environment is activated (see Installation Step 2).

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. The application will open in your default web browser at:
   ```
   http://localhost:8501
   ```

---

## 📖 Usage Examples

### Example 1: General Question
```
User: What is the capital of France?
Assistant: The capital of France is Paris. It is located in the north-central part of the country...
```

### Example 2: Technical Question
```
User: Explain how Python list comprehensions work.
Assistant: List comprehensions are a concise way to create lists in Python. They follow the syntax...
```

### Example 3: Creative Request
```
User: Write a short poem about technology.
Assistant: In circuits and code, innovation flows,
           Where human dreams and silicon rose...
```

---

## ⚙️ Tech Stack

| Technology | Version | Purpose |
|-----------|---------|----------|
| Python | 3.10+ | Programming Language |
| Streamlit | 1.40.2+ | Web UI Framework |
| LangChain | 0.3.16+ | AI Framework |
| OpenAI API | Latest | Language Model Provider |
| python-dotenv | 1.0.1+ | Environment Variable Management |

---

## ⚙️ Configuration

Key configurations can be modified in `src/config.py`:

```python
# Model Configuration
MODEL_NAME = "gpt-4o-mini"
CHATBOT_TEMPERATURE = 0.7  # Creativity level (0-1)
CHATBOT_MAX_TOKENS = 1024   # Max response length

# System Prompt
SYSTEM_PROMPT = "You are a helpful AI assistant..."
```

---

## 🐛 Troubleshooting

### Issue: "API key is invalid"
**Solution:**
- Verify your API key is correct from https://platform.openai.com/api-keys
- Ensure the key starts with `sk-`
- Check that the `.env` file is in the AI_Chatbot directory
- Try entering the key directly in the UI input field

### Issue: "ModuleNotFoundError: No module named 'langchain'"
**Solution:**
- Ensure your virtual environment is activated
- Run: `pip install -r requirements.txt`
- Verify Python version is 3.10 or higher: `python --version`

### Issue: "Connection timeout or error"
**Solution:**
- Check your internet connection
- Verify OpenAI API is accessible
- Try again after a few moments (may be a temporary service issue)
- Check if your API key has usage limits

### Issue: "Port 8501 already in use"
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### Issue: "Empty or No Response"
**Solution:**
- Verify API key is valid and has available credits
- Try a simpler question first
- Check the Streamlit terminal for error messages

---

## 📝 Code Quality

This project follows:
- **PEP 8** Python style guide
- **DRY** (Don't Repeat Yourself) principle
- **Modular design** for maintainability
- **Comprehensive docstrings** for all functions
- **Type hints** where applicable
- **Error handling** throughout

---

## 🔒 Security Best Practices

1. **Never commit `.env` file** - It's in `.gitignore` for a reason
2. **Use environment variables** for sensitive data
3. **Rotate API keys regularly** from OpenAI dashboard
4. **Monitor API usage** to prevent unexpected charges
5. **Review code before deployment** to production

---

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Python 3.10 Documentation](https://docs.python.org/3.10/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**Pushpanjali0410**

For questions or support, please open an issue on GitHub.

---

## 🎉 Enjoy!

Happy chatting! Feel free to explore the capabilities of your new AI assistant.
