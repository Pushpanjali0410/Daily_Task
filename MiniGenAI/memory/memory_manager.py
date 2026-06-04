from langchain.memory import ConversationBufferWindowMemory

def get_memory():
    return ConversationBufferWindowMemory(
        k=5,
        return_messages=False  # ← Change this from True to False
    )
