from langchain.memory import ConversationBufferWindowMemory


def get_memory():

    return ConversationBufferWindowMemory(
        k=5,
        return_messages=True
    )