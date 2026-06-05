from langchain.prompts import PromptTemplate, ChatPromptTemplate
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant.\n"
    "Provide accurate, concise, and well-structured responses.\n"
    "If information is uncertain, clearly mention it.\n"
    "Always be respectful and professional in your interactions."
)

TECHNICAL_ASSISTANT_PROMPT = (
    "You are a technical AI assistant specializing in software development and engineering.\n"
    "Provide detailed technical explanations, code examples when relevant.\n"
    "Always prioritize clarity and accuracy in technical discussions."
)

CREATIVE_ASSISTANT_PROMPT = (
    "You are a creative AI assistant designed to help with writing, brainstorming, and ideation.\n"
    "Encourage creative thinking while maintaining factual accuracy.\n"
    "Provide diverse perspectives and innovative suggestions."
)
def get_default_chat_prompt():
    """Get the default chat prompt template.
    
    Returns:
        ChatPromptTemplate: The default chat prompt template.
    """
    return ChatPromptTemplate.from_messages([
        ("system", DEFAULT_SYSTEM_PROMPT),
        ("human", "{user_input}")
    ])


def get_technical_chat_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", TECHNICAL_ASSISTANT_PROMPT),
        ("human", "{user_input}")
    ])


def get_creative_chat_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", CREATIVE_ASSISTANT_PROMPT),
        ("human", "{user_input}")
    ])

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Provide a concise summary of the following text:\n\n{text}"
)

TRANSLATION_PROMPT = PromptTemplate(
    input_variables=["text", "language"],
    template="Translate the following text to {language}:\n\n{text}"
)

CODE_REVIEW_PROMPT = PromptTemplate(
    input_variables=["code"],
    template=(
        "Review the following code and provide suggestions for improvement, "
        "highlighting any issues or optimizations:\n\n{code}"
    )
)
