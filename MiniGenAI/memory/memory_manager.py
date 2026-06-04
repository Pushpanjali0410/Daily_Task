from langchain_huggingface import HuggingFacePipeline

def load_llm():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        temperature=0.7,
        device=-1  # Force CPU usage
    )
    return HuggingFacePipeline(pipeline=pipe)
