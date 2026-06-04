from langchain.vectorstores import FAISS


def create_vector_store(
    documents,
    embeddings
):

    db = FAISS.from_documents(
        documents,
        embeddings
    )

    return db