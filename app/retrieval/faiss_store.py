import os

from langchain_community.vectorstores import FAISS


def load_faiss_store(faiss_path, embeddings):
    """
    Load an existing FAISS vector store from disk.
    """

    if not os.path.exists(faiss_path):
        return None

    vector_store = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✓ Existing FAISS vector store loaded.")

    return vector_store


def create_faiss_store(chunks, embeddings):
    """
    Create a new FAISS vector store from document chunks.
    """

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("✓ FAISS vector store created.")

    return vector_store


def save_faiss_store(vector_store, faiss_path):
    """
    Save FAISS vector store to disk.
    """

    vector_store.save_local(faiss_path)

    print("✓ FAISS vector store saved.")