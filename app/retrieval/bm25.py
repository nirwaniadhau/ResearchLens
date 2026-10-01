from langchain_community.retrievers import BM25Retriever


def create_bm25_retriever(vector_store, k=3):
    """
    Create a BM25 retriever using the documents
    stored inside the FAISS vector store.
    """

    stored_documents = list(
        vector_store.docstore._dict.values()
    )

    bm25_retriever = BM25Retriever.from_documents(
        stored_documents
    )

    bm25_retriever.k = k

    print("✓ BM25 retriever created.")

    return bm25_retriever