from langchain_classic.retrievers import EnsembleRetriever


def create_hybrid_retriever(
    vector_store,
    bm25_retriever,
    k=3
):
    """
    Create a hybrid retriever combining
    FAISS semantic search and BM25 keyword search.
    """

    faiss_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            faiss_retriever,
            bm25_retriever
        ],
        weights=[
            0.5,
            0.5
        ]
    )

    print("✓ Hybrid retriever created.")

    return hybrid_retriever