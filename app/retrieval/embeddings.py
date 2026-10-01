from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():
    """
    Create and return the HuggingFace embedding model
    used by ResearchLens.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings