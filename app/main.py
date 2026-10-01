import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.retrieval.embeddings import create_embedding_model
from app.retrieval.faiss_store import load_faiss_store
from app.retrieval.bm25 import create_bm25_retriever
from app.retrieval.hybrid import create_hybrid_retriever

from app.generation.llm import (
    create_groq_client,
    answer_question
)

load_dotenv()
# ============================================================
# CONFIGURATION
# ============================================================

FAISS_PATH = "vectorstore/researchlens_faiss"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ResearchLens API",
    description="Research paper question-answering API",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================

print("\n========================================")
print("INITIALIZING RESEARCHLENS API")
print("========================================")


embeddings = create_embedding_model()

print("✓ Embedding model loaded")


if not os.path.exists(FAISS_PATH):

    raise RuntimeError(
        "FAISS vector store not found. "
        "Run the ingestion pipeline first."
    )


vector_store = load_faiss_store(
    FAISS_PATH,
    embeddings
)

print("✓ FAISS vector store loaded")


bm25_retriever = create_bm25_retriever(
    vector_store,
    k=3
)

print("✓ BM25 retriever created")


hybrid_retriever = create_hybrid_retriever(
    vector_store,
    bm25_retriever,
    k=3
)

print("✓ Hybrid retriever created")


groq_client = create_groq_client()

print("✓ Groq client initialized")


print("\n========================================")
print("RESEARCHLENS API READY")
print("========================================")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "ResearchLens API is running",
        "status": "healthy"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        answer, retrieved_docs = answer_question(
            question,
            hybrid_retriever,
            groq_client
        )

        pages = []

        for doc in retrieved_docs:

            page = doc.metadata.get("page")

            if page is not None and page not in pages:
                pages.append(page)

        return {
            "question": question,
            "answer": answer,
            "pages": pages
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )