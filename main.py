import os
import pymupdf

from dotenv import load_dotenv

from app.ingestion.parser import parse_pdf
from app.ingestion.visual_analyzer import (
    create_gemini_client,
    analyze_page_image
)
from app.ingestion.chunker import (
    create_documents,
    chunk_documents
)

from app.retrieval.embeddings import (
    create_embedding_model
)
from app.retrieval.faiss_store import (
    load_faiss_store,
    create_faiss_store,
    save_faiss_store
)
from app.retrieval.bm25 import (
    create_bm25_retriever
)
from app.retrieval.hybrid import (
    create_hybrid_retriever
)

from app.generation.llm import (
    create_groq_client,
    answer_question
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

PDF_PATH = "D:\\ResearchLens\\data\\DLP Detecting Sensitive Information Leakage.docx (1).pdf"

FAISS_PATH = "vectorstore/researchlens_faiss"


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

print("\n========================================")
print("INITIALIZING RESEARCHLENS")
print("========================================")

embeddings = create_embedding_model()

print("✓ Embedding model loaded")


# ============================================================
# LOAD OR CREATE FAISS
# ============================================================

if os.path.exists(FAISS_PATH):

    print("\n========================================")
    print("LOADING EXISTING VECTOR STORE")
    print("========================================")

    vector_store = load_faiss_store(
        FAISS_PATH,
        embeddings
    )

else:

    print("\n========================================")
    print("STARTING DOCUMENT INGESTION")
    print("========================================")

    # --------------------------------------------------------
    # STEP 1: Parse PDF using LlamaCloud
    # --------------------------------------------------------

    parsed_pages = parse_pdf(
        PDF_PATH
    )

    # --------------------------------------------------------
    # STEP 2: Create Gemini client
    # --------------------------------------------------------

    gemini_client = create_gemini_client()

    # --------------------------------------------------------
    # STEP 3: Visual analysis using PyMuPDF + Gemini
    # --------------------------------------------------------

    print("\n========================================")
    print("ANALYZING PAGE VISUALS")
    print("========================================")

    doc = pymupdf.open(
        PDF_PATH
    )

    page_contents = []

    for page_index in range(
        len(doc)
    ):

        page = doc[page_index]

        page_number = page_index + 1

        print(
            f"\nProcessing page {page_number}/{len(doc)}..."
        )

        # ----------------------------------------------------
        # Extract normal text from LlamaCloud
        # ----------------------------------------------------

        parsed_page = parsed_pages[
            page_index
        ]

        page_text = parsed_page.markdown

        # ----------------------------------------------------
        # Detect images and drawings
        # ----------------------------------------------------

        images = page.get_images(
            full=True
        )

        drawings = page.get_drawings()

        has_visuals = (
            len(images) > 0
            or len(drawings) > 0
        )

        visual_description = ""

        # ----------------------------------------------------
        # Render page and analyze with Gemini
        # ----------------------------------------------------

        if has_visuals:

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(
                    2,
                    2
                )
            )

            image_bytes = pix.tobytes(
                "png"
            )

            visual_description = (
                analyze_page_image(
                    gemini_client,
                    image_bytes
                )
            )

            print(
                "✓ Visual analysis completed"
            )

        else:

            print(
                "No significant visuals detected"
            )

        page_contents.append(
            {
                "page_number": page_number,
                "text": page_text,
                "visual_description":
                    visual_description
            }
        )

    doc.close()

    print("\n✓ PDF visual processing completed")

    # --------------------------------------------------------
    # STEP 4: Convert pages into Documents
    # --------------------------------------------------------

    documents = create_documents(
        page_contents,
        PDF_PATH
    )

    print(
        f"✓ Pages converted to Documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # STEP 5: Chunk documents
    # --------------------------------------------------------

    chunks = chunk_documents(
        documents
    )

    print(
        f"✓ Number of chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # STEP 6: Create FAISS
    # --------------------------------------------------------

    vector_store = create_faiss_store(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # STEP 7: Save FAISS
    # --------------------------------------------------------

    save_faiss_store(
        vector_store,
        FAISS_PATH
    )


# ============================================================
# CREATE BM25 RETRIEVER
# ============================================================

print("\n========================================")
print("SETTING UP RETRIEVAL")
print("========================================")

bm25_retriever = create_bm25_retriever(
    vector_store,
    k=3
)


# ============================================================
# CREATE HYBRID RETRIEVER
# ============================================================

hybrid_retriever = create_hybrid_retriever(
    vector_store,
    bm25_retriever,
    k=3
)


print("✓ Hybrid retrieval ready")


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

groq_client = create_groq_client()

print("✓ Groq client initialized")


# ============================================================
# ASK QUESTIONS
# ============================================================

print("\n========================================")
print("RESEARCHLENS READY")
print("========================================")

print(
    "Ask questions about your research paper."
)

print(
    "Type 'exit' to stop."
)


while True:

    question = input(
        "\nYou: "
    )

    if question.lower() == "exit":
        print(
            "\nResearchLens stopped."
        )
        break

    if not question.strip():
        continue

    try:

        answer, retrieved_docs = answer_question(
            question,
            hybrid_retriever,
            groq_client
        )

        print(
            "\nResearchLens:"
        )

        print(
            answer
        )

        print(
            "\nRetrieved pages:"
        )

        pages = []

        for doc in retrieved_docs:

            page = doc.metadata.get(
                "page"
            )

            if page not in pages:
                pages.append(page)

        print(
            pages
        )

    except Exception as e:

        print("\nError while answering:")

        print(e)