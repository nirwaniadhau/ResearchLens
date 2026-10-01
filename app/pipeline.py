import pymupdf

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
    create_faiss_store
)
from app.retrieval.bm25 import (
    create_bm25_retriever
)
from app.retrieval.hybrid import (
    create_hybrid_retriever
)


def build_retrieval_pipeline(pdf_path):
    """
    Process an uploaded PDF and create the complete
    hybrid retrieval pipeline.

    PDF
      ↓
    LlamaCloud
      ↓
    PyMuPDF + Gemini
      ↓
    Documents
      ↓
    Chunks
      ↓
    FAISS
      ↓
    BM25
      ↓
    Hybrid Retriever
    """

    print("\n========================================")
    print("STARTING DOCUMENT PROCESSING")
    print("========================================")

    # ========================================================
    # 1. CREATE EMBEDDING MODEL
    # ========================================================

    embeddings = create_embedding_model()

    print("✓ Embedding model loaded")

    # ========================================================
    # 2. PARSE PDF USING LLAMACLOUD
    # ========================================================

    print("\nParsing PDF with LlamaCloud...")

    parsed_pages = parse_pdf(
        pdf_path
    )

    print(
        f"✓ PDF parsed: {len(parsed_pages)} pages"
    )

    # ========================================================
    # 3. CREATE GEMINI CLIENT
    # ========================================================

    gemini_client = create_gemini_client()

    print("✓ Gemini client initialized")

    # ========================================================
    # 4. PROCESS VISUAL INFORMATION
    # ========================================================

    print("\n========================================")
    print("ANALYZING PDF VISUALS")
    print("========================================")

    pdf_document = pymupdf.open(
        pdf_path
    )

    page_contents = []

    for page_index in range(
        len(pdf_document)
    ):

        page = pdf_document[
            page_index
        ]

        page_number = page_index + 1

        print(
            f"\nProcessing page "
            f"{page_number}/{len(pdf_document)}..."
        )

        # ----------------------------------------------------
        # Text extracted by LlamaCloud
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
        # Analyze visual information with Gemini
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

    pdf_document.close()

    print(
        "\n✓ Visual processing completed"
    )

    # ========================================================
    # 5. CREATE DOCUMENTS
    # ========================================================

    documents = create_documents(
        page_contents,
        pdf_path
    )

    print(
        f"✓ Documents created: "
        f"{len(documents)}"
    )

    # ========================================================
    # 6. CHUNK DOCUMENTS
    # ========================================================

    chunks = chunk_documents(
        documents
    )

    print(
        f"✓ Chunks created: "
        f"{len(chunks)}"
    )

    # ========================================================
    # 7. CREATE FAISS
    # ========================================================

    vector_store = create_faiss_store(
        chunks,
        embeddings
    )

    print(
        "✓ FAISS vector store created"
    )

    # ========================================================
    # 8. CREATE BM25
    # ========================================================

    bm25_retriever = create_bm25_retriever(
        vector_store,
        k=3
    )

    print(
        "✓ BM25 retriever created"
    )

    # ========================================================
    # 9. CREATE HYBRID RETRIEVER
    # ========================================================

    hybrid_retriever = create_hybrid_retriever(
        vector_store,
        bm25_retriever,
        k=3
    )

    print(
        "✓ Hybrid retriever created"
    )

    print("\n========================================")
    print("DOCUMENT PROCESSING COMPLETE")
    print("========================================")

    return hybrid_retriever