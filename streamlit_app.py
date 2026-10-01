import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR / ".env"
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.pipeline import build_retrieval_pipeline

from app.generation.llm import (
    create_groq_client,
    answer_question
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResearchLens",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application width */

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }


    /* Header */

    .researchlens-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .researchlens-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .researchlens-subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
    }


    /* Empty state */

    .empty-state {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }

    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }

    .empty-title {
        font-size: 1.8rem;
        font-weight: 600;
    }

    .empty-description {
        color: #9CA3AF;
        max-width: 600px;
        margin: auto;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #252B3A;
    }


    /* Chat messages */

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }


    /* Source text */

    .source-text {
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }


    /* Status card */

    .status-card {
        padding: 0.8rem;
        border-radius: 10px;
        background: #151A26;
        border: 1px solid #252B3A;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    return create_groq_client()


groq_client = get_groq_client()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔬 ResearchLens", unsafe_allow_html=True
    )

    st.caption(
        "AI-powered research paper assistant"
    )

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    st.markdown(
        "### 📄 Document", unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a research paper",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        st.caption(
            f"{uploaded_file.name}"
        )

        st.caption(
            f"{uploaded_file.size / 1024:.1f} KB"
        )

        process_button = st.button(
            "🚀 Process Document",
            use_container_width=True,
            type="primary"
        )

        if process_button:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                pdf_path = temp_file.name

            try:

                with st.status(
                    "Processing research paper...",
                    expanded=True
                ) as status:

                    st.write(
                        "📖 Parsing document"
                    )

                    st.write(
                        "🖼️ Analyzing visual content"
                    )

                    st.write(
                        "🧩 Creating document chunks"
                    )

                    st.write(
                        "🔎 Building hybrid retrieval index"
                    )

                    hybrid_retriever = (
                        build_retrieval_pipeline(
                            pdf_path
                        )
                    )

                    st.session_state.hybrid_retriever = (
                        hybrid_retriever
                    )

                    st.session_state.document_processed = True

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.session_state.messages = []

                    status.update(
                        label="Document ready!",
                        state="complete",
                        expanded=False
                    )

            except Exception as e:

                st.error(
                    f"Error while processing document: {e}"
                )

            finally:

                try:

                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

                except PermissionError:

                    pass

    # --------------------------------------------------------
    # DOCUMENT STATUS
    # --------------------------------------------------------

    if st.session_state.document_processed:

        st.markdown(
            f"""
            <div class="status-card">
            🟢 <b>Ready</b><br>
            <small>{st.session_state.document_name}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.caption(
            "Upload a PDF to begin."
        )

    st.divider()

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    with st.expander(
        "ℹ️ About ResearchLens"
    ):

        st.write(
            """
            ResearchLens is a conversational RAG
            assistant for research papers.

            **Retrieval**
            - FAISS semantic search
            - BM25 keyword search
            - Hybrid retrieval

            **AI**
            - LlamaCloud
            - Gemini Vision
            - Groq LLM
            """
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="researchlens-title">
        🔬 ResearchLens
    </div>

    <div class="researchlens-subtitle">
        Understand your research papers through conversation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.document_processed:

    st.markdown(
        """
        <div class="empty-icon">
            📚
        </div>

        <div class="empty-title">
            Your research assistant is ready.
        </div>

        <div class="empty-description">
            Upload a research paper from the sidebar,
            then ask questions and explore the document
            through a conversational interface.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHAT
# ============================================================

else:

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        avatar = (
            "🧑‍💻"
            if message["role"] == "user"
            else "🔬"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":

                pages = message.get(
                    "pages",
                    []
                )

                if pages:

                    st.caption(
                        "📚 Sources: "
                        + ", ".join(
                            f"Page {page}"
                            for page in pages
                        )
                    )

    # --------------------------------------------------------
    # SUGGESTED QUESTIONS
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.markdown(
            "##### 💡 Try asking"
            ,unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "What is the main contribution?",
                use_container_width=True
            ):

                st.session_state.pending_question = (
                    "What is the main contribution?"
                )

                st.rerun()

        with col2:

            if st.button(
                "What are the key findings?",
                use_container_width=True
            ):

                st.session_state.pending_question = (
                    "What are the key findings?"
                )

                st.rerun()

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your paper..."
    )

    # Handle suggested question
    if (
        "pending_question" in st.session_state
        and st.session_state.pending_question
    ):

        question = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None

    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message(
            "user",
            avatar="🧑‍💻"
        ):

            st.markdown(
                question
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # ----------------------------------------------------
        # ASSISTANT
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🔬"
        ):

            with st.status(
                "ResearchLens is searching the paper...",
                expanded=False
            ):

                try:

                    answer, retrieved_docs = (
                        answer_question(
                            question,
                            st.session_state.hybrid_retriever,
                            groq_client,
                            st.session_state.messages[:-1]
                        )
                    )

                except Exception as e:

                    st.error(
                        f"Error while answering: {e}"
                    )

                    answer = None
                    retrieved_docs = []

            if answer:

                st.markdown(
                    answer
                )

                # --------------------------------------------
                # SOURCE PAGES
                # --------------------------------------------

                pages = []

                for doc in retrieved_docs:

                    page = doc.metadata.get(
                        "page"
                    )

                    if (
                        page is not None
                        and page not in pages
                    ):

                        pages.append(page)

                if pages:

                    st.caption(
                        "📚 Sources: "
                        + ", ".join(
                            f"Page {page}"
                            for page in pages
                        )
                    )

                # --------------------------------------------
                # SAVE ASSISTANT MESSAGE
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "pages": pages
                    }
                )