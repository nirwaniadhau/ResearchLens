# ============================================
# 1. IMPORTS
# ============================================

import os
import time
import pymupdf

from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from google import genai
from google.genai import types
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# ============================================
# 2. CONFIGURATION
# ============================================

load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

client = LlamaCloud(
    api_key=LLAMA_CLOUD_API_KEY
)

pdf_path = r"D:\ResearchLens\data\DLP Detecting Sensitive Information Leakage.docx (1).pdf"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(
    api_key=GROQ_API_KEY
)
FAISS_PATH = "vectorstore/researchlens_faiss"
# ============================================
# 3. EMBEDDING MODEL
# ============================================

print("\n========================================")
print("CREATING EMBEDDING MODEL")
print("========================================")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✓ Embedding model loaded")


# ============================================
# 4. LOAD OR CREATE VECTOR STORE
# ============================================

if os.path.exists(FAISS_PATH):

    # ----------------------------------------
    # EXISTING VECTOR STORE
    # ----------------------------------------

    print("\n========================================")
    print("EXISTING VECTOR STORE FOUND")
    print("SKIPPING PDF INGESTION")
    print("========================================")

    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✓ FAISS index loaded successfully")


else:

    # ----------------------------------------
    # FIRST-TIME PDF INGESTION
    # ----------------------------------------

    print("\n========================================")
    print("NO VECTOR STORE FOUND")
    print("STARTING PDF INGESTION")
    print("========================================")


    # ========================================
    # 5. PDF PARSING — LLAMACLOUD
    # ========================================

    print("\n========================================")
    print("STARTING PDF PARSING")
    print("========================================")

    file = client.files.create(
        file=pdf_path,
        purpose="parse"
    )

    print("✓ PDF uploaded")
    print("File ID:", file.id)

    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"]
    )

    print("✓ PDF parsed successfully")

    parsed_pages = result.markdown.pages

    print(
        "Number of parsed pages:",
        len(parsed_pages)
    )


    # ========================================
    # 6. IMAGE UNDERSTANDING — GEMINI
    # ========================================

    def analyze_page_image(image_bytes):

        prompt = """
You are analyzing a page from a research paper.

Describe the important visual information on this page that
may not be fully captured by OCR/text extraction.

Focus on:
- diagrams and their relationships
- charts and graphs
- tables and their important values
- figures and screenshots
- flowcharts or architectures
- captions and labels
- visual structure that helps understand the research

Do NOT simply repeat normal paragraph text.

Return a concise but informative description in Markdown.

If there is no meaningful visual information, say:
"No significant visual information."
"""

        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                ),
                prompt
            ]
        )

        return response.text


    # ========================================
    # 7. PDF VISUAL PROCESSING — PYMUPDF
    # ========================================

    print("\n========================================")
    print("STARTING VISUAL PROCESSING")
    print("========================================")

    doc = pymupdf.open(pdf_path)

    print("Number of PDF pages:", len(doc))

    page_contents = []

    for page_index in range(len(doc)):

        page = doc[page_index]
        page_number = page_index + 1

        # ------------------------------------
        # Get text from LlamaCloud
        # ------------------------------------

        text_content = ""

        if page_index < len(parsed_pages):
            text_content = parsed_pages[page_index].markdown

        # ------------------------------------
        # Detect visuals
        # ------------------------------------

        images = page.get_images(full=True)
        drawings = page.get_drawings()

        has_visuals = (
            len(images) > 0 or
            len(drawings) > 0
        )

        print(
            f"\nPage {page_number}: "
            f"Images={len(images)}, "
            f"Drawings={len(drawings)}, "
            f"Visuals={has_visuals}"
        )

        # ------------------------------------
        # Render + analyze visual page
        # ------------------------------------

        image_bytes = None
        visual_description = ""

        if has_visuals:

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2)
            )

            image_bytes = pix.tobytes("png")

            print("  ✓ Page rendered")

            try:

                print(
                    "  → Sending page to Gemini..."
                )

                visual_description = analyze_page_image(
                    image_bytes
                )

                print(
                    "  ✓ Gemini analysis complete"
                )

            except Exception as e:

                print(
                    "  ✗ Gemini analysis failed:",
                    e
                )

                visual_description = ""

        # ------------------------------------
        # Store page information
        # ------------------------------------

        page_contents.append({
            "page_number": page_number,
            "text": text_content,
            "has_visuals": has_visuals,
            "visual_description": visual_description
        })

    doc.close()


    # ========================================
    # 8. INGESTION SUMMARY
    # ========================================

    print("\n========================================")
    print("INGESTION COMPLETE")
    print("========================================")

    print(
        "Pages processed:",
        len(page_contents)
    )

    for page in page_contents:

        print(
            f"Page {page['page_number']}: "
            f"text={len(page['text'])} chars, "
            f"visual={page['has_visuals']}"
        )

        if page["visual_description"]:

            print(
                "  Visual description:",
                page["visual_description"][:200],
                "..."
            )


    # ========================================
    # 9. COMBINE TEXT + VISUAL INFORMATION
    # ========================================

    print("\n========================================")
    print("PREPARING DOCUMENT FOR RAG")
    print("========================================")

    documents = []

    for page in page_contents:

        combined_page_content = (
            f"--- PAGE {page['page_number']} ---\n\n"
            f"[TEXT]\n"
            f"{page['text']}\n"
        )

        if page["visual_description"]:

            combined_page_content += (
                "\n[VISUAL ANALYSIS]\n"
                f"{page['visual_description']}\n"
                "[END VISUAL ANALYSIS]\n"
            )

        document = Document(
            page_content=combined_page_content,
            metadata={
                "source": pdf_path,
                "page": page["page_number"]
            }
        )

        documents.append(document)

    print(
        "Pages converted to Documents:",
        len(documents)
    )


    # ========================================
    # 10. TEXT CHUNKING
    # ========================================

    print("\n========================================")
    print("CHUNKING DOCUMENTS")
    print("========================================")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        "Number of chunks:",
        len(chunks)
    )


    # ========================================
    # 11. CREATE FAISS VECTOR STORE
    # ========================================

    print("\n========================================")
    print("CREATING FAISS VECTOR STORE")
    print("========================================")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(
        FAISS_PATH
    )

    print(
        "✓ FAISS index created and saved successfully"
    )


# ============================================
# VECTOR STORE SUMMARY
# ============================================

print(
    "Number of indexed chunks:",
    vector_store.index.ntotal
)

# ============================================
# 11. TEST SEMANTIC RETRIEVAL
# ============================================

print("\n========================================")
print("TESTING SEMANTIC SEARCH")
print("========================================")

query = "What accuracy did the steganalysis model achieve?"

results = vector_store.similarity_search(
    query,
    k=3
)

for i, doc in enumerate(results):

    print(f"\n--- RESULT {i + 1} ---")

    print("Page:", doc.metadata["page"])

    print(doc.page_content[:800])
    
    
    
    # ============================================
# 12. BM25 KEYWORD RETRIEVER
# ============================================

# ============================================
# BM25 RETRIEVER
# ============================================

stored_documents = list(
    vector_store.docstore._dict.values()
)

print("Documents available for BM25:", len(stored_documents))

bm25_retriever = BM25Retriever.from_documents(
    stored_documents
)

bm25_retriever.k = 3


# ============================================
# 13. TEST BM25 RETRIEVAL
# ============================================

print("\n========================================")
print("TESTING BM25 SEARCH")
print("========================================")

query = "What accuracy did the steganalysis model achieve?"

bm25_results = bm25_retriever.invoke(query)

for i, doc in enumerate(bm25_results):

    print(f"\n--- BM25 RESULT {i + 1} ---")

    print("Page:", doc.metadata["page"])

    print(doc.page_content[:800])
    
    
    # ============================================
# 14. HYBRID RETRIEVER
# ============================================

print("\n========================================")
print("BUILDING HYBRID RETRIEVER")
print("========================================")

faiss_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

bm25_retriever.k = 3

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

print("✓ Hybrid retriever created")

# ============================================
# 15. TEST HYBRID RETRIEVAL
# ============================================

print("\n========================================")
print("TESTING HYBRID RETRIEVAL")
print("========================================")

query = "What accuracy did the steganalysis model achieve?"

hybrid_results = hybrid_retriever.invoke(query)

for i, doc in enumerate(hybrid_results[:5]):

    print(f"\n--- HYBRID RESULT {i + 1} ---")

    print("Page:", doc.metadata["page"])

    print(doc.page_content[:800])
    
    

# ============================================
# 16. RAG — LLM ANSWER GENERATION
# ============================================

def answer_question(question):

    print("\n========================================")
    print("RETRIEVING RELEVANT CONTEXT")
    print("========================================")

    # Retrieve relevant chunks
    retrieved_docs = hybrid_retriever.invoke(question)

    print(
        "Retrieved chunks:",
        len(retrieved_docs)
    )

    # ----------------------------------------
    # Build context
    # ----------------------------------------

    context_parts = []

    for doc in retrieved_docs:

        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"[PAGE {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # ----------------------------------------
    # Create RAG prompt
    # ----------------------------------------

    prompt = f"""
You are ResearchLens, a research paper assistant.

Answer the user's question using ONLY the provided
research-paper context.

If the answer cannot be found in the context,
say that the information is not available in
the provided document.

Do not invent facts.

Always mention the relevant page number(s).

RESEARCH PAPER CONTEXT:
-----------------------
{context}
-----------------------

USER QUESTION:
{question}

Provide a clear and concise answer.
"""

    # ----------------------------------------
    # Send context + question to Groq
    # ----------------------------------------

    print("\nGenerating answer...")

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    return answer, retrieved_docs



# ============================================
# 17. TEST RAG
# ============================================

question = "What accuracy did the steganalysis model achieve?"

answer, sources = answer_question(question)

print("\n========================================")
print("FINAL ANSWER")
print("========================================")

print(answer)

print("\n========================================")
print("SOURCE PAGES")
print("========================================")

for doc in sources:

    print(
        "Page:",
        doc.metadata.get("page", "Unknown")
    )
    
    
    
# ============================================================
# TARGET 11 — RETRIEVAL EVALUATION
# ============================================================

import json

print("\n========================================")
print("RETRIEVAL EVALUATION")
print("========================================")


# ------------------------------------------------------------
# 1. Load evaluation dataset
# ------------------------------------------------------------

with open("evaluation_dataset.json", "r", encoding="utf-8") as f:
    evaluation_dataset = json.load(f)


print("Evaluation questions:", len(evaluation_dataset))


# ------------------------------------------------------------
# 2. Create evaluation retriever
# ------------------------------------------------------------
# We retrieve up to 10 candidates from each retriever.
# We will later evaluate the first 1, 3, 5 and 10 results.

evaluation_faiss_retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)

evaluation_bm25_retriever = BM25Retriever.from_documents(
    stored_documents
)
evaluation_bm25_retriever.k = 10


evaluation_hybrid_retriever = EnsembleRetriever(
    retrievers=[
        evaluation_faiss_retriever,
        evaluation_bm25_retriever
    ],
    weights=[
        0.5,
        0.5
    ]
)


# ------------------------------------------------------------
# 3. Evaluation function
# ------------------------------------------------------------

def evaluate_retrieval():

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    total_questions = len(evaluation_dataset)

    detailed_results = []
    

    for item in evaluation_dataset:

        question = item["question"]

        relevant_pages = set(
            item["relevant_pages"]
        )

        # Unanswerable questions have no relevant pages.
        # We skip them for retrieval recall.
        if not relevant_pages:
            continue


        # Retrieve candidates
        retrieved_docs = evaluation_hybrid_retriever.invoke(
            question
        )


        # Extract page numbers
        retrieved_pages = []

        for doc in retrieved_docs:

            page = doc.metadata.get("page")

            if page is not None:
                retrieved_pages.append(page)


        # Remove duplicate pages while preserving order
        unique_pages = []

        for page in retrieved_pages:

            if page not in unique_pages:
                unique_pages.append(page)


        # ----------------------------------------------------
        # Recall@K
        # ----------------------------------------------------

        hit_at_1 = any(
            page in relevant_pages
            for page in unique_pages[:1]
        )

        hit_at_3 = any(
            page in relevant_pages
            for page in unique_pages[:3]
        )

        hit_at_5 = any(
            page in relevant_pages
            for page in unique_pages[:5]
        )

        hit_at_10 = any(
            page in relevant_pages
            for page in unique_pages[:10]
        )


        recall_at_1 += int(hit_at_1)
        recall_at_3 += int(hit_at_3)
        recall_at_5 += int(hit_at_5)
        recall_at_10 += int(hit_at_10)


        detailed_results.append({
            "question": question,
            "relevant_pages": list(relevant_pages),
            "retrieved_pages": unique_pages[:10],
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10
        })


    # --------------------------------------------------------
    # Calculate percentages
    # --------------------------------------------------------

    evaluated_questions = len(detailed_results)

    recall1 = (
        recall_at_1 / evaluated_questions
    ) * 100

    recall3 = (
        recall_at_3 / evaluated_questions
    ) * 100

    recall5 = (
        recall_at_5 / evaluated_questions
    ) * 100

    recall10 = (
        recall_at_10 / evaluated_questions
    ) * 100


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n========================================")
    print("RETRIEVAL RESULTS")
    print("========================================")

    print(
        f"Evaluated questions : {evaluated_questions}"
    )

    print(
        f"Recall@1            : {recall1:.2f}%"
    )

    print(
        f"Recall@3            : {recall3:.2f}%"
    )

    print(
        f"Recall@5            : {recall5:.2f}%"
    )

    print(
        f"Recall@10           : {recall10:.2f}%"
    )
    failed_recall_at_1 = [
        result for result in detailed_results
        if not result["hit_at_1"]
    ]
    print("\n========================================")
    print("RECALL@1 FAILURES")
    print("========================================")

    if not failed_recall_at_1:
        print("No Recall@1 failures.")

    for failure in failed_recall_at_1:

        print("\nQuestion:")
        print(failure["question"])

        print("Relevant pages:")
        print(failure["relevant_pages"])

        print("Retrieved pages:")
        print(failure["retrieved_pages"])

    return detailed_results

    
evaluation_results = evaluate_retrieval()

    # ============================================================
# RETRIEVER COMPARISON
# FAISS vs BM25 vs HYBRID
# ============================================================

print("\n========================================")
print("RETRIEVER COMPARISON")
print("========================================")


def evaluate_retriever(retriever, evaluation_data):

    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    recall_10 = 0

    evaluated = 0

    for item in evaluation_data:

        relevant_pages = item["relevant_pages"]

        # Skip unanswerable questions
        if not relevant_pages:
            continue

        evaluated += 1

        question = item["question"]

        retrieved_docs = retriever.invoke(question)

        # Get unique pages
        retrieved_pages = []

        for doc in retrieved_docs:

            page = doc.metadata.get("page")

            if page not in retrieved_pages:
                retrieved_pages.append(page)

        # Recall@1
        if any(page in relevant_pages
               for page in retrieved_pages[:1]):
            recall_1 += 1

        # Recall@3
        if any(page in relevant_pages
               for page in retrieved_pages[:3]):
            recall_3 += 1

        # Recall@5
        if any(page in relevant_pages
               for page in retrieved_pages[:5]):
            recall_5 += 1

        # Recall@10
        if any(page in relevant_pages
               for page in retrieved_pages[:10]):
            recall_10 += 1

    return {
        "Recall@1": recall_1 / evaluated * 100,
        "Recall@3": recall_3 / evaluated * 100,
        "Recall@5": recall_5 / evaluated * 100,
        "Recall@10": recall_10 / evaluated * 100
    }


# ------------------------------------------------------------
# FAISS
# ------------------------------------------------------------

faiss_results = evaluate_retriever(
    evaluation_faiss_retriever,
    evaluation_dataset
)


# ------------------------------------------------------------
# BM25
# ------------------------------------------------------------

bm25_results = evaluate_retriever(
    evaluation_bm25_retriever,
    evaluation_dataset
)


# ------------------------------------------------------------
# HYBRID
# ------------------------------------------------------------

hybrid_results = evaluate_retriever(
    evaluation_hybrid_retriever,
    evaluation_dataset
)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print("\nRetriever Performance")
print("----------------------------------------")

print(
    f"FAISS   | "
    f"R@1: {faiss_results['Recall@1']:.2f}% | "
    f"R@3: {faiss_results['Recall@3']:.2f}% | "
    f"R@5: {faiss_results['Recall@5']:.2f}% | "
    f"R@10: {faiss_results['Recall@10']:.2f}%"
)

print(
    f"BM25    | "
    f"R@1: {bm25_results['Recall@1']:.2f}% | "
    f"R@3: {bm25_results['Recall@3']:.2f}% | "
    f"R@5: {bm25_results['Recall@5']:.2f}% | "
    f"R@10: {bm25_results['Recall@10']:.2f}%"
)

print(
    f"Hybrid  | "
    f"R@1: {hybrid_results['Recall@1']:.2f}% | "
    f"R@3: {hybrid_results['Recall@3']:.2f}% | "
    f"R@5: {hybrid_results['Recall@5']:.2f}% | "
    f"R@10: {hybrid_results['Recall@10']:.2f}%"
)


# ============================================================
# ANSWER EVALUATION
# ============================================================

print("\n========================================")
print("ANSWER EVALUATION")
print("========================================")


def normalize_text(text):
    """
    Normalize text so that small formatting differences
    do not affect evaluation.
    """

    text = text.lower()

    # Remove common punctuation
    punctuation = ".,!?;:()[]{}\"'"

    for char in punctuation:
        text = text.replace(char, " ")

    # Remove extra spaces
    text = " ".join(text.split())

    return text

def evaluate_answer(question, ground_truth):
    
    # Generate answer using your existing RAG system
    answer, retrieved_docs = answer_question(question)

    normalized_answer = normalize_text(answer)
    normalized_ground_truth = normalize_text(ground_truth)

    # Split ground truth into important words
    ground_truth_words = normalized_ground_truth.split()

    # Count how many ground-truth words appear in the answer
    matched_words = 0

    for word in ground_truth_words:

        if word in normalized_answer:
            matched_words += 1

    if len(ground_truth_words) == 0:
        score = 0

    else:
        score = (
            matched_words /
            len(ground_truth_words)
        ) * 100

    return score, answer


answer_scores = []

for item in evaluation_dataset:

    question = item["question"]
    ground_truth = item["ground_truth"]

    # Skip questions where there is no answer
    if not ground_truth:
        continue

    score, answer = evaluate_answer(
        question,
        ground_truth
    )

    answer_scores.append(score)

    print("\n----------------------------------------")
    print("Question:")
    print(question)

    print("\nExpected:")
    print(ground_truth)

    print("\nGenerated:")
    print(answer)

    print(f"\nScore: {score:.2f}%")
    
    average_answer_score = (
    sum(answer_scores) /
    len(answer_scores)
)

print("\n========================================")
print("ANSWER EVALUATION RESULTS")
print("========================================")

print(
    f"Evaluated answers : {len(answer_scores)}"
)

print(
    f"Average score     : {average_answer_score:.2f}%"
)


# ============================================================
# LLM-BASED ANSWER EVALUATION
# ============================================================

def evaluate_answer_with_llm(question, ground_truth, generated_answer):
    
    evaluation_prompt = f"""
You are evaluating the answer generated by a RAG system
for a research paper.

Evaluate the generated answer against the ground-truth answer.

QUESTION:
{question}

GROUND-TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{generated_answer}

Evaluate the generated answer on:

1. CORRECTNESS
- Does the answer give the correct information?
- Does it agree with the ground truth?

2. COMPLETENESS
- Does it include the important information from the ground truth?

Return ONLY this format:

CORRECTNESS: <score from 0 to 1>
COMPLETENESS: <score from 0 to 1>
REASON: <one short sentence>

Scoring:
1.0 = fully correct
0.5 = partially correct
0.0 = incorrect
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": evaluation_prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
# ============================================================
# FULL LLM ANSWER EVALUATION
# ============================================================

print("\n========================================")
print("FULL LLM ANSWER EVALUATION")
print("========================================")

correctness_scores = []
completeness_scores = []

for index, item in enumerate(evaluation_dataset):

    question = item["question"]
    ground_truth = item["ground_truth"]

    print("\n----------------------------------------")
    print(f"Evaluating question {index + 1}/{len(evaluation_dataset)}")
    print("----------------------------------------")

    # Generate answer using ResearchLens
    generated_answer, retrieved_docs = answer_question(question)

    # Evaluate generated answer
    evaluation_result = evaluate_answer_with_llm(
        question,
        ground_truth,
        generated_answer
    )

    print("\nQuestion:")
    print(question)

    print("\nGenerated Answer:")
    print(generated_answer)

    print("\nLLM Evaluation:")
    print(evaluation_result)

    # Extract scores
    try:
        lines = evaluation_result.splitlines()

        correctness = None
        completeness = None

        for line in lines:

            if line.startswith("CORRECTNESS:"):
                correctness = float(
                    line.split(":")[1].strip()
                )

            elif line.startswith("COMPLETENESS:"):
                completeness = float(
                    line.split(":")[1].strip()
                )

        if correctness is not None:
            correctness_scores.append(correctness)

        if completeness is not None:
            completeness_scores.append(completeness)

    except Exception as e:
        print("Could not parse evaluation score:", e)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========================================")
print("LLM ANSWER EVALUATION RESULTS")
print("========================================")

print(
    f"Questions evaluated : {len(correctness_scores)}"
)

if correctness_scores:

    average_correctness = (
        sum(correctness_scores)
        / len(correctness_scores)
    )

    print(
        f"Average correctness : "
        f"{average_correctness:.2f}"
    )

if completeness_scores:

    average_completeness = (
        sum(completeness_scores)
        / len(completeness_scores)
    )

    print(
        f"Average completeness: "
        f"{average_completeness:.2f}"
    )