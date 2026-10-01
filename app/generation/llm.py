import os
from groq import Groq


def create_groq_client():
    """
    Create and return the Groq client.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")

    client = Groq(
        api_key=groq_api_key
    )

    return client


def answer_question(
    question,
    hybrid_retriever,
    groq_client,
    chat_history=None
):
    """
    Answer a question using conversational RAG.

    Previous conversation is included during retrieval
    and answer generation so follow-up questions can
    refer to earlier messages.
    """

    if chat_history is None:
        chat_history = []

    # ========================================================
    # BUILD CONTEXTUAL RETRIEVAL QUERY
    # ========================================================

    retrieval_query = question

    if chat_history:

        recent_history = chat_history[-6:]

        history_text = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in recent_history
            ]
        )

        retrieval_query = f"""
Previous conversation:

{history_text}

Current question:

{question}
"""

    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    retrieved_docs = hybrid_retriever.invoke(
        retrieval_query
    )

    # ========================================================
    # BUILD RESEARCH PAPER CONTEXT
    # ========================================================

    context_parts = []

    for doc in retrieved_docs:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"[PAGE {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(
        context_parts
    )

    # ========================================================
    # BUILD CHAT HISTORY
    # ========================================================

    conversation_text = ""

    if chat_history:

        conversation_text = "\n".join(
            [
                f"{message['role'].upper()}: "
                f"{message['content']}"
                for message in chat_history[-6:]
            ]
        )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are ResearchLens, a research paper assistant.

Answer the user's question using ONLY the provided
research-paper context.

You are having a conversation with the user, so use
the previous conversation to understand follow-up
questions and references.

If the answer cannot be found in the research-paper
context, say that the information is not available
in the provided document.

Do not invent facts.

Always mention the relevant page number(s).

PREVIOUS CONVERSATION:
----------------------
{conversation_text}
----------------------

RESEARCH PAPER CONTEXT:
-----------------------
{context}
-----------------------

CURRENT USER QUESTION:
{question}

Provide a clear and concise answer.
"""

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

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