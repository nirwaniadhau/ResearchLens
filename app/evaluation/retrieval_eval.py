import json


def load_evaluation_dataset(dataset_path):
    """
    Load the evaluation questions from JSON.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    retriever,
    evaluation_dataset
):
    """
    Evaluate retrieval using Recall@K.

    The evaluation dataset should contain:
        question
        relevant_pages
    """

    results = []

    for item in evaluation_dataset:

        question = item["question"]
        relevant_pages = set(
            item.get("relevant_pages", [])
        )

        # Skip questions with no known relevant pages
        if not relevant_pages:
            continue

        retrieved_docs = retriever.invoke(question)

        retrieved_pages = []

        for doc in retrieved_docs:
            page = doc.metadata.get("page")

            if page is not None:
                retrieved_pages.append(page)

        results.append({
            "question": question,
            "relevant_pages": relevant_pages,
            "retrieved_pages": retrieved_pages
        })

    return results


def calculate_recall_at_k(
    results,
    k
):
    """
    Calculate Recall@K.

    Returns the percentage of questions
    for which at least one relevant page
    appears in the top K retrieved results.
    """

    if not results:
        return 0.0

    hits = 0

    for result in results:

        relevant_pages = result["relevant_pages"]

        retrieved_pages = result["retrieved_pages"][:k]

        if any(
            page in relevant_pages
            for page in retrieved_pages
        ):
            hits += 1

    recall = (
        hits / len(results)
    ) * 100

    return recall


def run_retrieval_evaluation(
    retriever,
    dataset_path
):
    """
    Run complete retrieval evaluation.
    """

    dataset = load_evaluation_dataset(
        dataset_path
    )

    results = evaluate_retrieval(
        retriever,
        dataset
    )

    metrics = {
        "Recall@1": calculate_recall_at_k(
            results, 1
        ),
        "Recall@3": calculate_recall_at_k(
            results, 3
        ),
        "Recall@5": calculate_recall_at_k(
            results, 5
        ),
        "Recall@10": calculate_recall_at_k(
            results, 10
        )
    }

    return metrics, results