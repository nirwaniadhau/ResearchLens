import json
import re


def load_evaluation_dataset(dataset_path):
    """
    Load the evaluation dataset.
    """

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def normalize_text(text):
    """
    Normalize text for lexical comparison.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s%.-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def lexical_overlap_score(
    answer,
    ground_truth
):
    """
    Calculate a simple lexical overlap score.

    This is a baseline metric and does NOT
    represent factual accuracy.
    """

    answer_tokens = set(
        normalize_text(answer).split()
    )

    truth_tokens = set(
        normalize_text(ground_truth).split()
    )

    if not truth_tokens:
        return 0.0

    overlap = (
        answer_tokens.intersection(
            truth_tokens
        )
    )

    score = (
        len(overlap) /
        len(truth_tokens)
    ) * 100

    return score


def evaluate_answers_lexically(
    answers,
    evaluation_dataset
):
    """
    Evaluate generated answers using
    lexical overlap.
    """

    results = []

    for item in evaluation_dataset:

        question_id = item["id"]

        if question_id not in answers:
            continue

        answer = answers[question_id]

        ground_truth = item["ground_truth"]

        score = lexical_overlap_score(
            answer,
            ground_truth
        )

        results.append({
            "id": question_id,
            "question": item["question"],
            "answer": answer,
            "ground_truth": ground_truth,
            "score": score
        })

    if results:

        average_score = (
            sum(
                item["score"]
                for item in results
            )
            / len(results)
        )

    else:
        average_score = 0.0

    return results, average_score


def evaluate_answer_with_llm(
    question,
    answer,
    ground_truth,
    groq_client
):
    """
    Use an LLM as a judge for answer correctness
    and completeness.
    """

    prompt = f"""
You are evaluating an answer produced by a
research-paper question answering system.

Evaluate the answer against the provided
ground truth.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{answer}

Return exactly this format:

CORRECTNESS: <0 to 1>
COMPLETENESS: <0 to 1>
REASON: <short explanation>

Correctness measures whether the answer is
factually consistent with the ground truth.

Completeness measures whether the answer
contains the important information expected
from the ground truth.
"""

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

    return response.choices[0].message.content