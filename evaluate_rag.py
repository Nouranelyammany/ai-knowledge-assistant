
import json
import csv
from pathlib import Path
from datetime import datetime

from rag_pipeline import (
    initialize_components,
    answer_question,
)

PROJECT_DIR = Path(__file__).resolve().parent
TEST_FILE = PROJECT_DIR / "test_questions.json"
REPORT_FILE = PROJECT_DIR / "evaluation_results.csv"


def evaluate_retrieval(passages, expected_source):
    """Check whether the expected PDF appears in retrieved sources."""
    if expected_source is None:
        return "N/A"

    for passage in passages:
        if passage["source"] == expected_source:
            return "PASS"

    return "FAIL"


def evaluate_abstention(answer, should_abstain):
    """Basic check for unsupported-question abstention."""
    if not should_abstain:
        return "N/A"

    abstention_phrases = [
        "couldn't find sufficiently relevant information",
        "do not contain enough information",
        "don't contain enough information",
        "not enough information",
        "cannot find",
        "can't find",
        "unable to answer",
        "do not provide",
        "does not provide",
        "not available in the documents",
        "not in the documents",
    ]

    answer_lower = answer.lower()

    if any(phrase in answer_lower for phrase in abstention_phrases):
        return "PASS"

    return "REVIEW"


def main():
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_questions = json.load(file)

    print("Initializing RAG components...")

    client, embedding_model, collection = initialize_components()

    results = []

    for index, test in enumerate(test_questions, start=1):
        question = test["question"]

        print(f"\n[{index}/{len(test_questions)}] {question}")

        try:
            answer, passages = answer_question(
                question,
                client,
                embedding_model,
                collection,
            )

            retrieval_status = evaluate_retrieval(
                passages,
                test["expected_source"],
            )

            abstention_status = evaluate_abstention(
                answer,
                test["should_abstain"],
            )

            result = {
                "id": test["id"],
                "question": question,
                "expected_source": test["expected_source"] or "",
                "retrieved_sources": "; ".join(
                    sorted({p["source"] for p in passages})
                ),
                "retrieval_status": retrieval_status,
                "should_abstain": test["should_abstain"],
                "abstention_status": abstention_status,
                "answer": answer,
                "error": "",
            }

        except Exception as error:
            result = {
                "id": test["id"],
                "question": question,
                "expected_source": test["expected_source"] or "",
                "retrieved_sources": "",
                "retrieval_status": "ERROR",
                "should_abstain": test["should_abstain"],
                "abstention_status": "ERROR",
                "answer": "",
                "error": str(error),
            }

            print(f"ERROR: {error}")

        results.append(result)

        print(f"Retrieval: {result['retrieval_status']}")
        print(f"Abstention: {result['abstention_status']}")

    fieldnames = list(results[0].keys())

    with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    answerable = [
        result for result in results
        if result["expected_source"]
    ]

    retrieval_passes = sum(
        result["retrieval_status"] == "PASS"
        for result in answerable
    )

    unsupported = [
        result for result in results
        if result["should_abstain"]
    ]

    abstention_passes = sum(
        result["abstention_status"] == "PASS"
        for result in unsupported
    )

    print("\n" + "=" * 45)
    print("RAG EVALUATION SUMMARY")
    print("=" * 45)
    print(f"Total questions: {len(results)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Correct-source retrieval: {retrieval_passes}/{len(answerable)}")
    print(f"Unsupported questions: {len(unsupported)}")
    print(f"Abstention checks passed: {abstention_passes}/{len(unsupported)}")
    print(f"CSV report saved to: {REPORT_FILE}")
    print("\nReview evaluation_results.csv for individual results.")


if __name__ == "__main__":
    main()
