"""Run the evaluation pipeline over a QA dataset JSON file."""
from __future__ import annotations
import json
from pathlib import Path
from app.retrieval.retriever import retrieve
from app.generation.pipeline import answer_question
from app.evaluation.metrics import context_relevance, answer_faithfulness, retrieval_hit
from app.models.schemas import QueryRequest
from app.core.logging import logger


def evaluate(dataset_path: Path) -> list[dict]:
    """
    dataset_path: JSON file with list of:
        { "question": "...", "expected_source": "doc.pdf", "reference_answer": "..." }

    Returns a list of per-example result dicts.
    """
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = []

    for i, example in enumerate(data):
        question = example["question"]
        expected_source = example.get("expected_source", "")

        # Retrieve
        retrieved = retrieve(question)
        chunk_texts = [c.text for c, _ in retrieved]
        retrieved_sources = [c.source for c, _ in retrieved]

        # Generate
        req = QueryRequest(question=question)
        response = answer_question(req)

        # Metrics
        cr = context_relevance(question, chunk_texts)
        af = answer_faithfulness(response.answer, chunk_texts)
        rh = retrieval_hit(expected_source, retrieved_sources) if expected_source else None

        result = {
            "id": i + 1,
            "question": question,
            "answer": response.answer,
            "context_relevance": cr,
            "answer_faithfulness": af,
            "retrieval_hit": rh,
            "retrieved_sources": retrieved_sources,
        }
        results.append(result)
        logger.info(f"[{i+1}/{len(data)}] CR={cr:.2f} AF={af:.2f} RH={rh}")

    # Summary
    avg_cr = sum(r["context_relevance"] for r in results) / len(results)
    avg_af = sum(r["answer_faithfulness"] for r in results) / len(results)
    hits = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    avg_rh = sum(hits) / len(hits) if hits else None

    logger.info(
        f"=== Evaluation Summary === "
        f"Avg CR={avg_cr:.3f} | Avg AF={avg_af:.3f} | Avg RH={avg_rh}"
    )

    return results
