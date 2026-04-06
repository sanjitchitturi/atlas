"""Tests for evaluation metrics."""
import pytest
from app.evaluation.metrics import context_relevance, answer_faithfulness, retrieval_hit


def test_context_relevance_full_match():
    score = context_relevance(
        question="heat exchanger temperature",
        chunk_texts=["The heat exchanger operates at high temperature.", "Pressure valve specs."],
    )
    assert score == 1.0


def test_context_relevance_no_match():
    score = context_relevance(
        question="quantum entanglement theory",
        chunk_texts=["The pump flow rate is 200 L/min.", "Steel frame load capacity 500 kN."],
    )
    assert score == 0.0


def test_context_relevance_empty_chunks():
    assert context_relevance("any question", []) == 0.0


def test_answer_faithfulness_grounded():
    answer = "The heat exchanger operates at 350 degrees Celsius maximum temperature."
    chunks = ["The heat exchanger maximum operating temperature is 350 degrees Celsius."]
    score = answer_faithfulness(answer, chunks)
    assert score > 0.5


def test_answer_faithfulness_ungrounded():
    answer = "The system uses quantum blockchain neural networks for distributed magic."
    chunks = ["The pump moves water through pipes at 50 bar."]
    score = answer_faithfulness(answer, chunks)
    assert score < 0.5


def test_retrieval_hit_found():
    assert retrieval_hit("engineering_overview.txt", ["engineering_overview.txt", "other.pdf"]) == 1


def test_retrieval_hit_not_found():
    assert retrieval_hit("missing_doc.pdf", ["engineering_overview.txt"]) == 0


def test_retrieval_hit_partial_match():
    assert retrieval_hit("overview", ["engineering_overview.txt"]) == 1
