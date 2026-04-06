"""
Script: run evaluation pipeline over evaluation_data/qa_dataset.json.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --dataset evaluation_data/qa_dataset.json
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.evaluator import evaluate
from app.core.logging import logger


def main():
    parser = argparse.ArgumentParser(description="Run Atlas RAG evaluation")
    parser.add_argument(
        "--dataset",
        default="evaluation_data/qa_dataset.json",
        help="Path to QA dataset JSON file",
    )
    parser.add_argument(
        "--output",
        default="evaluation_data/results.json",
        help="Where to save evaluation results",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    logger.info(f"Running evaluation on {dataset_path} …")
    results = evaluate(dataset_path)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'#':<4} {'Question':<40} {'CR':>6} {'AF':>6} {'RH':>4}")
    print("-" * 70)
    for r in results:
        rh = str(r["retrieval_hit"]) if r["retrieval_hit"] is not None else "N/A"
        print(
            f"{r['id']:<4} {r['question'][:38]:<40} "
            f"{r['context_relevance']:>6.2f} {r['answer_faithfulness']:>6.2f} {rh:>4}"
        )
    print("=" * 70)
    avg_cr = sum(r["context_relevance"] for r in results) / len(results)
    avg_af = sum(r["answer_faithfulness"] for r in results) / len(results)
    print(f"{'AVERAGE':<44} {avg_cr:>6.2f} {avg_af:>6.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
