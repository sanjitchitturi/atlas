"""
Script: run a quick demo query from the command line (no Streamlit needed).

Usage:
    python scripts/demo_query.py "What is the operating pressure limit?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.pipeline import answer_question
from app.models.schemas import QueryRequest
from app.retrieval.index import get_index


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the main topic of the documents?"

    index = get_index()
    if index.size == 0:
        print("❌  No documents indexed. Run: python scripts/ingest_sample_data.py")
        sys.exit(1)

    print(f"\n🔍 Question: {question}\n")
    result = answer_question(QueryRequest(question=question))

    print("💡 Answer:")
    print(f"   {result.answer}\n")
    print("📚 Sources:")
    for i, src in enumerate(result.sources, 1):
        print(f"   [{i}] {src.source} (page {src.page}, chunk #{src.chunk_index})")
        print(f"       {src.snippet[:120]}…")
    print(f"\nMode: {result.mode}")


if __name__ == "__main__":
    main()
