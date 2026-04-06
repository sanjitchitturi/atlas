"""
Script: ingest all documents in sample_data/ into the FAISS index.

Usage:
    python scripts/ingest_sample_data.py
"""
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from app.ingestion.pipeline import ingest_directory
from app.retrieval.embedder import embed_texts
from app.retrieval.index import get_index
from app.core.logging import logger


def main():
    sample_dir = Path("sample_data")
    if not sample_dir.exists():
        logger.error(f"sample_data/ directory not found at {sample_dir.resolve()}")
        sys.exit(1)

    logger.info(f"Ingesting documents from {sample_dir.resolve()}…")
    chunks, meta_list = ingest_directory(sample_dir)

    if not chunks:
        logger.error("No chunks extracted — check that sample_data/ contains PDF/DOCX/TXT/MD files.")
        sys.exit(1)

    logger.info(f"Embedding {len(chunks)} chunks…")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    index = get_index()
    index.add(embeddings, chunks)
    index.save()

    logger.info("=" * 50)
    logger.info(f"Ingestion complete!")
    logger.info(f"  Documents : {len(meta_list)}")
    logger.info(f"  Chunks    : {len(chunks)}")
    logger.info(f"  Index size: {index.size}")
    for m in meta_list:
        logger.info(f"  • {m.source} ({m.num_chunks} chunks)")
    logger.info("=" * 50)
    logger.info("You can now run the API and Streamlit frontend.")


if __name__ == "__main__":
    main()
