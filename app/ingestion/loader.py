"""Load raw text from PDF, DOCX, TXT, and MD files."""
from __future__ import annotations
from pathlib import Path
from app.core.logging import logger


def load_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples for any supported file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix == ".docx":
        return _load_docx(path)
    elif suffix in (".txt", ".md", ".markdown"):
        return _load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    logger.debug(f"PDF '{path.name}': {len(pages)} pages extracted")
    return pages


def _load_docx(path: Path) -> list[tuple[int, str]]:
    from docx import Document
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    logger.debug(f"DOCX '{path.name}': {len(text)} chars extracted")
    return [(1, text)]


def _load_text(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    logger.debug(f"TXT/MD '{path.name}': {len(text)} chars")
    return [(1, text)]
