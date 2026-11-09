"""File utility helpers."""

from __future__ import annotations
from uuid import uuid4
from pathlib import Path
from app.core.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_upload(file_bytes: bytes, filename: str) -> Path:
    name = Path(filename.replace("\\", "/")).name
    if not name or not is_allowed(name):
        raise ValueError("Unsupported filename")
    dest = settings.data_dir / (uuid4().hex + "_" + name)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)
    return dest


def list_documents() -> list[str]:
    if not settings.data_dir.exists():
        return []
    return sorted(
        f.name
        for f in settings.data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )
