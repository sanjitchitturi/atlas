"""Text cleaning utilities."""
import re


def clean_text(text: str) -> str:
    """Remove noise: excessive whitespace, control chars, repeated punctuation."""
    # Normalize unicode whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove non-printable control characters (keep newlines/tabs)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
