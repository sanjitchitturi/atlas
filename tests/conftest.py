"""Shared pytest fixtures."""
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def ensure_logs_dir():
    """Make sure the logs/ directory exists so the logger doesn't crash during tests."""
    Path("logs").mkdir(exist_ok=True)
