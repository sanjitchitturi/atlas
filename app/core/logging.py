"""Loguru-based logging setup."""
import sys
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> – <level>{message}</level>",
        colorize=True,
    )
    logger.add(
        "logs/atlas.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


setup_logging()

__all__ = ["logger"]
