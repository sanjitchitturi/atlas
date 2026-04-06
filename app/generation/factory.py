"""Return the correct generator based on GENERATION_MODE setting."""
from __future__ import annotations
from app.generation.base import BaseGenerator
from app.core.config import settings
from app.core.logging import logger


def get_generator() -> BaseGenerator:
    mode = settings.generation_mode.lower()
    logger.info(f"Generator mode: {mode}")

    if mode == "openai":
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set — falling back to local mode")
            mode = "local"
        else:
            from app.generation.openai_generator import OpenAIGenerator
            return OpenAIGenerator()

    if mode == "anthropic":
        if not settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set — falling back to local mode")
            mode = "local"
        else:
            from app.generation.anthropic_generator import AnthropicGenerator
            return AnthropicGenerator()

    from app.generation.local_generator import LocalGenerator
    return LocalGenerator()
