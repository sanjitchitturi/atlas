"""Central settings loaded from environment / .env file."""

from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("./data/documents")
    index_dir: Path = Path("./data/index")
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = Field(512, ge=1, le=4096)
    chunk_overlap: int = Field(64, ge=0)
    top_k: int = 5
    generation_mode: Literal["local", "openai", "anthropic"] = "local"
    embedding_backend: Literal["hash", "sentence-transformers"] = "hash"
    max_upload_mb: int = Field(25, ge=1, le=100)
    openai_api_key: str = Field("", repr=False)
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = Field("", repr=False)
    anthropic_model: str = "claude-3-haiku-20240307"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
