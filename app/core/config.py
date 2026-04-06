"""Central settings loaded from environment / .env file."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("./data/documents")
    index_dir: Path = Path("./data/index")
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    generation_mode: str = "local"
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
