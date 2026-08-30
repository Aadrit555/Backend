"""Configuration — BIBLE §39, ARCHITECTURE.md Locked Decisions.

Central settings for the Unified platform.  Local-first, single-user:
all paths resolve under one storage root, no cloud config, no auth
secrets beyond the Groq API key.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Loaded from environment / .env file."""

    # --- Groq orchestrator ---
    groq_api_key_1: str = ""
    groq_api_key_2: str = ""
    groq_api_key_3: str = ""
    groq_api_key_4: str = ""
    groq_api_key_5: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # --- Storage (ARCHITECTURE.md §6) ---
    storage_root: Path = Path(__file__).resolve().parent / "storage"

    # --- Database ---
    database_url: str = ""  # computed in model_post_init

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Cloud Models ---
    openrouter_api_key: str = ""

    model_config = {"env_file": str(Path(__file__).resolve().parent / ".env"), "env_file_encoding": "utf-8", "extra": "ignore"}

    def model_post_init(self, __context: object) -> None:
        if not self.database_url:
            db_path = self.storage_root / "db" / "unified.sqlite3"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{db_path}"

    # --- Derived paths ---
    @property
    def raw_dir(self) -> Path:
        return self.storage_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.storage_root / "processed"

    @property
    def models_dir(self) -> Path:
        return self.storage_root / "models"

    @property
    def experiments_dir(self) -> Path:
        return self.storage_root / "experiments"

    @property
    def logs_dir(self) -> Path:
        return self.storage_root / "logs"


settings = Settings()
