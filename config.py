"""Configuration utilities for environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Thresholds:
    """Confidence thresholds for AI decisions."""

    high: float = 0.75
    low: float = 0.4


@dataclass(frozen=True)
class Config:
    """Global configuration loaded from environment variables."""

    telegram_token: str
    google_project_id: Optional[str]
    sheets_id: str
    calendar_id: Optional[str]
    google_credentials_file: Optional[Path]
    dialog_log_path: Path
    ai_model: str
    thresholds: Thresholds = Thresholds()

    @staticmethod
    def load() -> "Config":
        token = os.getenv("TELEGRAM_TOKEN")
        if not token:
            raise RuntimeError("TELEGRAM_TOKEN is required in environment")
        sheets_id = os.getenv("GOOGLE_SHEETS_ID")
        if not sheets_id:
            raise RuntimeError("GOOGLE_SHEETS_ID is required in environment")
        calendar_id = os.getenv("GOOGLE_CALENDAR_ID")
        creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        log_path = Path(os.getenv("DIALOG_LOG_PATH", "dialog_log.jsonl")).resolve()
        ai_model = os.getenv("GENAI_MODEL", "gemini-3-pro-preview")
        return Config(
            telegram_token=token,
            google_project_id=os.getenv("GOOGLE_PROJECT_ID"),
            sheets_id=sheets_id,
            calendar_id=calendar_id,
            google_credentials_file=Path(creds_file) if creds_file else None,
            dialog_log_path=log_path,
            ai_model=ai_model,
        )


CONFIG = Config.load()
