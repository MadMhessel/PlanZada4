"""Configuration utilities for environment-driven settings.

The module centralises all environment-driven settings and threshold values
to keep the rest of the code declarative. Paths are resolved relative to the
project root to make the bot resilient for auto-start scenarios where the
working directory may differ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


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
    ai_high_confidence: float = 0.75
    ai_low_confidence: float = 0.40
    reminder_interval_seconds: int = 300

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
        project_root = Path(__file__).resolve().parent
        log_path = Path(os.getenv("DIALOG_LOG_PATH", project_root / "dialog_log.jsonl")).resolve()
        ai_model = os.getenv("GENAI_MODEL", "gemini-3-pro-preview")
        reminder_interval = int(os.getenv("REMINDER_INTERVAL_SECONDS", "300"))
        return Config(
            telegram_token=token,
            google_project_id=os.getenv("GOOGLE_PROJECT_ID"),
            sheets_id=sheets_id,
            calendar_id=calendar_id,
            google_credentials_file=Path(creds_file) if creds_file else None,
            dialog_log_path=log_path,
            ai_model=ai_model,
            reminder_interval_seconds=reminder_interval,
        )


CONFIG = Config.load()
