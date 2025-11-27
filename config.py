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
    calendar_provider: str
    yandex_caldav_url: Optional[str]
    yandex_calendar_login: Optional[str]
    yandex_calendar_password: Optional[str]
    yandex_calendar_name: Optional[str]
    dialog_log_path: Path
    ai_model: str
    ai_high_confidence: float = 0.75
    ai_low_confidence: float = 0.40
    reminder_interval_seconds: int = 300

    @property
    def AI_MODEL(self) -> str:
        """Совместимое имя модели для вызова ИИ (верхний регистр для удобства)."""

        return self.ai_model

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
        calendar_provider = os.getenv("CALENDAR_PROVIDER", "yandex").lower()
        yandex_caldav_url = os.getenv("YANDEX_CALDAV_URL", "https://calendar.yandex.ru/")
        yandex_calendar_login = os.getenv("YANDEX_CALENDAR_LOGIN")
        yandex_calendar_password = os.getenv("YANDEX_CALENDAR_PASSWORD")
        yandex_calendar_name = os.getenv("YANDEX_CALENDAR_NAME")
        project_root = Path(__file__).resolve().parent
        log_path = Path(os.getenv("DIALOG_LOG_PATH", project_root / "dialog_log.jsonl")).resolve()
        ai_model = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
        reminder_interval = int(os.getenv("REMINDER_INTERVAL_SECONDS", "300"))
        return Config(
            telegram_token=token,
            google_project_id=os.getenv("GOOGLE_PROJECT_ID"),
            sheets_id=sheets_id,
            calendar_id=calendar_id,
            google_credentials_file=Path(creds_file) if creds_file else None,
            calendar_provider=calendar_provider,
            yandex_caldav_url=yandex_caldav_url,
            yandex_calendar_login=yandex_calendar_login,
            yandex_calendar_password=yandex_calendar_password,
            yandex_calendar_name=yandex_calendar_name,
            dialog_log_path=log_path,
            ai_model=ai_model,
            reminder_interval_seconds=reminder_interval,
        )


CONFIG = Config.load()
