"""Debug flag handling for users."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Dict
from zoneinfo import ZoneInfo

import google_service

logger = logging.getLogger(__name__)

_user_debug_flags: Dict[int, bool] = {}


def is_debug_enabled(user_id: int) -> bool:
    return _user_debug_flags.get(user_id, False)


def debug_on(profile: dict, **_: object) -> str:
    try:
        user_id = int(profile.get("telegram_user_id"))
    except (TypeError, ValueError):
        logger.warning("Invalid telegram_user_id for debug_on: %r", profile.get("telegram_user_id"))
        return "Не удалось включить debug режим: некорректный идентификатор пользователя."
    _user_debug_flags[user_id] = True
    logger.info("Debug enabled for user %s", user_id)
    return "Debug режим включен."


def debug_off(profile: dict, **_: object) -> str:
    try:
        user_id = int(profile.get("telegram_user_id"))
    except (TypeError, ValueError):
        logger.warning("Invalid telegram_user_id for debug_off: %r", profile.get("telegram_user_id"))
        return "Не удалось выключить debug режим: некорректный идентификатор пользователя."
    _user_debug_flags[user_id] = False
    logger.info("Debug disabled for user %s", user_id)
    return "Debug режим выключен."


def debug_status(profile: dict, **_: object) -> str:
    try:
        user_id = int(profile.get("telegram_user_id"))
    except (TypeError, ValueError):
        logger.warning("Invalid telegram_user_id for debug_status: %r", profile.get("telegram_user_id"))
        return "Не удалось определить статус debug режима."
    return f"Debug режим {'включен' if is_debug_enabled(user_id) else 'выключен'}."


def create_test_calendar_event(profile: dict, **_: object) -> str:
    tz = ZoneInfo(profile.get("timezone") or "Europe/Moscow")
    now = dt.datetime.now(tz)
    start = now + dt.timedelta(minutes=10)
    end = start + dt.timedelta(minutes=30)
    response = google_service.create_or_update_event(
        profile,
        title="Тестовое событие",
        description="Проверка интеграции календаря",
        start_datetime=start.isoformat(),
        end_datetime=end.isoformat(),
    )
    return f"{response}\nНачало: {start.isoformat()}\nОкончание: {end.isoformat()}"


def show_today_agenda(profile: dict, **_: object) -> str:
    tz = ZoneInfo(profile.get("timezone") or "Europe/Moscow")
    today = dt.datetime.now(tz).date()
    start = dt.datetime.combine(today, dt.time.min, tzinfo=tz)
    end = dt.datetime.combine(today, dt.time.max, tzinfo=tz)
    return google_service.show_calendar_agenda(profile, start.isoformat(), end.isoformat())


async def default_help(profile: dict, **_: object) -> str:
    return (
        "Я умею создавать и обновлять личные и командные задачи, заметки и события в календаре. "
        "Примеры: 'создай задачу позвонить клиенту завтра в 18:00', "
        "'добавь заметку про бюджет', 'покажи мои задачи', 'добавь событие завтра в 10:00 встреча с командой'."
    )
