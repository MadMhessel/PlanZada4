"""Debug flag handling for users."""
from __future__ import annotations

import logging
from typing import Dict

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


async def default_help(profile: dict, **_: object) -> str:
    return (
        "Я умею создавать и обновлять личные и командные задачи, заметки и события в календаре. "
        "Примеры: 'создай задачу позвонить клиенту завтра в 18:00', "
        "'добавь заметку про бюджет', 'покажи мои задачи', 'добавь событие завтра в 10:00 встреча с командой'."
    )
