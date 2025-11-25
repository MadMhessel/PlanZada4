"""High level dispatcher executing AI plans."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import ai_service
import debug_service
import google_service
from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    user_visible_answer: Optional[str] = None
    extra_data: Optional[dict] = None


async def _run_sync(func: Callable, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def _default_help() -> str:
    return (
        "Я могу создавать и обновлять личные и командные задачи, заметки и события в календаре. "
        "Например: 'создай задачу позвонить клиенту завтра в 18:00', "
        "'добавь заметку про бюджет', 'покажи мои задачи'."
    )


METHOD_MAP: Dict[str, Callable] = {
    "write_personal_note": google_service.create_personal_note,
    "read_personal_notes": google_service.read_personal_notes,
    "search_personal_notes": google_service.search_personal_notes,
    "update_personal_note": google_service.update_personal_note,
    "delete_personal_note": google_service.delete_personal_note,
    "create_personal_task": google_service.create_personal_task,
    "update_personal_task": google_service.update_personal_task,
    "list_personal_tasks": google_service.list_personal_tasks,
    "create_team_task": google_service.create_team_task,
    "update_team_task": google_service.update_team_task,
    "list_team_tasks": google_service.list_team_tasks,
    "create_or_update_calendar_event": google_service.create_or_update_event,
    "show_calendar_agenda": google_service.show_calendar_agenda,
    "show_help": _default_help,
    "debug_on": debug_service.debug_on,
    "debug_off": debug_service.debug_off,
    "debug_status": debug_service.debug_status,
}


async def execute_plan(profile: dict, plan: dict) -> CommandResult:
    """Execute plan returned by AI with threshold handling."""
    method = plan.get("method") or "clarify"
    confidence = float(plan.get("confidence", 0))
    params = plan.get("params") or {}
    clarify_question = plan.get("clarify_question")

    if confidence < CONFIG.thresholds.low:
        return CommandResult(user_visible_answer=clarify_question or "Нужно уточнить запрос.", extra_data=plan)

    if CONFIG.thresholds.low <= confidence < CONFIG.thresholds.high and clarify_question:
        return CommandResult(user_visible_answer=clarify_question, extra_data=plan)

    handler = METHOD_MAP.get(method)
    if handler:
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(profile, **params)
            else:
                result = await _run_sync(handler, profile, **params)
            if isinstance(result, CommandResult):
                return result
            if isinstance(result, str):
                return CommandResult(user_visible_answer=result, extra_data=plan)
            return CommandResult(user_visible_answer=plan.get("user_visible_answer") or "Готово.", extra_data=result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Command failed: %s", exc)
            return CommandResult(user_visible_answer=f"Ошибка выполнения: {exc}")

    # fallback: chat
    chat_answer = await ai_service.free_chat(plan.get("user_visible_answer") or "Расскажи подробнее", profile)
    return CommandResult(user_visible_answer=chat_answer)
