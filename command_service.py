"""High level dispatcher executing AI plans."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ai_service import free_chat
import ai_service
import debug_service
import google_service
from action_log import log_action
from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    user_visible_answer: Optional[str] = None
    extra_data: Optional[dict] = None


async def _run_sync(func: Callable, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def chat_handler(
    profile: dict,
    question: str | None = None,
    context_text: str | None = None,
    **params,
):
    text = await free_chat(profile, question=question, context_text=context_text)
    return text


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
    "show_help": debug_service.default_help,
    "debug_on": debug_service.debug_on,
    "debug_off": debug_service.debug_off,
    "debug_status": debug_service.debug_status,
    "chat": chat_handler,
}


def _log_action_safe(profile: dict, method: str, params: dict) -> None:
    action_type_map = {
        "create_personal_task": "TASK_CREATED",
        "update_personal_task": "TASK_UPDATED",
        "create_team_task": "TASK_CREATED",
        "update_team_task": "TASK_UPDATED",
        "create_or_update_calendar_event": "CALENDAR_EVENT_CREATED",
        "write_personal_note": "NOTE_CREATED",
        "update_personal_note": "NOTE_UPDATED",
        "delete_personal_note": "NOTE_DELETED",
    }
    action_type = action_type_map.get(method)
    if not action_type:
        return
    try:
        user_id_raw = profile.get("telegram_user_id", 0)
        try:
            user_id = int(user_id_raw)
        except (TypeError, ValueError):
            logger.debug("Invalid telegram_user_id for logging: %r", user_id_raw)
            return
        log_action(user_id, action_type, params)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to log action", exc_info=True)


async def execute_plan(profile: dict, plan: dict) -> CommandResult:
    """Execute plan returned by AI with threshold handling and safety."""

    try:
        method = plan.get("method") or "chat"
        params = plan.get("params") or {}
        clarify_question = plan.get("clarify_question")
        confidence = float(plan.get("confidence", 0)) if plan.get("confidence") is not None else 0.0
    except Exception:  # noqa: BLE001
        logger.exception("Invalid plan structure: %s", plan)
        return CommandResult(user_visible_answer="Не удалось обработать запрос. Попробуйте ещё раз.")

    if not method:
        method = "chat"
        params.setdefault(
            "question",
            plan.get("original_question")
            or params.get("question")
            or plan.get("user_visible_answer")
            or "",
        )

    # Быстрый путь: если ИИ уже вернул готовый текст для чата, не дергаем модель второй раз
    if method == "chat" and plan.get("user_visible_answer"):
        command_result = CommandResult(user_visible_answer=str(plan["user_visible_answer"]))

        # Сохранить поведение debug-режима: добавить debug-хвост при необходимости
        try:
            debug_user_id = int(profile.get("telegram_user_id", 0))
        except (TypeError, ValueError):
            debug_user_id = None

        if debug_user_id is not None and debug_service.is_debug_enabled(debug_user_id):
            debug_info = (
                f"\n\n[debug] method={method}; confidence={confidence:.2f}; "
                f"params_keys={list((plan.get('params') or {}).keys())}"
            )
            command_result.user_visible_answer = (command_result.user_visible_answer or "") + debug_info

        # Для простого чата можно не писать action_log, чтобы не засорять журнал техническим шумом
        return command_result

    if method == "clarify" or clarify_question:
        question_text = clarify_question or plan.get("user_visible_answer")
        fallback_text = "Я не уверен, что правильно понял запрос. Уточните, пожалуйста."
        return CommandResult(user_visible_answer=str(question_text or fallback_text))

    handler = METHOD_MAP.get(method)
    if handler is None:
        handler = chat_handler
        params = {
            "question": params.get("question") or plan.get("user_visible_answer") or plan.get("original_question") or "",
            "context_text": params.get("context_text") or ai_service.build_context_for_user(profile),
        }

    try:
        if asyncio.iscoroutinefunction(handler):
            result = await handler(profile, **params)
        else:
            result = await _run_sync(handler, profile, **params)
        command_result = result if isinstance(result, CommandResult) else CommandResult(user_visible_answer=str(result))
        _log_action_safe(profile, method, params)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Command failed: %s", exc)
        return CommandResult(
            user_visible_answer="Во время обработки запроса произошла ошибка. Я записал её в лог и продолжу работу.",
            extra_data={"plan": plan},
        )

    try:
        debug_user_id = int(profile.get("telegram_user_id", 0))
    except (TypeError, ValueError):
        debug_user_id = None

    if debug_user_id is not None and debug_service.is_debug_enabled(debug_user_id):
        debug_info = (
            f"\n\n[debug] method={method}; confidence={confidence:.2f}; params_keys={list(params.keys())}"
        )
        command_result.user_visible_answer = (command_result.user_visible_answer or "") + debug_info
    return command_result
