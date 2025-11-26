"""High level dispatcher executing AI plans."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

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
    "chat": ai_service.free_chat,
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
        log_action(int(profile.get("telegram_user_id", 0)), action_type, params)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to log action", exc_info=True)


async def execute_plan(profile: dict, plan: dict) -> CommandResult:
    """Execute plan returned by AI with threshold handling and safety."""
    try:
        method = plan.get("method") or "chat"
        confidence = float(plan.get("confidence", 0))
        params = plan.get("params") or {}
        clarify_question = plan.get("clarify_question")
    except Exception:  # noqa: BLE001
        logger.exception("Invalid plan structure: %s", plan)
        return CommandResult(user_visible_answer="Не удалось обработать запрос. Попробуйте ещё раз.")

    if method == "clarify":
        question = plan.get("user_visible_answer") or clarify_question or "Нужны уточнения, чтобы продолжить."
        return CommandResult(user_visible_answer=str(question))

    if clarify_question:
        return CommandResult(user_visible_answer=str(clarify_question))

    if confidence < CONFIG.ai_low_confidence:
        return CommandResult(
            user_visible_answer="Я не до конца уверен, как лучше обработать запрос. Попробуйте переформулировать или уточнить детали.",
            extra_data=plan,
        )

    handler = METHOD_MAP.get(method)
    if handler:
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
                user_visible_answer="Во время обработки запроса произошла ошибка. Я записал её в лог и попробую продолжить работу.",
            )
    else:
        context_text = ai_service.build_context_for_user(profile)
        chat_answer = await ai_service.free_chat(
            profile, question=plan.get("user_visible_answer") or params.get("question", ""), context_text=context_text
        )
        command_result = CommandResult(user_visible_answer=chat_answer)

    if debug_service.is_debug_enabled(int(profile.get("telegram_user_id", 0))):
        debug_info = (
            f"\n\n[debug] method={method}; topic={plan.get('topic')}; "
            f"confidence={confidence:.2f}; params_keys={list(params.keys())}"
        )
        command_result.user_visible_answer = (command_result.user_visible_answer or "") + debug_info
    return command_result
