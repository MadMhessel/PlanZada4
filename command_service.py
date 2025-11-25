"""High-level routing between AI decisions and storage."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

from ai_schemas import Classification, PlannedAction
from crypto_service import decrypt_text, encrypt_text
import google_service

logger = logging.getLogger(__name__)

user_sessions: Dict[int, str] = {}


def login(user_id: int, password: str) -> str:
    user_sessions[user_id] = password
    return "Пароль сохранён для этой сессии."


def logout(user_id: int) -> str:
    user_sessions.pop(user_id, None)
    return "Пароль удалён из сессии."


def _require_password(user_id: int) -> str | None:
    if user_id not in user_sessions:
        return "Установите пароль командой /login <пароль> для доступа к личным данным."
    return None


def process(classification: Classification, action: PlannedAction, user_id: int) -> Tuple[str, Dict]:
    google_service.ensure_structures()
    debug_meta = {
        "kind": classification.kind.value,
        "topic": classification.topic.value,
        "confidence": classification.confidence,
        "method": action.method,
        "params": action.params,
    }
    method = action.method
    params = action.safe_params()
    try:
        if method == "write_personal_note":
            msg = _require_password(user_id)
            if msg:
                return msg, debug_meta
            encrypted = encrypt_text(user_sessions[user_id], params.get("text", ""))
            note_id = google_service.add_personal_note(user_id, encrypted, ",".join(params.get("tags", [])))
            return f"Заметка сохранена (id={note_id}).", debug_meta
        if method == "read_personal_notes":
            msg = _require_password(user_id)
            if msg:
                return msg, debug_meta
            notes = google_service.list_personal_notes(user_id, limit=int(params.get("limit", 5)))
            if not notes:
                return "Заметок пока нет.", debug_meta
            out_lines = []
            for n in notes:
                try:
                    text = decrypt_text(user_sessions[user_id], n["text"])
                except ValueError:
                    text = "Не удалось расшифровать"
                out_lines.append(f"• {text} (теги: {n['tags']})")
            return "\n".join(out_lines), debug_meta
        if method == "search_personal_notes":
            msg = _require_password(user_id)
            if msg:
                return msg, debug_meta
            results = google_service.search_personal_notes(user_id, params.get("search_query", ""))
            if not results:
                return "Ничего не найдено.", debug_meta
            out = []
            for n in results:
                try:
                    text = decrypt_text(user_sessions[user_id], n["text"])
                except ValueError:
                    text = "Не удалось расшифровать"
                out.append(f"• {text}")
            return "\n".join(out), debug_meta
        if method == "write_task":
            msg = _require_password(user_id)
            if params.get("scope") == "personal":
                if msg:
                    return msg, debug_meta
                event_id = google_service.create_or_update_calendar_event(
                    params.get("title", ""), params.get("description", ""), params.get("due_date", ""), None
                ) if params.get("add_to_calendar") else None
                task_id = google_service.add_personal_task(user_id, {**params, "calendar_event_id": event_id})
                return f"Личная задача создана (id={task_id}).", debug_meta
            event_id = google_service.create_or_update_calendar_event(
                params.get("title", ""), params.get("description", ""), params.get("due_date", ""), None
            ) if params.get("add_to_calendar") else None
            task_id = google_service.add_team_task({**params, "calendar_event_id": event_id})
            return f"Командная задача создана (id={task_id}).", debug_meta
        if method == "update_task_status":
            if params.get("scope") == "personal":
                ok = google_service.update_personal_task_status(params.get("id", ""), params.get("status", "todo"))
            else:
                ok = google_service.update_team_task_status(params.get("id", ""), params.get("status", "todo"))
            return ("Статус обновлён." if ok else "Задача не найдена."), debug_meta
        if method == "update_task_due":
            ok = google_service.update_personal_task_due(params.get("id", ""), params.get("due_date", ""))
            return ("Срок обновлён." if ok else "Задача не найдена."), debug_meta
        if method == "list_tasks":
            msg = _require_password(user_id)
            if params.get("scope") == "personal":
                if msg:
                    return msg, debug_meta
                tasks = google_service.list_personal_tasks(user_id, params.get("status"))
            else:
                tasks = google_service.list_team_tasks_by_assignee(params.get("assignee", ""))
            if not tasks:
                return "Задачи не найдены.", debug_meta
            lines = [f"• {t['title']} [{t.get('status','')}] до {t.get('due_date','')}" for t in tasks]
            return "\n".join(lines), debug_meta
        if method == "calendar_create_or_update":
            event_id = google_service.create_or_update_calendar_event(
                params.get("title", ""), params.get("description", ""), params.get("due_date", ""), params.get("calendar_event_id")
            )
            return "Событие календаря обновлено." if event_id else "Не удалось обновить календарь.", debug_meta
        if method == "clarify":
            question = params.get("clarify_question") or "Поясните задачу."
            return question, debug_meta
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process action")
        return f"Ошибка обработки: {exc}", debug_meta
    return "Команда не распознана.", debug_meta
