"""AI orchestration using Gemini for intent detection and reminders."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
from string import Template
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from config import CONFIG

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

_model: Optional[genai.GenerativeModel] = None


SYSTEM_PROMPT_MAIN = Template(
    """
Ты — интеллектуальный секретарь и диспетчер задач внутри Telegram-бота.
Работаешь строго по схеме: анализируешь входной текст пользователя и
возвращаешь ОДИН валидный JSON-объект с планом действия.

Входные данные:
- user_text: "$user_text"
- optional_context: "$optional_context"

Всегда учитывай профиль пользователя (имя, email, timezone). При обработке
относительных дат/времени ("завтра вечером", "через час") используй текущий
момент из контекста. Формат дат и времени в ответе: "YYYY-MM-DD HH:MM".

ТВОЙ ВЫХОД — только JSON без пояснений и форматирования:
{
  "kind": "CREATE | READ | UPDATE | DELETE | OTHER",
  "topic": "PERSONAL_NOTE | PERSONAL_TASK | TEAM_TASK | CALENDAR | SYSTEM | OTHER",
  "confidence": 0.0-1.0,
  "method": "строковый_идентификатор_метода",
  "params": { ... },
  "clarify_question": null | "строка",
  "user_visible_answer": null | "строка"
}

Допустимые method:
- write_personal_note, search_personal_notes, read_personal_notes,
  update_personal_note, delete_personal_note
- create_personal_task, update_personal_task, list_personal_tasks
- create_team_task, update_team_task, list_team_tasks
- create_or_update_calendar_event, show_calendar_agenda
- show_help, login, logout, debug_on, debug_off, debug_status
- clarify, chat

Порог уверенности:
- если уверен ≥0.75 — давай готовый план без уточнений;
- если 0.40–0.74 — предложи план и один clarify_question;
- если <0.40 — метод = "clarify", задай один короткий вопрос.

Структуры params (минимально необходимые поля):
- write_personal_note: {"note_text": str, "tags": [str]}
- search_personal_notes: {"query": str, "limit": int}
- read_personal_notes: {"limit": int}
- update_personal_note/delete_personal_note: {"note_id": str, "fields": {...}}

- create_personal_task/create_team_task:
  {"title": str, "description": str, "status": "todo|in_progress|done",
   "priority": "low|medium|high", "due_datetime": "YYYY-MM-DD HH:MM"|null,
   "tags": [str], "assignees": [str], "assignee_user_ids": [str]}
- update_personal_task/update_team_task:
  {"task_id": str, "fields": {"status":..., "priority":..., "due_datetime":..., "title":..., "description":..., "assignees": [str]}}
- list_personal_tasks/list_team_tasks: {"status": str|null}

- create_or_update_calendar_event:
  {"title": str, "description": str, "start_datetime": str, "end_datetime": str,
   "link_task_id": str|null, "attendees": ["email"]}
- show_calendar_agenda: {"from_datetime": str|null, "to_datetime": str|null}

- login: {"password": str}
- chat: {"question": str}

Никаких пояснений, только JSON.
"""
)

SYSTEM_PROMPT_REMINDERS = Template(
    """
Ты — дружелюбный секретарь. Составь короткое напоминание о задачах.
Текущее время: "$current_time".
Список задач в JSON:
$tasks_json

Ответь обычным текстом (русский), без JSON и без форматирования Markdown.
Сначала скажи, что это напоминание, затем перечисли просроченные и ближайшие
задачи, максимум по сути. Не выдумывай новых данных.
"""
)


def _get_model() -> Optional[genai.GenerativeModel]:
    global _model
    if _model is None:
        try:
            _model = genai.GenerativeModel(CONFIG.ai_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to heuristic AI: %s", exc)
            _model = None
    return _model


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


async def _call_model(prompt: str) -> Optional[dict]:
    model = _get_model()
    if not model:
        return None

    def _sync_call() -> Optional[dict]:
        resp = model.generate_content(prompt)
        content = getattr(resp, "text", None)
        if not content:
            return None
        return _safe_json_loads(content)

    return await asyncio.to_thread(_sync_call)


async def _call_model_text(prompt: str) -> Optional[str]:
    model = _get_model()
    if not model:
        return None

    def _sync_call() -> Optional[str]:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None)

    return await asyncio.to_thread(_sync_call)


def _build_context_text(user_profile: Dict[str, Any] | None, context: Dict[str, Any] | None) -> str:
    profile_part = ""
    if user_profile:
        profile_part = (
            f"Пользователь: {user_profile.get('display_name','')} | "
            f"роль: {user_profile.get('role','')} | "
            f"часовой пояс: {user_profile.get('timezone','')} | "
            f"email: {user_profile.get('email','')}"
        )
    ctx_text = context or {}
    summary_part = ctx_text.get("summary") if isinstance(ctx_text, dict) else None
    now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{profile_part}. Текущее время: {now_text}. Доп. контекст: {summary_part or ctx_text}".strip()


async def analyze_and_plan(user_profile: dict, user_text: str, context: dict | None) -> dict:
    """Return AI plan for a user request."""
    prompt = SYSTEM_PROMPT_MAIN.substitute(
        user_text=user_text,
        optional_context=_build_context_text(user_profile, context),
    )
    data = await _call_model(prompt)
    if data:
        if isinstance(data, dict):
            return data
        loaded = _safe_json_loads(str(data))
        if loaded:
            return loaded
    # fallback clarify
    return {
        "kind": "OTHER",
        "topic": "OTHER",
        "confidence": 0.3,
        "method": "clarify",
        "params": {},
        "clarify_question": "Что нужно сделать?",
        "user_visible_answer": None,
    }


async def free_chat(user_text: str, user_profile: Optional[dict] = None) -> str:
    """Fallback chat-style response."""
    context = _build_context_text(user_profile, None)
    prompt = (
        "Ты — дружелюбный ассистент. Ответь кратко и по делу на русском.\n"
        f"Контекст: {context}\n"
        f"Вопрос: {user_text}"
    )
    text = await _call_model_text(prompt)
    return text or "Готово."


async def build_reminder_text(tasks: List[dict]) -> str:
    current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
    prompt = SYSTEM_PROMPT_REMINDERS.substitute(current_time=current_time, tasks_json=tasks_json)
    text = await _call_model_text(prompt)
    return text or "На сейчас у вас нет задач, требующих внимания."
