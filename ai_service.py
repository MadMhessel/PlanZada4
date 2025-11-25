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
Ты — автономный диспетчер задач внутри Telegram-бота.
Вся логика принятия решений лежит на тебе. Анализируй запрос пользователя и
возвращай ОДИН валидный JSON со следующей схемой:
{
  "kind": "CREATE|READ|UPDATE|DELETE|OTHER",
  "topic": "PERSONAL_NOTE|PERSONAL_TASK|TEAM_TASK|CALENDAR|SYSTEM|OTHER",
  "confidence": float 0..1,
  "method": "string",
  "params": { ... },
  "clarify_question": null | "string",
  "user_visible_answer": null | "string"
}

Доступные методы (строки):
# Личные заметки
- write_personal_note, search_personal_notes, read_personal_notes,
  update_personal_note, delete_personal_note
# Личные задачи
- create_personal_task, update_personal_task, list_personal_tasks
# Командные задачи
- create_team_task, update_team_task, list_team_tasks
# Календарь
- create_or_update_calendar_event, show_calendar_agenda
# Система
- show_help, debug_on, debug_off, debug_status
# Прочее
- clarify (задать уточнение пользователю), chat (просто ответ)

Порог уверенности:
- >=0.75: можно выполнять без уточнений;
- 0.40..0.74: дай clarify_question при необходимости;
- <0.40: лучше запросить уточнение.

Правила дат/времени: учитывай timezone пользователя: "$timezone". Форматируй
даты и время как ISO "YYYY-MM-DD HH:MM". Для относительных выражений опирайся
на текущее время "$now".

Входные данные:
- user_text: "$user_text"
- контекст: $optional_context

Ответь ТОЛЬКО JSON без Markdown и без дополнительных комментариев.
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
    """Lazy init model with graceful degradation."""
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
    except json.JSONDecodeError:
        return None


async def _call_model(prompt: str) -> Optional[str]:
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


def _normalize_plan(plan: dict, fallback_question: str) -> dict:
    plan = plan or {}
    method = plan.get("method") or "chat"
    params = plan.get("params") or {"question": fallback_question}
    try:
        confidence = float(plan.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    clarify_question = plan.get("clarify_question")
    return {
        "kind": plan.get("kind") or "OTHER",
        "topic": plan.get("topic") or "OTHER",
        "confidence": confidence,
        "method": method,
        "params": params,
        "clarify_question": clarify_question,
        "user_visible_answer": plan.get("user_visible_answer"),
    }


async def analyze_and_plan(user_profile: dict, user_text: str, context: dict | None) -> dict:
    """Return AI plan for a user request with safe fallbacks."""
    prompt = SYSTEM_PROMPT_MAIN.substitute(
        user_text=user_text,
        optional_context=_build_context_text(user_profile, context),
        timezone=user_profile.get("timezone", "UTC"),
        now=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    raw_response = await _call_model(prompt)
    plan: dict | None = None
    if raw_response:
        plan = _safe_json_loads(raw_response)
    if plan is None:
        logger.warning("AI returned invalid JSON, fallback to chat: %s", raw_response)
        plan = {
            "kind": "OTHER",
            "topic": "OTHER",
            "confidence": 1.0,
            "method": "chat",
            "params": {"question": user_text},
            "clarify_question": None,
            "user_visible_answer": None,
        }
    return _normalize_plan(plan, user_text)


async def free_chat(profile: dict, question: str, **_: Any) -> str:
    """Fallback chat-style response."""
    context = _build_context_text(profile, None)
    prompt = (
        "Ты — дружелюбный ассистент. Ответь кратко и по делу на русском.\n"
        f"Контекст: {context}\n"
        f"Вопрос: {question}"
    )
    text = await _call_model(prompt)
    return text or "Готово."


async def build_reminder_text(tasks: List[dict], profile: Optional[dict] = None) -> str:
    current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
    prompt = SYSTEM_PROMPT_REMINDERS.substitute(current_time=current_time, tasks_json=tasks_json)
    text = await _call_model(prompt)
    if text:
        return text
    # fallback formatting
    overdue = [t for t in tasks if t.get("is_overdue")]
    upcoming = [t for t in tasks if not t.get("is_overdue")]
    parts: List[str] = ["Напоминание о задачах:"]
    if overdue:
        parts.append("Просроченные:")
        parts.extend(f"- {t.get('title')} (до {t.get('due_datetime')})" for t in overdue)
    if upcoming:
        parts.append("Ближайшие:")
        parts.extend(f"- {t.get('title')} (до {t.get('due_datetime')})" for t in upcoming)
    return "\n".join(parts) if len(parts) > 1 else "На сейчас у вас нет задач, требующих внимания."
