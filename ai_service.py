"""AI orchestration using Gemini for multi-stage planning."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from action_log import get_recent_actions_summary
from config import CONFIG
from dialog_history import get_recent_history

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

_model: Optional[genai.GenerativeModel] = None


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


async def _call_model(prompt: str, *, temperature: float = 0.2, max_output_tokens: int = 512) -> Optional[str]:
    """Call generative model safely in a thread."""
    model = _get_model()
    if not model:
        return None

    def _sync_call() -> Optional[str]:
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model call failed: %s", exc)
            return None
        return getattr(response, "text", None)

    return await asyncio.to_thread(_sync_call)


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(0.0, min(1.0, result))


def build_context_for_user(profile: dict) -> str:
    """Compose context text from profile, dialog history and recent actions."""
    history = get_recent_history(int(profile.get("telegram_user_id", 0)), limit=8)
    actions_summary = get_recent_actions_summary(int(profile.get("telegram_user_id", 0)), limit=5)

    profile_lines = [
        "Профиль пользователя:",
        f"- Имя: {profile.get('display_name', '')}",
        f"- Часовой пояс: {profile.get('timezone', 'UTC')}",
        f"- Email для календаря: {profile.get('email', '')}",
    ]

    history_lines = ["Краткая история последних сообщений:"]
    if history:
        for item in history:
            role = "Пользователь" if item.get("role") == "user" else "Ассистент"
            history_lines.append(f"{role}: {item.get('text', '')}")
    else:
        history_lines.append("(История пуста)")

    actions_lines = ["Краткое резюме последних действий ассистента:"]
    actions_lines.append(actions_summary or "(Нет зафиксированных действий)")

    return "\n".join(profile_lines + ["", *history_lines, "", *actions_lines])


async def analyze_intent(profile: dict, user_text: str, context_text: str) -> dict:
    """Quickly determine topic, intent and rough method."""
    prompt = (
        "Ты — системный классификатор намерений. Цель: определить тему и намерение запроса пользователя.\n"
        "Используй CONTEXT для разрешения местоимений вроде 'эта задача', ссылайся на последние действия.\n"
        "Верни строго JSON с полями: {"
        "\"topic\": \"PERSONAL_TASK|TEAM_TASK|PERSONAL_NOTE|CALENDAR|CHAT|OTHER\"," \
        "\"intent\": \"CREATE|READ|UPDATE|DELETE|OTHER\", "
        "\"rough_method\": \"string\", \"complexity\": \"simple|medium|complex\", \"confidence\": 0..1 }.\n"
        f"CONTEXT:\n{context_text}\n"
        f"USER_TEXT:\n{user_text}\n"
        "Отвечай только JSON без пояснений."
    )
    raw = await _call_model(prompt, temperature=0.1, max_output_tokens=300)
    parsed = _safe_json_loads(raw or "")
    if not parsed:
        return {
            "topic": "CHAT",
            "intent": "OTHER",
            "rough_method": "chat",
            "complexity": "simple",
            "confidence": 0.5,
        }
    return {
        "topic": parsed.get("topic", "OTHER"),
        "intent": parsed.get("intent", "OTHER"),
        "rough_method": parsed.get("rough_method", "chat"),
        "complexity": parsed.get("complexity", "simple"),
        "confidence": _coerce_confidence(parsed.get("confidence"), 0.5),
    }


async def extract_structure(profile: dict, user_text: str, context_text: str, intent: dict) -> dict:
    """Extract structured fields depending on topic."""
    prompt = (
        "Ты извлекаешь структурированные поля из запроса пользователя.\n"
        "Для PERSONAL_TASK или TEAM_TASK нужны поля: title, description, due_datetime_local (ISO), priority (low|medium|high), "
        "tags (list of strings), assignees (list).\n"
        "Для CALENDAR: summary, start_datetime_local, end_datetime_local, all_day (bool).\n"
        "Для PERSONAL_NOTE: title, body.\n"
        "Используй CONTEXT, чтобы понять ссылки на предыдущие задачи/события и местоимения.\n"
        "Верни JSON только с актуальными для topic полями. Пропущенные поля делай null.\n"
        f"CONTEXT:\n{context_text}\n"
        f"INTENT (текстом): {json.dumps(intent, ensure_ascii=False)}\n"
        f"USER_TEXT:\n{user_text}\n"
    )
    raw = await _call_model(prompt, temperature=0.2, max_output_tokens=400)
    parsed = _safe_json_loads(raw or "") or {}
    result: Dict[str, Any] = {
        "title": parsed.get("title"),
        "description": parsed.get("description"),
        "due_datetime_local": parsed.get("due_datetime_local"),
        "priority": parsed.get("priority"),
        "tags": parsed.get("tags"),
        "assignees": parsed.get("assignees"),
        "summary": parsed.get("summary"),
        "start_datetime_local": parsed.get("start_datetime_local"),
        "end_datetime_local": parsed.get("end_datetime_local"),
        "all_day": parsed.get("all_day"),
        "body": parsed.get("body"),
    }
    return result


async def make_plan(
    profile: dict,
    user_text: str,
    context_text: str,
    intent: dict,
    structured: dict,
) -> dict:
    """Convert intent + structured data into executable plan."""
    allowed_methods = {
        "create_personal_task",
        "update_personal_task",
        "list_personal_tasks",
        "create_team_task",
        "create_or_update_calendar_event",
        "write_personal_note",
        "update_personal_note",
        "delete_personal_note",
        "list_team_tasks",
        "chat",
        "clarify",
    }
    methods_description = (
        "create_personal_task: новая личная задача; "
        "update_personal_task: изменить личную задачу по id; "
        "list_personal_tasks: показать список личных задач; "
        "create_team_task: новая командная задача; "
        "create_or_update_calendar_event: создать/обновить событие календаря; "
        "write_personal_note/update_personal_note/delete_personal_note: операции с заметками; "
        "list_team_tasks: показать командные задачи; "
        "chat: когда запрос не о задачах/календаре; "
        "clarify: задать уточняющий вопрос."
    )
    prompt = (
        "Ты планировщик действий. На основе intent и извлечённой структуры выбери метод программы и параметры.\n"
        f"Доступные методы: {methods_description}\n"
        "Если запрос не про задачи/календарь/заметки — выбирай method='chat'.\n"
        "Формат JSON ответа: {\"method\": string, \"params\": {...}, \"user_visible_answer\": string, \"confidence\": 0..1, \"clarify_question\": null|string}.\n"
        "Используй CONTEXT для разрешения местоимений и последних действий.\n"
        f"CONTEXT:\n{context_text}\n"
        f"INTENT:\n{json.dumps(intent, ensure_ascii=False)}\n"
        f"STRUCTURED:\n{json.dumps(structured, ensure_ascii=False)}\n"
        f"USER_TEXT:\n{user_text}\n"
        "Отвечай только JSON."
    )
    raw = await _call_model(prompt, temperature=0.15, max_output_tokens=500)
    parsed = _safe_json_loads(raw or "") or {}
    method = parsed.get("method")
    if method not in allowed_methods:
        method = "chat"
    plan = {
        "method": method,
        "params": parsed.get("params") or {},
        "user_visible_answer": parsed.get("user_visible_answer") or "",
        "confidence": _coerce_confidence(parsed.get("confidence"), 0.5 if method == "chat" else 0.7),
        "clarify_question": parsed.get("clarify_question"),
    }
    return plan


async def review_plan(profile: dict, user_text: str, context_text: str, plan: dict) -> dict:
    """Review plan for obvious issues and missing data."""
    prompt = (
        "Ты ревизор плана. Проверь, хватает ли данных в plan.params для безопасного выполнения.\n"
        "Особенно оцени даты/время, понятность задачи/события и соответствие intent выбранному методу.\n"
        "Верни JSON {\"quality\":0..1, \"problems\":[...], \"clarify_question\": null|string}.\n"
        "Используй CONTEXT (история и последние действия) для понимания местоимений.\n"
        f"CONTEXT:\n{context_text}\n"
        f"PLAN:\n{json.dumps(plan, ensure_ascii=False)}\n"
        f"USER_TEXT:\n{user_text}\n"
        "Отвечай только JSON."
    )
    raw = await _call_model(prompt, temperature=0.1, max_output_tokens=300)
    parsed = _safe_json_loads(raw or "")
    if not parsed:
        return {
            "quality": 0.4,
            "problems": ["Модель вернула некорректный формат."],
            "clarify_question": "Я не уверен, что правильно понял ваш запрос. Пожалуйста, уточните детали.",
        }
    return {
        "quality": _coerce_confidence(parsed.get("quality"), 0.5),
        "problems": parsed.get("problems", []),
        "clarify_question": parsed.get("clarify_question"),
    }


async def process_user_request(profile: dict, user_text: str) -> dict:
    """Orchestrate all AI stages and return final plan."""
    context_text = build_context_for_user(profile)
    intent = await analyze_intent(profile, user_text, context_text)

    if intent.get("topic") == "CHAT":
        chat_answer = await free_chat(profile, user_text, context_text)
        return {
            "method": "chat",
            "params": {},
            "user_visible_answer": chat_answer,
            "confidence": 1.0,
            "clarify_question": None,
        }

    structured = await extract_structure(profile, user_text, context_text, intent)
    plan = await make_plan(profile, user_text, context_text, intent, structured)
    review = await review_plan(profile, user_text, context_text, plan)

    quality = review.get("quality", 0.0)
    clarify_question = review.get("clarify_question")

    if quality < 0.5:
        return {
            "method": "clarify",
            "params": {},
            "user_visible_answer": clarify_question
            or "Я не уверен, что правильно понял ваш запрос. Пожалуйста, уточните детали.",
            "confidence": quality,
            "clarify_question": clarify_question,
        }

    if quality < 0.8 and clarify_question:
        return {
            "method": "clarify",
            "params": {},
            "user_visible_answer": clarify_question,
            "confidence": quality,
            "clarify_question": clarify_question,
        }

    return plan


async def free_chat(profile: dict, question: str, context_text: str | None = None) -> str:
    """Fallback chat-style response with context awareness."""
    resolved_context = context_text or build_context_for_user(profile)
    prompt = (
        "Ты — дружелюбный ассистент. Отвечай кратко и по делу на русском.\n"
        f"Контекст по пользователю:\n{resolved_context}\n"
        f"Вопрос: {question}"
    )
    text = await _call_model(prompt, temperature=0.3, max_output_tokens=500)
    return text or "Готово."


async def build_reminder_text(tasks: List[dict], profile: Optional[dict] = None) -> str:
    current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
    prompt = (
        "Ты — дружелюбный секретарь. Составь короткое напоминание о задачах.\n"
        f"Текущее время: {current_time}.\n"
        f"Список задач в JSON:\n{tasks_json}\n"
        "Ответь обычным текстом (русский), без JSON и без форматирования Markdown."
        "Сначала скажи, что это напоминание, затем перечисли просроченные и ближайшие задачи."
    )
    text = await _call_model(prompt, temperature=0.2, max_output_tokens=400)
    if text:
        return text
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
