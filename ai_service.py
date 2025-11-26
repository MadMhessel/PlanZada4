"""AI orchestration using Gemini for multi-stage planning."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import google.generativeai as genai

from action_log import get_recent_actions_summary
from config import CONFIG
from dialog_history import get_recent_history

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


async def _call_model(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 512,
) -> Optional[str]:
    """Безопасный вызов модели Gemini с обработкой всех вариантов ответа."""

    def _sync_call() -> Optional[str]:
        try:
            model = genai.GenerativeModel(CONFIG.ai_model)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка инициализации модели: %s", exc)
            return None

        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка вызова модели: %s", exc)
            return None

        try:
            return response.text  # может бросить ValueError
        except Exception as exc:  # noqa: BLE001
            logger.warning("Модель не вернула .text, пробуем разобрать кандидатов: %s", exc)

        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                logger.warning("Модель вернула пустой список candidates")
                return None

            cand0 = candidates[0]
            finish_reason = getattr(cand0, "finish_reason", None)
            if finish_reason not in (1, "STOP", None):
                logger.warning(
                    "Модель завершила генерацию с finish_reason=%r, текст не используем",
                    finish_reason,
                )
                return None

            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None

            texts: list[str] = []
            if parts:
                for part in parts:
                    text_part = getattr(part, "text", None)
                    if text_part:
                        texts.append(text_part)

            if not texts:
                logger.warning("Кандидат без текстовых частей, cand0=%r", cand0)
                return None

            return "\n".join(texts)
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось извлечь текст из ответа модели")
            return None

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


async def build_reminder_text(tasks: list[dict], user: dict) -> str:
    """Формирует текст напоминания с использованием модели и безопасным фолбэком."""

    user_name = user.get("display_name") or user.get("telegram_full_name") or "Коллега"

    prompt = f"""
Ты — персональный ассистент и секретарь.

Пользователь: {user_name}
Нужно составить краткое напоминание о его задачах.

Список задач:
{_format_tasks_for_prompt(tasks)}

Сделай:

вежливое обращение к пользователю по имени;

короткий текст напоминания;

перечисли задачи по пунктам.

Ответ отдай одним цельным текстом на русском языке.
""".strip()

    text = await _call_model(prompt, temperature=0.2, max_output_tokens=400)

    if text:
        return text.strip()

    lines: list[str] = []
    lines.append(f"{user_name}, напоминаю о ваших задачах:")

    if not tasks:
        lines.append("На данный момент у вас нет активных задач.")
        return "\n".join(lines)

    for idx, task in enumerate(tasks, start=1):
        title = task.get("title") or task.get("name") or "Задача без названия"
        due = task.get("due") or task.get("due_datetime_local") or ""
        if due:
            lines.append(f"{idx}) {title} — срок: {due}")
        else:
            lines.append(f"{idx}) {title}")

    return "\n".join(lines)


def _format_tasks_for_prompt(tasks: list[dict]) -> str:
    """Подготавливает список задач для передачи в промт модели."""

    if not tasks:
        return "Нет активных задач."

    lines: list[str] = []
    for idx, task in enumerate(tasks[:10], start=1):
        title = task.get("title") or task.get("name") or "Задача без названия"
        due = task.get("due") or task.get("due_datetime_local") or "срок не указан"
        status = task.get("status") or "open"
        lines.append(f"{idx}) [{status}] {title} (срок: {due})")

    return "\n".join(lines)
