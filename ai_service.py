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
    max_output_tokens: int | None = 1024,
    extra_config: dict | None = None,

    retry_without_context: bool = False,
) -> Optional[str]:
    """Безопасный вызов модели Gemini с обработкой всех вариантов ответа."""

    prompt = truncate_text(prompt, 12000)

    def _sync_call(current_prompt: str, current_max_tokens: int | None) -> tuple[Optional[str], dict]:
        try:
            model = genai.GenerativeModel(CONFIG.AI_MODEL)
        except Exception as e:  # noqa: BLE001
            logger.exception("Ошибка инициализации модели: %s", e)
            return None, {}
        try:
            generation_config = {"temperature": temperature, **(extra_config or {})}
            if current_max_tokens is not None:
                generation_config["max_output_tokens"] = current_max_tokens
            response = model.generate_content(
                current_prompt,
                generation_config=generation_config,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Ошибка вызова модели: %s", e)
            return None, {}

        finish_reason = None
        meta: dict = {}
        try:
            finish_reason = getattr(response, "candidates", [None])[0].finish_reason  # type: ignore[index]
        except Exception:
            pass
        meta["finish_reason"] = finish_reason
        meta["usage_metadata"] = getattr(response, "usage_metadata", None)
        meta["prompt_feedback"] = getattr(response, "prompt_feedback", None)

        try:
            text = response.text  # может бросить ValueError
            if text:
                return text, meta
        except Exception as e:  # noqa: BLE001
            logger.debug("Модель не вернула .text, пробуем кандидатов: %s", e)

        try:
            candidates = list(getattr(response, "candidates", None) or [])
            if not candidates:
                logger.warning("Модель вернула пустой список candidates")
                return None, meta

            cand0 = candidates[0]
            finish_reason = getattr(cand0, "finish_reason", None)

            bad_reasons = {"SAFETY", "BLOCKED", "OTHER", "RECITATION"}
            bad_codes = {3, 4}

            warning_logged = False
            if finish_reason in bad_reasons or finish_reason in bad_codes:
                logger.warning(
                    "Модель завершилась с проблемной finish_reason=%r, текст не используем",
                    finish_reason,
                )
                warning_logged = True
                return None, meta

            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None

            texts: list[str] = []
            if parts:
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        texts.append(t)

            if finish_reason == "MAX_TOKENS" and not texts:
                logger.warning(
                    "finish_reason=MAX_TOKENS без текстовых частей | usage=%r | prompt_feedback=%r",
                    meta.get("usage_metadata"),
                    meta.get("prompt_feedback"),
                )
                return None, meta

            if not texts:
                if not warning_logged and finish_reason:
                    logger.warning(
                        "Кандидат без текста, finish_reason=%r | usage=%r | prompt_feedback=%r",
                        finish_reason,
                        meta.get("usage_metadata"),
                        meta.get("prompt_feedback"),
                    )
                else:
                    logger.debug("Кандидат без текстовых частей, cand0=%r", cand0)
                return None, meta

            return "\n".join(texts), meta
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось извлечь текст из ответа модели")
            return None, meta

    def _extract_user_request(text: str) -> str:
        for marker in ["=== Новый запрос пользователя ===", "=== ЗАПРОС ===", "=== ЗАПРОС ПОЛЬЗОВАТЕЛЯ ==="]:
            if marker in text:
                return text.split(marker, 1)[-1].strip()
        return text

    try:
        result, meta = await asyncio.to_thread(_sync_call, prompt, max_output_tokens)
        finish_reason = meta.get("finish_reason")
        if result:
            return result

        if finish_reason == "MAX_TOKENS":
            logger.warning(
                "Пустой ответ от модели: finish_reason=MAX_TOKENS | usage=%r | prompt_feedback=%r",
                meta.get("usage_metadata"),
                meta.get("prompt_feedback"),
            )
            expanded_tokens = None if max_output_tokens is None else max(max_output_tokens * 2, 1024)
            try:
                retry_result, retry_meta = await asyncio.to_thread(
                    _sync_call, prompt, expanded_tokens
                )
                if retry_result:
                    return retry_result
                logger.warning(
                    "Повторный вызов после MAX_TOKENS тоже без текста | finish_reason=%r | usage=%r | prompt_feedback=%r",
                    retry_meta.get("finish_reason"),
                    retry_meta.get("usage_metadata"),
                    retry_meta.get("prompt_feedback"),
                )
            except Exception:  # noqa: BLE001
                logger.exception("Ошибка при повторном вызове модели после MAX_TOKENS")

        if not result:
            logger.warning(
                "Пустой ответ от модели: finish_reason=%r | usage=%r | prompt_feedback=%r",
                finish_reason,
                meta.get("usage_metadata"),
                meta.get("prompt_feedback"),
            )
            return (
                "Не удалось сгенерировать ответ из-за ограничения по длине. Попробуйте задать вопрос короче."
                if finish_reason == "MAX_TOKENS"
                else None
            )

        if retry_without_context:
            user_request = _extract_user_request(prompt)
            simple_prompt = (
                "Ответь по-русски на следующий запрос кратко и по делу:\n\n"
                f"{user_request}"
            )
            return await asyncio.to_thread(
                lambda p, t: _sync_call(p, t)[0],
                truncate_text(simple_prompt, 4000),
                200,
            )
    except Exception:  # noqa: BLE001
        logger.exception("Ошибка при вызове модели")
    return None


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to parse JSON from model output: %r", text, exc_info=True)
        return None


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(0.0, min(1.0, result))


def truncate_text(text: str, max_chars: int) -> str:
    """
    Если текст длиннее max_chars — обрезать его, сохранив начало и конец.
    Пример:
    <первые 3000 символов>\n...\n<последние 1000 символов>
    """

    if len(text) <= max_chars:
        return text

    separator = "\n...\n"
    tail_len = min(1000, max_chars // 3)
    head_len = max_chars - tail_len - len(separator)
    if head_len <= 0:
        head_len = max_chars // 2
        tail_len = max_chars - head_len - len(separator)
    return f"{text[:head_len]}{separator}{text[-tail_len:]}"


def build_context_for_user(profile: dict) -> str:
    """Compose context text from profile, dialog history and recent actions."""

    user_id = 0
    try:
        user_id = int(profile.get("telegram_user_id", 0))
    except (TypeError, ValueError):
        logger.debug("Invalid telegram_user_id in profile: %r", profile.get("telegram_user_id"))

    history = get_recent_history(user_id, limit=6)
    actions_summary = get_recent_actions_summary(user_id, limit=3)

    display_name = (
        profile.get("display_name")
        or profile.get("telegram_full_name")
        or profile.get("telegram_username")
        or "Пользователь"
    )
    timezone = profile.get("timezone") or "UTC"
    email = profile.get("calendar_email") or profile.get("email") or "не указан"

    history_lines = ["Краткая история последних сообщений:"]
    if history:
        for item in history:
            role = "Пользователь" if item.get("role") == "user" else "Ассистент"
            history_lines.append(f"{role}: {item.get('text', '')}")
    else:
        history_lines.append("(История пуста)")

    actions_lines = ["Краткое резюме последних действий ассистента:"]
    actions_lines.append(actions_summary or "(Нет зафиксированных действий)")

    context = (
        "Профиль пользователя:\n"
        f"- Имя: {display_name}\n"
        f"- Часовой пояс: {timezone}\n"
        f"- Email для календаря: {email}\n\n"
        + "\n".join(history_lines)
        + "\n\n"
        + "\n".join(actions_lines)
    )
    return truncate_text(context, 4000)


async def analyze_intent(profile: dict, user_text: str, context_text: str) -> dict:
    """Определяет тему и намерение запроса."""

    prompt = (
        "Ты — системный классификатор намерений. Твоя задача: определить тему и намерение запроса пользователя.\n"
        "Верни JSON строго в формате: {"
        "\"topic\": \"PERSONAL_TASK|TEAM_TASK|PERSONAL_NOTE|CALENDAR|CHAT|OTHER\","
        " \"intent\": \"CREATE|READ|UPDATE|DELETE|OTHER\","
        " \"rough_method\": \"string\", \"complexity\": \"simple|medium|complex\", \"confidence\": 0..1 }.\n"
        "Никаких пояснений и текста кроме JSON.\n"
        "Используй контекст и историю для понимания местоимений.\n"
        f"=== КОНТЕКСТ ===\n{context_text}\n"
        f"=== ЗАПРОС ===\n{user_text}\n"
    )
    raw = await _call_model(prompt, temperature=0.1, max_output_tokens=200)
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
    """Извлекает структурированные данные в зависимости от темы."""

    prompt = (
        "Ты извлекаешь структурированные поля из запроса пользователя.\n"
        "Ответь строго JSON. Для разных topic поля такие:\n"
        "- PERSONAL_TASK/TEAM_TASK: title, description, due_datetime_local (ISO с учетом часового пояса пользователя),"
        " priority (low|medium|high), tags (list[str]), assignees (list[str]) для командных задач.\n"
        "- CALENDAR: summary, start_datetime_local, end_datetime_local, all_day (bool).\n"
        "- PERSONAL_NOTE: title (может быть пустым), body.\n"
        "Заполняй только актуальные для topic поля, остальные делай null или пустыми.\n"
        "Используй CONTEXT для разрешения местоимений и ссылок на предыдущие действия.\n"
        f"=== КОНТЕКСТ ===\n{context_text}\n"
        f"INTENT (JSON): {json.dumps(intent, ensure_ascii=False)}\n"
        f"=== ЗАПРОС ===\n{user_text}\n"
    )
    try:
        raw = await _call_model(prompt, temperature=0.2, max_output_tokens=400)
        parsed = _safe_json_loads(raw or "") or {}
    except Exception:  # noqa: BLE001
        logger.warning("extract_structure failed, using defaults", exc_info=True)
        parsed = {}

    tags = parsed.get("tags") if isinstance(parsed.get("tags"), list) else None
    assignees = parsed.get("assignees") if isinstance(parsed.get("assignees"), list) else None

    result: Dict[str, Any] = {
        "title": parsed.get("title"),
        "description": parsed.get("description"),
        "due_datetime_local": parsed.get("due_datetime_local"),
        "priority": parsed.get("priority"),
        "tags": tags,
        "assignees": assignees,
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
    """Преобразует intent и структуру в исполняемый план."""

    allowed_methods = [
        "create_personal_task",
        "update_personal_task",
        "list_personal_tasks",
        "create_team_task",
        "update_team_task",
        "list_team_tasks",
        "create_or_update_calendar_event",
        "write_personal_note",
        "search_personal_notes",
        "chat",
        "clarify",
        "show_help",
    ]

    prompt = (
        "Ты планировщик действий. На основе intent и извлечённых данных выбери метод и параметры.\n"
        "Доступные методы: create_personal_task, update_personal_task, list_personal_tasks, create_team_task,"
        " update_team_task, list_team_tasks, create_or_update_calendar_event, write_personal_note, search_personal_notes,"
        " chat, clarify, show_help.\n"
        "Если это просто разговор — method='chat'. Если данных недостаточно — method='clarify' и задай clarify_question.\n"
        "Формат JSON ответа: {\"method\": string, \"params\": {...}, \"user_visible_answer\": string,"
        " \"confidence\": 0..1, \"clarify_question\": null|string}.\n"
        f"=== КОНТЕКСТ ===\n{context_text}\n"
        f"INTENT:\n{json.dumps(intent, ensure_ascii=False)}\n"
        f"STRUCTURED:\n{json.dumps(structured, ensure_ascii=False)}\n"
        f"=== ЗАПРОС ===\n{user_text}\n"
        "Отвечай только JSON."
    )
    fallback_plan = {
        "method": "chat",
        "params": {"question": user_text},
        "user_visible_answer": None,
        "confidence": 0.5,
        "clarify_question": None,
    }

    raw = await _call_model(prompt, temperature=0.15, max_output_tokens=512)
    parsed = _safe_json_loads(raw or "") or {}

    if not raw or not parsed:
        return fallback_plan

    method = parsed.get("method") if parsed.get("method") in allowed_methods else "chat"
    plan = {
        "method": method,
        "params": parsed.get("params") or {},
        "user_visible_answer": parsed.get("user_visible_answer") or None,
        "confidence": _coerce_confidence(parsed.get("confidence"), 0.5 if method == "chat" else 0.7),
        "clarify_question": parsed.get("clarify_question"),
    }
    if method == "chat" and not plan["params"].get("question"):
        plan["params"]["question"] = user_text
    if method == "clarify" and not plan["clarify_question"]:
        plan["clarify_question"] = "Мне нужно уточнить детали, чтобы продолжить."

    return plan


async def review_plan(profile: dict, user_text: str, context_text: str, plan: dict) -> dict:
    """Оценивает качество плана и необходимость уточнений."""

    prompt = (
        "Ты ревизор плана. Проверь, хватает ли данных в plan.params для безопасного выполнения."
        " Оцени дату/время, полноту описания и соответствие intent выбранному методу.\n"
        "Верни JSON {\"quality\":0..1, \"problems\":[...], \"clarify_question\": null|string}.\n"
        "Если есть сомнения — предлагай уточнить.\n"
        f"=== КОНТЕКСТ ===\n{context_text}\n"
        f"PLAN:\n{json.dumps(plan, ensure_ascii=False)}\n"
        f"=== ЗАПРОС ===\n{user_text}\n"
        "Отвечай только JSON."
    )
    raw = await _call_model(prompt, temperature=0.1, max_output_tokens=256)
    parsed = _safe_json_loads(raw or "")
    if not parsed:
        return {
            "quality": 0.5,
            "problems": [],
            "clarify_question": None,
        }
    return {
        "quality": _coerce_confidence(parsed.get("quality"), 0.5),
        "problems": parsed.get("problems") or [],
        "clarify_question": parsed.get("clarify_question"),
    }


async def free_chat(
    profile: dict,
    question: str | None = None,
    context_text: str | None = None,
    **kwargs: Any,
) -> str:
    """Свободный диалог с учётом контекста и безопасным фолбэком."""

    resolved_question = question or "Продолжай диалог со мной."
    resolved_context = truncate_text(context_text or build_context_for_user(profile), 2500)
    prompt = (
        "Ты — дружелюбный ассистент. Отвечай кратко и по делу на русском языке. "
        "Используй контекст только если он действительно нужен.\n"
        f"Контекст:\n{resolved_context}\n"
        f"Запрос: {resolved_question}\n"
    )
    text = await _call_model(
        prompt,
        temperature=0.3,
        max_output_tokens=400,
        extra_config=kwargs.get("extra_config"),
        retry_without_context=True,
    )
    return text or (
        "Сейчас у меня не получается получить ответ от модели, но я продолжу помогать с задачами и "
        "напоминаниями. Попробуйте переформулировать вопрос проще."
    )


async def process_user_request(profile: dict, user_text: str) -> dict:
    """Оркеструет все этапы AI и возвращает итоговый план."""

    fallback_plan = {
        "method": "chat",
        "params": {"question": user_text},
        "user_visible_answer": None,
        "confidence": 0.3,
        "clarify_question": None,
        "original_question": user_text,
    }

    try:
        context_text = build_context_for_user(profile)
        intent = await analyze_intent(profile, user_text, context_text)

        if intent.get("topic") == "CHAT":
            reply = await free_chat(profile, question=user_text, context_text=context_text)
            return {
                "method": "chat",
                "params": {"question": user_text, "context_text": context_text},
                "user_visible_answer": reply,
                "confidence": 1.0,
                "clarify_question": None,
                "original_question": user_text,
            }

        structured = await extract_structure(profile, user_text, context_text, intent)
        plan = await make_plan(profile, user_text, context_text, intent, structured)
        plan.setdefault("params", {})
        plan["original_question"] = user_text
        review = await review_plan(profile, user_text, context_text, plan)

        quality = review.get("quality", 0.0)
        clarify_question = review.get("clarify_question")

        if (
            _coerce_confidence(intent.get("confidence"), 0.0) < 0.3
            or not plan.get("method")
            or _coerce_confidence(quality, 0.0) < 0.3
        ):
            return fallback_plan

        if quality < 0.5:
            return {
                "method": "clarify",
                "params": {},
                "user_visible_answer": clarify_question
                or "Я не уверен, что правильно понял запрос. Уточните, пожалуйста.",
                "confidence": quality,
                "clarify_question": clarify_question,
                "original_question": user_text,
            }

        if 0.5 <= quality < 0.8 and clarify_question:
            return {
                "method": "clarify",
                "params": {},
                "user_visible_answer": clarify_question,
                "confidence": quality,
                "clarify_question": clarify_question,
                "original_question": user_text,
            }

        return plan
    except Exception:  # noqa: BLE001
        logger.exception("Ошибка в process_user_request")
        return fallback_plan


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

    text = await _call_model(prompt, temperature=0.2, max_output_tokens=300)

    if text:
        return text.strip()

    lines: list[str] = []
    lines.append(f"{user_name}, напоминаю о ваших задачах:")

    if not tasks:
        lines.append("На данный момент у вас нет активных задач.")
        return "\n".join(lines)

    for idx, task in enumerate(tasks, start=1):
        title = task.get("title") or task.get("name") or "Задача без названия"
        due = task.get("due") or task.get("due_datetime_local") or task.get("due_datetime") or ""
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
        due = task.get("due") or task.get("due_datetime_local") or task.get("due_datetime") or "срок не указан"
        status = task.get("status") or "open"
        lines.append(f"{idx}) [{status}] {title} (срок: {due})")

    return "\n".join(lines)
