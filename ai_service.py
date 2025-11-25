"""AI orchestration using Gemini with graceful degradation."""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import google.generativeai as genai

from ai_schemas import ActionKind, Classification, PlannedAction, Topic
from config import CONFIG

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            _model = genai.GenerativeModel(CONFIG.ai_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to heuristic AI: %s", exc)
            _model = False
    return _model


def _call_model(prompt: str) -> Optional[dict]:
    model = _get_model()
    if not model:
        return None
    try:
        resp = model.generate_content(prompt)
        if hasattr(resp, "text"):
            return json.loads(resp.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model call failed: %s", exc)
    return None


def _heuristic_classification(text: str) -> Classification:
    lower = text.lower()
    if any(cmd in lower for cmd in ["/start", "/help", "/login", "/logout"]):
        return Classification(ActionKind.OTHER, Topic.SYSTEM, 0.9)
    if "заметк" in lower:
        return Classification(ActionKind.CREATE, Topic.PERSONAL_NOTE, 0.6)
    if "команд" in lower:
        return Classification(ActionKind.CREATE, Topic.TEAM_TASK, 0.55)
    if "задач" in lower:
        return Classification(ActionKind.CREATE, Topic.PERSONAL_TASK, 0.55)
    if "календар" in lower or "напом" in lower:
        return Classification(ActionKind.CREATE, Topic.CALENDAR, 0.55)
    return Classification(ActionKind.OTHER, Topic.OTHER, 0.3)


def analyze_request(text: str, context_data: str | None = None) -> Tuple[Classification, PlannedAction]:
    prompt = f"""
Ты — автономный ассистент. Определи классификацию запроса и детальный план действия.
Верни JSON строго формата:
{{
  "classification": {{"kind": "CREATE|READ|UPDATE|DELETE|OTHER", "topic": "PERSONAL_NOTE|PERSONAL_TASK|TEAM_TASK|CALENDAR|SYSTEM|OTHER", "confidence": 0-1}},
  "action": {{"method": "string", "confidence": 0-1, "params": {{}} }}
}}
Учитывай контекст: {context_data or "нет"}.
Запрос: {text}
"""
    data = _call_model(prompt)
    if data:
        try:
            cls_data = data.get("classification", {})
            action_data = data.get("action", {})
            cls = Classification(ActionKind(cls_data.get("kind", "OTHER")), Topic(cls_data.get("topic", "OTHER")), float(cls_data.get("confidence", 0)))
            action = PlannedAction(
                method=str(action_data.get("method", "clarify")),
                confidence=float(action_data.get("confidence", 0)),
                params=action_data.get("params", {}) or {},
            )
            return cls, action
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse model output: %s", exc)
    heuristic_cls = _heuristic_classification(text)
    heuristic_action = _heuristic_action(text, heuristic_cls.topic)
    return heuristic_cls, heuristic_action


def _heuristic_action(text: str, topic: Topic) -> PlannedAction:
    lower = text.lower()
    if topic == Topic.PERSONAL_NOTE:
        return PlannedAction("write_personal_note", 0.55, {"text": text, "tags": []})
    if topic == Topic.PERSONAL_TASK:
        return PlannedAction("write_task", 0.55, {"scope": "personal", "title": text[:60], "status": "todo"})
    if topic == Topic.TEAM_TASK:
        return PlannedAction("write_team_task", 0.55, {"title": text[:60], "assignees": ""})
    if topic == Topic.CALENDAR:
        return PlannedAction("calendar_create_or_update", 0.5, {"title": text[:60], "due_date": ""})
    return PlannedAction("clarify", 0.3, {"clarify_question": "Что нужно сделать?"})


def build_confirmation(action: PlannedAction) -> str:
    parts = [f"Метод: {action.method}", f"Уверенность: {action.confidence:.2f}"]
    if action.params:
        parts.append(f"Параметры: {json.dumps(action.params, ensure_ascii=False)}")
    return "\n".join(parts)


def build_reminder_text(tasks: List[dict]) -> str:
    if not tasks:
        return """Нет срочных задач."""
    prompt = """
Сформулируй краткое напоминание пользователю о ближайших задачах. Верни обычный текст.
Задачи:
"""
    for t in tasks:
        prompt += json.dumps(t, ensure_ascii=False) + "\n"
    data = _call_model(prompt)
    if isinstance(data, str):
        return data
    return "Ближайшие задачи:\n" + "\n".join(f"- {t['title']} (до {t.get('due_date','')})" for t in tasks)
