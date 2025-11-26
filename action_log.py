"""Lightweight in-memory action log for user actions."""
from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from typing import Deque, Dict, List

_ACTIONS: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=50))


def log_action(user_id: int, action_type: str, payload: dict) -> None:
    """Append an action entry for a user."""
    try:
        safe_user_id = int(user_id)
    except (TypeError, ValueError):
        return
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    _ACTIONS[safe_user_id].append({"timestamp": timestamp, "action_type": action_type, "payload": payload})


def get_recent_actions_summary(user_id: int, limit: int = 5) -> str:
    """Return a short textual summary of recent actions for prompt context."""
    actions = list(_ACTIONS.get(user_id, deque()))
    if limit > 0:
        actions = actions[-limit:]
    lines: List[str] = []
    for idx, item in enumerate(actions, start=1):
        payload = item.get("payload", {}) or {}
        title = payload.get("title") or payload.get("summary") or payload.get("body")
        due = payload.get("due_datetime") or payload.get("start_datetime")
        lines.append(
            f"{idx}) {item.get('timestamp')} — {item.get('action_type')} {title or ''} {f'({due})' if due else ''}".strip()
        )
    return "\n".join(lines)
