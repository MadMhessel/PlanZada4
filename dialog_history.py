"""In-memory dialog history per user."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List

_HISTORY: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=50))


def append_message(user_id: int, role: str, text: str) -> None:
    """Store a dialog message for a user."""
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    _HISTORY[user_id].append({"role": role, "text": text})


def get_recent_history(user_id: int, limit: int = 8) -> List[dict]:
    """Return recent dialog messages for a user."""
    history = list(_HISTORY.get(user_id, deque()))
    if limit > 0:
        history = history[-limit:]
    return history
