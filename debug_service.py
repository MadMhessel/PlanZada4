"""Debug flag handling for users."""
from __future__ import annotations

from typing import Dict


class DebugService:
    def __init__(self) -> None:
        self._flags: Dict[int, bool] = {}

    def set_debug(self, user_id: int, enabled: bool) -> None:
        self._flags[user_id] = enabled

    def is_debug(self, user_id: int) -> bool:
        return self._flags.get(user_id, False)

    def status_text(self, user_id: int) -> str:
        return "включен" if self.is_debug(user_id) else "выключен"

    def debug_on(self, profile: dict, **_: object) -> str:  # noqa: ANN003
        self.set_debug(int(profile.get("telegram_user_id")), True)
        return "Debug режим включен."

    def debug_off(self, profile: dict, **_: object) -> str:  # noqa: ANN003
        self.set_debug(int(profile.get("telegram_user_id")), False)
        return "Debug режим выключен."

    def debug_status(self, profile: dict, **_: object) -> str:  # noqa: ANN003
        return f"Debug режим {self.status_text(int(profile.get('telegram_user_id')))}."


debug_service = DebugService()
