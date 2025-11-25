"""Simple JSONL dialog logger."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from config import CONFIG


class DialogLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


dialog_logger = DialogLogger(CONFIG.dialog_log_path)
