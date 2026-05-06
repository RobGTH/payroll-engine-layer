"""Append-only audit logging primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_AUDIT_LOG_PATH


def append_audit_event(event: dict[str, Any], path: Path | None = None) -> Path:
    """Append one audit event as a JSON line and return the log path used."""
    log_path = path or DEFAULT_AUDIT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return log_path
