"""Append-only audit logging primitives."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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


def iso_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_engine_audit_event(
    *,
    event_type: str,
    engine_name: str,
    engine_call_id: str,
    correlation_id: str,
    org_id: str,
    actor: dict[str, Any],
    subject: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consistent audit event envelope for engine calls."""
    return {
        "schema_version": "1.0.0",
        "engine_call_id": engine_call_id,
        "correlation_id": correlation_id,
        "org_id": org_id,
        "event": {
            "event_id": f"{engine_call_id}:{event_type}:{engine_name}",
            "event_type": event_type,
            "event_time": iso_now(),
            "actor": actor,
            "subject": subject,
            "engine": {"engine_name": engine_name},
            "details": details or {},
        },
    }
