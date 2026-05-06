"""Runtime configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional local convenience dependency
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
DEFAULT_SCHEMA_VERSION = "1.0.0"
DEFAULT_ENGINE_VERSION = "2026-03-01"
DEFAULT_AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "audit/audit_events.jsonl"))
