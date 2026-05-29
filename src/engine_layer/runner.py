"""OpenAI-backed engine runner with strict schema validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .audit import append_audit_event, build_engine_audit_event
from .config import DEFAULT_MODEL

if TYPE_CHECKING:
    from openai import OpenAI

PROMPTS_DIR = Path(__file__).parent / "prompts"
T = TypeVar("T", bound=BaseModel)


class EngineRunError(RuntimeError):
    """Raised when an engine input or output fails validation."""


def load_system_prompt(engine_name: str) -> str:
    path = PROMPTS_DIR / f"{engine_name}_system.txt"
    if not path.exists():
        raise EngineRunError(f"Missing system prompt for {engine_name}: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_response_format(output_model: type[BaseModel], schema_name: str) -> dict[str, Any]:
    """Wrap a Pydantic schema in the JSON schema format expected by Responses API."""
    schema = output_model.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "name": schema_name,
        "strict": True,
        "schema": schema,
    }


def run_engine(
    *,
    client: "OpenAI",
    engine_name: str,
    input_model: type[BaseModel],
    output_model: type[T],
    input_obj: dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> T:
    """Validate input, call OpenAI with a strict schema, and validate the output."""
    try:
        validated_input = input_model.model_validate(input_obj)
    except ValidationError as exc:
        raise EngineRunError(f"{engine_name} input validation failed") from exc

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": load_system_prompt(engine_name)},
            {
                "role": "user",
                "content": json.dumps(validated_input.model_dump(mode="json")),
            },
        ],
        text={
            "format": build_response_format(
                output_model, f"{engine_name}_output"
            )
        },
    )

    raw_text = response.output_text
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise EngineRunError(f"{engine_name} returned invalid JSON") from exc

    try:
        return output_model.model_validate(payload)
    except ValidationError as exc:
        raise EngineRunError(f"{engine_name} output validation failed") from exc


def default_audit_subject(input_obj: dict[str, Any]) -> dict[str, Any]:
    """Build a compact subject from common engine input fields."""
    employee_record = input_obj.get("employee_record")
    if isinstance(employee_record, dict) and employee_record.get("employee_id"):
        return {"employee_id": employee_record["employee_id"]}
    return {}


def run_engine_with_audit(
    *,
    client: "OpenAI",
    engine_name: str,
    input_model: type[BaseModel],
    output_model: type[T],
    input_obj: dict[str, Any],
    model: str = DEFAULT_MODEL,
    audit_path: Path | None = None,
    subject: dict[str, Any] | None = None,
    started_details: dict[str, Any] | None = None,
    completed_details: Callable[[T], dict[str, Any]] | None = None,
) -> T:
    """Run an engine and append started/completed/failed audit events."""
    audit_subject = subject if subject is not None else default_audit_subject(input_obj)
    actor = input_obj.get("actor", {})

    common_event = {
        "engine_name": engine_name,
        "engine_call_id": str(input_obj.get("engine_call_id", "unknown")),
        "correlation_id": str(input_obj.get("correlation_id", "unknown")),
        "org_id": str(input_obj.get("org_id", "unknown")),
        "actor": actor if isinstance(actor, dict) else {},
        "subject": audit_subject,
    }

    append_audit_event(
        build_engine_audit_event(
            event_type="engine_call_started",
            details=started_details or {},
            **common_event,
        ),
        path=audit_path,
    )

    try:
        output = run_engine(
            client=client,
            engine_name=engine_name,
            input_model=input_model,
            output_model=output_model,
            input_obj=input_obj,
            model=model,
        )
    except Exception as exc:
        append_audit_event(
            build_engine_audit_event(
                event_type="engine_call_failed",
                details={"error": str(exc)},
                **common_event,
            ),
            path=audit_path,
        )
        raise

    append_audit_event(
        build_engine_audit_event(
            event_type="engine_call_completed",
            details=completed_details(output) if completed_details else {},
            **common_event,
        ),
        path=audit_path,
    )
    return output
