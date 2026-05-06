"""Run Engine 1 with a sample payload and append audit events."""

from __future__ import annotations

import json
import sys

from engine_layer.audit import append_audit_event, build_engine_audit_event
from engine_layer.demo_data import sample_engine1_input
from engine_layer.runner import EngineRunError, run_engine
from engine_layer.schemas.engine1 import Engine1Input, Engine1Output


def main() -> int:
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("The 'openai' package is not installed. Install project dependencies first.")
        return 1

    input_obj = sample_engine1_input()
    subject = {"employee_id": input_obj["employee_record"]["employee_id"]}
    append_audit_event(
        build_engine_audit_event(
            event_type="engine_call_started",
            engine_name="engine1",
            engine_call_id=input_obj["engine_call_id"],
            correlation_id=input_obj["correlation_id"],
            org_id=input_obj["org_id"],
            actor=input_obj["actor"],
            subject=subject,
            details={"request_type": input_obj["request_type"]},
        )
    )

    try:
        output = run_engine(
            client=OpenAI(),
            engine_name="engine1",
            input_model=Engine1Input,
            output_model=Engine1Output,
            input_obj=input_obj,
        )
    except EngineRunError as exc:
        append_audit_event(
            build_engine_audit_event(
                event_type="engine_call_failed",
                engine_name="engine1",
                engine_call_id=input_obj["engine_call_id"],
                correlation_id=input_obj["correlation_id"],
                org_id=input_obj["org_id"],
                actor=input_obj["actor"],
                subject=subject,
                details={"error": str(exc)},
            )
        )
        print(f"Engine run failed: {exc}")
        return 1

    append_audit_event(
        build_engine_audit_event(
            event_type="engine_call_completed",
            engine_name="engine1",
            engine_call_id=input_obj["engine_call_id"],
            correlation_id=input_obj["correlation_id"],
            org_id=input_obj["org_id"],
            actor=input_obj["actor"],
            subject=subject,
            details={
                "record_validity": output.record_validity,
                "onboarding_status": output.onboarding_status,
                "missing_fields_count": len(output.missing_fields),
            },
        )
    )
    print(json.dumps(output.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
