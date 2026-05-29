import json
from types import SimpleNamespace

import pytest

from engine_layer.demo_data import sample_engine1_input
from engine_layer.runner import EngineRunError, run_engine, run_engine_with_audit
from engine_layer.schemas.engine1 import Engine1Input, Engine1Output


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def valid_engine1_output():
    input_obj = sample_engine1_input()
    return {
        "schema_version": input_obj["schema_version"],
        "engine_version": input_obj["engine_version"],
        "engine_call_id": input_obj["engine_call_id"],
        "correlation_id": input_obj["correlation_id"],
        "org_id": input_obj["org_id"],
        "employee_id": input_obj["employee_record"]["employee_id"],
        "onboarding_status": "incomplete",
        "record_validity": "invalid",
        "hard_errors": [],
        "soft_warnings": [],
        "missing_fields": [],
        "document_status": {
            "required_docs_total": 1,
            "required_docs_received": 0,
            "required_docs_verified": 0,
            "docs": [
                {
                    "doc_type": "right_to_work",
                    "required": True,
                    "status": "not_requested",
                }
            ],
        },
        "risk_flags": [],
        "next_actions": [],
        "employee_record_patch": {},
    }


def audit_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_engine_returns_validated_output_and_sends_strict_schema():
    client = FakeClient(json.dumps(valid_engine1_output()))

    output = run_engine(
        client=client,
        engine_name="engine1",
        input_model=Engine1Input,
        output_model=Engine1Output,
        input_obj=sample_engine1_input(),
    )

    assert output.employee_id == "emp-001"
    assert client.responses.calls[0]["text"]["format"]["strict"] is True


def test_run_engine_rejects_invalid_input_before_model_call():
    client = FakeClient(json.dumps(valid_engine1_output()))
    input_obj = sample_engine1_input()
    input_obj.pop("schema_version")

    with pytest.raises(EngineRunError, match="input validation failed"):
        run_engine(
            client=client,
            engine_name="engine1",
            input_model=Engine1Input,
            output_model=Engine1Output,
            input_obj=input_obj,
        )

    assert client.responses.calls == []


def test_run_engine_rejects_invalid_json_response():
    client = FakeClient("not json")

    with pytest.raises(EngineRunError, match="returned invalid JSON"):
        run_engine(
            client=client,
            engine_name="engine1",
            input_model=Engine1Input,
            output_model=Engine1Output,
            input_obj=sample_engine1_input(),
        )


def test_run_engine_rejects_invalid_output_schema():
    payload = valid_engine1_output()
    payload["record_validity"] = "maybe"
    client = FakeClient(json.dumps(payload))

    with pytest.raises(EngineRunError, match="output validation failed"):
        run_engine(
            client=client,
            engine_name="engine1",
            input_model=Engine1Input,
            output_model=Engine1Output,
            input_obj=sample_engine1_input(),
        )


def test_run_engine_with_audit_writes_started_and_completed_events(tmp_path):
    audit_path = tmp_path / "audit_events.jsonl"
    client = FakeClient(json.dumps(valid_engine1_output()))

    output = run_engine_with_audit(
        client=client,
        engine_name="engine1",
        input_model=Engine1Input,
        output_model=Engine1Output,
        input_obj=sample_engine1_input(),
        audit_path=audit_path,
        started_details={"request_type": "validate_only"},
        completed_details=lambda result: {"record_validity": result.record_validity},
    )

    events = audit_events(audit_path)
    assert output.employee_id == "emp-001"
    assert [event["event"]["event_type"] for event in events] == [
        "engine_call_started",
        "engine_call_completed",
    ]
    assert events[0]["event"]["details"] == {"request_type": "validate_only"}
    assert events[1]["event"]["details"] == {"record_validity": "invalid"}


def test_run_engine_with_audit_writes_failed_event(tmp_path):
    audit_path = tmp_path / "audit_events.jsonl"
    client = FakeClient("not json")

    with pytest.raises(EngineRunError, match="returned invalid JSON"):
        run_engine_with_audit(
            client=client,
            engine_name="engine1",
            input_model=Engine1Input,
            output_model=Engine1Output,
            input_obj=sample_engine1_input(),
            audit_path=audit_path,
        )

    events = audit_events(audit_path)
    assert [event["event"]["event_type"] for event in events] == [
        "engine_call_started",
        "engine_call_failed",
    ]
    assert "returned invalid JSON" in events[1]["event"]["details"]["error"]
