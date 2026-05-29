import json
from types import SimpleNamespace

import pytest

from engine_layer.demo_data import sample_engine2_input
from engine_layer.runner import EngineRunError, run_engine, run_engine_with_audit
from engine_layer.schemas.engine2 import Engine2Input, Engine2Output


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


def valid_engine2_output():
    input_obj = sample_engine2_input()
    return {
        "schema_version": input_obj["schema_version"],
        "engine_version": input_obj["engine_version"],
        "engine_call_id": input_obj["engine_call_id"],
        "correlation_id": input_obj["correlation_id"],
        "org_id": input_obj["org_id"],
        "employee_id": input_obj["employee_record"]["employee_id"],
        "pay_period_id": input_obj["pay_period"]["pay_period_id"],
        "payroll_prep_status": "blocked",
        "include_in_pay_run": False,
        "blocking_issues": [
            {
                "code": "banking_not_verified",
                "path": "employee_record.banking.banking_status",
                "severity": "blocking",
                "message": "Banking details must be verified before payroll calculation.",
            }
        ],
        "warnings": [],
        "next_actions": [
            {
                "action": "verify_banking",
                "target_path": "employee_record.banking",
                "priority": "high",
            }
        ],
        "payroll_prep_summary": {
            "employee_id": input_obj["employee_record"]["employee_id"],
            "pay_period_id": input_obj["pay_period"]["pay_period_id"],
            "pay_schedule_group": input_obj["pay_period"]["pay_schedule_group"],
            "pay_basis": input_obj["earnings"]["pay_basis"],
            "estimated_gross_pay": None,
            "deductions_total": None,
            "ready_for_calculation": False,
            "requires_manual_review": False,
        },
    }


def audit_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_engine_accepts_valid_engine2_output():
    client = FakeClient(json.dumps(valid_engine2_output()))

    output = run_engine(
        client=client,
        engine_name="engine2",
        input_model=Engine2Input,
        output_model=Engine2Output,
        input_obj=sample_engine2_input(),
    )

    assert output.employee_id == "emp-001"
    assert output.payroll_prep_status == "blocked"
    assert output.include_in_pay_run is False
    assert client.responses.calls[0]["text"]["format"]["name"] == "engine2_output"


def test_run_engine_rejects_invalid_engine2_input_before_model_call():
    client = FakeClient(json.dumps(valid_engine2_output()))
    input_obj = sample_engine2_input()
    input_obj["pay_period"].pop("pay_period_id")

    with pytest.raises(EngineRunError, match="input validation failed"):
        run_engine(
            client=client,
            engine_name="engine2",
            input_model=Engine2Input,
            output_model=Engine2Output,
            input_obj=input_obj,
        )

    assert client.responses.calls == []


def test_run_engine_rejects_invalid_engine2_output_schema():
    payload = valid_engine2_output()
    payload["payroll_prep_status"] = "ready"
    client = FakeClient(json.dumps(payload))

    with pytest.raises(EngineRunError, match="output validation failed"):
        run_engine(
            client=client,
            engine_name="engine2",
            input_model=Engine2Input,
            output_model=Engine2Output,
            input_obj=sample_engine2_input(),
        )


def test_run_engine_with_audit_writes_engine2_events(tmp_path):
    audit_path = tmp_path / "audit_events.jsonl"
    client = FakeClient(json.dumps(valid_engine2_output()))

    output = run_engine_with_audit(
        client=client,
        engine_name="engine2",
        input_model=Engine2Input,
        output_model=Engine2Output,
        input_obj=sample_engine2_input(),
        audit_path=audit_path,
        started_details={"pay_period_id": "pay-2026-02-01"},
        completed_details=lambda result: {
            "payroll_prep_status": result.payroll_prep_status,
            "include_in_pay_run": result.include_in_pay_run,
        },
    )

    events = audit_events(audit_path)
    assert output.payroll_prep_status == "blocked"
    assert [event["event"]["event_type"] for event in events] == [
        "engine_call_started",
        "engine_call_completed",
    ]
    assert events[0]["event"]["engine"]["engine_name"] == "engine2"
    assert events[1]["event"]["details"] == {
        "payroll_prep_status": "blocked",
        "include_in_pay_run": False,
    }
