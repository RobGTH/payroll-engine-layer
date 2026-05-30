import json
from types import SimpleNamespace

import pytest

from engine_layer.demo_data import sample_engine1_input, sample_engine2_input
from engine_layer.pipeline import (
    PipelineRunError,
    build_engine2_input_from_engine1,
    build_pipeline_output,
    run_payroll_readiness_pipeline,
)
from engine_layer.readiness import evaluate_deterministic_readiness
from engine_layer.schemas.engine1 import Engine1Output
from engine_layer.schemas.engine2 import Engine2Output


class FakeResponses:
    def __init__(self, output_texts):
        self.output_texts = list(output_texts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_texts.pop(0))


class FakeClient:
    def __init__(self, output_texts):
        self.responses = FakeResponses(output_texts)


def engine1_payload():
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
        "hard_errors": [
            {
                "code": "tax_not_verified",
                "path": "employee_record.tax.tax_profile_status",
                "message": "Tax profile is not verified.",
            }
        ],
        "soft_warnings": [
            {
                "code": "manager_missing",
                "path": "employee_record.employment.manager_employee_id",
                "message": "Manager is not assigned.",
            }
        ],
        "missing_fields": [
            {
                "path": "employee_record.consents.privacy_acknowledged",
                "severity": "blocking",
                "reason": "Privacy acknowledgement is required.",
            }
        ],
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
        "next_actions": [
            {
                "action": "request_field",
                "target_path": "employee_record.consents.privacy_acknowledged",
                "doc_type": None,
                "priority": "high",
            }
        ],
        "employee_record_patch": {},
    }


def engine2_payload():
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
                "message": "Banking details must be verified.",
            }
        ],
        "warnings": [
            {
                "code": "hours_need_review",
                "path": "earnings.hours.regular_hours",
                "severity": "warning",
                "message": "Hours should be reviewed before calculation.",
            }
        ],
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
            "requires_manual_review": True,
        },
    }


def audit_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pipeline_success_runs_engine1_then_engine2_and_writes_audit(tmp_path):
    audit_path = tmp_path / "audit_events.jsonl"
    client = FakeClient([json.dumps(engine1_payload()), json.dumps(engine2_payload())])

    output = run_payroll_readiness_pipeline(
        client=client,
        engine1_input=sample_engine1_input(),
        engine2_input=sample_engine2_input(),
        pipeline_call_id="pipe-001",
        audit_path=audit_path,
    )

    assert output.pipeline_status == "blocked"
    assert output.deterministic_readiness.has_blockers is True
    assert output.deterministic_readiness.blocking_issue_count == 5
    assert output.employee_id == "emp-001"
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0]["text"]["format"]["name"] == "engine1_output"
    assert client.responses.calls[1]["text"]["format"]["name"] == "engine2_output"

    engine2_user_payload = json.loads(client.responses.calls[1]["input"][1]["content"])
    assert engine2_user_payload["engine1_readiness"] == {
        "onboarding_status": "incomplete",
        "record_validity": "invalid",
        "blocking_issue_count": 2,
        "warning_count": 1,
    }

    events = audit_events(audit_path)
    assert [event["event"]["event_type"] for event in events] == [
        "pipeline_started",
        "engine_call_started",
        "engine_call_completed",
        "engine_call_started",
        "engine_call_completed",
        "pipeline_completed",
    ]
    assert events[-1]["event"]["details"]["deterministic_blockers_count"] == 5


def test_pipeline_engine1_failure_stops_before_engine2_and_writes_failure(tmp_path):
    audit_path = tmp_path / "audit_events.jsonl"
    client = FakeClient(["not json"])

    with pytest.raises(PipelineRunError, match="payroll readiness pipeline failed"):
        run_payroll_readiness_pipeline(
            client=client,
            engine1_input=sample_engine1_input(),
            engine2_input=sample_engine2_input(),
            audit_path=audit_path,
        )

    assert len(client.responses.calls) == 1
    events = audit_events(audit_path)
    assert [event["event"]["event_type"] for event in events] == [
        "pipeline_started",
        "engine_call_started",
        "engine_call_failed",
        "pipeline_failed",
    ]


def test_pipeline_engine2_failure_writes_pipeline_failure(tmp_path):
    audit_path = tmp_path / "audit_events.jsonl"
    client = FakeClient([json.dumps(engine1_payload()), "not json"])

    with pytest.raises(PipelineRunError, match="payroll readiness pipeline failed"):
        run_payroll_readiness_pipeline(
            client=client,
            engine1_input=sample_engine1_input(),
            engine2_input=sample_engine2_input(),
            audit_path=audit_path,
        )

    assert len(client.responses.calls) == 2
    events = audit_events(audit_path)
    assert [event["event"]["event_type"] for event in events] == [
        "pipeline_started",
        "engine_call_started",
        "engine_call_completed",
        "engine_call_started",
        "engine_call_failed",
        "pipeline_failed",
    ]


def test_pipeline_summary_combines_deterministic_engine_blockers_warnings_and_actions():
    engine1_output = Engine1Output.model_validate(engine1_payload())
    engine2_output = Engine2Output.model_validate(engine2_payload())
    deterministic = evaluate_deterministic_readiness(
        engine1_input=sample_engine1_input(),
        engine2_input=sample_engine2_input(),
        engine1_output=engine1_output,
    )

    output = build_pipeline_output(
        pipeline_call_id="pipe-001",
        engine1_output=engine1_output,
        engine2_output=engine2_output,
        deterministic=deterministic,
    )

    assert output.pipeline_status == "blocked"
    assert output.onboarding_readiness.blocking_issue_count == 2
    assert output.payroll_readiness.blocking_issue_count == 1
    assert output.deterministic_readiness.blocking_issue_count == 5
    assert [issue.source for issue in output.blocking_issues[:5]] == [
        "deterministic",
        "deterministic",
        "deterministic",
        "deterministic",
        "deterministic",
    ]
    assert [issue.category for issue in output.warnings] == ["warning", "warning"]
    assert "deterministic" in [action.source for action in output.recommended_next_actions]


def test_build_engine2_input_from_engine1_does_not_mutate_original():
    engine2_input = sample_engine2_input()
    engine1_output = Engine1Output.model_validate(engine1_payload())

    prepared = build_engine2_input_from_engine1(engine2_input, engine1_output)

    assert prepared["engine1_readiness"]["blocking_issue_count"] == 2
    assert engine2_input["engine1_readiness"]["blocking_issue_count"] == 3
