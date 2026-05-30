from copy import deepcopy

from engine_layer.demo_data import sample_engine1_input, sample_engine2_input
from engine_layer.readiness import evaluate_deterministic_readiness
from engine_layer.schemas.engine1 import Engine1Output


def engine1_output_payload(*, ready: bool = False):
    input_obj = sample_engine1_input()
    return {
        "schema_version": input_obj["schema_version"],
        "engine_version": input_obj["engine_version"],
        "engine_call_id": input_obj["engine_call_id"],
        "correlation_id": input_obj["correlation_id"],
        "org_id": input_obj["org_id"],
        "employee_id": input_obj["employee_record"]["employee_id"],
        "onboarding_status": "ready_for_employment" if ready else "incomplete",
        "record_validity": "valid" if ready else "invalid",
        "hard_errors": [],
        "soft_warnings": [],
        "missing_fields": [],
        "document_status": {
            "required_docs_total": 1,
            "required_docs_received": 1 if ready else 0,
            "required_docs_verified": 1 if ready else 0,
            "docs": [
                {
                    "doc_type": "right_to_work",
                    "required": True,
                    "status": "verified" if ready else "not_requested",
                }
            ],
        },
        "risk_flags": [],
        "next_actions": [],
        "employee_record_patch": {},
    }


def ready_inputs():
    engine1_input = deepcopy(sample_engine1_input())
    employee_record = engine1_input["employee_record"]
    employee_record["tax"]["tax_profile_status"] = "verified"
    employee_record["super"]["super_profile_status"] = "verified"
    employee_record["banking"]["banking_status"] = "verified"
    employee_record["consents"]["privacy_acknowledged"] = True

    engine2_input = deepcopy(sample_engine2_input())
    engine2_input["employee_record"] = employee_record
    engine2_input["engine1_readiness"] = {
        "onboarding_status": "ready_for_employment",
        "record_validity": "valid",
        "blocking_issue_count": 0,
        "warning_count": 0,
    }
    return engine1_input, engine2_input


def test_deterministic_blockers_for_obvious_incomplete_inputs():
    result = evaluate_deterministic_readiness(
        engine1_input=sample_engine1_input(),
        engine2_input=sample_engine2_input(),
        engine1_output=Engine1Output.model_validate(engine1_output_payload()),
    )

    assert result.has_blockers is True
    assert [issue.code for issue in result.blockers] == [
        "engine1_record_invalid",
        "onboarding_incomplete",
        "banking_not_verified",
        "tax_not_verified",
        "super_not_verified",
    ]
    assert {action.action for action in result.next_actions} >= {
        "complete_onboarding",
        "verify_banking",
        "verify_tax",
        "verify_super",
    }


def test_deterministic_checks_clear_for_ready_inputs():
    engine1_input, engine2_input = ready_inputs()

    result = evaluate_deterministic_readiness(
        engine1_input=engine1_input,
        engine2_input=engine2_input,
        engine1_output=Engine1Output.model_validate(engine1_output_payload(ready=True)),
    )

    assert result.has_blockers is False
    assert result.blockers == []
    assert result.warnings == []
    assert [item.code for item in result.informational_items] == [
        "deterministic_checks_clear"
    ]


def test_deterministic_blocker_for_missing_hourly_hours_without_override():
    engine1_input, engine2_input = ready_inputs()
    engine2_input["earnings"]["hours"] = None

    result = evaluate_deterministic_readiness(
        engine1_input=engine1_input,
        engine2_input=engine2_input,
        engine1_output=Engine1Output.model_validate(engine1_output_payload(ready=True)),
    )

    assert [issue.code for issue in result.blockers] == ["hourly_hours_missing"]
    assert [action.action for action in result.next_actions] == ["review_hours"]


def test_deterministic_blocker_for_missing_pay_period_fields():
    engine1_input, engine2_input = ready_inputs()
    engine2_input["pay_period"]["pay_date"] = None

    result = evaluate_deterministic_readiness(
        engine1_input=engine1_input,
        engine2_input=engine2_input,
        engine1_output=Engine1Output.model_validate(engine1_output_payload(ready=True)),
    )

    assert [issue.code for issue in result.blockers] == ["pay_period_pay_date_missing"]
    assert [action.action for action in result.next_actions] == ["review_pay_period"]
