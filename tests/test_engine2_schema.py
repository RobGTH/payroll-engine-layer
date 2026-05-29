from engine_layer.demo_data import sample_engine2_input
from engine_layer.runner import build_response_format, load_system_prompt
from engine_layer.schemas.engine2 import Engine2Input, Engine2Output


def test_engine2_input_accepts_design_shaped_payload():
    parsed = Engine2Input.model_validate(sample_engine2_input())
    assert parsed.employee_record.employee_id == "emp-001"
    assert parsed.pay_period.pay_period_id == "pay-2026-02-01"
    assert parsed.earnings.pay_basis.amount == "35.00"


def test_engine2_output_schema_builds_strict_json_schema_wrapper():
    schema_wrapper = build_response_format(Engine2Output, "engine2_output")
    assert schema_wrapper["type"] == "json_schema"
    assert schema_wrapper["strict"] is True
    assert schema_wrapper["name"] == "engine2_output"


def test_engine2_prompt_exists():
    prompt = load_system_prompt("engine2")
    assert "Payroll Prep Core" in prompt
    assert "do not perform authoritative payroll" in prompt
