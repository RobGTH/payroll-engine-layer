from engine_layer.demo_data import sample_engine1_input
from engine_layer.runner import build_response_format, load_system_prompt
from engine_layer.schemas.engine1 import Engine1Input, Engine1Output


def test_engine1_input_accepts_design_shaped_payload():
    parsed = Engine1Input.model_validate(sample_engine1_input())
    assert parsed.employee_record.employee_id == "emp-001"
    assert parsed.employee_record.employment.pay_basis.amount == "35.00"


def test_runner_builds_strict_json_schema_wrapper():
    schema_wrapper = build_response_format(Engine1Output, "engine1_output")
    assert schema_wrapper["type"] == "json_schema"
    assert schema_wrapper["strict"] is True
    assert schema_wrapper["name"] == "engine1_output"


def test_engine1_prompt_exists():
    prompt = load_system_prompt("engine1")
    assert "Output exactly one JSON object" in prompt
