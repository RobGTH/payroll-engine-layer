import json
from pathlib import Path

from engine_layer.reports import pipeline_output_to_json, write_pipeline_report_json
from engine_layer.schemas.pipeline import PayrollReadinessPipelineOutput


def blocked_fixture_output() -> PayrollReadinessPipelineOutput:
    payload = json.loads(
        Path("examples/payroll_readiness_report.blocked.json").read_text(
            encoding="utf-8"
        )
    )
    return PayrollReadinessPipelineOutput.model_validate(payload)


def test_pipeline_output_to_json_returns_valid_pipeline_json():
    payload = json.loads(pipeline_output_to_json(blocked_fixture_output()))

    parsed = PayrollReadinessPipelineOutput.model_validate(payload)

    assert parsed.pipeline_status == "blocked"
    assert parsed.blocking_issues[0].source == "deterministic"


def test_write_pipeline_report_json_writes_valid_json(tmp_path):
    report_path = tmp_path / "reports" / "readiness.json"

    returned_path = write_pipeline_report_json(blocked_fixture_output(), report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    parsed = PayrollReadinessPipelineOutput.model_validate(payload)
    assert returned_path == report_path
    assert parsed.pipeline_status == "blocked"
