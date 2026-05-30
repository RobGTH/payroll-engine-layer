import json
from pathlib import Path

from engine_layer.schemas.pipeline import PayrollReadinessPipelineOutput


def test_blocked_readiness_report_fixture_matches_pipeline_schema():
    fixture_path = Path("examples/payroll_readiness_report.blocked.json")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    parsed = PayrollReadinessPipelineOutput.model_validate(payload)

    assert parsed.pipeline_status == "blocked"
    assert parsed.deterministic_readiness.has_blockers is True
    assert parsed.blocking_issues[0].source == "deterministic"
