import json
from pathlib import Path

import pytest

from engine_layer.schemas.pipeline import PayrollReadinessPipelineOutput


@pytest.mark.parametrize(
    ("fixture_name", "expected_status"),
    [
        ("payroll_readiness_report.blocked.json", "blocked"),
        ("payroll_readiness_report.needs_review.json", "needs_review"),
        ("payroll_readiness_report.ready.json", "ready"),
    ],
)
def test_readiness_report_fixtures_match_pipeline_schema(fixture_name, expected_status):
    fixture_path = Path("examples") / fixture_name
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    parsed = PayrollReadinessPipelineOutput.model_validate(payload)

    assert parsed.pipeline_status == expected_status
    assert parsed.employee_id
    assert parsed.pay_period_id
