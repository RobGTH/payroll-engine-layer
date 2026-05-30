"""Report export helpers for pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas.pipeline import PayrollReadinessPipelineOutput


def pipeline_output_to_json(output: PayrollReadinessPipelineOutput) -> str:
    """Return a stable, human-readable JSON representation of a pipeline output."""
    return json.dumps(output.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_pipeline_report_json(
    output: PayrollReadinessPipelineOutput,
    path: str | Path,
) -> Path:
    """Write a pipeline output as JSON and return the path used."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(pipeline_output_to_json(output), encoding="utf-8")
    return report_path
