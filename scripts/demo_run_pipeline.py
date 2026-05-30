"""Run the Engine 1 -> Engine 2 payroll readiness pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engine_layer.demo_data import sample_engine1_input, sample_engine2_input
from engine_layer.pipeline import PipelineRunError, run_payroll_readiness_pipeline
from engine_layer.reports import pipeline_output_to_json, write_pipeline_report_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the payroll readiness pipeline and emit JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON readiness report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("The 'openai' package is not installed. Install project dependencies first.")
        return 1

    try:
        output = run_payroll_readiness_pipeline(
            client=OpenAI(),
            engine1_input=sample_engine1_input(),
            engine2_input=sample_engine2_input(),
        )
    except PipelineRunError as exc:
        print(f"Pipeline run failed: {exc}")
        return 1

    if args.output:
        report_path = write_pipeline_report_json(output, args.output)
        print(f"Wrote pipeline report to {report_path}")
    else:
        print(pipeline_output_to_json(output), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
