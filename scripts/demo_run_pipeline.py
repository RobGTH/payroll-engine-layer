"""Run the Engine 1 -> Engine 2 payroll readiness pipeline."""

from __future__ import annotations

import json
import sys

from engine_layer.demo_data import sample_engine1_input, sample_engine2_input
from engine_layer.pipeline import PipelineRunError, run_payroll_readiness_pipeline


def main() -> int:
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

    print(json.dumps(output.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
