"""Run Engine 2 with a sample payroll-prep payload and append audit events."""

from __future__ import annotations

import json
import sys

from engine_layer.demo_data import sample_engine2_input
from engine_layer.runner import EngineRunError, run_engine_with_audit
from engine_layer.schemas.engine2 import Engine2Input, Engine2Output


def engine2_completion_details(output: Engine2Output) -> dict[str, object]:
    return {
        "payroll_prep_status": output.payroll_prep_status,
        "include_in_pay_run": output.include_in_pay_run,
        "blocking_issues_count": len(output.blocking_issues),
        "warnings_count": len(output.warnings),
    }


def main() -> int:
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("The 'openai' package is not installed. Install project dependencies first.")
        return 1

    input_obj = sample_engine2_input()

    try:
        output = run_engine_with_audit(
            client=OpenAI(),
            engine_name="engine2",
            input_model=Engine2Input,
            output_model=Engine2Output,
            input_obj=input_obj,
            started_details={"pay_period_id": input_obj["pay_period"]["pay_period_id"]},
            completed_details=engine2_completion_details,
        )
    except EngineRunError as exc:
        print(f"Engine run failed: {exc}")
        return 1

    print(json.dumps(output.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
