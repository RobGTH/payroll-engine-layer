"""Run Engine 1 with a sample payload and append audit events."""

from __future__ import annotations

import json
import sys

from engine_layer.demo_data import sample_engine1_input
from engine_layer.runner import EngineRunError, run_engine_with_audit
from engine_layer.schemas.engine1 import Engine1Input, Engine1Output


def engine1_completion_details(output: Engine1Output) -> dict[str, object]:
    return {
        "record_validity": output.record_validity,
        "onboarding_status": output.onboarding_status,
        "missing_fields_count": len(output.missing_fields),
    }


def main() -> int:
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("The 'openai' package is not installed. Install project dependencies first.")
        return 1

    input_obj = sample_engine1_input()

    try:
        output = run_engine_with_audit(
            client=OpenAI(),
            engine_name="engine1",
            input_model=Engine1Input,
            output_model=Engine1Output,
            input_obj=input_obj,
            started_details={"request_type": input_obj["request_type"]},
            completed_details=engine1_completion_details,
        )
    except EngineRunError as exc:
        print(f"Engine run failed: {exc}")
        return 1

    print(json.dumps(output.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
