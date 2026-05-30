# payroll-engine-layer

Minimal Python foundation for a modular payroll engine layer.

This repository starts with the smallest runnable slice from the design document:

- Engine 1 prompt loading
- Engine 1 input and output schemas
- a schema-driven OpenAI runner
- an append-only audit event sink
- a demo script for one end-to-end Engine 1 run
- focused tests around schema and audit behavior

## Layout

```text
src/engine_layer/
  audit.py
  config.py
  demo_data.py
  runner.py
  prompts/
    engine1_system.txt
  schemas/
    common.py
    engine1.py
scripts/
  demo_run_engine1.py
tests/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

To run the live OpenAI-backed Engine 1 demo, set `OPENAI_API_KEY` in your environment or
in a local `.env` file and then run:

```bash
PYTHONPATH=src python scripts/demo_run_engine1.py
```

The demo appends JSONL audit events to `audit/audit_events.jsonl` and prints the Engine 1
output JSON to the terminal.

## Pipeline report export

The payroll readiness pipeline can print JSON to the terminal or write the same report to a file:

```bash
PYTHONPATH=src python scripts/demo_run_pipeline.py --output output/payroll-readiness-report.json
```

The report remains JSON-first so it can be inspected directly or passed to a future dashboard.
