# payroll-engine-layer

Minimal Python foundation for a modular payroll engine layer.

This repository starts with the smallest runnable slice from the design document:

- Engine 1 prompt loading
- Engine 1 input and output schemas
- a schema-driven OpenAI runner
- an append-only audit event sink
- focused tests around schema and audit behavior

## Layout

```text
src/engine_layer/
  audit.py
  config.py
  runner.py
  prompts/
    engine1_system.txt
  schemas/
    common.py
    engine1.py
tests/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

To run the live OpenAI-backed engine later, set `OPENAI_API_KEY` in your environment or
in a local `.env` file before calling `engine_layer.runner.run_engine`.
