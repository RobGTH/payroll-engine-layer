import importlib.util
from pathlib import Path
from types import SimpleNamespace

from engine_layer.schemas.pipeline import PayrollReadinessPipelineOutput


def load_demo_module():
    script_path = Path("scripts/demo_run_pipeline.py")
    spec = importlib.util.spec_from_file_location("demo_run_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def blocked_fixture_output() -> PayrollReadinessPipelineOutput:
    import json

    payload = json.loads(
        Path("examples/payroll_readiness_report.blocked.json").read_text(
            encoding="utf-8"
        )
    )
    return PayrollReadinessPipelineOutput.model_validate(payload)


def test_demo_main_writes_output_file(monkeypatch, tmp_path, capsys):
    demo_run_pipeline = load_demo_module()
    report_path = tmp_path / "readiness.json"

    monkeypatch.setattr(
        demo_run_pipeline,
        "parse_args",
        lambda: SimpleNamespace(output=report_path),
    )
    monkeypatch.setattr(
        demo_run_pipeline,
        "run_payroll_readiness_pipeline",
        lambda **kwargs: blocked_fixture_output(),
    )
    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=lambda: object()))

    assert demo_run_pipeline.main() == 0

    assert report_path.exists()
    assert "Wrote pipeline report" in capsys.readouterr().out


def test_demo_main_prints_json(monkeypatch, capsys):
    demo_run_pipeline = load_demo_module()

    monkeypatch.setattr(
        demo_run_pipeline,
        "parse_args",
        lambda: SimpleNamespace(output=None),
    )
    monkeypatch.setattr(
        demo_run_pipeline,
        "run_payroll_readiness_pipeline",
        lambda **kwargs: blocked_fixture_output(),
    )
    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=lambda: object()))

    assert demo_run_pipeline.main() == 0

    assert '"pipeline_status": "blocked"' in capsys.readouterr().out
