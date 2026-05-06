import json

from engine_layer.audit import append_audit_event


def test_append_audit_event_writes_one_json_line(tmp_path):
    log_path = tmp_path / "audit_events.jsonl"
    event = {
        "schema_version": "1.0.0",
        "engine_call_id": "call-001",
        "event": {"event_type": "engine_call_started"},
    }

    append_audit_event(event, path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == event
