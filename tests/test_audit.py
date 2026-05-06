import json

from engine_layer.audit import append_audit_event, build_engine_audit_event


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


def test_build_engine_audit_event_shapes_expected_envelope():
    event = build_engine_audit_event(
        event_type="engine_call_started",
        engine_name="engine1",
        engine_call_id="call-001",
        correlation_id="corr-001",
        org_id="org-abc",
        actor={"actor_type": "system", "actor_id": "sys-1", "source": "test"},
        subject={"employee_id": "emp-001"},
        details={"request_type": "validate_only"},
    )

    assert event["event"]["event_type"] == "engine_call_started"
    assert event["event"]["engine"]["engine_name"] == "engine1"
    assert event["event"]["subject"]["employee_id"] == "emp-001"
