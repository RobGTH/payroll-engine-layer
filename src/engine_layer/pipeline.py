"""Synchronous orchestration for payroll readiness pipelines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from .audit import append_audit_event, build_engine_audit_event
from .config import DEFAULT_ENGINE_VERSION, DEFAULT_MODEL, DEFAULT_SCHEMA_VERSION
from .readiness import DeterministicReadinessResult, evaluate_deterministic_readiness
from .runner import run_engine_with_audit
from .schemas.engine1 import Engine1Input, Engine1Output
from .schemas.engine2 import Engine1ReadinessSummary, Engine2Input, Engine2Output
from .schemas.pipeline import (
    DeterministicReadinessSummary,
    OnboardingReadinessSummary,
    PayrollReadinessPipelineOutput,
    PayrollReadinessSummary,
    PipelineIssue,
    PipelineNextAction,
)

if TYPE_CHECKING:
    from openai import OpenAI


class PipelineRunError(RuntimeError):
    """Raised when a pipeline run fails."""


def _pipeline_audit_event(
    *,
    event_type: str,
    pipeline_call_id: str,
    correlation_id: str,
    org_id: str,
    actor: dict,
    subject: dict,
    details: dict | None = None,
) -> dict:
    return build_engine_audit_event(
        event_type=event_type,
        engine_name="payroll_readiness_pipeline",
        engine_call_id=pipeline_call_id,
        correlation_id=correlation_id,
        org_id=org_id,
        actor=actor,
        subject=subject,
        details=details,
    )


def _engine1_readiness_summary(output: Engine1Output) -> Engine1ReadinessSummary:
    blocking_issue_count = len(output.hard_errors) + sum(
        1 for field in output.missing_fields if field.severity == "blocking"
    )
    return Engine1ReadinessSummary(
        onboarding_status=output.onboarding_status,
        record_validity=output.record_validity,
        blocking_issue_count=blocking_issue_count,
        warning_count=len(output.soft_warnings),
    )


def build_engine2_input_from_engine1(
    engine2_input: dict,
    engine1_output: Engine1Output,
) -> dict:
    """Carry Engine 1 readiness into the Engine 2 payload."""
    prepared_input = deepcopy(engine2_input)
    prepared_input["engine1_readiness"] = _engine1_readiness_summary(
        engine1_output
    ).model_dump(mode="json")
    return prepared_input


def _pipeline_status(
    engine1_output: Engine1Output,
    engine2_output: Engine2Output,
    deterministic: DeterministicReadinessResult,
) -> str:
    if (
        deterministic.has_blockers
        or engine1_output.record_validity == "invalid"
        or engine1_output.onboarding_status == "incomplete"
        or engine2_output.payroll_prep_status == "blocked"
    ):
        return "blocked"
    if (
        engine1_output.onboarding_status == "compliance_pending"
        or engine2_output.payroll_prep_status == "needs_review"
    ):
        return "needs_review"
    return "ready"


def _engine1_blockers(output: Engine1Output) -> list[PipelineIssue]:
    issues = [
        PipelineIssue(
            source="engine1",
            category="blocker",
            code=issue.code,
            path=issue.path,
            message=issue.message,
        )
        for issue in output.hard_errors
    ]
    issues.extend(
        PipelineIssue(
            source="engine1",
            category="blocker" if field.severity == "blocking" else "warning",
            code="missing_field",
            path=field.path,
            message=field.reason,
        )
        for field in output.missing_fields
        if field.severity == "blocking"
    )
    return issues


def _engine1_warnings(output: Engine1Output) -> list[PipelineIssue]:
    warnings = [
        PipelineIssue(
            source="engine1",
            category="warning",
            code=issue.code,
            path=issue.path,
            message=issue.message,
        )
        for issue in output.soft_warnings
    ]
    warnings.extend(
        PipelineIssue(
            source="engine1",
            category="warning",
            code="missing_field",
            path=field.path,
            message=field.reason,
        )
        for field in output.missing_fields
        if field.severity == "non_blocking"
    )
    return warnings


def _engine2_issues(output: Engine2Output, category: str) -> list[PipelineIssue]:
    source_items = output.blocking_issues if category == "blocker" else output.warnings
    return [
        PipelineIssue(
            source="engine2",
            category=category,
            code=issue.code,
            path=issue.path,
            message=issue.message,
        )
        for issue in source_items
    ]


def _next_actions(
    engine1_output: Engine1Output,
    engine2_output: Engine2Output,
    deterministic: DeterministicReadinessResult,
) -> list[PipelineNextAction]:
    actions = [
        PipelineNextAction(
            source="engine1",
            action=action.action,
            target_path=action.target_path,
            priority=action.priority,
        )
        for action in engine1_output.next_actions
    ]
    actions.extend(
        PipelineNextAction(
            source="engine2",
            action=action.action,
            target_path=action.target_path,
            priority=action.priority,
        )
        for action in engine2_output.next_actions
    )
    actions.extend(deterministic.next_actions)
    return actions


def build_pipeline_output(
    *,
    pipeline_call_id: str,
    engine1_output: Engine1Output,
    engine2_output: Engine2Output,
    deterministic: DeterministicReadinessResult | None = None,
) -> PayrollReadinessPipelineOutput:
    """Combine engine outputs into one JSON-first readiness report."""
    deterministic = deterministic or DeterministicReadinessResult()
    onboarding = OnboardingReadinessSummary(
        status=engine1_output.onboarding_status,
        record_validity=engine1_output.record_validity,
        blocking_issue_count=len(_engine1_blockers(engine1_output)),
        warning_count=len(_engine1_warnings(engine1_output)),
    )
    payroll = PayrollReadinessSummary(
        status=engine2_output.payroll_prep_status,
        include_in_pay_run=engine2_output.include_in_pay_run,
        blocking_issue_count=len(engine2_output.blocking_issues),
        warning_count=len(engine2_output.warnings),
    )
    deterministic_summary = DeterministicReadinessSummary(
        blocking_issue_count=len(deterministic.blockers),
        warning_count=len(deterministic.warnings),
        informational_count=len(deterministic.informational_items),
        has_blockers=deterministic.has_blockers,
    )
    return PayrollReadinessPipelineOutput(
        schema_version=DEFAULT_SCHEMA_VERSION,
        engine_version=DEFAULT_ENGINE_VERSION,
        engine_call_id=pipeline_call_id,
        correlation_id=engine1_output.correlation_id,
        org_id=engine1_output.org_id,
        employee_id=engine1_output.employee_id,
        pay_period_id=engine2_output.pay_period_id,
        pipeline_status=_pipeline_status(engine1_output, engine2_output, deterministic),
        onboarding_readiness=onboarding,
        payroll_readiness=payroll,
        deterministic_readiness=deterministic_summary,
        blocking_issues=deterministic.blockers
        + _engine1_blockers(engine1_output)
        + _engine2_issues(engine2_output, "blocker"),
        warnings=deterministic.warnings
        + _engine1_warnings(engine1_output)
        + _engine2_issues(engine2_output, "warning"),
        informational_items=deterministic.informational_items,
        recommended_next_actions=_next_actions(engine1_output, engine2_output, deterministic),
        engine1_output=engine1_output,
        engine2_output=engine2_output,
    )


def run_payroll_readiness_pipeline(
    *,
    client: "OpenAI",
    engine1_input: dict,
    engine2_input: dict,
    pipeline_call_id: str = "pipeline-001",
    model: str = DEFAULT_MODEL,
    audit_path: Path | None = None,
) -> PayrollReadinessPipelineOutput:
    """Run Engine 1 then Engine 2 and return a combined readiness report."""
    subject = {"employee_id": engine1_input["employee_record"]["employee_id"]}
    actor = engine1_input.get("actor", {})
    common_pipeline_event = {
        "pipeline_call_id": pipeline_call_id,
        "correlation_id": engine1_input["correlation_id"],
        "org_id": engine1_input["org_id"],
        "actor": actor if isinstance(actor, dict) else {},
        "subject": subject,
    }
    append_audit_event(
        _pipeline_audit_event(
            event_type="pipeline_started",
            details={"pipeline_name": "payroll_readiness"},
            **common_pipeline_event,
        ),
        path=audit_path,
    )

    try:
        engine1_output = run_engine_with_audit(
            client=client,
            engine_name="engine1",
            input_model=Engine1Input,
            output_model=Engine1Output,
            input_obj=engine1_input,
            model=model,
            audit_path=audit_path,
            subject=subject,
            started_details={"pipeline_call_id": pipeline_call_id},
            completed_details=lambda output: {
                "onboarding_status": output.onboarding_status,
                "record_validity": output.record_validity,
            },
        )
        deterministic = evaluate_deterministic_readiness(
            engine1_input=engine1_input,
            engine2_input=engine2_input,
            engine1_output=engine1_output,
        )
        prepared_engine2_input = build_engine2_input_from_engine1(
            engine2_input,
            engine1_output,
        )
        engine2_output = run_engine_with_audit(
            client=client,
            engine_name="engine2",
            input_model=Engine2Input,
            output_model=Engine2Output,
            input_obj=prepared_engine2_input,
            model=model,
            audit_path=audit_path,
            subject=subject,
            started_details={"pipeline_call_id": pipeline_call_id},
            completed_details=lambda output: {
                "payroll_prep_status": output.payroll_prep_status,
                "include_in_pay_run": output.include_in_pay_run,
            },
        )
        output = build_pipeline_output(
            pipeline_call_id=pipeline_call_id,
            engine1_output=engine1_output,
            engine2_output=engine2_output,
            deterministic=deterministic,
        )
    except Exception as exc:
        append_audit_event(
            _pipeline_audit_event(
                event_type="pipeline_failed",
                details={"error": str(exc)},
                **common_pipeline_event,
            ),
            path=audit_path,
        )
        raise PipelineRunError("payroll readiness pipeline failed") from exc

    append_audit_event(
        _pipeline_audit_event(
            event_type="pipeline_completed",
            details={
                "pipeline_status": output.pipeline_status,
                "blocking_issues_count": len(output.blocking_issues),
                "warnings_count": len(output.warnings),
                "deterministic_blockers_count": output.deterministic_readiness.blocking_issue_count,
            },
            **common_pipeline_event,
        ),
        path=audit_path,
    )
    return output
