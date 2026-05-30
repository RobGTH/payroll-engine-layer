"""Deterministic payroll readiness checks for obvious factual blockers."""

from __future__ import annotations

from .schemas.pipeline import PipelineIssue, PipelineNextAction


class DeterministicReadinessResult:
    """Container for deterministic readiness findings."""

    def __init__(
        self,
        *,
        blockers: list[PipelineIssue] | None = None,
        warnings: list[PipelineIssue] | None = None,
        informational_items: list[PipelineIssue] | None = None,
        next_actions: list[PipelineNextAction] | None = None,
    ) -> None:
        self.blockers = blockers or []
        self.warnings = warnings or []
        self.informational_items = informational_items or []
        self.next_actions = next_actions or []

    @property
    def has_blockers(self) -> bool:
        return bool(self.blockers)


def _issue(category: str, code: str, path: str, message: str) -> PipelineIssue:
    return PipelineIssue(
        source="deterministic",
        category=category,
        code=code,
        path=path,
        message=message,
    )


def _action(action: str, target_path: str, priority: str) -> PipelineNextAction:
    return PipelineNextAction(
        source="deterministic",
        action=action,
        target_path=target_path,
        priority=priority,
    )


def _is_blank(value: object) -> bool:
    return value is None or value == ""


def evaluate_deterministic_readiness(
    *,
    engine1_input: dict,
    engine2_input: dict,
    engine1_output: object | None = None,
) -> DeterministicReadinessResult:
    """Return transparent local readiness findings from supplied records."""
    blockers: list[PipelineIssue] = []
    warnings: list[PipelineIssue] = []
    informational_items: list[PipelineIssue] = []
    next_actions: list[PipelineNextAction] = []

    employee_record = engine1_input.get("employee_record", {})
    banking = employee_record.get("banking", {})
    tax = employee_record.get("tax", {})
    super_profile = employee_record.get("super", {})
    employment = employee_record.get("employment", {})
    pay_period = engine2_input.get("pay_period", {})
    earnings = engine2_input.get("earnings", {})
    pay_basis = earnings.get("pay_basis") or employment.get("pay_basis")

    if getattr(engine1_output, "record_validity", None) == "invalid":
        blockers.append(
            _issue(
                "blocker",
                "engine1_record_invalid",
                "engine1_output.record_validity",
                "Engine 1 marked the employee record invalid.",
            )
        )
        next_actions.append(
            _action("complete_onboarding", "employee_record", "high")
        )

    if getattr(engine1_output, "onboarding_status", None) == "incomplete":
        blockers.append(
            _issue(
                "blocker",
                "onboarding_incomplete",
                "engine1_output.onboarding_status",
                "Engine 1 marked onboarding incomplete.",
            )
        )
        next_actions.append(
            _action("complete_onboarding", "employee_record", "high")
        )

    if banking.get("banking_status") != "verified":
        blockers.append(
            _issue(
                "blocker",
                "banking_not_verified",
                "employee_record.banking.banking_status",
                "Banking details must be verified before payroll preparation can proceed.",
            )
        )
        next_actions.append(_action("verify_banking", "employee_record.banking", "high"))

    if tax.get("tax_profile_status") != "verified":
        blockers.append(
            _issue(
                "blocker",
                "tax_not_verified",
                "employee_record.tax.tax_profile_status",
                "Tax profile must be verified before payroll preparation can proceed.",
            )
        )
        next_actions.append(_action("verify_tax", "employee_record.tax", "high"))

    if super_profile.get("super_profile_status") != "verified":
        blockers.append(
            _issue(
                "blocker",
                "super_not_verified",
                "employee_record.super.super_profile_status",
                "Super profile must be verified before payroll preparation can proceed.",
            )
        )
        next_actions.append(_action("verify_super", "employee_record.super", "high"))

    for field in ("pay_period_id", "start_date", "end_date", "pay_date"):
        if _is_blank(pay_period.get(field)):
            blockers.append(
                _issue(
                    "blocker",
                    f"pay_period_{field}_missing",
                    f"pay_period.{field}",
                    f"Pay period field `{field}` is required for payroll preparation.",
                )
            )
            next_actions.append(_action("review_pay_period", f"pay_period.{field}", "high"))

    if not isinstance(pay_basis, dict):
        blockers.append(
            _issue(
                "blocker",
                "pay_basis_missing",
                "earnings.pay_basis",
                "Pay basis is required before payroll preparation can proceed.",
            )
        )
        next_actions.append(_action("review_pay_basis", "earnings.pay_basis", "high"))
    elif pay_basis.get("rate_type") == "hourly" and earnings.get("gross_pay_override") is None:
        hours = earnings.get("hours")
        regular_hours = hours.get("regular_hours") if isinstance(hours, dict) else None
        overtime_hours = hours.get("overtime_hours") if isinstance(hours, dict) else None
        if _is_blank(regular_hours) and _is_blank(overtime_hours):
            blockers.append(
                _issue(
                    "blocker",
                    "hourly_hours_missing",
                    "earnings.hours",
                    "Hourly employees require hours when no gross pay override is supplied.",
                )
            )
            next_actions.append(_action("review_hours", "earnings.hours", "high"))

    if not blockers:
        informational_items.append(
            _issue(
                "informational",
                "deterministic_checks_clear",
                "pipeline.deterministic_readiness",
                "No deterministic readiness blockers were found.",
            )
        )

    return DeterministicReadinessResult(
        blockers=blockers,
        warnings=warnings,
        informational_items=informational_items,
        next_actions=next_actions,
    )
