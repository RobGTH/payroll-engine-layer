"""Engine 2 input and output schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import Actor, EngineBaseModel, EngineMeta, Money, PayBasis
from .engine1 import EmployeeRecord


class PayPeriod(EngineBaseModel):
    pay_period_id: str
    start_date: str
    end_date: str
    pay_date: str
    pay_schedule_group: str | None = None


class HoursSummary(EngineBaseModel):
    regular_hours: str = "0"
    overtime_hours: str = "0"
    unpaid_leave_hours: str = "0"


class EarningsInput(EngineBaseModel):
    pay_basis: PayBasis
    hours: HoursSummary | None = None
    gross_pay_override: Money | None = None


class DeductionInput(EngineBaseModel):
    deduction_code: str
    amount: Money
    description: str | None = None


class PayrollPrepConfig(EngineBaseModel):
    require_verified_banking: bool = True
    require_verified_tax: bool = True
    require_verified_super: bool = True
    block_if_onboarding_incomplete: bool = True
    allow_manual_review_to_proceed: bool = False


class Engine1ReadinessSummary(EngineBaseModel):
    onboarding_status: Literal[
        "incomplete",
        "ready_for_employment",
        "compliance_pending",
    ]
    record_validity: Literal["valid", "invalid"]
    blocking_issue_count: int = 0
    warning_count: int = 0


class Engine2Input(EngineMeta):
    actor: Actor
    employee_record: EmployeeRecord
    engine1_readiness: Engine1ReadinessSummary | None = None
    pay_period: PayPeriod
    earnings: EarningsInput
    deductions: list[DeductionInput] = Field(default_factory=list)
    config: PayrollPrepConfig


class PayrollPrepIssue(EngineBaseModel):
    code: str
    path: str
    severity: Literal["blocking", "warning"]
    message: str


class PayrollPrepNextAction(EngineBaseModel):
    action: Literal[
        "complete_onboarding",
        "verify_banking",
        "verify_tax",
        "verify_super",
        "review_hours",
        "review_manual_override",
        "approve_for_pay_run",
    ]
    target_path: str
    priority: Literal["low", "medium", "high"]


class PayrollPrepSummary(EngineBaseModel):
    employee_id: str
    pay_period_id: str
    pay_schedule_group: str | None = None
    pay_basis: PayBasis
    estimated_gross_pay: Money | None = None
    deductions_total: Money | None = None
    ready_for_calculation: bool
    requires_manual_review: bool


class Engine2Output(EngineMeta):
    employee_id: str
    pay_period_id: str
    payroll_prep_status: Literal[
        "blocked",
        "needs_review",
        "ready_for_calculation",
    ]
    include_in_pay_run: bool
    blocking_issues: list[PayrollPrepIssue] = Field(default_factory=list)
    warnings: list[PayrollPrepIssue] = Field(default_factory=list)
    next_actions: list[PayrollPrepNextAction] = Field(default_factory=list)
    payroll_prep_summary: PayrollPrepSummary
