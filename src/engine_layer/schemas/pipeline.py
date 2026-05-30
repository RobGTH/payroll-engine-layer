"""Pipeline output schemas for end-to-end readiness workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import EngineBaseModel, EngineMeta
from .engine1 import Engine1Output
from .engine2 import Engine2Output

PipelineSource = Literal["engine1", "engine2", "pipeline", "deterministic"]


class PipelineIssue(EngineBaseModel):
    source: PipelineSource
    category: Literal["blocker", "warning", "informational"]
    code: str
    path: str
    message: str


class PipelineNextAction(EngineBaseModel):
    source: PipelineSource
    action: str
    target_path: str
    priority: Literal["low", "medium", "high"]


class OnboardingReadinessSummary(EngineBaseModel):
    status: Literal["incomplete", "ready_for_employment", "compliance_pending"]
    record_validity: Literal["valid", "invalid"]
    blocking_issue_count: int
    warning_count: int


class PayrollReadinessSummary(EngineBaseModel):
    status: Literal["blocked", "needs_review", "ready_for_calculation"]
    include_in_pay_run: bool
    blocking_issue_count: int
    warning_count: int


class DeterministicReadinessSummary(EngineBaseModel):
    blocking_issue_count: int
    warning_count: int
    informational_count: int
    has_blockers: bool


class PayrollReadinessPipelineOutput(EngineMeta):
    employee_id: str
    pay_period_id: str
    pipeline_status: Literal["blocked", "needs_review", "ready"]
    onboarding_readiness: OnboardingReadinessSummary
    payroll_readiness: PayrollReadinessSummary
    deterministic_readiness: DeterministicReadinessSummary
    blocking_issues: list[PipelineIssue] = Field(default_factory=list)
    warnings: list[PipelineIssue] = Field(default_factory=list)
    informational_items: list[PipelineIssue] = Field(default_factory=list)
    recommended_next_actions: list[PipelineNextAction] = Field(default_factory=list)
    engine1_output: Engine1Output
    engine2_output: Engine2Output
