"""Engine 1 input and output schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import Actor, EngineBaseModel, EngineMeta, PayBasis


class Identity(EngineBaseModel):
    legal_first_name: str
    legal_last_name: str
    date_of_birth: str
    email: str
    phone: str | None = None


class WorkLocation(EngineBaseModel):
    country_code: str
    state_code: str | None = None
    timezone: str | None = None


class Employment(EngineBaseModel):
    start_date: str
    employment_type: Literal["full_time", "part_time", "casual", "contractor"]
    role_title: str
    department: str | None = None
    manager_employee_id: str | None = None
    pay_basis: PayBasis
    standard_hours_per_week: str | None = None
    pay_schedule_group: str | None = None


class TaxProfile(EngineBaseModel):
    tax_profile_status: Literal["not_provided", "provided", "verified"]
    tax_id_last4: str | None = None
    withholding_declaration_received: bool | None = None


class SuperProfile(EngineBaseModel):
    super_profile_status: Literal["not_provided", "provided", "verified"]
    fund_name: str | None = None
    member_number_last4: str | None = None
    contribution_rate_percent: str | None = None


class BankingProfile(EngineBaseModel):
    banking_status: Literal["not_provided", "provided", "verified"]
    account_name: str | None = None
    bsb_last3: str | None = None
    account_number_last4: str | None = None


class DocumentRecord(EngineBaseModel):
    doc_type: str
    required: bool
    status: Literal["not_requested", "requested", "received", "verified", "rejected"]
    received_at: str | None = None
    verified_at: str | None = None
    notes: str | None = None


class Consents(EngineBaseModel):
    privacy_acknowledged: bool | None = None
    policy_acknowledged: bool | None = None


class StatusOverrides(EngineBaseModel):
    force_compliance_pending: bool | None = None
    force_incomplete: bool | None = None


class RecordMetadata(EngineBaseModel):
    created_at: str | None = None
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)


class EmployeeRecord(EngineBaseModel):
    employee_id: str
    identity: Identity
    work_location: WorkLocation
    employment: Employment
    tax: TaxProfile
    super: SuperProfile
    banking: BankingProfile
    documents: list[DocumentRecord] = Field(default_factory=list)
    consents: Consents
    status_overrides: StatusOverrides
    metadata: RecordMetadata


class RequiredFieldsConfig(EngineBaseModel):
    identity: list[str] = Field(default_factory=list)
    employment: list[str] = Field(default_factory=list)
    tax: list[str] = Field(default_factory=list)
    super: list[str] = Field(default_factory=list)
    banking: list[str] = Field(default_factory=list)
    consents: list[str] = Field(default_factory=list)


class RequiredDocumentConfig(EngineBaseModel):
    doc_type: str
    required: bool


class Engine1Config(EngineBaseModel):
    required_fields: RequiredFieldsConfig
    required_documents: list[RequiredDocumentConfig] = Field(default_factory=list)
    allow_partial_record_save: bool = True


class Engine1Input(EngineMeta):
    actor: Actor
    request_type: Literal["create", "update", "validate_only"]
    employee_record: EmployeeRecord
    config: Engine1Config


class IssueItem(EngineBaseModel):
    code: str
    path: str
    message: str


class MissingField(EngineBaseModel):
    path: str
    severity: Literal["blocking", "non_blocking"]
    reason: str


class DocumentStatusItem(EngineBaseModel):
    doc_type: str
    required: bool
    status: Literal["not_requested", "requested", "received", "verified", "rejected"]


class DocumentStatusSummary(EngineBaseModel):
    required_docs_total: int
    required_docs_received: int
    required_docs_verified: int
    docs: list[DocumentStatusItem]


class RiskFlag(EngineBaseModel):
    flag: str
    severity: Literal["low", "medium", "high", "critical"]


class NextAction(EngineBaseModel):
    action: Literal["request_field", "request_document", "verify_document", "verify_field"]
    target_path: str
    doc_type: str | None = None
    priority: Literal["low", "medium", "high"]


class EmployeeRecordPatch(EngineBaseModel):
    updated_at: str | None = None
    normalized_fields: dict[str, str] = Field(default_factory=dict)


class Engine1Output(EngineMeta):
    employee_id: str
    onboarding_status: Literal["incomplete", "ready_for_employment", "compliance_pending"]
    record_validity: Literal["valid", "invalid"]
    hard_errors: list[IssueItem] = Field(default_factory=list)
    soft_warnings: list[IssueItem] = Field(default_factory=list)
    missing_fields: list[MissingField] = Field(default_factory=list)
    document_status: DocumentStatusSummary
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list)
    employee_record_patch: EmployeeRecordPatch
