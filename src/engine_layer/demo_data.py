"""Sample payloads for local demos."""

from __future__ import annotations


def sample_engine1_input() -> dict:
    """Return a design-aligned Engine 1 payload with intentional gaps."""
    return {
        "schema_version": "1.0.0",
        "engine_version": "2026-03-01",
        "engine_call_id": "call-001",
        "correlation_id": "corr-001",
        "org_id": "org-abc",
        "actor": {
            "actor_type": "payroll_admin",
            "actor_id": "pay-admin-1",
            "source": "demo_script",
        },
        "request_type": "validate_only",
        "employee_record": {
            "employee_id": "emp-001",
            "identity": {
                "legal_first_name": "Sam",
                "legal_last_name": "Lee",
                "date_of_birth": "1999-01-01",
                "email": "sam@example.com",
                "phone": None,
            },
            "work_location": {
                "country_code": "AU",
                "state_code": "NSW",
                "timezone": "Australia/Sydney",
            },
            "employment": {
                "start_date": "2026-02-01",
                "employment_type": "casual",
                "role_title": "Support",
                "department": None,
                "manager_employee_id": None,
                "pay_basis": {
                    "rate_type": "hourly",
                    "amount": "35.00",
                    "currency": "AUD",
                    "period": "hour",
                },
                "standard_hours_per_week": None,
                "pay_schedule_group": None,
            },
            "tax": {
                "tax_profile_status": "not_provided",
                "tax_id_last4": None,
                "withholding_declaration_received": None,
            },
            "super": {
                "super_profile_status": "not_provided",
                "fund_name": None,
                "member_number_last4": None,
                "contribution_rate_percent": None,
            },
            "banking": {
                "banking_status": "not_provided",
                "account_name": None,
                "bsb_last3": None,
                "account_number_last4": None,
            },
            "documents": [
                {
                    "doc_type": "right_to_work",
                    "required": True,
                    "status": "not_requested",
                    "received_at": None,
                    "verified_at": None,
                    "notes": None,
                }
            ],
            "consents": {
                "privacy_acknowledged": None,
                "policy_acknowledged": None,
            },
            "status_overrides": {
                "force_compliance_pending": None,
                "force_incomplete": None,
            },
            "metadata": {
                "created_at": None,
                "updated_at": None,
                "tags": [],
            },
        },
        "config": {
            "required_fields": {
                "identity": [
                    "legal_first_name",
                    "legal_last_name",
                    "date_of_birth",
                    "email",
                ],
                "employment": [
                    "start_date",
                    "employment_type",
                    "role_title",
                    "pay_basis",
                ],
                "tax": ["tax_profile_status"],
                "super": ["super_profile_status"],
                "banking": ["banking_status"],
                "consents": ["privacy_acknowledged"],
            },
            "required_documents": [{"doc_type": "right_to_work", "required": True}],
            "allow_partial_record_save": True,
        },
    }
