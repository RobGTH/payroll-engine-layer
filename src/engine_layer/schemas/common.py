"""Shared schema components."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Actor(EngineBaseModel):
    actor_type: Literal["employee", "payroll_admin", "manager", "system"]
    actor_id: str
    source: str


class EngineMeta(EngineBaseModel):
    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    engine_version: str
    engine_call_id: str
    correlation_id: str
    org_id: str


class Money(EngineBaseModel):
    amount: str
    currency: str = "AUD"


class PayBasis(EngineBaseModel):
    rate_type: Literal["hourly", "salary"]
    amount: str
    currency: str = "AUD"
    period: Literal["hour", "year"]
