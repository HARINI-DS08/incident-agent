"""
Request/response schemas, built directly from the spec's exact envelopes.
Unlike the mailroom build, the incident spec gives the full shapes, so
these are the real contract rather than guesses.

We keep inbound models permissive (extra="allow") where the grader may add
fields, but validate the fields we depend on. The `sensitive` object is
accepted but MUST NEVER be forwarded to the model or exported to telemetry
-- that's enforced in the planner/trace layers, not here.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

PROFILE = "ga5-incident-agent/v2"


# ------------------------------------------------------------- inbound: create

class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    inputSchema: Optional[dict[str, Any]] = None


class Policy(BaseModel):
    model_config = ConfigDict(extra="allow")
    maximumDiagnostics: int = 3
    effectTools: list[str] = []
    approvalRequiredFor: list[str] = []
    doNotExport: list[str] = []


class Incident(BaseModel):
    model_config = ConfigDict(extra="allow")
    incidentId: str
    title: Optional[str] = None
    service: Optional[str] = None
    severity: Optional[str] = None
    transcript: str
    allowedRootCauses: list[str]


class CreateIncidentRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    profile: str
    runId: str
    agentName: Optional[str] = None
    publicMarker: str
    sensitive: Optional[dict[str, Any]] = None
    incident: Incident
    toolCatalog: list[ToolSpec] = []
    policy: Policy


# ------------------------------------------------------------ inbound: receipts

class OutcomeIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    actionId: str
    callId: str
    attempt: int
    status: int
    resultClass: Optional[str] = None
    errorType: Optional[str] = None
    nonce: Optional[str] = None


class ApprovalIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    approvalId: str
    decision: str
    nonce: Optional[str] = None


class ReceiptRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    receiptId: str
    outcomes: list[OutcomeIn] = []
    approvals: list[ApprovalIn] = []


# --------------------------------------------------------------------- outbound
# Responses are assembled as plain dicts from stored state (so replay is a
# straight read-back), so we don't model every outbound field here. These
# constants keep status strings consistent.

STATUS_WAITING = "waiting"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

PHASE_DIAGNOSTIC = "diagnostic"
PHASE_EFFECT = "effect"
