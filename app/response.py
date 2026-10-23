"""
Response assembly. Every response is a pure read of stored state, so an
identical replay returns byte-equivalent JSON. Nothing here calls the model
or mutates state.

Sensitive input is never stored (see statemachine.content_fingerprint /
_sanitize_incident), so it cannot appear here -- responses are structurally
incapable of echoing it.
"""
from __future__ import annotations

from typing import Any

from .trace import build_trace


def _pending_dispatches(state: dict) -> list[dict]:
    """Dispatches for attempts still awaiting an outcome, in issue order."""
    pending_keys = set()
    for a in state["logicalActions"]:
        for att in a["attempts"]:
            if att["status"] is None:
                pending_keys.add((a["actionId"], a["callId"], att["attempt"]))
    out = []
    for d in state["actionLog"]:
        key = (d["actionId"], d["callId"], d.get("attempt", 1))
        if key in pending_keys:
            out.append(d)
    return out


def _approval_requests(state: dict) -> list[dict]:
    appr = state.get("approval")
    if appr and appr.get("status") == "pending":
        return [{
            "approvalId": appr["approvalId"],
            "actionId": appr["actionId"],
            "toolName": appr["toolName"],
            "argumentsDigest": appr["argumentsDigest"],
        }]
    return []


def build_response(state: dict) -> dict:
    """The response for a create or receipt call, reflecting current phase."""
    if state["phase"] == "terminal":
        return build_final(state)

    body: dict[str, Any] = {
        "runId": state["runId"],
        "status": "waiting",
        "diagnosis": state["diagnosis"],
        "dispatches": _pending_dispatches(state),
        "approvals": _approval_requests(state),
    }
    return body


def build_final(state: dict) -> dict:
    """Terminal/GET response: full stored result. No pending work."""
    body: dict[str, Any] = {
        "runId": state["runId"],
        "status": state["status"],
        "diagnosis": state["diagnosis"],
        "chosenEffect": state.get("chosenEffect"),
        "suppressed": state.get("suppressed", []),
        "actionLog": state["actionLog"],
        "receiptLog": state["receiptLog"],
        "otlp": build_trace(state),
    }
    # A final response has no pending work: include empty arrays for clarity.
    body["dispatches"] = []
    body["approvals"] = []
    return body


def build_get(state: dict) -> dict:
    """GET returns current stored state as a response envelope."""
    if state["phase"] == "terminal":
        return build_final(state)
    return build_response(state)
