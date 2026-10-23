"""
The run state machine. Pure-ish orchestration over stored state:
`create_run` runs the single model call and lays out the initial trace +
diagnostic dispatches; `process_receipt` advances the run as the grader
posts outcomes/approvals. Trace/response assembly is delegated to trace.py
and response.py so this module only owns transitions.

All span IDs and logical-action records are allocated here (at first
creation) and stored, so replay/GET reproduce identical output without
re-planning or re-generating anything.
"""
from __future__ import annotations

from typing import Any

from . import idgen
from .idgen import arguments_digest, format_traceparent, new_span_id
from .planner import PlannerError, plan

APPROVAL_TOOLS_DEFAULT = {"rollback_deployment", "disable_feature"}


def _sanitize_incident(incident: dict) -> dict:
    """Copy of the incident WITHOUT anything sensitive. The `sensitive`
    object lives at the request top level, not inside incident, but we also
    defensively drop any key literally named 'sensitive' here."""
    return {k: v for k, v in incident.items() if k != "sensitive"}


def content_fingerprint(req: dict) -> str:
    """Fingerprint of the request content that must stay stable for a runId.
    Excludes volatile per-run fields (markers, trace context, nonces) but
    includes the incident, catalog, and policy that define the decision.
    The sensitive object is excluded (never fingerprinted or stored)."""
    material = {
        "profile": req.get("profile"),
        "agentName": req.get("agentName"),
        "publicMarker": req.get("publicMarker"),
        "incident": _sanitize_incident(req.get("incident", {})),
        "toolCatalog": req.get("toolCatalog", []),
        "policy": req.get("policy", {}),
    }
    return idgen.fingerprint(material)


def create_run(req: dict, incoming_traceparent: str | None,
               incoming_tracestate: str | None = None) -> dict:
    """Build the initial stored run state: run the planner once, allocate the
    trace skeleton, and lay out the first diagnostic dispatches. Returns the
    full stored-state dict (caller persists it)."""
    run_id = req["runId"]
    marker = req["publicMarker"]
    incident = _sanitize_incident(req.get("incident", {}))
    catalog = [t if isinstance(t, dict) else t.model_dump() for t in req.get("toolCatalog", [])]
    policy = req.get("policy", {}) or {}

    # --- single model call
    the_plan = plan(incident, catalog, policy)  # may raise PlannerError

    # --- trace skeleton: allocate all "structural" span ids up front
    parsed = idgen.parse_traceparent(incoming_traceparent)
    trace_id = parsed["trace_id"] if parsed else idgen.new_trace_id()
    trace = {
        "trace_id": trace_id,
        "incoming_parent_id": parsed["parent_id"] if parsed else None,
        "tracestate": incoming_tracestate if parsed else None,
        "server_span_id": new_span_id(),
        "agent_span_id": new_span_id(),
        "chat_span_id": new_span_id(),
        "join_span_id": new_span_id(),
        "approval_span_id": new_span_id(),
        "model_name": _model_name(),
    }

    # --- diagnostic logical actions + first attempt each
    approval_required = set(policy.get("approvalRequiredFor", [])) | APPROVAL_TOOLS_DEFAULT
    logical_actions: list[dict] = []
    dispatches: list[dict] = []
    evidence = the_plan["evidence"]

    for i, diag in enumerate(the_plan["diagnostics"]):
        action_id = idgen.stable_id("act", run_id, "diag", str(i))
        call_id = idgen.stable_id("call", run_id, "diag", str(i))
        client_span = new_span_id()
        cites = diag["evidence"] or [evidence[0]]
        action = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": diag["toolName"],
            "arguments": diag["arguments"],
            "evidence": list(dict.fromkeys(cites)),
            "execute_span_id": new_span_id(),
            "attempts": [{"attempt": 1, "client_span_id": client_span,
                          "status": None, "errorType": None, "receiptId": None, "nonce": None}],
            "state": "pending",  # pending | succeeded | failed
        }
        logical_actions.append(action)
        dispatches.append(_dispatch_from_action(
            action, trace_id, attempt=1, tracestate=trace.get("tracestate")))

    state: dict[str, Any] = {
        "runId": run_id,
        "agentName": req.get("agentName") or "incident-response",
        "publicMarker": marker,
        "status": "waiting",
        "diagnosis": {"rootCause": the_plan["rootCause"], "evidence": evidence},
        "plannedEffect": the_plan["effect"],   # dispatched later, after diagnostics
        "approvalRequiredFor": sorted(approval_required),
        "logicalActions": logical_actions,
        "trace": trace,
        "receiptLog": [],
        "actionLog": list(dispatches),          # every dispatch exactly as issued
        "approval": None,                        # populated when approval gate opens
        "chosenEffect": None,
        "suppressed": [],
        "phase": "diagnostic",                   # diagnostic | approval | effect | terminal
        "contentFingerprint": content_fingerprint(req),
        "seenReceiptFingerprints": {},           # receiptId -> content fp (idempotency/409)
    }
    return state


def _model_name() -> str:
    from .planner import MODEL_NAME
    return MODEL_NAME


def _dispatch_from_action(action: dict, trace_id: str, attempt: int,
                          tracestate: str | None = None) -> dict:
    """Build the outbound dispatch object for a given attempt of an action.
    The traceparent's span id is that attempt's CLIENT span id."""
    att = next(a for a in action["attempts"] if a["attempt"] == attempt)
    d = {
        "actionId": action["actionId"],
        "callId": action["callId"],
        "phase": action["phase"],
        "toolName": action["toolName"],
        "arguments": action["arguments"],
        "evidence": action.get("evidence", []),
        "attempt": attempt,
        "traceparent": format_traceparent(trace_id, att["client_span_id"], "01"),
    }
    if tracestate:
        d["tracestate"] = tracestate
    if action["phase"] == "effect" and action.get("approvalId"):
        d["approvalId"] = action["approvalId"]
        d["approvalNonce"] = action.get("approvalNonce")
    # diagnostic dispatches carry no approval fields
    return {k: v for k, v in d.items() if v is not None}


# ------------------------------------------------------------- receipt handling

def process_receipt(state: dict, receipt: dict) -> dict:
    """Advance the run given a posted receipt. Mutates and returns state.
    Handles diagnostic outcomes (200/503/timeout), approval decisions, and
    the effect outcome, then computes the next transition."""
    if state["phase"] == "terminal":
        return state  # nothing pending; replay handled by caller before this

    # --- approvals
    for appr in receipt.get("approvals", []):
        _apply_approval(state, receipt, appr)

    # --- tool outcomes
    for outcome in receipt.get("outcomes", []):
        _apply_outcome(state, receipt, outcome)

    _advance(state)
    return state


def _find_action(state: dict, action_id: str, call_id: str) -> dict | None:
    for a in state["logicalActions"]:
        if a["actionId"] == action_id and a["callId"] == call_id:
            return a
    return None


def _apply_outcome(state: dict, receipt: dict, outcome: dict) -> None:
    action = _find_action(state, outcome.get("actionId"), outcome.get("callId"))
    if action is None:
        return  # accept outcomes only for known calls
    att_num = outcome.get("attempt")
    att = next((a for a in action["attempts"] if a["attempt"] == att_num), None)
    if att is None or att["status"] is not None:
        return  # accept only for a pending (not-yet-resolved) attempt

    status = outcome.get("status")
    error_type = outcome.get("errorType")
    att["status"] = status
    att["errorType"] = error_type
    att["receiptId"] = receipt.get("receiptId")
    att["nonce"] = outcome.get("nonce")

    # record in receiptLog (shape A)
    log_entry = {
        "receiptId": receipt.get("receiptId"),
        "actionId": action["actionId"],
        "callId": action["callId"],
        "attempt": att_num,
        "status": status,
        "resultClass": outcome.get("resultClass"),
        "nonce": outcome.get("nonce"),
    }
    # status 0 is ambiguous on its own; echo the error class so the timeout in
    # the receipt log can be correlated with the span's error.type=timeout.
    if error_type:
        log_entry["errorType"] = error_type
    elif status == 503:
        log_entry["errorType"] = "503"
    elif status == 0:
        log_entry["errorType"] = "timeout"
    state["receiptLog"].append(log_entry)

    if error_type == "timeout" or status == 0:
        action["state"] = "failed"
        if action["actionId"] not in state["suppressed"]:
            pass  # suppression recorded at effect stage
    elif status == 503:
        # exactly one retry allowed
        already_retried = any(a["attempt"] > 1 for a in action["attempts"])
        if not already_retried:
            new_attempt = att_num + 1
            action["attempts"].append({
                "attempt": new_attempt, "client_span_id": new_span_id(),
                "status": None, "errorType": None, "receiptId": None, "nonce": None,
            })
            action["state"] = "pending"
            # emit the retry dispatch
            state["actionLog"].append(_dispatch_from_action(
                action, state["trace"]["trace_id"], new_attempt,
                tracestate=state["trace"].get("tracestate")))
        else:
            action["state"] = "failed"
    elif isinstance(status, int) and 200 <= status < 300:
        action["state"] = "succeeded"
    else:
        action["state"] = "failed"


def _apply_approval(state: dict, receipt: dict, appr: dict) -> None:
    pending = state.get("approval")
    if not pending or pending.get("status") != "pending":
        return
    if appr.get("approvalId") != pending["approvalId"]:
        return
    decision = appr.get("decision")
    pending["decision"] = decision
    pending["nonce"] = appr.get("nonce")
    pending["receiptId"] = receipt.get("receiptId")
    pending["status"] = "resolved"

    # record approval receipt (shape B)
    state["receiptLog"].append({
        "receiptId": receipt.get("receiptId"),
        "approvalId": pending["approvalId"],
        "decision": decision,
        "nonce": appr.get("nonce"),
    })

    if decision == "approved":
        _dispatch_effect(state, approved=True)
    else:
        # destructive effect not approved -> terminal failed, no effect
        state["phase"] = "terminal"
        state["status"] = "failed"


# ----------------------------------------------------------------- transitions

def _advance(state: dict) -> None:
    """Compute the next phase after applying receipts."""
    if state["phase"] == "diagnostic":
        diags = [a for a in state["logicalActions"] if a["phase"] == "diagnostic"]
        if any(a["state"] == "pending" for a in diags):
            return  # still waiting on diagnostics/retries
        # all diagnostics resolved
        any_timeout = any(
            (att.get("errorType") == "timeout" or att.get("status") == 0)
            for a in diags for att in a["attempts"]
        )
        all_ok = all(a["state"] == "succeeded" for a in diags)
        if any_timeout or not all_ok:
            # timeout suppression: do NOT send the effect
            eff = state["plannedEffect"]["toolName"]
            state["suppressed"] = list(dict.fromkeys(state["suppressed"] + [eff]))
            state["phase"] = "terminal"
            state["status"] = "failed"
            return
        # diagnostics confirmed -> effect stage
        effect_tool = state["plannedEffect"]["toolName"]
        if effect_tool in set(state["approvalRequiredFor"]):
            _open_approval_gate(state)
        else:
            _dispatch_effect(state, approved=False)

    elif state["phase"] == "effect":
        eff_actions = [a for a in state["logicalActions"] if a["phase"] == "effect"]
        if eff_actions and all(a["state"] in ("succeeded", "failed") for a in eff_actions):
            state["phase"] = "terminal"
            state["status"] = "completed" if any(a["state"] == "succeeded" for a in eff_actions) else "failed"


def _open_approval_gate(state: dict) -> None:
    run_id = state["runId"]
    effect = state["plannedEffect"]
    action_id = idgen.stable_id("act", run_id, "effect")  # reserved for the later effect dispatch
    approval_id = idgen.stable_id("appr", run_id, "effect")
    state["approval"] = {
        "approvalId": approval_id,
        "actionId": action_id,
        "toolName": effect["toolName"],
        "argumentsDigest": arguments_digest(effect["arguments"]),
        "status": "pending",
        "decision": None,
        "nonce": None,
    }
    state["phase"] = "approval"


def _dispatch_effect(state: dict, approved: bool) -> None:
    run_id = state["runId"]
    effect = state["plannedEffect"]
    trace_id = state["trace"]["trace_id"]
    # reuse the reserved action id if approval happened, else a fresh stable one
    if state.get("approval"):
        action_id = state["approval"]["actionId"]
    else:
        action_id = idgen.stable_id("act", run_id, "effect")
    call_id = idgen.stable_id("call", run_id, "effect")

    action = {
        "actionId": action_id,
        "callId": call_id,
        "phase": "effect",
        "toolName": effect["toolName"],
        "arguments": effect["arguments"],
        "evidence": state["diagnosis"]["evidence"][:1],
        "execute_span_id": new_span_id(),
        "attempts": [{"attempt": 1, "client_span_id": new_span_id(),
                      "status": None, "errorType": None, "receiptId": None, "nonce": None}],
        "state": "pending",
    }
    if approved and state.get("approval"):
        action["approvalId"] = state["approval"]["approvalId"]
        action["approvalNonce"] = state["approval"]["nonce"]

    state["logicalActions"].append(action)
    state["chosenEffect"] = effect["toolName"]
    state["actionLog"].append(_dispatch_from_action(
        action, trace_id, attempt=1, tracestate=state["trace"].get("tracestate")))
    state["phase"] = "effect"
