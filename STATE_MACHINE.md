# Incident agent — state machine (derived from spec)

One run moves through these phases. Every transition is driven by a POST to
`/receipts`; the run's stored state is the single source of truth. Replay =
read stored state, never recompute.

## Phase 0 — CREATE (`POST /v2/incidents`)
- Validate profile == "ga5-incident-agent/v2" else 400/422, create nothing.
- If runId already stored with SAME content → return stored response (replay).
- If runId already stored with DIFFERENT content → 409.
- Else: run the model planner ONCE.
  - Planner picks rootCause ∈ allowedRootCauses, cites 2–4 evidence IDs.
  - Planner picks 1–3 diagnostic tools from catalog with incident-specific args.
- Persist diagnosis + planned diagnostics.
- Emit diagnostic dispatches (phase="diagnostic", attempt=1), each with a
  CLIENT span id in its traceparent, each citing ≥1 of the diagnosis evidence.
- Response: status="waiting", diagnosis, dispatches=[...], approvals=[].
- **No model call ever again after this** for this run.

## Phase 1 — DIAGNOSTIC RECEIPTS (`POST .../receipts`)
Grader posts outcomes for pending diagnostic calls. For each outcome:
- Accept ONLY if it matches a pending (actionId, callId, attempt). Else ignore.
- status 200 → mark that diagnostic confirmed/succeeded.
- status 503 → allow EXACTLY ONE retry: re-dispatch same actionId/callId,
  attempt+1, NEW client span id, resend_count=1. A retry response must NOT
  contain an approval request.
- status 0 & errorType "timeout" → that diagnostic FAILS. Any effect that
  depended on it must NOT be sent (timeout suppression).
- Record receipt in receiptLog.

Transition once all diagnostics are resolved (succeeded, or failed/suppressed):
- Determine the single justified effect tool from the catalog.
- If effect ∈ policy.approvalRequiredFor (or is rollback_deployment/
  disable_feature) → go to Phase 2 (APPROVAL).
- Else → dispatch the one effect (phase="effect"), go to Phase 3 pending its
  receipt.
- If the effect's prerequisite diagnostic timed out → suppress the effect,
  go to terminal (failed or completed-with-suppression), emit no effect.

## Phase 2 — APPROVAL GATE
- Response: status="waiting", dispatches=[], approvals=[{approvalId, actionId,
  toolName, argumentsDigest}]. actionId here RESERVES the logical action id
  the later effect dispatch will reuse.
- argumentsDigest = lowercase sha256 over recursively key-sorted compact JSON
  of the effect's arguments.
- Grader posts approval receipt {approvalId, decision:"approved", nonce}.
  - Accept only for the exact pending approvalId.
  - On approved → dispatch effect with matching approvalId + approvalNonce,
    phase="effect". Record approval receipt (second receiptLog shape).
  - (decision != approved handling: no destructive effect; terminal.)

## Phase 3 — EFFECT RECEIPT
- Grader posts outcome for the effect call. Record it.
- Transition to terminal: status="completed" (or "failed").

## Phase 4 — TERMINAL / FINAL
Final response (and GET) has NO pending work: dispatches/approvals omitted or
empty arrays. Contains:
- runId, status completed|failed
- diagnosis {rootCause, evidence}
- chosenEffect
- suppressed [...]
- actionLog: every dispatch exactly as issued
- receiptLog: tool-outcome receipts (shape A) + approval receipt (shape B)
- otlp: {resourceSpans:[{scopeSpans:[{spans:[...]}]}]}

## Conflict / replay rules (all phases)
- Same runId + changed content → 409.
- Same receiptId + changed content → 409.
- Identical request / identical receipt replay → same stored JSON, no model
  call, no new action.
- GET → current stored state.
- Error & replay responses must not echo sensitive input.

## Hard caps to respect
- Wrong destructive target / unapproved destructive call / effect after its
  prereq timed out / sensitive leak → 0.5/4.
- No valid action attempt observed in current run → 0.
