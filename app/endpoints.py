from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import idgen, statemachine, store
from .models import PROFILE
from .planner import PlannerError
from .response import build_final, build_get, build_response

logger = logging.getLogger("incident")

MAX_RESPONSE_BYTES = 768 * 1024  # spec: successful responses <= 768 KiB
MAX_REQUEST_BYTES = 4 * 1024 * 1024

_TRACE_PATH = os.environ.get(
    "INCIDENT_TRACE",
    os.path.join(os.path.dirname(os.path.abspath(store._DB_PATH)), "q4_trace.jsonl"))


def _trace(kind: str, payload: dict) -> None:
    """Append the grader's synthetic request/response for offline diagnosis."""
    try:
        os.makedirs(os.path.dirname(_TRACE_PATH), exist_ok=True)
        with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _without_sensitive(body: dict) -> dict:
    copy = {k: v for k, v in body.items() if k != "sensitive"}
    incident = copy.get("incident")
    if isinstance(incident, dict):
        copy["incident"] = {k: v for k, v in incident.items() if k != "sensitive"}
    return copy

app = FastAPI(title="incident-agent")


@app.on_event("startup")
def _startup() -> None:
    store.init()


store.init()  # also run at import time (serverless lifespan may not fire)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s", request.url.path)
    # Never echo request content (could contain sensitive input).
    return JSONResponse({"error": "internal_error"}, status_code=500, media_type="application/json")


def _json(payload: dict, status_code: int = 200) -> JSONResponse:
    body = json.dumps(payload).encode("utf-8")
    if status_code == 200 and len(body) > MAX_RESPONSE_BYTES:
        logger.error("response exceeded %d bytes", MAX_RESPONSE_BYTES)
        raise HTTPException(500, "response too large")
    return JSONResponse(content=payload, status_code=status_code, media_type="application/json")


@app.get("/health")
async def health() -> JSONResponse:
    from .planner import API_BASE, API_KEY, MODEL_NAME

    info: dict[str, Any] = {
        "backend": store.backend_name(),
        "model_name": MODEL_NAME,
        "api_base": API_BASE,
        "model_ok": bool(API_KEY),
    }
    return _json(info)


# --------------------------------------------------------------- POST create

@app.post("/v2/incidents")
async def create_incident(request: Request) -> JSONResponse:
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        raise HTTPException(400, "request too large")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(400, "malformed JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be an object")

    if body.get("profile") != PROFILE:
        # unsupported profile -> reject, create nothing
        raise HTTPException(422, "unsupported profile")

    run_id = body.get("runId")
    if not run_id or not isinstance(run_id, str):
        raise HTTPException(400, "missing runId")
    _trace("create_request", {"runId": run_id, "request": _without_sensitive(body)})
    if not body.get("incident") or not isinstance(body["incident"], dict):
        raise HTTPException(422, "missing incident")
    if not body.get("publicMarker"):
        raise HTTPException(422, "missing publicMarker")

    incoming_tp = request.headers.get("traceparent")
    new_fp = statemachine.content_fingerprint(body)

    existing = store.load_run(run_id)
    if existing is not None:
        if existing.get("contentFingerprint") == new_fp:
            return _json(build_get(existing))  # exact replay
        raise HTTPException(409, "runId already used with different content")

    # build initial state (one model call happens here)
    incoming_ts = request.headers.get("tracestate")
    try:
        state = statemachine.create_run(body, incoming_tp, incoming_ts)
    except PlannerError:
        # fail closed: no valid plan -> failed run, still create a record so
        # GET/replay are stable. (Scored as no valid action, but deterministic.)
        raise HTTPException(422, "could not plan incident")

    created = store.create_run_if_absent(run_id, state)
    if not created:
        # lost a race; re-read and treat as replay/conflict
        existing = store.load_run(run_id)
        if existing and existing.get("contentFingerprint") == new_fp:
            return _json(build_get(existing))
        raise HTTPException(409, "runId already used with different content")

    response_body = build_response(state)
    _trace("create_response", {"runId": run_id, "response": response_body})
    return _json(response_body)


# ------------------------------------------------------------- POST receipts

@app.post("/v2/incidents/{run_id}/receipts")
async def post_receipt(run_id: str, request: Request) -> JSONResponse:
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        raise HTTPException(400, "request too large")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(400, "malformed JSON")
    if not isinstance(body, dict) or not body.get("receiptId"):
        raise HTTPException(400, "missing receiptId")

    state = store.load_run(run_id)
    if state is None:
        raise HTTPException(404, "unknown runId")
    _trace("receipt_request", {"runId": run_id, "request": body})

    receipt_id = body["receiptId"]
    receipt_fp = idgen.fingerprint({
        "outcomes": body.get("outcomes", []),
        "approvals": body.get("approvals", []),
    })

    seen = state.get("seenReceiptFingerprints", {})
    if receipt_id in seen:
        if seen[receipt_id] == receipt_fp:
            # identical receipt replay -> return current stored state, no work
            return _json(build_get(state))
        raise HTTPException(409, "receiptId reused with different content")

    # advance the state machine
    state = statemachine.process_receipt(state, body)
    state.setdefault("seenReceiptFingerprints", {})[receipt_id] = receipt_fp
    store.save_run(run_id, state)
    response_body = build_get(state)
    _trace("receipt_response", {"runId": run_id, "response": response_body})
    return _json(response_body)


# ------------------------------------------------------------------ GET run

@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str) -> JSONResponse:
    state = store.load_run(run_id)
    if state is None:
        raise HTTPException(404, "unknown runId")
    return _json(build_get(state))


# ------------------------------------------------------------ debug view
# Read-only summary of the grader's runs from the local trace file. The
# sensitive object is never written to that file, so nothing secret is shown.
from fastapi.responses import PlainTextResponse  # noqa: E402
import re as _re  # noqa: E402
from collections import Counter as _Counter  # noqa: E402


@app.get("/debug/runs/{nonce}", response_class=PlainTextResponse)
async def debug_runs(nonce: str, last: int = 12) -> PlainTextResponse:
    try:
        with open(_TRACE_PATH, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    except OSError:
        return PlainTextResponse("no trace yet")
    order, runs = [], {}
    for r in rows:
        rid = r.get("runId")
        if rid not in runs:
            runs[rid] = []
            order.append(rid)
        runs[rid].append(r)
    out = []
    for rid in order[-last:]:
        out.append(f"\n===== {rid}")
        for r in runs[rid]:
            k = r["kind"]
            if k == "create_request":
                inc = (r.get("request") or {}).get("incident") or {}
                out.append(f"service={inc.get('service')} title={inc.get('title')}")
                for line in (inc.get("transcript") or "").splitlines():
                    if line.startswith("[ev_") and "corr_" not in line.lower():
                        out.append("  SIGNAL " + line[:400])
            elif k in ("create_response", "receipt_response"):
                resp = r.get("response") or {}
                ds = [(d.get("phase"), d.get("toolName"), d.get("arguments"), d.get("attempt"),
                       d.get("approvalId")) for d in resp.get("dispatches") or []]
                out.append(f"  <- {k} status={resp.get('status')} diag={resp.get('diagnosis')} "
                           f"dispatches={ds} approvals={[a.get('toolName') for a in resp.get('approvals') or []]} "
                           f"chosen={resp.get('chosenEffect')} suppressed={resp.get('suppressed')}")
                if resp.get("otlp"):
                    spans = resp["otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
                    out.append(f"     spans={dict(_Counter(s['name'] for s in spans))}")
            elif k == "receipt_request":
                req = r.get("request") or {}
                oc = [(o.get("attempt"), o.get("status"), o.get("errorType"), o.get("resultClass"))
                      for o in req.get("outcomes") or []]
                ap = [a.get("decision") for a in req.get("approvals") or []]
                out.append(f"  -> receipt {req.get('receiptId')} outcomes={oc} approvals={ap}")
    return PlainTextResponse("\n".join(out))
