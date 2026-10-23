# incident-agent (GA5 v2)

AI incident-response agent: reads an evidence-tagged transcript, picks a
root cause + minimal diagnostics via one model call, runs a receipt-driven
state machine (retries, timeout suppression, destructive-effect approval),
and emits a stored result plus an exact OTLP trace.

Runs locally and passes a 12-scenario end-to-end smoke test with a stubbed
planner (no API key needed): `python3 test_smoke.py`.

## Layout
```
app/
  endpoints.py    the 3 endpoints (+ /health), replay/409/validation
  statemachine.py create_run + receipt processing; retries, suppression, approval
  planner.py      the ONE model call; sensitive never sent; outputs validated
  trace.py        OTLP builder — pure function of stored state
  response.py     waiting/terminal/GET responses, pure reads of stored state
  store.py        Upstash Redis REST (prod) / SQLite (local), auto-selected
  idgen.py        canonical JSON, args digest, opaque/stable IDs, traceparent
  models.py       request schemas from the exact spec envelopes
api/index.py      Vercel entrypoint
vercel.json       routing + maxDuration
STATE_MACHINE.md  phase-by-phase mapping to the spec
test_smoke.py     offline end-to-end test
```

## Deploy (Vercel)

1. **Storage — Upstash Redis** (required; multi-request runs need durable
   state across cold starts). Vercel dashboard → Storage → add **Upstash
   for Redis** from the Marketplace and connect it to this project. That
   injects `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` as env
   vars; `store.py` auto-detects them and uses Redis. Without them it falls
   back to `/tmp` SQLite, which will NOT survive between requests on Vercel
   and will cause spurious 409s / lost pending calls — so don't skip this.

2. **Model — Groq.** Set `INCIDENT_API_KEY` = your `gsk_...` key (Settings →
   Environment Variables → Production). Base URL and model default to Groq;
   override `INCIDENT_MODEL` / `INCIDENT_API_BASE` if needed. (It also
   accepts `MAILROOM_API_KEY` as a fallback so you can reuse the same key.)

3. **Redeploy**, then verify BEFORE spending a Check:
   ```
   curl https://YOUR-URL/health
   ```
   Want `"backend":"upstash-redis"` and `"model_ok":true`. If backend says
   `sqlite`, the Upstash env vars aren't set. If `model_ok:false`, the
   `error` field says why.

4. Submit the base URL (no path/query/fragment), e.g.
   `https://your-app.vercel.app/`

## Config (env vars)
```
UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN   durable store (Marketplace)
INCIDENT_API_KEY        Groq key (or MAILROOM_API_KEY fallback)
INCIDENT_MODEL          default llama-3.3-70b-versatile
INCIDENT_API_BASE       default https://api.groq.com/openai/v1
```

## Design decisions worth knowing

- **One model call per run**, at create time. The planner picks rootCause,
  evidence, diagnostics AND the effect up front; the effect is *stored* and
  only *dispatched* later (after diagnostics confirm), so there's still
  exactly one `chat incident-plan` span per run and timeout-suppression
  still works.
- **Trace is a pure function of stored state** — all span IDs are allocated
  when a dispatch is created and stored, so GET and replay reproduce a
  byte-identical trace with no model call and no re-generation.
- **Sensitive never touches model or telemetry.** The `sensitive` object is
  dropped before planning, excluded from the content fingerprint, and never
  stored — responses are structurally incapable of echoing it.
- **incident.join** fires only when ≥2 diagnostics fan out, with `links` to
  each diagnostic's `execute_tool` span.
- **Approval gate**: destructive effects (rollback_deployment,
  disable_feature, + anything in policy.approvalRequiredFor) return zero
  dispatches and one approval request first; the reserved `actionId` is
  reused by the later effect dispatch, which carries the approvalId + nonce.

## Known assumptions to verify against real grading

The spec is far more complete than the mailroom one, so most fields are
exact. Two areas are judgment calls the grader may score differently:

- **CLIENT-span attribute names** for the observed HTTP status. The spec
  names `http.request.method`, `http.request.resend_count`, `error.type`,
  and span status codes explicitly (all implemented). It says "the observed
  HTTP/error status" without pinning the exact key for the success status
  code; I used `http.response.status_code`. Adjust in `trace._build_client_
  attempt_span` if grading indicates otherwise.
- **Effect selection** is model-chosen from the catalog (the spec gives no
  rootCause→effect mapping). If a specific incident wants a specific effect,
  that comes from the model reading the transcript, not a hardcoded rule.
```
