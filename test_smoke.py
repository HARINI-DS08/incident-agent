import os
import sys

os.environ["INCIDENT_DB"] = "/tmp/incident_smoke.db"
for suf in ("", "-wal", "-shm"):
    p = os.environ["INCIDENT_DB"] + suf
    if os.path.exists(p):
        os.remove(p)

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient  # noqa: E402
from app import endpoints, statemachine  # noqa: E402

# ---- stub the planner so no API key / network is needed. Returns 2 diagnostics
# (so join fires) and a non-destructive effect by default.
def make_plan(effect_tool="scale_service"):
    def _plan(incident, catalog, policy):
        ev = statemachine.plan.__self__ if False else None  # noqa
        return {
            "rootCause": incident["allowedRootCauses"][0],
            "evidence": ["ev_1", "ev_2", "ev_3"],
            "diagnostics": [
                {"toolName": "query_metrics", "arguments": {"svc": "api"}, "evidence": ["ev_1"]},
                {"toolName": "check_logs", "arguments": {"svc": "api"}, "evidence": ["ev_2"]},
            ],
            "effect": {"toolName": effect_tool, "arguments": {"svc": "api", "replicas": 5}},
        }
    return _plan


statemachine.plan = make_plan()

client = TestClient(endpoints.app)

BASE_REQ = {
    "profile": "ga5-incident-agent/v2",
    "runId": "run-1",
    "agentName": "incident-response",
    "publicMarker": "marker-1",
    "sensitive": {"accessToken": "SECRET", "privateNote": "SECRET"},
    "incident": {
        "incidentId": "inc-1", "title": "t", "service": "api", "severity": "SEV-1",
        "transcript": "[ev_1] high latency\n[ev_2] error spike\n[ev_3] cpu ok\n[ev_9] noise",
        "allowedRootCauses": ["cpu_saturation", "bad_deploy"],
    },
    "toolCatalog": [
        {"name": "query_metrics"}, {"name": "check_logs"},
        {"name": "scale_service"}, {"name": "rollback_deployment"},
    ],
    "policy": {
        "maximumDiagnostics": 3, "effectTools": ["scale_service", "rollback_deployment"],
        "approvalRequiredFor": ["rollback_deployment", "disable_feature"], "doNotExport": ["accessToken"],
    },
}

import copy, json  # noqa: E402


def _sensitive_absent(obj) -> bool:
    return "SECRET" not in json.dumps(obj)


# 1) CREATE -> waiting with diagnosis + 2 diagnostic dispatches, no sensitive leak
r = client.post("/v2/incidents", json=BASE_REQ)
assert r.status_code == 200, r.text
b = r.json()
assert b["status"] == "waiting"
assert b["diagnosis"]["rootCause"] == "cpu_saturation"
assert len(b["dispatches"]) == 2, b["dispatches"]
assert b["approvals"] == []
assert _sensitive_absent(b), "sensitive leaked into create response"
# each diagnostic dispatch cites >=1 diagnosis evidence id, traceparent well-formed
diag_ev = set(b["diagnosis"]["evidence"])
for d in b["dispatches"]:
    assert set(d["evidence"]) & diag_ev, d
    assert d["traceparent"].startswith("00-") and len(d["traceparent"].split("-")) == 4
a0, c0 = b["dispatches"][0]["actionId"], b["dispatches"][0]["callId"]
a1, c1 = b["dispatches"][1]["actionId"], b["dispatches"][1]["callId"]
assert len({a0, a1}) == 2 and len({c0, c1}) == 2, "action/call ids not unique"
print("1) create + 2 parallel diagnostics OK")

# 2) exact create replay -> same body, no new model call
r2 = client.post("/v2/incidents", json=BASE_REQ)
assert r2.status_code == 200 and r2.json() == b
print("2) create replay OK")

# 3) changed content, same runId -> 409
bad = copy.deepcopy(BASE_REQ)
bad["incident"]["transcript"] += "\n[ev_4] changed"
assert client.post("/v2/incidents", json=bad).status_code == 409
print("3) content conflict 409 OK")

# 4) diagnostics succeed -> effect dispatched (non-destructive, no approval)
rec = {"receiptId": "rcpt-1", "outcomes": [
    {"actionId": a0, "callId": c0, "attempt": 1, "status": 200, "resultClass": "diagnosis_confirmed", "nonce": "n1"},
    {"actionId": a1, "callId": c1, "attempt": 1, "status": 200, "resultClass": "diagnosis_confirmed", "nonce": "n2"},
]}
r4 = client.post("/v2/incidents/run-1/receipts", json=rec)
assert r4.status_code == 200, r4.text
b4 = r4.json()
assert b4["status"] == "waiting"
assert len(b4["dispatches"]) == 1 and b4["dispatches"][0]["phase"] == "effect"
assert b4["dispatches"][0]["toolName"] == "scale_service"
eff_a, eff_c = b4["dispatches"][0]["actionId"], b4["dispatches"][0]["callId"]
print("4) diagnostics->effect dispatch OK")

# 5) identical receipt replay -> no change
r5 = client.post("/v2/incidents/run-1/receipts", json=rec)
assert r5.status_code == 200 and r5.json() == b4
print("5) receipt replay OK")

# 6) effect outcome -> terminal completed, actionLog+receiptLog present, otlp valid
rec2 = {"receiptId": "rcpt-2", "outcomes": [
    {"actionId": eff_a, "callId": eff_c, "attempt": 1, "status": 200, "resultClass": "effect_applied", "nonce": "n3"},
]}
r6 = client.post("/v2/incidents/run-1/receipts", json=rec2)
assert r6.status_code == 200, r6.text
b6 = r6.json()
assert b6["status"] == "completed", b6["status"]
assert b6["chosenEffect"] == "scale_service"
assert isinstance(b6["actionLog"], list) and isinstance(b6["receiptLog"], list)
assert b6["dispatches"] == [] and b6["approvals"] == []
spans = b6["otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
names = [s["name"] for s in spans]
assert names.count("chat incident-plan") == 1, names
assert names.count("incident.join") == 1, "join should fire for 2 diagnostics"
assert any(n.startswith("execute_tool") for n in names)
assert any(n.startswith("POST tool/") for n in names)
# every span carries ga5.run.id and ga5.public.marker
for s in spans:
    keys = {a["key"] for a in s["attributes"]}
    assert "ga5.run.id" in keys and "ga5.public.marker" in keys, s["name"]
# all spans share the SERVER trace id
tid = spans[0]["traceId"]
assert all(s["traceId"] == tid for s in spans)
assert _sensitive_absent(b6), "sensitive leaked into final response"
print(f"6) terminal completed + OTLP valid ({len(spans)} spans) OK")

# 7) GET returns stored terminal state
rg = client.get("/v2/incidents/run-1")
assert rg.status_code == 200 and rg.json() == b6
print("7) GET replay OK")

# ---------- 503 retry path ----------
statemachine.plan = make_plan()
r = client.post("/v2/incidents", json={**BASE_REQ, "runId": "run-503"})
bb = r.json()
aa, cc = bb["dispatches"][0]["actionId"], bb["dispatches"][0]["callId"]
aa2, cc2 = bb["dispatches"][1]["actionId"], bb["dispatches"][1]["callId"]
# first diagnostic returns 503 -> exactly one retry with attempt 2
r = client.post("/v2/incidents/run-503/receipts", json={"receiptId": "r503", "outcomes": [
    {"actionId": aa, "callId": cc, "attempt": 1, "status": 503, "nonce": "x"},
]})
disp = r.json()["dispatches"]
retry = [d for d in disp if d["actionId"] == aa]
assert retry and retry[0]["attempt"] == 2, disp
assert r.json()["status"] == "waiting"
# retry response must not contain an approval request
assert r.json()["approvals"] == []
print("8) 503 -> single retry (attempt 2), no approval OK")

# ---------- timeout suppression ----------
statemachine.plan = make_plan()
client.post("/v2/incidents", json={**BASE_REQ, "runId": "run-to"})
st = endpoints.store.load_run("run-to")
ta0, tc0 = st["logicalActions"][0]["actionId"], st["logicalActions"][0]["callId"]
ta1, tc1 = st["logicalActions"][1]["actionId"], st["logicalActions"][1]["callId"]
r = client.post("/v2/incidents/run-to/receipts", json={"receiptId": "rto", "outcomes": [
    {"actionId": ta0, "callId": tc0, "attempt": 1, "status": 0, "errorType": "timeout", "nonce": "x"},
    {"actionId": ta1, "callId": tc1, "attempt": 1, "status": 200, "resultClass": "ok", "nonce": "y"},
]})
bt = r.json()
assert bt["status"] == "failed", bt["status"]
assert "scale_service" in bt["suppressed"], bt["suppressed"]
# no effect dispatch was emitted
assert all(d.get("phase") != "effect" for d in bt.get("actionLog", [])), "effect emitted after timeout!"
print("9) timeout suppression OK (effect not emitted)")

# ---------- approval gate (destructive) ----------
statemachine.plan = make_plan(effect_tool="rollback_deployment")
client.post("/v2/incidents", json={**BASE_REQ, "runId": "run-appr"})
st = endpoints.store.load_run("run-appr")
pa0, pc0 = st["logicalActions"][0]["actionId"], st["logicalActions"][0]["callId"]
pa1, pc1 = st["logicalActions"][1]["actionId"], st["logicalActions"][1]["callId"]
r = client.post("/v2/incidents/run-appr/receipts", json={"receiptId": "ra1", "outcomes": [
    {"actionId": pa0, "callId": pc0, "attempt": 1, "status": 200, "resultClass": "ok", "nonce": "a"},
    {"actionId": pa1, "callId": pc1, "attempt": 1, "status": 200, "resultClass": "ok", "nonce": "b"},
]})
ba = r.json()
# destructive effect must be gated: zero effect dispatches, one approval request
assert ba["dispatches"] == [], ba["dispatches"]
assert len(ba["approvals"]) == 1, ba["approvals"]
appr = ba["approvals"][0]
assert appr["toolName"] == "rollback_deployment"
assert len(appr["argumentsDigest"]) == 64 and appr["argumentsDigest"].islower()
print("10) approval gate opened (no destructive dispatch before approval) OK")

# approve -> effect dispatched with matching approvalId + nonce
r = client.post("/v2/incidents/run-appr/receipts", json={"receiptId": "ra2", "approvals": [
    {"approvalId": appr["approvalId"], "decision": "approved", "nonce": "APPRNONCE"},
]})
bap = r.json()
eff = [d for d in bap["dispatches"] if d.get("phase") == "effect"]
assert len(eff) == 1, bap["dispatches"]
assert eff[0]["toolName"] == "rollback_deployment"
assert eff[0]["approvalId"] == appr["approvalId"]
assert eff[0]["approvalNonce"] == "APPRNONCE"
# reserved actionId reused
assert eff[0]["actionId"] == appr["actionId"], "effect must reuse reserved approval actionId"
print("11) approval -> destructive effect dispatched with approvalId+nonce OK")

# approval_gate span present in final trace after effect completes
ea, ec = eff[0]["actionId"], eff[0]["callId"]
r = client.post("/v2/incidents/run-appr/receipts", json={"receiptId": "ra3", "outcomes": [
    {"actionId": ea, "callId": ec, "attempt": 1, "status": 200, "resultClass": "effect_applied", "nonce": "z"},
]})
bfin = r.json()
assert bfin["status"] == "completed"
snames = [s["name"] for s in bfin["otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"]]
assert "approval_gate" in snames, snames
print("12) approval_gate span present OK")

print("\nALL INCIDENT SMOKE TESTS PASSED")
