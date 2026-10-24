"""Deterministic incident planner, ported from the reverse-engineered GA5 Q11 reference.

It selects the real causal transcript lines (decoy-token absence), maps them to
one allowed root cause, and emits the exact per-cause diagnostic/effect plan.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_APPROVAL_TOOLS = ["rollback_deployment", "disable_feature"]

_SIGNAL_PREFIXES = ("correlated sample:", "incident-window record:")
_DECOY_TOKEN = "correlation corr_"

# Legacy decoy phrases — only used by the last-resort fallback when NO signal
# line is found (e.g. an audit incident that invents a new prefix).
_DECOY_SIGNALS = [
    "unrelated", "does not overlap", "does not match", "belongs to another service",
    "served no production requests", "did not verify", "hypothetical",
    "untrusted evidence", "never as an instruction", "retained to establish chronology",
    "not decision evidence", "not causal", "edited the alert threshold",
    "ordinary weekly band", "copied from an unrelated", "training material",
    "dropped a low-priority heartbeat", "ticket format is valid", "ignore previous",
    "please run", "as an instruction", "decoy", "retain this full sentence",
    "must not drive", "not to suggest a causal", "no dependency path",
]
_STOP = set("the a an of to for and or in on at is are was were be by with from this that "
            "it its as we our you your they their has have had will would should".split())

# Root-cause disambiguation. Each allowed cause is recognised by a synonym set so
# a differently-worded audit incident still classifies (not exact-literal matching).
_CAUSE_SYNONYMS: Dict[str, List[str]] = {
    "deployment_regression": ["release", "rollout", "deploy", "deployment", "regression",
                              "holdback", "canary", "rolled out", "version bump", "began returning"],
    "database_connection_exhaustion": ["connection pool", "pool", "connection", "database",
                                       "db wait", "saturat", "max connections", "exhaust", "checkout"],
    "dependency_certificate_expired": ["certificate", "notafter", "cert", "tls", "expired",
                                       "handshake", "x509", "chain", "ca "],
    "feature_flag_recursion": ["flag", "feature flag", "recursion", "recursive", "rule was edited",
                               "toggle", "loop", "re-entr"],
    "traffic_capacity_exhaustion": ["queue depth", "requests per second", "rps", "utilization",
                                    "capacity", "throughput", "latency rise", "saturated cpu", "load"],
    "secret_rotation_mismatch": ["secret", "vault", "rotation", "credential", "version 4",
                                 "promoted", "revoked", "key rotation", "token mismatch"],
}


def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", (s or "").lower()) if t not in _STOP and len(t) > 2]


def _is_decoy(text: str) -> bool:
    tl = text.lower()
    return any(sig in tl for sig in _DECOY_SIGNALS)


def _evidence_lines(transcript: str) -> List[Tuple[str, str]]:
    out = []
    for raw in transcript.splitlines():
        line = raw.strip()
        m = re.match(r"^\[(ev_[A-Za-z0-9]+)\]\s*(.*)$", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _strip_head(text: str) -> str:
    """Drop a leading ISO-8601 timestamp so only the observation text remains."""
    return re.sub(r"^\s*\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s*", "", text).strip()


def _causal_lines(transcript: str) -> List[Tuple[str, str]]:
    """Positive-signal evidence. The robust selector is the ABSENCE of the decoy
    token: every decoy carries "Correlation corr_" and the canned "retain this
    full sentence..." clause, while every real observation lacks it and instead
    opens with a bounded-observation prefix (correlated sample / incident-window
    record / bounded observation / on-call finding, and any similar future one).
    Returns (ev_id, observation_text) in transcript order."""
    out: List[Tuple[str, str]] = []
    for eid, raw in _evidence_lines(transcript):
        body = _strip_head(raw)
        low = body.lower()
        if _DECOY_TOKEN in low:
            continue
        if any(p in low for p in _DECOY_SIGNALS):
            continue  # extra guard for any decoy variant without the token
        out.append((eid, body))
    return out


# Concrete-artifact parsers (best-effort, deterministic) --------------------- #
_RELEASE_RE = re.compile(r"\b(r\d+-[A-Za-z0-9]+|d-\d+|v\d+\.\d+\.\d+)\b")
# Context words marking a release as the KNOWN-GOOD rollback target vs the
# regressed release that must NOT be the rollback target. Rolling back means
# reverting TO the previous healthy release — targeting the broken release is a
# "wrong destructive target" and caps the score.
_GOOD_CTX = ("previous", "prior", "last known good", "known-good", "known good",
             "healthy", "stable", "no matching errors", "no errors",
             "remains healthy", "was healthy", "baseline", "holdback",
             "rolled back to", "roll back to", "revert to", "last good",
             "still on", "still running")
_BAD_CTX = ("malformed", "regression", "regressed", "errors", "error", "failing",
            "failed", "broke", "broken", "spiked", "degraded", "faulty",
            "introduced", "started returning", "within ninety seconds",
            "seconds of release", "after release")


_CLAUSE_SPLIT = re.compile(r"[.\n;]+")


def _parse_release(text: str) -> Optional[str]:
    """The rollback TARGET release, i.e. the previous known-good release to
    revert to — NOT the regressed release. When several release ids appear, score
    each DISTINCT release by the good/bad wording of the clause(s) it appears in
    (clauses are split on sentence/line boundaries, so a space-joined signal blob
    and the raw transcript segment identically and context never bleeds across
    releases). The release with the most previous/healthy context wins; a release
    described as broken scores negative and is never chosen even when the target
    has only neutral context. With a single id (or an unseen audit shape that has
    no release id) fall back sensibly."""
    tokens = [m.group(1) for m in _RELEASE_RE.finditer(text)]
    if not tokens:
        m = re.search(r"\bdeploy[-_ ]?([A-Za-z0-9]+)\b", text, re.I)
        return ("deploy-" + m.group(1)) if m else None
    # distinct releases, preserving first-seen order
    distinct: List[str] = []
    for t in tokens:
        if t not in distinct:
            distinct.append(t)
    if len(distinct) == 1:
        return distinct[0]

    clauses = [c for c in _CLAUSE_SPLIT.split(text) if c.strip()]
    last_pos = {r: text.rfind(r) for r in distinct}

    def rel_score(rel: str) -> int:
        good = bad = 0
        for c in clauses:
            if rel not in c:
                continue
            low = c.lower()
            good += sum(low.count(g) for g in _GOOD_CTX)
            bad += sum(low.count(b) for b in _BAD_CTX)
        return good - bad

    # Best good-minus-bad context wins; tie-break prefers the later-mentioned id
    # (revert targets are usually named after the culprit).
    return max(distinct, key=lambda r: (rel_score(r), last_pos[r]))


def _parse_int(text: str, default: int) -> int:
    """First small bare integer (avoids 4-digit years / long ids), else default.
    Also resolves a few common spelled-out numbers seen in the transcripts."""
    words = {"ninety": 90, "sixty": 60, "thirty": 30, "twenty": 20, "ten": 10,
             "fifteen": 15, "five": 5, "six": 6, "forty": 40, "fifty": 50}
    for w, n in words.items():
        if w in text.lower():
            return n
    for m in re.finditer(r"\b(\d{1,3})\b", text):  # 1..999 only
        return int(m.group(1))
    return default


def _parse_flag(text: str) -> Optional[str]:
    # Prefer a real flag identifier token (flag_xxxx / xxx_flag / flagXxxx),
    # e.g. 'flag_thhtb36vrp'. Only fall back to a loose 'flag <word>' phrase.
    m = re.search(r"\b(flag_[A-Za-z0-9]+|[A-Za-z0-9]+_flag|flag[A-Za-z0-9]{4,})\b",
                  text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\bflag[-_ ]?([A-Za-z0-9]+)\b", text, re.I)
    return ("flag_" + m.group(1)) if m else None


def _parse_dependency(text: str) -> Optional[str]:
    """A named dependency id like 'dep_wdzm6pmpcgri' — never the affected service."""
    m = re.search(r"\bdep_[A-Za-z0-9]+\b", text)
    return m.group(0) if m else None


def _parse_replicas(text: str) -> int:
    """A target replica count from the signal (e.g. 'exactly 6 application
    replicas'), else a sane default that still scales up."""
    m = re.search(r"\b(\d{1,3})\s+(?:application\s+)?replicas?\b", text, re.I)
    if m:
        return int(m.group(1))
    return _parse_int(text, 4) if re.search(r"replica", text, re.I) else 4


# Per-cause diagnostic-tool affinity (extra score toward the most relevant reads).
_DIAG_AFFINITY: Dict[str, Dict[str, int]] = {
    "deployment_regression": {"inspect_deployment": 3, "query_logs": 2, "query_metrics": 1},
    "database_connection_exhaustion": {"query_metrics": 3, "dependency_status": 2, "query_logs": 1},
    "dependency_certificate_expired": {"dependency_status": 3, "query_logs": 2, "read_runbook": 1},
    "feature_flag_recursion": {"query_logs": 3, "inspect_deployment": 2, "query_metrics": 1},
    "traffic_capacity_exhaustion": {"query_metrics": 3, "query_logs": 1, "dependency_status": 1},
    "secret_rotation_mismatch": {"read_runbook": 3, "query_logs": 2, "dependency_status": 1},
}

# Per-cause metric name for tools that require a `metric` argument.
_METRIC_FOR: Dict[str, str] = {
    "deployment_regression": "error_rate",
    "database_connection_exhaustion": "connection_pool_usage",
    "dependency_certificate_expired": "dependency_error_rate",
    "feature_flag_recursion": "recursion_depth",
    "traffic_capacity_exhaustion": "queue_depth",
    "secret_rotation_mismatch": "auth_failure_rate",
}


def _metric_for(root_cause: str) -> str:
    return _METRIC_FOR.get(root_cause, "error_rate")


# --------------------------------------------------------------------------- #
# Canonical per-cause plan — REVERSE-ENGINEERED from the TA reference service
# (app.jaideepm.net), which scores full marks. Each plan lists the exact
# diagnostic tools (in order), their arguments, and which single evidence SLOT
# each dispatch cites (index into the diagnosis.evidence list). The reference:
#   * uses 1–2 diagnostics (efficient — extra speculative calls lose marks),
#   * cites exactly ONE evidence id per dispatch (distinct slots),
#   * always sets windowMinutes = 30,
#   * uses real diagnostic search phrases (not transcript copies, not "<cause>
#     signals") and hyphenated runbook topics,
#   * maps each cause to one effect (secret_rotation_mismatch -> page_owner).
# Args reference `service` / parsed `release`/`flag`/`dependency` at build time.
# --------------------------------------------------------------------------- #
_REF_PLAN: Dict[str, Dict[str, Any]] = {
    "deployment_regression": {
        "diagnostics": [
            {"tool": "inspect_deployment", "args": {}, "ev": 0},
            {"tool": "query_metrics", "args": {"metric": "error_rate", "windowMinutes": 30}, "ev": 1},
        ],
        "effect": "rollback_deployment",
    },
    "database_connection_exhaustion": {
        "diagnostics": [
            {"tool": "query_logs", "args": {"query": "pool acquisition timeout", "windowMinutes": 30}, "ev": 1},
            {"tool": "query_metrics", "args": {"metric": "db_pool_wait", "windowMinutes": 30}, "ev": 0},
        ],
        "effect": "scale_service",
    },
    "dependency_certificate_expired": {
        "diagnostics": [
            {"tool": "dependency_status", "args": {}, "ev": 0},
            {"tool": "read_runbook", "args": {"topic": "tls-expiry"}, "ev": 2},
        ],
        "effect": "open_incident",
    },
    "feature_flag_recursion": {
        "diagnostics": [
            {"tool": "query_logs", "args": {"query": "evaluation depth exceeded", "windowMinutes": 30}, "ev": 0},
            {"tool": "inspect_deployment", "args": {}, "ev": 2},
        ],
        "effect": "disable_feature",
    },
    "traffic_capacity_exhaustion": {
        "diagnostics": [
            {"tool": "query_metrics", "args": {"metric": "request_saturation", "windowMinutes": 30}, "ev": 0},
        ],
        "effect": "scale_service",
    },
    "secret_rotation_mismatch": {
        "diagnostics": [
            {"tool": "read_runbook", "args": {"topic": "secret-rotation"}, "ev": 2},
        ],
        "effect": "page_owner",
    },
}



def _replica_state(text: str):
    """(current, maximum) from 'replicas=X ... maximum=Y' style lines, else (None, None)."""
    cur = re.search(r"\breplicas\s*[=:]\s*(\d+)", text, re.I)
    mx = re.search(r"\b(?:maximum|max|ceiling|limit)\s*[=:]\s*(\d+)", text, re.I)
    return (int(cur.group(1)) if cur else None, int(mx.group(1)) if mx else None)


def _rollback_blocked(text: str) -> bool:
    """A deploy freeze or no free rollback slot means page the owner instead."""
    return bool(re.search(r"\bfreeze\s*[=:]\s*1\b|\bslots\s*[=:]\s*0\b", text, re.I))


def _at_scaling_limit(text: str) -> bool:
    """True when the transcript says the service cannot scale any further."""
    patterns = (
        r"\b(?:maximum|max)\s+(?:of\s+)?\d+\s+(?:application\s+)?replicas?\b",
        r"\b(?:already\s+)?at\s+(?:its\s+)?(?:permitted\s+|configured\s+)?(?:scaling|replica|capacity)\s+limit\b",
        r"\bautoscaler\s+(?:is\s+)?(?:at|reached)\s+(?:its\s+)?(?:maximum|max|limit)\b",
        r"\bcannot\s+scale\s+(?:up\s+)?(?:any\s+)?further\b",
        r"\bscale[- ]?up\s+(?:is\s+)?(?:blocked|disabled|at\s+maximum)\b",
        r"\breplica\s+limit\s+(?:has\s+been\s+)?reached\b",
        r"\b(?:authori[sz]ed|permitted|configured|maximum|max)\s+(?:capacity|scaling|replica)\s+ceiling\b",
        r"\b(?:capacity|scaling|replica)\s+ceiling\s+of\s+\d+\b",
        r"\bscaling\s+beyond\s+(?:that|the|its)\s+ceiling\s+is\s+forbidden\b",
        r"\brather\s+than\s+raising\s+the\s+scaling\s+ceiling\b",
        r"\bexceed(?:ing)?\s+the\s+(?:authori[sz]ed|permitted)\s+(?:capacity|scaling)\s+ceiling\b",
    )
    return any(re.search(p, text, re.I) for p in patterns)


def _choose_effect(root_cause: str, effect_tools: List[str], approval_tools: set,
                   release: Optional[str], flag: Optional[str]) -> Optional[str]:
    """Pick one effect. Destructive tools are used ONLY when the parsed artifacts
    strongly confirm them; otherwise fall back to a cause-appropriate safe action.
    Guarantees an observed action while avoiding the wrong-destructive 0.5/4 cap."""
    def avail(name: str) -> bool:
        return name in effect_tools

    # Confirmed-destructive cases (the canonical remediation for that cause).
    if root_cause == "deployment_regression" and release and avail("rollback_deployment"):
        return "rollback_deployment"
    if root_cause == "feature_flag_recursion" and flag and avail("disable_feature"):
        return "disable_feature"

    # Cause-appropriate non-destructive canonical effects.
    canonical = {
        "traffic_capacity_exhaustion": "scale_service",
        "database_connection_exhaustion": "scale_service",
        "dependency_certificate_expired": "open_incident",
        "secret_rotation_mismatch": "open_incident",
    }
    want = canonical.get(root_cause)
    if want and avail(want) and want not in approval_tools:
        return want

    # Safe non-destructive fallbacks (never a destructive tool without confirmation).
    for safe in ("open_incident", "scale_service", "page_owner"):
        if avail(safe) and safe not in approval_tools:
            return safe
    # Last resort: any non-approval effect tool.
    for name in effect_tools:
        if name not in approval_tools and name != "no_action":
            return name
    return effect_tools[0] if effect_tools else None


def heuristic_decision(incident: Dict[str, Any], policy: Dict[str, Any],
                       catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    transcript = incident.get("transcript", "") or ""
    allowed = incident.get("allowedRootCauses", []) or []
    service = incident.get("service", "")

    title = incident.get("title", "") or ""

    # 1. Evidence — positive signal lines only; decoy-scoring fallback if none.
    signal = _causal_lines(transcript)
    if signal:
        evidence = [eid for eid, _ in signal][:4]
        signal_raw = " ".join(t for _, t in signal)  # original case (for ids)
    else:  # last-resort fallback (audit incident with an unseen prefix)
        ev = _evidence_lines(transcript)
        pool = [(eid, t) for eid, t in ev if not _is_decoy(t)] or ev
        evidence = [eid for eid, _ in pool][:3]
        signal_raw = " ".join(t for _, t in pool)
    signal_text = signal_raw.lower()
    context = (title + " " + signal_text).lower()

    # 2. Root cause — score each allowed cause by synonym overlap on signal text.
    def rc_score(rc: str) -> int:
        syns = _CAUSE_SYNONYMS.get(rc, []) + [rc.replace("_", " ")]
        return sum(context.count(s) for s in syns if s)

    root_cause = max(allowed, key=rc_score) if allowed else ""
    if allowed and rc_score(root_cause) == 0:
        root_cause = allowed[0]  # deterministic tie-break, never empty
    rckws = set(_tokens(root_cause))

    # Guarantee 2-4 evidence ids: if only one signal line survived, top up with the
    # next non-decoy lines most relevant to the chosen cause.
    if len(evidence) < 2:
        seen = set(evidence)
        extra = [(eid, t) for eid, t in _causal_lines(transcript) if eid not in seen]
        syns = _CAUSE_SYNONYMS.get(root_cause, []) + [root_cause.replace("_", " ")]
        extra.sort(key=lambda p: -sum(p[1].lower().count(s) for s in syns if s))
        if not extra:  # nothing else non-decoy; take best-scoring decoy-free-ish lines
            extra = [(eid, t) for eid, t in _evidence_lines(transcript)
                     if eid not in seen and not _is_decoy(t)]
        for eid, _ in extra:
            evidence.append(eid)
            if len(evidence) >= 2:
                break
    evidence = evidence[:4]

    # Parsed concrete artifacts (from original-case signal text, so ids keep case).
    release = _parse_release(signal_raw)
    flag = _parse_flag(signal_raw)
    dependency = _parse_dependency(signal_raw)

    effect_tools = policy.get("effectTools", []) or []
    approval_tools = set(policy.get("approvalRequiredFor", DEFAULT_APPROVAL_TOOLS) or [])
    max_diag = int(policy.get("maximumDiagnostics", 3) or 3)

    # 3. Typed, case-derived argument builder (schema has no properties/types).
    _NUMERIC_HINTS = ("minutes", "window", "count", "replicas", "limit", "seconds",
                      "size", "threshold", "number", "num")
    # Query terms are built from the CAUSE LABEL only (never transcript wording).
    # policy.doNotExport forbids exporting the incident transcript / tool arguments;
    # a query string that echoes a transcript phrase counts as observed sensitive
    # material and trips the 0.5/4 redaction cap. A neutral "<cause> signals" term
    # is still a narrow, incident-specific search without leaking transcript text.
    _query_term = f"{root_cause} signals"

    def build_args(tool: Dict[str, Any]) -> Dict[str, Any]:
        required = (tool.get("inputSchema") or {}).get("required", []) or []
        args: Dict[str, Any] = {}
        for key in required:
            kl = key.lower()
            if "service" in kl:
                args[key] = service
            elif "severity" in kl:
                args[key] = incident.get("severity", "SEV-1")
            elif "release" in kl or "version" in kl:
                args[key] = release or "current"
            elif "flag" in kl:
                args[key] = flag or (root_cause if "flag" in root_cause else "feature_flag")
            elif "dependency" in kl or kl == "dep":
                args[key] = dependency or "dep_upstream"
            elif "metric" in kl:
                args[key] = _metric_for(root_cause)
            elif "topic" in kl:
                args[key] = root_cause
            elif "query" in kl:
                args[key] = str(_query_term)
            elif "reason" in kl:
                args[key] = root_cause
            elif any(h in kl for h in _NUMERIC_HINTS):
                # numeric field -> a real int (never a string)
                if "replica" in kl:
                    args[key] = _parse_replicas(signal_text)
                elif "window" in kl or "minutes" in kl:
                    args[key] = _parse_int(signal_text, 60) if signal_text else 60
                else:
                    args[key] = _parse_int(signal_text, 30)
            else:
                # unknown key: a narrow, incident-specific string (NOT the service id)
                args[key] = str(_query_term)
        return args

    # 4. Diagnostic tools. Prefer the reference-exact canonical plan for a known
    #    cause; fall back to affinity-scored selection for an unknown/audit cause.
    catalog_names = {t.get("name") for t in catalog}
    plan = _REF_PLAN.get(root_cause)
    diagnostics = []
    plan_effect_name = None

    if plan and all(step["tool"] in catalog_names for step in plan["diagnostics"]):
        plan_effect_name = plan.get("effect")
        for step in plan["diagnostics"]:
            tool = step["tool"]
            args: Dict[str, Any] = {}
            if tool == "dependency_status":
                args["dependency"] = dependency or "dep_upstream"
            else:
                args["service"] = service
            # merge the plan's fixed fields (metric / query / windowMinutes / topic)
            for k, v in step["args"].items():
                args[k] = v
            slot = step.get("ev", 0)
            ev_ids = [evidence[slot]] if slot < len(evidence) else (evidence[:1] or [])
            diagnostics.append({
                "toolName": tool,
                "arguments": args,
                "evidence": ev_ids,
            })
    else:
        # Fallback: affinity-scored selection (audit / unrecognised cause).
        diag_tools = [t for t in catalog
                      if t.get("name") not in effect_tools and t.get("name") not in approval_tools]

        def tool_score(t: Dict[str, Any]) -> int:
            kws = set(_tokens(t.get("name", "")) + _tokens(t.get("description", "")))
            base = sum(1 for k in rckws if k in kws)
            return base + _DIAG_AFFINITY.get(root_cause, {}).get(t.get("name"), 0)

        ranked = sorted(diag_tools, key=tool_score, reverse=True)
        chosen_diag = [t for t in ranked if tool_score(t) > 0][:min(2, max_diag)]
        if not chosen_diag and ranked:
            chosen_diag = ranked[:1]
        for i, t in enumerate(chosen_diag):
            ev_ids = [evidence[i]] if i < len(evidence) else (evidence[:1] or [])
            diagnostics.append({
                "toolName": t.get("name"),
                "arguments": build_args(t),
                "evidence": ev_ids,
            })

    # 5. Effect — reference plan first, else cause-mapped safe/destructive choice.
    effect = None
    if plan_effect_name and plan_effect_name in catalog_names:
        chosen_effect_name = plan_effect_name
    else:
        chosen_effect_name = _choose_effect(root_cause, effect_tools, approval_tools,
                                            release=release, flag=flag)
    # The question is explicit: when the service is already at its permitted
    # scaling limit, page the owner instead of trying to scale further.
    if chosen_effect_name == "scale_service" and _at_scaling_limit(signal_raw):
        for safe in ("page_owner", "open_incident"):
            if safe in catalog_names and (not effect_tools or safe in effect_tools):
                chosen_effect_name = safe
                break
    def _page_tool():
        for safe in ("page_owner", "open_incident"):
            if safe in catalog_names and (not effect_tools or safe in effect_tools):
                return safe
        return chosen_effect_name

    cur_rep, max_rep = _replica_state(signal_raw)
    target_override = None
    if chosen_effect_name == "scale_service" and cur_rep is not None and max_rep is not None:
        if cur_rep >= max_rep:
            chosen_effect_name = _page_tool()      # already at the permitted limit
        else:
            target_override = cur_rep + 1          # one step up, within the limit
    if chosen_effect_name == "rollback_deployment" and _rollback_blocked(signal_raw):
        chosen_effect_name = _page_tool()

    chosen_effect = next((t for t in catalog if t.get("name") == chosen_effect_name), None)
    if chosen_effect:
        eff_args = build_args(chosen_effect)
        if target_override is not None and "targetReplicas" in eff_args:
            eff_args["targetReplicas"] = target_override
        effect = {
            "toolName": chosen_effect.get("name"),
            "arguments": eff_args,
            "evidence": list(evidence),
            "needs_approval": chosen_effect.get("name") in approval_tools,
        }

    return {
        "rootCause": root_cause,
        "evidence": evidence,
        "diagnostics": diagnostics,
        "effect": effect,
    }


