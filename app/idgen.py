"""
Deterministic primitives: canonical JSON, the arguments digest, opaque ID
generation, and W3C trace-context handling. Every rule the spec pins down
exactly lives here so there is one definition of each.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Any


# --------------------------------------------------------------- canonical json

def canonical_json(obj: Any) -> str:
    """Recursively key-sorted, compact JSON. Used for fingerprints and for
    the arguments digest (spec: 'SHA-256 over recursively key-sorted
    compact JSON arguments')."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def arguments_digest(arguments: dict) -> str:
    """Lowercase SHA-256 hex over recursively key-sorted compact JSON of the
    tool arguments. This exact value goes in the approval request as
    argumentsDigest."""
    return sha256_hex(canonical_json(arguments)).lower()


# ------------------------------------------------------------------- opaque ids

def opaque_id(prefix: str = "") -> str:
    """Opaque nonempty string >= 8 chars. Deterministic IDs (actionId,
    callId) should NOT use this -- they must be stable across replays, so
    derive them from stable inputs via stable_id() instead. Use opaque_id
    only for values that legitimately change per run if any."""
    token = secrets.token_hex(8)  # 16 chars
    return f"{prefix}{token}" if prefix else token


def stable_id(prefix: str, *parts: str) -> str:
    """Deterministic ID derived from stable inputs, so replays reproduce it
    byte-for-byte. >= 8 chars, lowercase hex body."""
    body = sha256_hex("|".join(parts))[:24]
    return f"{prefix}_{body}" if prefix else body


# --------------------------------------------------------- w3c trace context

_TRACEPARENT_RE = re.compile(
    r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16


def new_trace_id() -> str:
    """16 bytes -> 32 lowercase hex chars, guaranteed nonzero."""
    while True:
        tid = secrets.token_hex(16)
        if tid != _ZERO_TRACE:
            return tid


def new_span_id() -> str:
    """8 bytes -> 16 lowercase hex chars, guaranteed nonzero."""
    while True:
        sid = secrets.token_hex(8)
        if sid != _ZERO_SPAN:
            return sid


def parse_traceparent(value: str | None) -> dict | None:
    """Return {version, trace_id, parent_id, flags} for a VALID nonzero
    traceparent, else None. Zero trace/span IDs are invalid per W3C."""
    if not value or not isinstance(value, str):
        return None
    m = _TRACEPARENT_RE.match(value.strip())
    if not m:
        return None
    version, trace_id, parent_id, flags = m.groups()
    if version == "ff" or trace_id == _ZERO_TRACE or parent_id == _ZERO_SPAN:
        return None
    return {"version": version, "trace_id": trace_id, "parent_id": parent_id, "flags": flags}


def format_traceparent(trace_id: str, span_id: str, flags: str = "01") -> str:
    return f"00-{trace_id}-{span_id}-{flags}"
