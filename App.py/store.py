"""
Durable run storage.

Backend selection is automatic:
  * If UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN are set (injected by
    the Vercel <-> Upstash Marketplace integration), use Upstash Redis over
    its REST API. This survives cold starts and instance recycling, which
    matters because an incident RUN spans multiple requests (create ->
    receipts -> GET); losing state mid-run would cause spurious 409s.
  * Otherwise fall back to local SQLite (for `uvicorn` dev + the smoke test).

Only the functions at the bottom are called by the rest of the app, so the
backend is swappable without touching the state machine.

State is stored as one JSON blob per runId, plus a small set of auxiliary
keys for content-conflict detection and receipt idempotency.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import httpx

_UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
_UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
_USE_REDIS = bool(_UPSTASH_URL and _UPSTASH_TOKEN)

_DB_PATH = os.environ.get("INCIDENT_DB", "/tmp/incident.db")
_lock = threading.Lock()


# ============================================================ Upstash REST backend

def _redis_cmd(*args: Any) -> Any:
    """Execute one Redis command via Upstash REST. Returns the 'result'
    field. Raises on transport error so callers can surface it."""
    # Upstash REST accepts the command as a JSON array POSTed to the base URL.
    resp = httpx.post(
        _UPSTASH_URL,
        headers={"Authorization": f"Bearer {_UPSTASH_TOKEN}"},
        json=[str(a) for a in args],
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("result")


def _redis_get(key: str) -> Optional[str]:
    return _redis_cmd("GET", key)


def _redis_set(key: str, value: str) -> None:
    _redis_cmd("SET", key, value)


def _redis_setnx(key: str, value: str) -> bool:
    # returns 1 if set, 0 if key already existed
    return bool(_redis_cmd("SETNX", key, value))


# ================================================================= SQLite backend

@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _sqlite_init() -> None:
    with _lock, _conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")


def _sqlite_get(key: str) -> Optional[str]:
    with _lock, _conn() as conn:
        row = conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row["v"] if row else None


def _sqlite_set(key: str, value: str) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, value),
        )


def _sqlite_setnx(key: str, value: str) -> bool:
    with _lock, _conn() as conn:
        cur = conn.execute("INSERT OR IGNORE INTO kv (k, v) VALUES (?, ?)", (key, value))
        return cur.rowcount == 1


# ==================================================================== unified kv

def init() -> None:
    if not _USE_REDIS:
        _sqlite_init()


def _kv_get(key: str) -> Optional[str]:
    return _redis_get(key) if _USE_REDIS else _sqlite_get(key)


def _kv_set(key: str, value: str) -> None:
    (_redis_set if _USE_REDIS else _sqlite_set)(key, value)


def _kv_setnx(key: str, value: str) -> bool:
    return (_redis_setnx if _USE_REDIS else _sqlite_setnx)(key, value)


def backend_name() -> str:
    return "upstash-redis" if _USE_REDIS else "sqlite"


# ============================================================ run-state accessors

def _run_key(run_id: str) -> str:
    return f"run:{run_id}"


def load_run(run_id: str) -> Optional[dict]:
    raw = _kv_get(_run_key(run_id))
    return json.loads(raw) if raw else None


def save_run(run_id: str, state: dict) -> None:
    _kv_set(_run_key(run_id), json.dumps(state))


def create_run_if_absent(run_id: str, state: dict) -> bool:
    """Atomically create the run record. Returns True if created, False if a
    record already existed (caller then compares content for replay/409)."""
    return _kv_setnx(_run_key(run_id), json.dumps(state))
