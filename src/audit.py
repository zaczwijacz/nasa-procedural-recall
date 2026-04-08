"""
Audit logger for the NASA procedural recall assistant.

Each query run appends one JSON line to the audit log (JSONL format).
The log is the post-run traceability record: what was asked, what was
retrieved, what the model produced, what the safety scan decided, and
what was shown to the crew member.
"""

import json
import os
import time
from datetime import datetime, timezone


def write_audit_event(log_path: str, event: dict) -> None:
    """
    Append a structured audit event to the JSONL log.

    Adds ISO 8601 timestamps (ts_iso_start / ts_iso_end) alongside the
    raw Unix floats so the log is human-readable without extra tooling.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # Ensure Unix timestamps exist
    event.setdefault("ts_start", time.time())
    event.setdefault("ts_end",   event["ts_start"])

    # Guarantee retrieval provenance and failsafe flag are always present
    event.setdefault("retrieval_type",    "unknown")
    event.setdefault("failsafe_triggered", False)

    # Add ISO 8601 counterparts
    event["ts_iso_start"] = _to_iso(event["ts_start"])
    event["ts_iso_end"]   = _to_iso(event["ts_end"])

    # Derived duration_ms for quick performance inspection
    event["duration_ms"] = round((event["ts_end"] - event["ts_start"]) * 1000)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _to_iso(unix_ts: float) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
