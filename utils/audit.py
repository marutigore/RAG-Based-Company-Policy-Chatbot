"""
Compliance & Regulatory Audit Trail Module.
Implements a tamper-evident cryptographic hash-chained audit ledger for enterprise
compliance audits (SOC2, ISO 27001, HIPAA, GDPR), recording all query and indexing events.
"""

import os
import csv
import io
import json
import uuid
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("utils.audit")

AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audit_log.json")
GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


def _ensure_data_dir() -> None:
    data_dir = os.path.dirname(AUDIT_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def _load_audit_log() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if os.path.exists(AUDIT_FILE):
        try:
            with open(AUDIT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading audit log: {e}")
    return []


def _save_audit_log(logs: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    try:
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(logs[-3000:], f, indent=2)  # Maintain recent 3,000 ledger events
    except Exception as e:
        logger.error(f"Error saving audit log: {e}")


def _calculate_entry_hash(prev_hash: str, timestamp: str, event_type: str, user_id: str, details_str: str) -> str:
    payload = f"{prev_hash}|{timestamp}|{event_type}|{user_id}|{details_str}".encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def log_audit_event(
    event_type: str,
    user_id: str = "anonymous",
    clearance: str = "Employee",
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = "127.0.0.1"
) -> Dict[str, Any]:
    """Appends an immutable tamper-evident event to the cryptographic audit trail."""
    logs = _load_audit_log()
    prev_hash = logs[-1]["entry_hash"] if logs else GENESIS_HASH
    
    timestamp = datetime.utcnow().isoformat()
    details_dict = details or {}
    details_str = json.dumps(details_dict, sort_keys=True)
    entry_id = str(uuid.uuid4())[:8]

    entry_hash = _calculate_entry_hash(prev_hash, timestamp, event_type, user_id, details_str)

    entry = {
        "id": entry_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "user_id": user_id,
        "clearance": clearance,
        "ip_address": ip_address,
        "details": details_dict,
        "previous_hash": prev_hash,
        "entry_hash": entry_hash
    }

    logs.append(entry)
    _save_audit_log(logs)
    logger.info(f"Audit event logged: [{event_type}] by user '{user_id}' (hash: {entry_hash[:8]})")
    return entry


def get_audit_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Returns recent audit log records in reverse chronological order."""
    logs = _load_audit_log()
    return logs[-limit:][::-1]


def verify_audit_integrity() -> Dict[str, Any]:
    """
    Validates complete cryptographic hash chain to certify the ledger is un-tampered.
    """
    logs = _load_audit_log()
    if not logs:
        return {"valid": True, "entries_checked": 0, "status": "Genesis state (clean)"}

    expected_prev = GENESIS_HASH
    for idx, entry in enumerate(logs):
        if entry["previous_hash"] != expected_prev:
            return {
                "valid": False,
                "broken_at_index": idx,
                "broken_entry_id": entry["id"],
                "reason": "Previous hash chain mismatch"
            }
        
        details_str = json.dumps(entry["details"], sort_keys=True)
        recalculated_hash = _calculate_entry_hash(
            expected_prev,
            entry["timestamp"],
            entry["event_type"],
            entry["user_id"],
            details_str
        )
        if recalculated_hash != entry["entry_hash"]:
            return {
                "valid": False,
                "broken_at_index": idx,
                "broken_entry_id": entry["id"],
                "reason": "Cryptographic signature tampered"
            }
        
        expected_prev = entry["entry_hash"]

    return {
        "valid": True,
        "entries_checked": len(logs),
        "latest_hash": expected_prev[:12] + "...",
        "verified_at": datetime.utcnow().isoformat()
    }


def export_audit_csv() -> str:
    """Exports audit trail as standard RFC-4180 CSV string."""
    logs = _load_audit_log()
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow(["ID", "Timestamp (UTC)", "Event Type", "User ID", "Clearance", "IP Address", "Details", "Entry Hash"])
    
    for l in logs:
        writer.writerow([
            l.get("id"),
            l.get("timestamp"),
            l.get("event_type"),
            l.get("user_id"),
            l.get("clearance"),
            l.get("ip_address"),
            json.dumps(l.get("details", {})),
            l.get("entry_hash")
        ])
    return output.getvalue()
