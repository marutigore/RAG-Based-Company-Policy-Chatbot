"""
Document Version Control and Changelog Module.
Computes cryptographic SHA-256 document fingerprints, tracks incremental version numbers
(v1.0, v1.1, etc.), and archives past revision metadata for enterprise compliance.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("utils.versioning")

VERSIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "document_versions.json")


def _ensure_data_dir() -> None:
    data_dir = os.path.dirname(VERSIONS_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def _load_versions() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_data_dir()
    if os.path.exists(VERSIONS_FILE):
        try:
            with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading versions file: {e}")
    return {}


def _save_versions(versions: Dict[str, List[Dict[str, Any]]]) -> None:
    _ensure_data_dir()
    try:
        with open(VERSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(versions, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving document versions: {e}")


def compute_file_hash(file_path: str) -> str:
    """Calculates SHA-256 checksum of document bytes."""
    sha = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return ""


def register_document_version(
    filename: str,
    file_path: str,
    chunks_count: int,
    pages_count: int = 1,
    uploader: str = "admin",
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Registers a new or updated document version with cryptographic hash tracking."""
    versions_map = _load_versions()
    file_hash = compute_file_hash(file_path)
    
    doc_history = versions_map.get(filename, [])
    
    # Check if identical hash already registered
    if doc_history and doc_history[-1].get("hash") == file_hash:
        logger.info(f"Document {filename} matches latest version hash {file_hash[:8]}.")
        return doc_history[-1]

    major = len(doc_history) + 1
    version_tag = f"v{major}.0"

    version_record = {
        "filename": filename,
        "version": version_tag,
        "hash": file_hash,
        "chunks_count": chunks_count,
        "pages_count": pages_count,
        "uploader": uploader,
        "notes": notes or f"Policy revision ingested on {datetime.utcnow().strftime('%Y-%m-%d')}",
        "timestamp": datetime.utcnow().isoformat(),
        "is_active": True
    }

    # Mark past versions inactive
    for v in doc_history:
        v["is_active"] = False

    doc_history.append(version_record)
    versions_map[filename] = doc_history
    _save_versions(versions_map)

    logger.info(f"Registered document version {version_tag} for {filename} (hash: {file_hash[:8]})")
    return version_record


def get_document_versions(filename: Optional[str] = None) -> Any:
    """Returns version history for a given document or all documents."""
    versions_map = _load_versions()
    if filename:
        return versions_map.get(filename, [])
    return versions_map


def get_active_version_tag(filename: str) -> str:
    """Returns latest version tag for a document (e.g. 'v1.0')."""
    versions_map = _load_versions()
    history = versions_map.get(filename, [])
    if history:
        return history[-1].get("version", "v1.0")
    return "v1.0"
