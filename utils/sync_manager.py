"""
Automated Policy Sync & Folder Watcher Module.
Scans local policy repositories and shared drives to automatically detect new or updated
manuals, recalculate SHA-256 hashes, and incrementally ingest modified vectors.
"""

import os
import glob
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from utils.document_loader import load_document
from utils.chunker import split_documents
from utils.retriever import add_documents_to_db, delete_document_from_db
from utils.versioning import register_document_version, compute_file_hash, get_document_versions
from utils.audit import log_audit_event

logger = logging.getLogger("utils.sync_manager")

DEFAULT_WATCH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploaded_policies")
_last_sync_info = {
    "last_sync": None,
    "status": "idle",
    "scanned": 0,
    "ingested": 0,
    "logs": []
}


def scan_and_sync_policies(watch_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Scans the policy folder for modified or unindexed policy files and incrementally
    re-indexes them into the ChromaDB vector database.
    """
    global _last_sync_info
    target_dir = watch_dir or DEFAULT_WATCH_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    logs = [f"Starting policy sync scan on directory: {target_dir}"]
    scanned_count = 0
    new_ingested = 0
    updated_count = 0
    errors = []

    # Supported file types
    patterns = ["*.pdf", "*.docx", "*.xlsx", "*.csv", "*.txt", "*.md", "*.html"]
    candidate_files = []
    for p in patterns:
        candidate_files.extend(glob.glob(os.path.join(target_dir, p)))

    scanned_count = len(candidate_files)
    logs.append(f"Discovered {scanned_count} candidate policy documents.")

    for file_path in candidate_files:
        fn = os.path.basename(file_path)
        current_hash = compute_file_hash(file_path)
        existing_versions = get_document_versions(fn)

        # Check if identical hash is already indexed
        is_already_indexed = False
        for v in existing_versions:
            if v.get("hash") == current_hash:
                is_already_indexed = True
                break

        if is_already_indexed:
            logs.append(f"Document '{fn}' is up to date (hash {current_hash[:8]} matches active index).")
            continue

        try:
            logs.append(f"Processing updated/new document: '{fn}'...")
            pages = load_document(file_path, custom_filename=fn)
            chunks = split_documents(pages, chunk_size=512, chunk_overlap=64)

            # Delete prior chunks if updating existing document
            if existing_versions:
                try:
                    delete_document_from_db(fn)
                    updated_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete old chunks for {fn}: {e}")
            else:
                new_ingested += 1

            ver_info = register_document_version(
                filename=fn,
                file_path=file_path,
                chunks_count=len(chunks),
                pages_count=len(pages)
            )

            for c in chunks:
                c["metadata"]["version"] = ver_info.get("version", "v1.0")
                c["metadata"]["hash"] = ver_info.get("hash", "")

            add_documents_to_db(chunks)
            logs.append(f"Successfully ingested '{fn}' ({len(chunks)} chunks, ver {ver_info.get('version')}).")

            log_audit_event(
                event_type="DOCUMENT_AUTO_SYNCED",
                user_id="sync_daemon",
                clearance="Compliance Officer",
                details={"filename": fn, "chunks": len(chunks), "version": ver_info.get("version")}
            )
        except Exception as e:
            err_msg = f"Failed to sync '{fn}': {str(e)}"
            errors.append(err_msg)
            logs.append(f"ERROR: {err_msg}")
            logger.error(err_msg)

    logs.append(f"Sync complete. Scanned: {scanned_count}, Ingested: {new_ingested}, Updated: {updated_count}.")
    
    result = {
        "status": "completed",
        "timestamp": datetime.utcnow().isoformat(),
        "scanned_files": scanned_count,
        "new_indexed": new_ingested,
        "updated_files": updated_count,
        "errors": errors,
        "logs": logs
    }
    _last_sync_info = result
    return result


def get_sync_status() -> Dict[str, Any]:
    """Returns status of most recent folder sync."""
    return _last_sync_info
