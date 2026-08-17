"""
Enterprise PII Detection & Sensitive Data Redaction Guardrail Module.
Prevents leakage of Personally Identifiable Information (SSNs, Credit Cards,
Phone Numbers, API Keys, Private Emails) in both user queries and indexing contexts.
"""

import re
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger("utils.pii_guardrail")

PII_PATTERNS = [
    ("SSN", re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[REDACTED_SSN]"),
    ("CREDIT_CARD", re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'), "[REDACTED_CARD]"),
    ("API_SECRET", re.compile(r'\b(?:AIza[0-9A-Za-z-_]{35}|ghp_[0-9a-zA-Z]{36}|sk-[0-9a-zA-Z]{20,})\b'), "[REDACTED_SECRET]"),
    ("PHONE_NUMBER", re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), "[REDACTED_PHONE]"),
    ("EMAIL_ADDRESS", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "[REDACTED_EMAIL]")
]


def scan_pii(text: str) -> List[Dict[str, Any]]:
    """Identifies all PII entities present in the text."""
    if not text:
        return []
    
    findings = []
    for entity_type, pattern, _ in PII_PATTERNS:
        for match in pattern.finditer(text):
            findings.append({
                "type": entity_type,
                "start": match.start(),
                "end": match.end(),
                "value_masked": match.group()[:2] + "****" + match.group()[-2:] if len(match.group()) > 4 else "****"
            })
    return findings


def redact_pii(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Replaces sensitive PII occurrences with standard privacy tags.
    Returns the redacted text and a list of identified entities.
    """
    if not text:
        return "", []

    sanitized = text
    detected = []

    for entity_type, pattern, placeholder in PII_PATTERNS:
        matches = list(pattern.finditer(sanitized))
        if matches:
            for m in matches:
                detected.append({
                    "type": entity_type,
                    "placeholder": placeholder
                })
            sanitized = pattern.sub(placeholder, sanitized)

    if detected:
        logger.info(f"Guardrail redacted {len(detected)} sensitive PII tokens from text.")

    return sanitized, detected


def mask_sensitive_query(query: str) -> str:
    """Convenience function returning redacted query string."""
    redacted, _ = redact_pii(query)
    return redacted
