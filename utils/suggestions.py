"""
Suggested Questions & Smart Autocomplete Module.
Generates document-specific policy prompts and provides real-time prefix-matching
autocomplete to streamline employee inquiries and onboarding.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("utils.suggestions")

SUGGESTIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "suggestions.json")

DEFAULT_SUGGESTIONS = [
    "What is the standard vacation leave policy?",
    "What are the standard working hours and remote guidelines?",
    "How do I request maternity or paternity leave?",
    "What is the company policy on business expense reimbursement?",
    "What are the password security and MFA requirements?",
    "How does the performance review and bonus cycle work?",
    "What is the code of conduct for workplace harassment?",
    "How to submit IT equipment and software purchase requests?"
]


def _ensure_data_dir() -> None:
    data_dir = os.path.dirname(SUGGESTIONS_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def _load_suggestions() -> List[str]:
    _ensure_data_dir()
    if os.path.exists(SUGGESTIONS_FILE):
        try:
            with open(SUGGESTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data
        except Exception as e:
            logger.warning(f"Error loading suggestions file: {e}")
    return DEFAULT_SUGGESTIONS.copy()


def _save_suggestions(suggestions: List[str]) -> None:
    _ensure_data_dir()
    try:
        with open(SUGGESTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(dict.fromkeys(suggestions)), f, indent=2)
    except Exception as e:
        logger.error(f"Error saving suggestions file: {e}")


def generate_document_suggestions(document_text: str, filename: str) -> List[str]:
    """Generates contextual questions based on document content keywords."""
    generated = []
    text_lower = document_text.lower()
    fn_lower = filename.lower()

    if "leave" in text_lower or "leave" in fn_lower or "vacation" in text_lower:
        generated.extend([
            f"What are the annual leave entitlements in {filename}?",
            f"How is carryover leave calculated under {filename}?",
            "What documentation is required for sick leave approval?"
        ])
    if "security" in text_lower or "security" in fn_lower or "password" in text_lower:
        generated.extend([
            f"What are the data protection rules in {filename}?",
            "How should security incidents and data breaches be reported?",
            "What are the device encryption and VPN requirements?"
        ])
    if "remote" in text_lower or "travel" in text_lower or "expense" in text_lower:
        generated.extend([
            f"What expenses are eligible for reimbursement in {filename}?",
            "What is the daily per diem meal allowance during travel?",
            "What are the home office equipment stipends?"
        ])
    if "conduct" in text_lower or "ethics" in text_lower or "harassment" in text_lower:
        generated.extend([
            f"What is the reporting procedure for ethics violations in {filename}?",
            "What anti-retaliation protections exist for whistleblowers?"
        ])

    if not generated:
        generated.append(f"What are the key policy mandates specified in {filename}?")

    # Merge with current list
    current = _load_suggestions()
    for q in generated:
        if q not in current:
            current.insert(0, q)
    _save_suggestions(current)
    return generated


def get_all_suggestions(limit: int = 10) -> List[str]:
    """Returns top suggested questions."""
    suggestions = _load_suggestions()
    return suggestions[:limit]


def get_autocomplete_suggestions(prefix: str, limit: int = 6) -> List[str]:
    """Provides fast fuzzy/prefix matching for typing autocomplete."""
    if not prefix or not prefix.strip():
        return get_all_suggestions(limit=limit)

    p_lower = prefix.lower().strip()
    suggestions = _load_suggestions()
    
    matches = [s for s in suggestions if s.lower().startswith(p_lower)]
    contain_matches = [s for s in suggestions if p_lower in s.lower() and s not in matches]
    
    combined = matches + contain_matches
    return combined[:limit] if combined else DEFAULT_SUGGESTIONS[:limit]
