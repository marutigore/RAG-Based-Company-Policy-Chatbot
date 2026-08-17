"""
User Feedback and Reinforcement Module.
Collects, aggregates, and analyzes user satisfaction ratings, negative feedback flags,
and correction notes to refine enterprise retrieval accuracy.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("utils.feedback")

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feedback.json")


def _ensure_data_dir() -> None:
    data_dir = os.path.dirname(FEEDBACK_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def _load_feedback() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading feedback file: {e}")
    return []


def _save_feedback(records: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    try:
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(records[-1000:], f, indent=2)
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")


def record_feedback(
    query: str,
    answer: str,
    rating: int,  # +1 for helpful, -1 for unhelpful
    comments: Optional[str] = None,
    issue_type: Optional[str] = None,
    user_id: str = "anonymous",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Records a user rating and qualitative critique."""
    records = _load_feedback()
    
    item = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer[:300] + ("..." if len(answer) > 300 else ""),
        "rating": 1 if rating > 0 else -1,
        "comments": comments or "",
        "issue_type": issue_type or ("helpful" if rating > 0 else "general_issue"),
        "user_id": user_id,
        "session_id": session_id
    }
    
    records.append(item)
    _save_feedback(records)
    logger.info(f"Feedback recorded: id={item['id']} rating={item['rating']} user={user_id}")
    return item


def get_feedback_summary() -> Dict[str, Any]:
    """Returns aggregated feedback analytics."""
    records = _load_feedback()
    if not records:
        return {
            "total_reviews": 0,
            "positive_count": 0,
            "negative_count": 0,
            "satisfaction_rate": 100.0,
            "recent_flagged": []
        }

    total = len(records)
    pos = sum(1 for r in records if r.get("rating", 0) > 0)
    neg = sum(1 for r in records if r.get("rating", 0) < 0)
    rate = round((pos / total) * 100.0, 1) if total > 0 else 100.0

    flagged = [r for r in records if r.get("rating", 0) < 0][-5:]

    return {
        "total_reviews": total,
        "positive_count": pos,
        "negative_count": neg,
        "satisfaction_rate": rate,
        "recent_flagged": flagged
    }


def list_feedback_records(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns recent feedback items."""
    records = _load_feedback()
    return records[-limit:][::-1]
