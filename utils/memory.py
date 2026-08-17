"""
Conversation Memory Module.
Provides persistent multi-turn chat session management, historical message tracking,
and contextual query resolution for follow-up questions.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("utils.memory")

CONVERSATIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations.json")


def _ensure_data_dir() -> None:
    data_dir = os.path.dirname(CONVERSATIONS_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def _load_conversations() -> Dict[str, Dict[str, Any]]:
    _ensure_data_dir()
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading conversations: {e}")
    return {}


def _save_conversations(convs: Dict[str, Dict[str, Any]]) -> None:
    _ensure_data_dir()
    try:
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(convs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversations: {e}")


def create_session(user_id: str = "anonymous", title: Optional[str] = None) -> Dict[str, Any]:
    """Creates a new conversation session."""
    convs = _load_conversations()
    session_id = str(uuid.uuid4())[:8]
    session_data = {
        "id": session_id,
        "user_id": user_id,
        "title": title or f"Session {datetime.utcnow().strftime('%b %d, %H:%M')}",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "messages": []
    }
    convs[session_id] = session_data
    _save_conversations(convs)
    logger.info(f"Created conversation session {session_id} for user {user_id}")
    return session_data


def add_message(
    session_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    user_id: str = "anonymous"
) -> Dict[str, Any]:
    """Appends a message to an active conversation session."""
    convs = _load_conversations()
    if session_id not in convs:
        convs[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "title": content[:30] + ("..." if len(content) > 30 else ""),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "messages": []
        }
    
    session = convs[session_id]
    msg_id = str(uuid.uuid4())[:8]
    msg_data = {
        "id": msg_id,
        "role": role,
        "content": content,
        "citations": citations or [],
        "evaluation": evaluation or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    session["messages"].append(msg_data)
    session["updated_at"] = datetime.utcnow().isoformat()
    
    # Auto-generate title from first user query if still generic
    if role == "user" and session.get("title", "").startswith("Session "):
        session["title"] = content[:36] + ("..." if len(content) > 36 else "")

    _save_conversations(convs)
    return msg_data


def get_session_messages(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves chronological messages for a conversation session."""
    convs = _load_conversations()
    session = convs.get(session_id)
    if not session:
        return []
    return session.get("messages", [])[-limit:]


def list_sessions(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all stored conversation sessions."""
    convs = _load_conversations()
    results = []
    for s_id, s_data in convs.items():
        if user_id and s_data.get("user_id") not in [user_id, "anonymous", "admin"]:
            continue
        results.append({
            "id": s_data["id"],
            "title": s_data.get("title", "Untitled Session"),
            "user_id": s_data.get("user_id", "anonymous"),
            "message_count": len(s_data.get("messages", [])),
            "updated_at": s_data.get("updated_at", s_data.get("created_at"))
        })
    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results


def delete_session(session_id: str) -> bool:
    """Deletes a conversation session."""
    convs = _load_conversations()
    if session_id in convs:
        del convs[session_id]
        _save_conversations(convs)
        return True
    return False


def build_contextual_query(question: str, history: List[Dict[str, Any]]) -> str:
    """
    Reformulates follow-up queries with pronoun resolution using past dialogue context.
    """
    if not history:
        return question

    recent_user_turns = [m["content"] for m in history if m.get("role") == "user"]
    if not recent_user_turns:
        return question

    q_lower = question.lower().strip()
    follow_up_cues = ["what about", "how about", "and for", "why", "who is", "explain that", "does it apply", "is that also"]
    
    if any(q_lower.startswith(cue) for cue in follow_up_cues) or len(q_lower.split()) <= 4:
        last_turn = recent_user_turns[-1]
        return f"{last_turn} -> Contextual follow-up: {question}"

    return question
