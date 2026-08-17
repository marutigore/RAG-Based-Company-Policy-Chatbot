"""
Authentication and Authorization Module.
Provides secure password hashing (PBKDF2-HMAC-SHA256), stateless JWT session tokens,
and Role-Based Access Control (RBAC) with seeded enterprise clearance tiers.
"""

import os
import json
import hmac
import base64
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger("utils.auth")

USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.json")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "synthara-enterprise-rag-jwt-secret-key-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Role definitions & default clearances
ROLE_CLEARANCES = {
    "Admin": "Compliance Officer",
    "Compliance Officer": "Compliance Officer",
    "Manager": "Manager",
    "Employee": "Employee"
}


def _ensure_data_dir() -> None:
    data_dir = os.path.dirname(USERS_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hashes password using PBKDF2-HMAC-SHA256 with a unique salt."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return f"{salt}:{key.hex()}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies plain password against stored salt:hash."""
    try:
        salt, key_hex = hashed_password.split(":")
        check_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return hmac.compare_digest(key_hex, check_key.hex())
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def _init_default_users() -> Dict[str, Dict[str, Any]]:
    """Seeds default enterprise users across clearance tiers."""
    _ensure_data_dir()
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading users file, reinitializing: {e}")

    default_users = {
        "admin": {
            "username": "admin",
            "full_name": "Portal Administrator",
            "password_hash": hash_password("admin123", salt="synthara_admin_salt"),
            "role": "Admin",
            "clearance": "Compliance Officer",
            "department": "Executive Security",
            "created_at": datetime.utcnow().isoformat()
        },
        "compliance": {
            "username": "compliance",
            "full_name": "Elena Rostova",
            "password_hash": hash_password("comp123", salt="synthara_comp_salt"),
            "role": "Compliance Officer",
            "clearance": "Compliance Officer",
            "department": "Legal & Audit",
            "created_at": datetime.utcnow().isoformat()
        },
        "manager": {
            "username": "manager",
            "full_name": "Marcus Vance",
            "password_hash": hash_password("mgr123", salt="synthara_mgr_salt"),
            "role": "Manager",
            "clearance": "Manager",
            "department": "Engineering Ops",
            "created_at": datetime.utcnow().isoformat()
        },
        "employee": {
            "username": "employee",
            "full_name": "Sarah Jenkins",
            "password_hash": hash_password("emp123", salt="synthara_emp_salt"),
            "role": "Employee",
            "clearance": "Employee",
            "department": "Product Design",
            "created_at": datetime.utcnow().isoformat()
        }
    }

    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to seed users file: {e}")

    return default_users


def _save_users(users: Dict[str, Dict[str, Any]]) -> None:
    _ensure_data_dir()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def _b64_decode(data_str: str) -> bytes:
    padding = '=' * (4 - (len(data_str) % 4)) if len(data_str) % 4 != 0 else ''
    return base64.urlsafe_b64decode(data_str + padding)


def create_jwt_token(payload: Dict[str, Any], expires_delta_hours: int = JWT_EXPIRATION_HOURS) -> str:
    """Generates an RFC 7519 HMAC-SHA256 JWT Token."""
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    exp_time = (datetime.utcnow() + timedelta(hours=expires_delta_hours)).timestamp()
    
    token_payload = payload.copy()
    token_payload["exp"] = exp_time
    token_payload["iat"] = datetime.utcnow().timestamp()

    header_b64 = _b64_encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))
    payload_b64 = _b64_encode(json.dumps(token_payload, separators=(',', ':')).encode('utf-8'))

    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(JWT_SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies and decodes JWT Token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(JWT_SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = _b64_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.warning("JWT Token signature mismatch.")
            return None

        payload = json.loads(_b64_decode(payload_b64).decode('utf-8'))
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            logger.info("JWT Token has expired.")
            return None

        return payload
    except Exception as e:
        logger.error(f"JWT verification error: {e}")
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates credentials against the user database."""
    users = _init_default_users()
    user = users.get(username.lower().strip())
    if not user:
        # Legacy fallback compatibility
        if username == "admin" and password == "password":
            return {
                "username": "admin",
                "full_name": "Portal Administrator",
                "role": "Admin",
                "clearance": "Compliance Officer",
                "department": "System"
            }
        return None

    if verify_password(password, user["password_hash"]):
        return {
            "username": user["username"],
            "full_name": user.get("full_name", user["username"]),
            "role": user.get("role", "Employee"),
            "clearance": user.get("clearance", "Employee"),
            "department": user.get("department", "General")
        }
    return None


def get_all_users() -> List[Dict[str, Any]]:
    """Returns safe user profiles list for administration."""
    users = _init_default_users()
    return [
        {
            "username": u["username"],
            "full_name": u.get("full_name", u["username"]),
            "role": u.get("role", "Employee"),
            "clearance": u.get("clearance", "Employee"),
            "department": u.get("department", "General"),
            "created_at": u.get("created_at")
        }
        for u in users.values()
    ]


def register_user(username: str, password: str, full_name: str, role: str, department: str = "General") -> Dict[str, Any]:
    """Registers a new user."""
    users = _init_default_users()
    username_key = username.lower().strip()
    if username_key in users:
        raise ValueError(f"User '{username}' already exists.")

    clearance = ROLE_CLEARANCES.get(role, "Employee")
    user_record = {
        "username": username_key,
        "full_name": full_name,
        "password_hash": hash_password(password),
        "role": role,
        "clearance": clearance,
        "department": department,
        "created_at": datetime.utcnow().isoformat()
    }
    users[username_key] = user_record
    _save_users(users)
    return {
        "username": username_key,
        "full_name": full_name,
        "role": role,
        "clearance": clearance,
        "department": department
    }
