"""
Configuration Module for Project 1: Multi-Document RAG Company Policy Chatbot.
Loads settings from environment variables and validates API keys.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Setup basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("config")

# Load environment variables from .env file (force override to prioritize local settings)
try:
    load_dotenv(override=True)
    logger.info("Environment variables loaded from .env file (override=True).")
except Exception as e:
    logger.warning(f"Error loading .env file (using system environment): {e}")

# Sanitize key strings helper
def _clean_key(key_str: Optional[str]) -> str:
    if not key_str:
        return ""
    k = key_str.replace('\u2011', '-').replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-').strip()
    return k

raw_openai_key = _clean_key(os.getenv("OPENAI_API_KEY", ""))
raw_gemini_key = _clean_key(os.getenv("GEMINI_API_KEY", ""))

# Determine provider routing
is_valid_openai = bool(raw_openai_key and len(raw_openai_key) > 15 and "your-openai-api-key" not in raw_openai_key.lower() and "your_actual_key" not in raw_openai_key.lower() and not raw_openai_key.startswith("sk-<"))
is_valid_gemini = bool(raw_gemini_key and len(raw_gemini_key) > 15 and "your_actual_key" not in raw_gemini_key.lower())

if is_valid_openai:
    logger.info("Configuring standard OpenAI API endpoint for LLM completion.")
    OPENAI_API_KEY = raw_openai_key
    OPENAI_BASE_URL = None
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    # If LLM_MODEL in .env is set to a gemini model by accident while using OpenAI key, fix to gpt-4o-mini
    if "gemini" in LLM_MODEL.lower():
        LLM_MODEL = "gpt-4o-mini"
elif is_valid_gemini:
    logger.info("Configuring Gemini API (OpenAI-compatible) endpoint for LLM completion.")
    OPENAI_API_KEY = raw_gemini_key
    OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    env_model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    LLM_MODEL = env_model if "gemini" in env_model.lower() else "gemini-1.5-flash"
else:
    logger.info("No external LLM API key detected or invalid keys; local extractive synthesis will be used.")
    OPENAI_API_KEY = raw_openai_key or "sk-demo-key-local-only"
    OPENAI_BASE_URL = None
    LLM_MODEL = "gpt-4o-mini"

# We use sentence-transformers locally, so this setting is a fallback name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Parse integer configs with error recovery to defaults
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
except ValueError:
    logger.warning("Invalid CHUNK_SIZE in env, defaulting to 500")
    CHUNK_SIZE = 500

try:
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
except ValueError:
    logger.warning("Invalid CHUNK_OVERLAP in env, defaulting to 50")
    CHUNK_OVERLAP = 50

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_db")
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()

try:
    MAX_PAGES_LIMIT = int(os.getenv("MAX_PAGES_LIMIT", "100"))
except ValueError:
    logger.warning("Invalid MAX_PAGES_LIMIT in env, defaulting to 100")
    MAX_PAGES_LIMIT = 100

# Map log level string to logging levels
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)


def check_keys() -> bool:
    """
    Validates that a required API key is set and is not a placeholder.
    """
    if is_valid_openai or is_valid_gemini:
        return True
        
    logger.warning("No live API Key found. Operating in local extractive RAG mode.")
    return False


def get_openai_client():
    """
    Initializes and returns an OpenAI client instance routed to the configured provider.
    """
    import openai
    return openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_retries=0,
        timeout=4.0
    )
