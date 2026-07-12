"""
Configuration Module for Project 1: Multi-Document RAG Company Policy Chatbot.
Loads settings from environment variables and validates API keys.
"""

import os
import logging
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

# Check for Gemini key first (very generous free tier, no billing required)
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
raw_openai_key = os.getenv("OPENAI_API_KEY", "").strip()

# If Gemini Key is present and valid, dynamically route OpenAI client calls to Google Gemini API
if GEMINI_KEY and "your_actual_key" not in GEMINI_KEY.lower() and GEMINI_KEY != "":
    logger.info("Configuring Gemini API (OpenAI-compatible) endpoint for LLM completion.")
    OPENAI_API_KEY = GEMINI_KEY
    OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
else:
    logger.info("Configuring standard OpenAI API endpoint for LLM completion.")
    OPENAI_API_KEY = raw_openai_key.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-').strip()
    OPENAI_BASE_URL = None
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

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

# Map log level string to logging levels
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)


def check_keys() -> bool:
    """
    Validates that a required API key is set and is not a placeholder.
    """
    if GEMINI_KEY and "your_actual_key" not in GEMINI_KEY.lower() and GEMINI_KEY != "":
        return True
    
    if OPENAI_API_KEY and OPENAI_API_KEY.strip() != "" and "your-openai-api-key" not in OPENAI_API_KEY.lower() and "your_actual_key" not in OPENAI_API_KEY.lower() and "sk-<your" not in OPENAI_API_KEY.lower():
        return True
        
    logger.error("No valid API Key found. Please add a valid OPENAI_API_KEY or GEMINI_API_KEY to your .env file.")
    return False


def get_openai_client():
    """
    Initializes and returns an OpenAI client instance routed to the configured provider.
    """
    import openai
    return openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
