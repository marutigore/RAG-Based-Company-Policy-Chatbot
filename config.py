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

# Load environment variables from .env file if it exists
# This enables local development using a .env file
try:
    load_dotenv()
    logger.info("Environment variables loaded from .env file.")
except Exception as e:
    logger.warning(f"Error loading .env file (using system environment): {e}")

raw_key = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY: str = raw_key.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-').strip()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Parse integer configs with error recovery to defaults
try:
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
except ValueError:
    logger.warning("Invalid CHUNK_SIZE in env, defaulting to 500")
    CHUNK_SIZE = 500

try:
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
except ValueError:
    logger.warning("Invalid CHUNK_OVERLAP in env, defaulting to 50")
    CHUNK_OVERLAP = 50

VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "./chroma_db")
LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Map log level string to logging levels
LOG_LEVEL: int = getattr(logging, LOG_LEVEL_STR, logging.INFO)


def check_keys() -> bool:
    """
    Validates that the required OpenAI API key is set and is not a placeholder.

    Returns:
        bool: True if key is valid and usable, False otherwise.

    Example:
        >>> from config import check_keys
        >>> is_valid = check_keys()
        >>> print(is_valid)
        True
    """
    # Check if key is missing, empty, or still set to the placeholder string
    if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "" or "your-openai-api-key" in OPENAI_API_KEY.lower() or "your_actual_key" in OPENAI_API_KEY.lower():
        logger.error("OPENAI_API_KEY is not set or is still the placeholder value.")
        return False
    
    logger.info("OPENAI_API_KEY check passed successfully.")
    return True
