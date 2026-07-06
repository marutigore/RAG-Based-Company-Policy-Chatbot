"""
Utilities package for Multi-Document RAG.
Exposes loader, chunker, embedder, retriever, and validator modules.
"""

import logging
import sys
from config import LOG_LEVEL

# Configure root logger for the utils package
logger = logging.getLogger("utils")
logger.setLevel(LOG_LEVEL)

# Ensure handlers are added if not already present
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("Utils package initialized successfully.")
