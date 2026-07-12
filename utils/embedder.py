"""
Embedding Module.
Generates vector embeddings for text chunks completely LOCALLY using SentenceTransformers.
This ensures 100% free operation with zero quota limits or API keys required.
Pads embeddings to 1536 dimensions to remain compatible with standard database indices.
"""

import logging
from typing import List
import openai
from sentence_transformers import SentenceTransformer

# Initialize module logger
logger = logging.getLogger("utils.embedder")

# Global reference for local SentenceTransformer model instance
_model = None
TARGET_DIMENSIONS = 1536


def _get_model() -> SentenceTransformer:
    """
    Initializes and returns the local SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SentenceTransformer('all-MiniLM-L6-v2') locally on device: {device}...")
        # Load the lightweight MiniLM model (runs in milliseconds on CPU or CUDA)
        _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return _model


def _pad_vector(vector: List[float], target_dim: int = TARGET_DIMENSIONS) -> List[float]:
    """
    Pads a float vector with zeros to reach target dimensions.
    Mathematically preserves cosine similarity query results.
    """
    current_dim = len(vector)
    if current_dim >= target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - current_dim)


def get_embedding(text: str, model: str = None) -> List[float]:
    """
    Generates a dense vector embedding for a single text string locally.

    Args:
        text (str): Input text to embed.
        model (str): Ignored (always runs all-MiniLM-L6-v2 locally).

    Returns:
        List[float]: 1536-dimensional padded embedding vector.
    """
    if not text or not text.strip():
        logger.error("Attempted to embed empty or null text.")
        raise ValueError("Input text cannot be empty.")

    try:
        model_instance = _get_model()
        raw_embedding = model_instance.encode(text)
        if hasattr(raw_embedding, "tolist"):
            raw_embedding = raw_embedding.tolist()
        else:
            raw_embedding = list(raw_embedding)
        # Pad to 1536 dimensions to match database index constraints
        return _pad_vector(raw_embedding)
    except Exception as e:
        logger.error(f"Error in local get_embedding: {e}")
        raise


def get_embeddings_batch(texts: List[str], model: str = None, batch_size: int = 100) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of text strings in batches locally.

    Args:
        texts (List[str]): List of texts to embed.
        model (str): Ignored.
        batch_size (int): Max number of items to embed in a single batch.

    Returns:
        List[List[float]]: List of 1536-dimensional padded embedding vectors.
    """
    if not texts:
        return []

    try:
        model_instance = _get_model()
        raw_embeddings = model_instance.encode(texts, batch_size=batch_size, show_progress_bar=False)
        if hasattr(raw_embeddings, "tolist"):
            raw_embeddings = raw_embeddings.tolist()
        else:
            raw_embeddings = [list(vec) for vec in raw_embeddings]
        
        # Pad all vectors to 1536 dimensions
        padded_embeddings = [_pad_vector(vec) for vec in raw_embeddings]
        return padded_embeddings
    except Exception as e:
        logger.error(f"Error in local get_embeddings_batch: {e}")
        raise
