"""
Embedding Module.
Interfaces with OpenAI API to generate vector embeddings for text chunks.
"""

import logging
from typing import List
import openai
import config

# Initialize module logger
logger = logging.getLogger("utils.embedder")


def _get_client() -> openai.OpenAI:
    """
    Initializes and returns an OpenAI client instance.

    Returns:
        openai.OpenAI: Configured OpenAI client.
    """
    # Instantiate client with key loaded from config
    return openai.OpenAI(api_key=config.OPENAI_API_KEY)


def get_embedding(text: str, model: str = config.EMBEDDING_MODEL) -> List[float]:
    """
    Generates a dense vector embedding for a single text string.

    Args:
        text (str): Input text to embed.
        model (str): Target embedding model ID.

    Returns:
        List[float]: Embedding vector.

    Raises:
        ValueError: If configuration validation or input checking fails.
        openai.OpenAIError: For API call errors.

    Example:
        >>> vec = get_embedding("Hello world")
        >>> print(len(vec))
        1536
    """
    # Input validation
    if not text or not text.strip():
        logger.error("Attempted to embed empty or null text.")
        raise ValueError("Input text cannot be empty.")

    # Check API key before invoking OpenAI
    if not config.OPENAI_API_KEY:
        logger.error("Missing OpenAI API key configuration.")
        raise ValueError("OpenAI API key must be set in configuration.")

    try:
        logger.info(f"Generating embedding for text length {len(text)} characters.")
        client = _get_client()
        
        # API call to generate embedding
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        # Extract and return float list
        return response.data[0].embedding
        
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded: {e}")
        raise
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API authentication failed: {e}")
        raise
    except openai.APIConnectionError as e:
        logger.error(f"Failed to connect to OpenAI API server: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_embedding: {e}")
        raise


def get_embeddings_batch(texts: List[str], model: str = config.EMBEDDING_MODEL, batch_size: int = 100) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of text strings in batches.
    Batching reduces API calls and avoids hitting single-request overheads.

    Args:
        texts (List[str]): List of texts to embed.
        model (str): Target embedding model ID.
        batch_size (int): Max number of items to embed in a single batch call.

    Returns:
        List[List[float]]: List of embedding vectors matching input size.

    Raises:
        ValueError: If config is invalid.
        openai.OpenAIError: If API call fails.

    Example:
        >>> vectors = get_embeddings_batch(["Hello", "World"])
        >>> print(len(vectors))
        2
    """
    if not texts:
        return []

    if not config.OPENAI_API_KEY:
        logger.error("Missing OpenAI API key configuration.")
        raise ValueError("OpenAI API key must be set in configuration.")

    client = _get_client()
    embeddings: List[List[float]] = []

    logger.info(f"Generating embeddings for {len(texts)} chunks in batches of {batch_size}")

    # Process list in slices of size batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Skip empty strings within a batch to avoid API errors
        batch_clean = [t if t.strip() else "[empty]" for t in batch]
        
        try:
            logger.info(f"Sending batch {i // batch_size + 1} ({len(batch_clean)} texts) to OpenAI.")
            # Call API with batch list
            response = client.embeddings.create(
                input=batch_clean,
                model=model
            )
            # Append retrieved embeddings to the results list
            for item in response.data:
                embeddings.append(item.embedding)
                
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error in batch {i // batch_size + 1}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in batch processing: {e}")
            raise

    return embeddings
