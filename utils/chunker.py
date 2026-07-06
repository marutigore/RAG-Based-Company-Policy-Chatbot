"""
Document Chunker Module.
Splits text from loaded pages into overlapping chunks using token counts (tiktoken).
"""

import logging
from typing import List, Dict, Any
import tiktoken
import config

# Initialize module logger
logger = logging.getLogger("utils.chunker")


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Counts the number of tokens in a text string using the specified encoding.

    Args:
        text (str): Input text string.
        encoding_name (str): Tiktoken encoding name (default is cl100k_base).

    Returns:
        int: Number of tokens.

    Example:
        >>> tokens = count_tokens("Hello world!")
        >>> print(tokens)
        3
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error getting tiktoken encoding '{encoding_name}': {e}. Falling back to character estimation.")
        # Rough fallback calculation: 1 token ~ 4 characters
        return len(text) // 4


def split_text_by_tokens(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Splits a single text string into chunks of a given token length with overlap.

    Args:
        text (str): Raw string content of a page.
        chunk_size (int): Max tokens per chunk.
        chunk_overlap (int): Token overlap between consecutive chunks.
        encoding_name (str): Tiktoken encoding name.

    Returns:
        List[str]: List of text chunks.

    Example:
        >>> chunks = split_text_by_tokens("Very long text...", chunk_size=2, chunk_overlap=1)
    """
    # Validation checks
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size.")

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
    except Exception as e:
        logger.error(f"Tiktoken tokenization failed: {e}. Splitting by words instead.")
        # Fallback to word-based splitting if tiktoken fails
        words = text.split()
        chunks: List[str] = []
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunks.append(" ".join(words[i : i + chunk_size]))
        return chunks

    chunks_text: List[str] = []
    num_tokens = len(tokens)
    
    # Slide a window across the tokens to extract chunks
    start = 0
    while start < num_tokens:
        end = min(start + chunk_size, num_tokens)
        chunk_tokens = tokens[start:end]
        
        # Decode tokens back into a string
        chunk_str = encoding.decode(chunk_tokens)
        chunks_text.append(chunk_str)
        
        # Move the start pointer forward by step size (chunk_size - overlap)
        step = chunk_size - chunk_overlap
        
        # Guard against infinite loops if settings are misconfigured
        if step <= 0:
            logger.error("Step size resolved to <= 0. Breaking loop to prevent infinite run.")
            break
            
        start += step

    return chunks_text


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Takes a list of documents (pages) and splits their text into overlapping chunks,
    preserving page-level metadata and appending chunk-specific details.

    Args:
        documents (List[Dict[str, Any]]): List of dicts representing page documents.
        chunk_size (int): Target tokens per chunk.
        chunk_overlap (int): Target tokens of overlap.

    Returns:
        List[Dict[str, Any]]: List of chunked document segments.

    Example:
        >>> pages = [{'text': '...', 'metadata': {'source': 'doc.pdf', 'page': 1}}]
        >>> chunks = split_documents(pages)
        >>> print(chunks[0]['metadata'])
        {'source': 'doc.pdf', 'page': 1, 'chunk_idx': 0, 'token_count': 45}
    """
    chunked_docs: List[Dict[str, Any]] = []

    logger.info(f"Chunking {len(documents)} document pages with chunk_size={chunk_size}, overlap={chunk_overlap}")

    for doc in documents:
        text = doc.get("text", "")
        meta = doc.get("metadata", {}).copy()

        # Skip empty text strings
        if not text.strip():
            continue

        # Split current document's text into list of chunks
        chunks = split_text_by_tokens(text, chunk_size, chunk_overlap)

        # Build chunk dictionaries preserving base metadata
        for idx, chunk_content in enumerate(chunks):
            # Recalculate token count for each specific text chunk
            tok_count = count_tokens(chunk_content)
            
            chunk_meta = meta.copy()
            chunk_meta["chunk_idx"] = idx
            chunk_meta["token_count"] = tok_count

            chunked_docs.append({
                "text": chunk_content,
                "metadata": chunk_meta
            })

    logger.info(f"Generated {len(chunked_docs)} chunks from {len(documents)} pages.")
    return chunked_docs
