"""
Validator Module.
Contains input verification routines and RAG evaluation methods (Faithfulness and Answer Relevancy).
"""

import json
import logging
from typing import List, Dict, Any
import openai
import config

# Initialize module logger
logger = logging.getLogger("utils.validator")


def _sanitize(text: str) -> str:
    """Replaces problematic unicode characters with ASCII equivalents."""
    replacements = {
        '\u2011': '-', '\u2013': '-', '\u2014': '-',
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u00a0': ' ',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


def validate_query(query: str) -> str:
    """
    Validates a natural language user query.
    Ensures it is non-empty and fits within safe size limits.

    Args:
        query (str): Raw query string.

    Returns:
        str: Cleaned query string.

    Raises:
        ValueError: If query is invalid.

    Example:
        >>> validated = validate_query("   What is the policy?  ")
        >>> print(validated)
        "What is the policy?"
    """
    if not query or not query.strip():
        logger.error("Empty user query validation failed.")
        raise ValueError("Query string cannot be empty. Please type a question.")

    cleaned_query = query.strip()
    
    # Check length limits (e.g. 500 characters maximum)
    if len(cleaned_query) > 500:
        logger.warning(f"Query too long ({len(cleaned_query)} chars). Truncating.")
        cleaned_query = cleaned_query[:500]

    return cleaned_query


def validate_file_extension(filename: str) -> bool:
    """
    Verifies that the uploaded file has a valid PDF extension.

    Args:
        filename (str): Name of the file.

    Returns:
        bool: True if file is a PDF, False otherwise.
    """
    return filename.lower().endswith(".pdf")


def evaluate_faithfulness(contexts: List[str], answer: str) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer is fully grounded in the retrieved contexts (no hallucination).
    Uses a standard LLM-as-a-judge prompt.

    Args:
        contexts (List[str]): List of retrieved text chunks.
        answer (str): Generated LLM answer.

    Returns:
        Dict[str, Any]: Dictionary containing 'score' (0.0 to 1.0) and 'reasoning'.

    Example:
        >>> res = evaluate_faithfulness(["Context text here"], "Answer text here")
        >>> print(res['score'])
        1.0
    """
    # Guard against empty contexts or answers
    if not contexts or not answer:
        return {"score": 0.0, "reasoning": "Missing inputs to evaluate."}

    # Format retrieved contexts into a structured string
    context_str = "\n---\n".join([f"Context {i+1}:\n{_sanitize(c)}" for i, c in enumerate(contexts)])
    answer = _sanitize(answer)

    system_prompt = (
        "You are an objective evaluation auditor. Assess if the candidate answer is strictly grounded in the provided contexts.\n"
        "Do not use external knowledge. Every fact in the answer must exist in the contexts.\n"
        "Provide your evaluation in a JSON structure containing 'score' (a float from 0.0 to 1.0, where 1.0 means fully grounded and 0.0 means completely unsupported) and 'reasoning' (a brief explanation)."
    )

    user_prompt = (
        f"Contexts:\n{context_str}\n\n"
        f"Candidate Answer:\n{answer}\n\n"
        f"Output JSON formatting rule:\n"
        f"{{\n"
        f"  \"score\": 1.0,\n"
        f"  \"reasoning\": \"Explanation why it is grounded or not.\"\n"
        f"}}"
    )

    try:
        client = config.get_openai_client()
        logger.info("Calling OpenAI to evaluate answer Faithfulness...")
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Greedy search for consistent evaluations
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from LLM judge.")
            
        data = json.loads(content)
        # Ensure score is parsed as float
        data["score"] = float(data.get("score", 0.0))
        logger.info(f"Faithfulness evaluated: Score = {data['score']}")
        return data

    except Exception as e:
        logger.error(f"Error during faithfulness evaluation: {e}")
        return {"score": 0.0, "reasoning": f"Evaluation error: {str(e)}"}


def evaluate_answer_relevancy(question: str, answer: str) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer directly and completely addresses the user's question.
    Uses an LLM-as-a-judge prompt.

    Args:
        question (str): User's original question.
        answer (str): Generated LLM answer.

    Returns:
        Dict[str, Any]: Dictionary containing 'score' (0.0 to 1.0) and 'reasoning'.

    Example:
        >>> res = evaluate_answer_relevancy("What is x?", "x is y.")
        >>> print(res['score'])
        1.0
    """
    if not question or not answer:
        return {"score": 0.0, "reasoning": "Missing inputs to evaluate."}

    system_prompt = (
        "You are an objective evaluation auditor. Assess if the candidate answer directly answers the user's question.\n"
        "A relevant answer addresses the specific question asked without rambling or introducing unrelated information.\n"
        "Provide your evaluation in a JSON structure containing 'score' (a float from 0.0 to 1.0, where 1.0 means perfectly relevant and addressing all components, and 0.0 means completely off-topic) and 'reasoning' (a brief explanation)."
    )

    user_prompt = (
        f"Question:\n{_sanitize(question)}\n\n"
        f"Candidate Answer:\n{_sanitize(answer)}\n\n"
        f"Output JSON formatting rule:\n"
        f"{{\n"
        f"  \"score\": 1.0,\n"
        f"  \"reasoning\": \"Explanation of answer relevance.\"\n"
        f"}}"
    )

    try:
        client = config.get_openai_client()
        logger.info("Calling OpenAI to evaluate Answer Relevancy...")
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from LLM judge.")
            
        data = json.loads(content)
        data["score"] = float(data.get("score", 0.0))
        logger.info(f"Relevancy evaluated: Score = {data['score']}")
        return data

    except Exception as e:
        logger.error(f"Error during relevancy evaluation: {e}")
        return {"score": 0.0, "reasoning": f"Evaluation error: {str(e)}"}
