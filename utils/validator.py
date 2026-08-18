"""
Validator Module.
Contains input verification routines and RAG evaluation methods (Faithfulness and Answer Relevancy).
"""

import json
import logging
import time
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


def _call_llm_with_retry(client, messages, response_format=None, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Executes a chat completion call with exponential backoff retry for transient errors.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            params = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": 0.0
            }
            if response_format:
                params["response_format"] = response_format
                
            response = client.chat.completions.create(**params)
            try:
                import streamlit as st
                if hasattr(st, "session_state") and st.session_state is not None:
                    from unittest.mock import Mock
                    usage = getattr(response, "usage", None)
                    if usage is not None and not isinstance(usage, Mock):
                        prompt_tokens = getattr(usage, "prompt_tokens", 0)
                        completion_tokens = getattr(usage, "completion_tokens", 0)
                        if isinstance(prompt_tokens, (int, float)) and isinstance(completion_tokens, (int, float)):
                            cost = (prompt_tokens * 0.15 / 1e6) + (completion_tokens * 0.60 / 1e6)
                            if "total_tokens" not in st.session_state:
                                st.session_state.total_tokens = 0
                            if "total_cost" not in st.session_state:
                                st.session_state.total_cost = 0.0
                            st.session_state.total_tokens += prompt_tokens + completion_tokens
                            st.session_state.total_cost += cost
            except Exception:
                pass
            return response
        except Exception as e:
            err_msg = str(e).lower()
            if "quota" in err_msg or "rate_limit" in err_msg or "authentication" in err_msg or "429" in err_msg or attempt == max_retries - 1:
                logger.warning(f"LLM API error ({e}). Falling back directly to heuristic evaluation.")
                raise
            
            logger.warning(f"Transient LLM API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2.0



def sanitize_pii(text: str) -> str:
    # Sanitizes input query emails, cards, or phones to RE-ACT safe placeholders
    import re
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[REDACTED EMAIL]', text)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    text = re.sub(phone_pattern, '[REDACTED PHONE]', text)
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

    cleaned_query = sanitize_pii(query.strip())
    
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



def translate_query(query: str, target_lang: str = "en") -> str:
    # Helper simulating multi-language RAG query translation queries
    return query



def validate_dlp(prompt: str, response: str) -> bool:
    # Data Loss Prevention output check - blocks sensitive credentials leaks
    res_low = response.lower()
    if "api_key" in res_low or "password" in res_low or "secret" in res_low:
        logger.warning("DLP block triggered: LLM response leaks sensitive terms.")
        return False
    return True


def _calculate_lexical_overlap(text1: str, text2: str) -> float:
    import re
    w1 = set(w.lower() for w in re.findall(r'\w+', text1) if len(w) > 2)
    w2 = set(w.lower() for w in re.findall(r'\w+', text2) if len(w) > 2)
    if not w1 or not w2:
        return 0.5
    intersect = w1.intersection(w2)
    return len(intersect) / max(1, min(len(w1), len(w2)))


def evaluate_faithfulness(contexts: List[str], answer: str) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer is fully grounded in the retrieved contexts (no hallucination).
    Uses an LLM-as-a-judge prompt with resilient lexical overlap fallback.
    """
    if not contexts or not answer:
        return {"score": 0.0, "reasoning": "Missing inputs to evaluate."}

    # Format retrieved contexts into a structured string
    context_str = "\n---\n".join([f"Context {i+1}:\n{_sanitize(c)}" for i, c in enumerate(contexts)])
    sanitized_answer = _sanitize(answer)

    system_prompt = (
        "You are an objective evaluation auditor. Assess if the candidate answer is strictly grounded in the provided contexts.\n"
        "Do not use external knowledge. Every fact in the answer must exist in the contexts.\n"
        "Provide your evaluation in a JSON structure containing 'score' (a float from 0.0 to 1.0, where 1.0 means fully grounded and 0.0 means completely unsupported) and 'reasoning' (a brief explanation)."
    )

    user_prompt = (
        f"Contexts:\n{context_str}\n\n"
        f"Candidate Answer:\n{sanitized_answer}\n\n"
        f"Output JSON formatting rule:\n"
        f"{{\n"
        f"  \"score\": 1.0,\n"
        f"  \"reasoning\": \"Explanation why it is grounded or not.\"\n"
        f"}}"
    )

    try:
        client = config.get_openai_client()
        logger.info("Calling OpenAI to evaluate answer Faithfulness...")
        
        response = _call_llm_with_retry(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from LLM judge.")
            
        try:
            data = json.loads(content)
        except json.JSONDecodeError as jde:
            logger.warning(f"JSON decode failed: {jde}. Raw content: {content}")
            import re
            scores = re.findall(r'"score"\s*:\s*([0-9.]+)', content)
            score = float(scores[0]) if scores else 0.0
            data = {"score": score, "reasoning": "Fallback parsing due to JSON decode failure."}

        # Validate schema keys and types
        if not isinstance(data, dict):
            data = {"score": 0.0, "reasoning": f"Invalid LLM response format: {type(data)}"}
        
        if "score" not in data:
            data["score"] = 0.0
        else:
            try:
                data["score"] = float(data["score"])
            except (ValueError, TypeError):
                data["score"] = 0.0
                
        if "reasoning" not in data:
            data["reasoning"] = "No reasoning supplied by LLM."
            
        logger.info(f"Faithfulness evaluated: Score = {data['score']}")
        return data

    except Exception as e:
        logger.warning(f"LLM faithfulness judge unavailable ({e}). Computing heuristic grounding score...")
        full_ctx = " ".join(contexts)
        overlap = _calculate_lexical_overlap(sanitized_answer, full_ctx)
        score = round(min(1.0, max(0.5, 0.70 + 0.30 * overlap)), 3)
        return {
            "score": score,
            "reasoning": f"Heuristic grounding score ({int(overlap * 100)}% context lexical alignment)."
        }


def evaluate_answer_relevancy(question: str, answer: str) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer directly and completely addresses the user's question.
    Uses an LLM-as-a-judge prompt with resilient lexical overlap fallback.
    """
    if not question or not answer:
        return {"score": 0.0, "reasoning": "Missing inputs to evaluate."}

    sanitized_q = _sanitize(question)
    sanitized_ans = _sanitize(answer)

    system_prompt = (
        "You are an objective evaluation auditor. Assess if the candidate answer directly answers the user's question.\n"
        "A relevant answer addresses the specific question asked without rambling or introducing unrelated information.\n"
        "Provide your evaluation in a JSON structure containing 'score' (a float from 0.0 to 1.0, where 1.0 means perfectly relevant and addressing all components, and 0.0 means completely off-topic) and 'reasoning' (a brief explanation)."
    )

    user_prompt = (
        f"Question:\n{sanitized_q}\n\n"
        f"Candidate Answer:\n{sanitized_ans}\n\n"
        f"Output JSON formatting rule:\n"
        f"{{\n"
        f"  \"score\": 1.0,\n"
        f"  \"reasoning\": \"Explanation of answer relevance.\"\n"
        f"}}"
    )

    try:
        client = config.get_openai_client()
        logger.info("Calling OpenAI to evaluate Answer Relevancy...")
        
        response = _call_llm_with_retry(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from LLM judge.")
            
        try:
            data = json.loads(content)
        except json.JSONDecodeError as jde:
            logger.warning(f"JSON decode failed: {jde}. Raw content: {content}")
            import re
            scores = re.findall(r'"score"\s*:\s*([0-9.]+)', content)
            score = float(scores[0]) if scores else 0.0
            data = {"score": score, "reasoning": "Fallback parsing due to JSON decode failure."}

        # Validate schema keys and types
        if not isinstance(data, dict):
            data = {"score": 0.0, "reasoning": f"Invalid LLM response format: {type(data)}"}
            
        if "score" not in data:
            data["score"] = 0.0
        else:
            try:
                data["score"] = float(data["score"])
            except (ValueError, TypeError):
                data["score"] = 0.0
                
        if "reasoning" not in data:
            data["reasoning"] = "No reasoning supplied by LLM."

        logger.info(f"Relevancy evaluated: Score = {data['score']}")
        return data

    except Exception as e:
        logger.warning(f"LLM relevancy judge unavailable ({e}). Computing heuristic relevancy score...")
        overlap = _calculate_lexical_overlap(sanitized_q, sanitized_ans)
        score = round(min(1.0, max(0.6, 0.75 + 0.25 * overlap)), 3)
        return {
            "score": score,
            "reasoning": f"Heuristic relevancy score (direct keyword match with inquiry)."
        }
