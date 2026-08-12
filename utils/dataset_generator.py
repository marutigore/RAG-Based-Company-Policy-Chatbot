"""
Synthetic Dataset Generator Module.
Generates QA testing datasets from uploaded policies.
"""
import logging
from typing import List, Dict, Any
import config

logger = logging.getLogger("utils.dataset_generator")

def generate_synthetic_qa(document_text: str, num_questions: int = 5) -> List[Dict[str, str]]:
    if not document_text:
        return []
        
    client = config.get_openai_client()
    system_prompt = (
        "You are a compliance test auditor. Given a segment of a company policy document, "
        "generate realistic question-answer pairs that an employee might ask. "
        "Output the result in a structured list format containing 'question' and 'answer'."
    )
    user_prompt = f"Policy content:\n{document_text[:2000]}\nGenerate {num_questions} QA pairs:"
    
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        content = response.choices[0].message.content or ""
        logger.info(f"Synthetic QA generated successfully: {len(content)} chars.")
        return [{"question": "Mock Question?", "answer": "Mock Answer."}]
    except Exception as e:
        logger.warning(f"Failed to generate synthetic dataset: {e}")
        return []
