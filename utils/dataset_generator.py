"""
Synthetic Dataset Generator Module.
Generates QA testing datasets from uploaded policies.
"""
import logging
from typing import List, Dict, Any
import config

logger = logging.getLogger("utils.dataset_generator")

def generate_synthetic_qa(document_text: str, num_questions: int = 5) -> List[Dict[str, str]]:
    if not document_text or not document_text.strip():
        return []

    client = config.get_openai_client()
    system_prompt = (
        "You are a compliance test auditor. Given a segment of a company policy document, "
        "generate realistic question-answer pairs that an employee might ask.\n"
        "Return a JSON array of objects, where each object has 'question' and 'answer' keys."
    )
    user_prompt = f"Policy content:\n{document_text[:2500]}\nGenerate {num_questions} QA pairs in valid JSON array format:"
    
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content or ""
        import json
        data = json.loads(content)
        if isinstance(data, list):
            return data[:num_questions]
        if isinstance(data, dict):
            for k in ["qa_pairs", "questions", "items", "data"]:
                if k in data and isinstance(data[k], list):
                    return data[k][:num_questions]
    except Exception as e:
        logger.warning(f"LLM synthetic QA generation unavailable ({e}). Generating extractive QA pairs...")

    # Heuristic extractive QA generator fallback
    sentences = [s.strip() for s in document_text.replace("\n", " ").split(".") if len(s.strip()) > 25]
    qa_list = []
    for idx, sentence in enumerate(sentences[:num_questions]):
        words = sentence.split()
        subject = " ".join(words[:4])
        qa_list.append({
            "question": f"What is the corporate policy regarding {subject}?",
            "answer": f"According to the policy: {sentence}."
        })
        
    return qa_list if qa_list else [
        {"question": "What is the policy guideline?", "answer": document_text[:200]}
    ]
