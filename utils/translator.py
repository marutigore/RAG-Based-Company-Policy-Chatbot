"""
Multi-Language Detection and Response Localization Module.
Provides zero-dependency language identification and multilingual response directives
enabling global employees to query English corporate policies in their native language.
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("utils.translator")

SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English", "flag": "🇺🇸"},
    {"code": "es", "name": "Spanish (Español)", "flag": "🇪🇸"},
    {"code": "fr", "name": "French (Français)", "flag": "🇫🇷"},
    {"code": "de", "name": "German (Deutsch)", "flag": "🇩🇪"},
    {"code": "hi", "name": "Hindi (हिन्दी)", "flag": "🇮🇳"},
    {"code": "ja", "name": "Japanese (日本語)", "flag": "🇯🇵"},
    {"code": "zh", "name": "Chinese (中文)", "flag": "🇨🇳"},
    {"code": "pt", "name": "Portuguese (Português)", "flag": "🇧🇷"},
    {"code": "ar", "name": "Arabic (العربية)", "flag": "🇸🇦"}
]

# Lexical trigger indicators for high-precision detection
LANGUAGE_TRIGGERS = {
    "es": ["cuántos", "cuál", "cómo", "política", "vacaciones", "empleado", "permiso", "empresa", "días", "trabajo", "seguridad", "horario"],
    "fr": ["combien", "quel", "quelle", "politique", "congés", "vacances", "employé", "travail", "sécurité", "horaires", "jours", "règlement"],
    "de": ["wie", "was", "urlaub", "richtlinie", "mitarbeiter", "arbeitszeit", "tage", "unternehmen", "sicherheit", "krankheit"],
    "hi": ["छुट्टी", "अवकाश", "नीति", "कर्मचारी", "काम", "समय", "कंपनी", "नियम", "कितने", "वेतन"],
    "ja": ["休暇", "方針", "社員", "勤務", "時間", "会社", "規程", "有給", "日数", "セキュリティ"],
    "zh": ["假期", "请假", "员工", "政策", "公司", "工作时间", "规定", "薪资", "福利"],
    "pt": ["quantos", "qual", "como", "política", "férias", "funcionário", "trabalho", "segurança", "licença", "horário"],
    "ar": ["إجازة", "سياسة", "موظف", "عمل", "ساعات", "أيام", "شركة", "أمان", "راتب"]
}


def detect_language(text: str) -> Dict[str, Any]:
    """
    Detects natural language of the query using script Unicode ranges and lexical markers.
    """
    if not text or not text.strip():
        return {"code": "en", "name": "English", "flag": "🇺🇸"}

    t_lower = text.lower()

    # 1. Unicode script detection
    if re.search(r'[\u0900-\u097F]', text):
        return {"code": "hi", "name": "Hindi (हिन्दी)", "flag": "🇮🇳"}
    if re.search(r'[\u3040-\u30FF\u31F0-\u31FF]', text):
        return {"code": "ja", "name": "Japanese (日本語)", "flag": "🇯🇵"}
    if re.search(r'[\u4E00-\u9FFF]', text):
        return {"code": "zh", "name": "Chinese (中文)", "flag": "🇨🇳"}
    if re.search(r'[\u0600-\u06FF]', text):
        return {"code": "ar", "name": "Arabic (العربية)", "flag": "🇸🇦"}

    # 2. Latin lexical marker matching
    scores = {"es": 0, "fr": 0, "de": 0, "pt": 0}
    words = re.findall(r'\b\w+\b', t_lower)
    
    for lang, markers in LANGUAGE_TRIGGERS.items():
        if lang in scores:
            for word in words:
                if word in markers:
                    scores[lang] += 1

    top_lang = max(scores, key=scores.get)
    if scores[top_lang] > 0:
        names = {
            "es": ("Spanish (Español)", "🇪🇸"),
            "fr": ("French (Français)", "🇫🇷"),
            "de": ("German (Deutsch)", "🇩🇪"),
            "pt": ("Portuguese (Português)", "🇧🇷")
        }
        return {"code": top_lang, "name": names[top_lang][0], "flag": names[top_lang][1]}

    return {"code": "en", "name": "English", "flag": "🇺🇸"}


def get_supported_languages() -> List[Dict[str, Any]]:
    """Returns list of supported portal languages."""
    return SUPPORTED_LANGUAGES


def build_multilingual_system_prompt(detected_lang: Dict[str, Any]) -> str:
    """Builds prompt directive ensuring answers are returned in employee's native language."""
    base_prompt = (
        "You are an expert corporate policy assistant. Your goal is to answer the employee's question "
        "using ONLY the provided policy excerpts. If the information is not present in the excerpts, "
        "state that you cannot find the answer in the current policy documents. Do not hallucinate.\n\n"
        "At the end of your response, list the citations matching the Excerpt bracket numbers (e.g. [1], [2])."
    )
    
    if detected_lang["code"] != "en":
        base_prompt += (
            f"\n\nCRITICAL LOCALIZATION REQUIREMENT: The user is asking in {detected_lang['name']}. "
            f"You MUST answer the question accurately and fluently in {detected_lang['name']}, while keeping excerpt citation numbers [1], [2] intact."
        )
    return base_prompt
