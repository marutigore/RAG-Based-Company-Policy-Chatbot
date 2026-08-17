"""
Document Viewer & Citation Highlight Module.
Retrieves full page content from uploaded policies and dynamically injects
highlight markers into cited passages for side-by-side document verification.
"""

import os
import re
import html
import logging
from typing import Dict, Any, Optional
from utils.document_loader import load_document

logger = logging.getLogger("utils.document_viewer")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploaded_policies")


def highlight_passage(full_text: str, highlight_snippet: Optional[str] = None) -> str:
    """Escapes HTML and wraps matched citation text in animated glowing mark elements."""
    escaped_full = html.escape(full_text)
    if not highlight_snippet or not highlight_snippet.strip():
        # Convert newlines to paragraphs/breaks
        return escaped_full.replace("\n", "<br>")

    # Clean query words
    snippet_words = [re.escape(w) for w in highlight_snippet.strip().split() if len(w) > 3][:8]
    if not snippet_words:
        return escaped_full.replace("\n", "<br>")

    pattern = r'(' + r'|'.join(snippet_words) + r')'
    highlighted = re.sub(
        pattern,
        r'<mark style="background:rgba(99,102,241,0.35);color:#38bdf8;padding:2px 4px;border-radius:4px;border-bottom:2px solid #818cf8;font-weight:600;">\1</mark>',
        escaped_full,
        flags=re.IGNORECASE
    )
    return highlighted.replace("\n", "<br>")


def get_document_page_preview(
    source_filename: str,
    page_number: int = 1,
    highlight_snippet: Optional[str] = None
) -> Dict[str, Any]:
    """Loads and formats a document page with highlighted citations."""
    file_path = os.path.join(UPLOAD_DIR, source_filename)
    
    if not os.path.exists(file_path):
        # Return synthetic preview if file was indexed in a different session
        sample_text = f"[Document Preview: {source_filename} - Page {page_number}]\n\n" + (
            highlight_snippet or "Detailed company policy guidelines and regulatory provisions."
        )
        return {
            "source": source_filename,
            "page": page_number,
            "total_pages": page_number,
            "raw_text": sample_text,
            "content_html": highlight_passage(sample_text, highlight_snippet),
            "found_on_disk": False
        }

    try:
        pages = load_document(file_path, custom_filename=source_filename)
        target_page = None
        for p in pages:
            if p.get("metadata", {}).get("page") == int(page_number):
                target_page = p
                break
        
        if not target_page and pages:
            target_page = pages[min(max(0, int(page_number) - 1), len(pages) - 1)]

        raw_text = target_page.get("text", "") if target_page else "Page content unavailable."
        content_html = highlight_passage(raw_text, highlight_snippet)

        return {
            "source": source_filename,
            "page": target_page.get("metadata", {}).get("page", page_number) if target_page else page_number,
            "total_pages": len(pages),
            "raw_text": raw_text,
            "content_html": content_html,
            "found_on_disk": True
        }
    except Exception as e:
        logger.error(f"Error generating preview for {source_filename} p.{page_number}: {e}")
        return {
            "source": source_filename,
            "page": page_number,
            "total_pages": 1,
            "raw_text": f"Error loading preview: {e}",
            "content_html": f"<p style='color:#f87171;'>Error loading page: {html.escape(str(e))}</p>",
            "found_on_disk": False
        }
