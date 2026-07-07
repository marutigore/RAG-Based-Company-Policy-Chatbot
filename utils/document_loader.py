"""
Document Loader Module.
Extracts clean text and metadata from PDF files using PyMuPDF (fitz).
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

# Initialize module logger
logger = logging.getLogger("utils.document_loader")


def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace, resolving line wraps,
    and removing non-printable control characters.

    Args:
        text (str): Raw text extracted from the document page.

    Returns:
        str: Cleaned and normalized text.

    Example:
        >>> cleaned = clean_text("Hello-\\nworld   from PyMuPDF.")
        >>> print(cleaned)
        "Hello-world from PyMuPDF."
    """
    # Replace carriage returns with newlines
    cleaned = text.replace("\r", "\n")
    
    # Normalize unicode characters to avoid ASCII encode failures and match queries better
    cleaned = cleaned.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    
    # Resolve hyphenated line wraps (e.g. "multi-\nfunctional" to "multi-functional")
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)
    
    # Replace double or multiple newlines with single newlines
    cleaned = re.sub(r"\n+", "\n", cleaned)
    
    # Replace multiple spaces with a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    # Strip leading/trailing whitespaces
    return cleaned.strip()


def load_pdf(file_path: str, custom_filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads a PDF document page-by-page and extracts text and metadata.

    Args:
        file_path (str): Absolute or relative path to the PDF file.

    Returns:
        List[Dict[str, Any]]: List of pages, where each page is represented as a dictionary
                              with 'text' and 'metadata' keys.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the file is not a valid PDF or is corrupted.

    Example:
        >>> pages = load_pdf("C:/docs/policy.pdf")
        >>> print(pages[0]['metadata'])
        {'source': 'policy.pdf', 'page': 1}
    """
    # Check if the file actually exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Validate that it has a PDF extension
    if not file_path.lower().endswith(".pdf"):
        logger.error(f"Invalid file extension: {file_path}")
        raise ValueError("Unsupported file format. Only PDF files are supported.")

    pages_data: List[Dict[str, Any]] = []
    raw_filename = custom_filename or os.path.basename(file_path)
    filename = raw_filename.replace('\u2011', '-').encode('ascii', 'replace').decode('ascii').replace('?', '_')

    try:
        logger.info(f"Opening PDF document: {file_path}")
        # Open document with fitz
        doc = fitz.open(file_path)
        
        # Iterate page by page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract plain text
            raw_text = page.get_text()
            
            # Run text cleaning
            cleaned_page_text = clean_text(raw_text)
            
            # If the page is empty, try to extract text using Vision-based OCR fallback (gpt-4o-mini)
            if not cleaned_page_text:
                import config
                if config.OPENAI_API_KEY:
                    try:
                        logger.warning(f"Page {page_num + 1} is empty or image-only. Attempting Vision-based OCR fallback...")
                        import base64
                        import openai
                        
                        # Render page to PNG bytes
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        base64_image = base64.b64encode(img_bytes).decode('utf-8')
                        
                        # Send image to the configured model
                        client = config.get_openai_client()
                        response = client.chat.completions.create(
                            model=config.LLM_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Extract all readable text, tables, or diagram details from this document page image. Output only the raw text content without markdown codeblocks, notes, or explanations."},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_tokens=1000
                        )
                        ocr_text = response.choices[0].message.content or ""
                        cleaned_page_text = clean_text(ocr_text)
                        if cleaned_page_text:
                            logger.info(f"Successfully extracted {len(cleaned_page_text)} characters via Vision OCR on page {page_num + 1}")
                    except Exception as ocr_err:
                        logger.error(f"Vision OCR fallback failed for page {page_num + 1}: {ocr_err}")
                
                # If it's still empty, skip it
                if not cleaned_page_text:
                    logger.warning(f"Skipping empty or image-only page {page_num + 1} in {filename}")
                    continue
            
            # Create page dictionary with metadata
            page_info = {
                "text": cleaned_page_text,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,  # Store as 1-indexed page number
                    "total_pages": len(doc)
                }
            }
            pages_data.append(page_info)
            logger.debug(f"Loaded page {page_num + 1} from {filename}")
            
        logger.info(f"Successfully loaded {len(pages_data)} pages from {filename}")
        doc.close()
        
    except fitz.FileDataError as e:
        logger.error(f"PyMuPDF error reading {file_path}: {e}")
        raise ValueError(f"Corrupted or invalid PDF file: {file_path}. Details: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        raise Exception(f"Failed to read PDF: {e}")

    return pages_data
