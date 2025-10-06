"""
ingestion.py — Resume Parsing & Candidate Profile Builder
----------------------------------------------------------
This module handles:
  ✅ Reading and extracting text from various document types (PDF, DOCX, TXT)
  ✅ Running OCR on scanned PDF pages
  ✅ Extracting basic metadata (name, email, phone)
  ✅ Generating candidate summaries and skill lists using Cohere LLM
  ✅ Preparing unified content for vector embeddings

It converts raw resumes into structured candidate profiles ready for indexing.
"""

import re
import fitz  # PyMuPDF for PDF text extraction
from pathlib import Path
from pdf2image import convert_from_path  # For OCR fallback
import pytesseract  # OCR engine
import docx2txt  # For reading Word documents
from cohere_client import CohereClient
from config import COHERE_CHAT_MODEL

# -------------------------------------------------------------------------
# 🔍 Regular Expressions — for email and phone extraction
# -------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d[\d \-\(\)]{7,}\d)")

# -------------------------------------------------------------------------
# 📄 Text Extraction from Different File Types
# -------------------------------------------------------------------------
def read_text_from_file(file_path: str) -> str:
    """
    Extract text from a file, supporting PDF, DOCX, and TXT formats.
    Uses OCR fallback for scanned PDFs with no extractable text.
    
    Args:
        file_path (str): Path to the resume file.

    Returns:
        str: Extracted text content.
    """
    ext = Path(file_path).suffix.lower()
    text = ""
    try:
        # Handle PDF files
        if ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                t = page.get_text().strip()

                # If no text found, use OCR as fallback
                if not t:
                    images = convert_from_path(file_path)
                    for img in images:
                        t += pytesseract.image_to_string(img)

                text += "\n" + t

        # Handle Word documents
        elif ext in [".docx", ".doc"]:
            text = docx2txt.process(file_path)

        # Handle plain text files
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")

    return text.strip()


# -------------------------------------------------------------------------
# 🧩 Extract Basic Candidate Information
# -------------------------------------------------------------------------
def extract_basic_fields(text: str):
    """
    Extract simple structured data such as email, phone, and a name guess.

    Heuristics:
      - The first short line with at least two words (and no 'CV'/'Resume') is likely the candidate name.

    Args:
        text (str): The raw resume text.

    Returns:
        dict: {email, phone, name_guess}
    """
    # Extract email and phone
    email = (EMAIL_RE.search(text) or [None]).group(0) if EMAIL_RE.search(text) else None
    phone = (PHONE_RE.search(text) or [None]).group(0) if PHONE_RE.search(text) else None

    # Guess candidate name from top lines
    candidate_name = None
    for line in text.splitlines():
        l = line.strip()
        if len(l.split()) >= 2 and len(l) < 80 and not any(k.lower() in l.lower() for k in ("resume", "cv", "curriculum")):
            candidate_name = l
            break

    return {"email": email, "phone": phone, "name_guess": candidate_name}


# -------------------------------------------------------------------------
# 🤖 Build Candidate Profile using Cohere
# -------------------------------------------------------------------------
def build_candidate_profile(file_path: str, cohere: CohereClient, model_embed: str, model_chat: str):
    """
    Build a structured candidate profile ready for vector indexing.

    Steps:
      1. Read and clean the resume text.
      2. Extract email, phone, and guessed name.
      3. Ask Cohere to extract technical skills.
      4. Ask Cohere for a short professional summary.
      5. Combine all into a unified embedding text block.

    Args:
        file_path (str): Path to the resume file.
        cohere (CohereClient): Wrapper for Cohere API.
        model_embed (str): Cohere embedding model name.
        model_chat (str): Cohere chat model for NLP tasks.

    Returns:
        dict: A full structured candidate profile ready to be indexed.
    """
    # Step 1: Extract full text
    text = read_text_from_file(file_path)

    # Step 2: Extract basic fields
    basic = extract_basic_fields(text)
    short_text = text[:6000]  # Limit for prompt token efficiency

    # Step 3: Generate skills & summary with LLM
    skills_prompt = "Extract a comma-separated list of technical skills and job-relevant keywords from the text."
    summary_prompt = "Give a concise (2-3 sentence) professional summary of this candidate."

    skills_text = cohere.extract_with_prompt(short_text, skills_prompt, model_chat)
    skills = [s.strip() for s in re.split(r"[,;\n]", skills_text) if s.strip()]
    summary = cohere.extract_with_prompt(short_text, summary_prompt, model_chat)

    # Step 4: Combine key info into a single text for embedding
    embed_text = "\n".join([
        basic.get("name_guess") or "",
        basic.get("email") or "",
        "Skills: " + ", ".join(skills),
        "Summary: " + (summary or ""),
        "FullResumeText: " + text[:20000]
    ])

    # Step 5: Return structured profile
    return {
        "file_path": str(file_path),
        "file_name": Path(file_path).name,
        "name": basic.get("name_guess"),
        "email": basic.get("email"),
        "phone": basic.get("phone"),
        "skills": skills,
        "summary": summary,
        "content_for_embedding": embed_text,  # Used for vector search
        "raw_text": text  # Retained for debugging or fine-tuning
    }
