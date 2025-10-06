"""
query_engine.py — Conversational Retrieval & AI Assistant Logic
---------------------------------------------------------------
This module powers the AI Hiring Assistant by handling:
  ✅ Retrieval of relevant resumes from the FAISS vector store
  ✅ Intelligent conversational Q&A about candidates
  ✅ Direct factual answers (email, phone, family name)
  ✅ Cohere-powered reasoning for open-ended HR questions
  ✅ Transparent citation of source files used

It combines embeddings, retrieval, and LLM reasoning into one coherent pipeline.
"""

from cohere_client import CohereClient
from vector_store import FaissStore
import numpy as np
import json
import os
import re
import faiss
from config import (
    COHERE_API_KEY,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBED_MODEL,
    COHERE_CHAT_MODEL,
    TOP_K,
)

# Initialize Cohere API client (shared instance)
co = CohereClient(COHERE_API_KEY)


# -------------------------------------------------------------------------
# 🧱 Load the FAISS Index and Metadata
# -------------------------------------------------------------------------
def load_store_and_dim():
    """
    Load the FAISS index and candidate metadata for vector search.

    Returns:
        (FaissStore, list[dict]): The loaded store and metadata list.
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("❌ Index or metadata not found. Please build the index first.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Read FAISS index to get vector dimension
    idx = faiss.read_index(FAISS_INDEX_PATH)
    d = idx.d

    # Initialize FaissStore wrapper
    store = FaissStore(dim=d, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)
    return store, metadata


# -------------------------------------------------------------------------
# 🔍 Candidate Search (Vector Retrieval)
# -------------------------------------------------------------------------
def search_candidates(query: str, top_k: int = TOP_K):
    """
    Convert the query to an embedding and search for the top-matching candidates.

    Args:
        query (str): The HR query (e.g., "Who has experience in NLP?")
        top_k (int): Number of top candidates to retrieve.

    Returns:
        list[dict]: Metadata for the top retrieved candidates.
    """
    store, metadata = load_store_and_dim()

    # Create query embedding and normalize it
    qemb = co.embed([query], model=EMBED_MODEL)[0]
    arr = np.array(qemb, dtype="float32")
    arr /= (np.linalg.norm(arr) + 1e-12)

    # Perform similarity search
    hits = store.search(arr, top_k=top_k)

    results = []
    for md, score in hits:
        md["score"] = float(score)
        results.append(md)
    return results


# -------------------------------------------------------------------------
# 💬 Main Conversational HR Assistant Logic
# -------------------------------------------------------------------------
def query_assistant(question: str, chat_history=None):
    """
    Handle HR's conversational queries with retrieval + Cohere reasoning.

    Modes of response:
      1. Direct factual lookup (email, phone, family name)
      2. LLM-generated reasoning for summaries, rankings, insights
      3. Always includes transparent list of source files used

    Args:
        question (str): The user's HR question.
        chat_history (list[dict], optional): Previous conversation turns.

    Returns:
        (str, list[dict]): The assistant’s final response and the list of unique sources.
    """
    try:
        # Retrieve relevant candidates via vector search
        candidates = search_candidates(question, top_k=5)
    except Exception as e:
        return f"⚠️ Error retrieving candidates: {e}", []

    if not candidates:
        return "No candidates found. Try building the index first.", []

    lower_q = question.lower()
    direct_answers = []
    matched_sources = []

    # ---------------------------------------------------------------------
    # 🎯 Direct factual question detection (faster, no LLM)
    # ---------------------------------------------------------------------
    for c in candidates:
        name = c.get("name", "") or c.get("file_name", "")
        if not name:
            continue

        # Check if the candidate is mentioned in the question
        if any(n in lower_q for n in name.lower().split()):
            # If HR asked for an email
            if "email" in lower_q and c.get("email"):
                direct_answers.append(f"{name}'s email address is {c['email']}.")
            # If HR asked for a phone number
            if "phone" in lower_q and c.get("phone"):
                direct_answers.append(f"{name}'s phone number is {c['phone']}.")
            # If HR asked for family name (last word of the full name)
            if "family name" in lower_q and len(name.split()) >= 2:
                direct_answers.append(f"{name}'s family name is {name.split()[-1]}.")

            matched_sources.append({
                "file_name": c.get("file_name", "Unknown"),
                "file_path": c.get("file_path", "Unknown"),
                "name": name
            })

    # If we found direct factual answers, return them immediately
    if direct_answers:
        unique_sources = {s["file_path"]: s for s in matched_sources}.values()
        return " ".join(direct_answers), list(unique_sources)

    # ---------------------------------------------------------------------
    # 🧠 Contextual question — build full candidate context for LLM
    # ---------------------------------------------------------------------
    context_blocks = []
    for c in candidates:
        block = (
            f"Name: {c.get('name') or c.get('file_name')}\n"
            f"Email: {c.get('email')}\n"
            f"Phone: {c.get('phone')}\n"
            f"Skills: {', '.join(c.get('skills', []))}\n"
            f"Summary: {c.get('summary')}\n"
            f"File: {c.get('file_path')}\n"
        )
        context_blocks.append(block)

    context = "\n---\n".join(context_blocks)

    # ---------------------------------------------------------------------
    # 🧩 System Prompt — defines the assistant’s persona & behavior
    # ---------------------------------------------------------------------
    system_prompt = (
        "You are an AI Hiring Assistant supporting an HR professional. "
        "You have access to structured resume data (names, emails, phones, skills, summaries). "
        "Use that data responsibly to answer HR questions precisely. "
        "If HR asks for ranking, provide it. "
        "If they ask for contact info or factual data, answer clearly and professionally. "
        "If the question is analytical or open-ended, respond naturally with context.\n\n"
        "Candidate Database Context:\n" + context
    )

    # ---------------------------------------------------------------------
    # 💬 Combine chat history with the current question
    # ---------------------------------------------------------------------
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    # ---------------------------------------------------------------------
    # ⚙️ Generate contextual answer using Cohere LLM
    # ---------------------------------------------------------------------
    answer = co.chat_text(messages=messages, model=COHERE_CHAT_MODEL)

    # ---------------------------------------------------------------------
    # 📁 Deduplicate and clean source file list
    # ---------------------------------------------------------------------
    seen = set()
    unique_sources = []
    for c in candidates:
        src = c.get("file_path", "Unknown")
        if src not in seen:
            seen.add(src)
            unique_sources.append({
                "file_name": c.get("file_name", "Unknown"),
                "file_path": src,
                "name": c.get("name")
            })

    # Return clean response and source citations
    return answer.strip(), unique_sources
