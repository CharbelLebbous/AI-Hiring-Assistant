"""
index_builder.py — Resume Index Construction Pipeline
-----------------------------------------------------
This module builds the FAISS vector index used by the AI Hiring Assistant.

It:
- Reads and parses all candidate resumes from a folder
- Extracts key information (name, email, phone, skills, summary)
- Generates embeddings via Cohere
- Stores vectors + metadata using FAISS for fast retrieval

The index can later be searched to rank and query candidates interactively.
"""

import os
import numpy as np
from pathlib import Path
from cohere_client import CohereClient
from vector_store import FaissStore
from ingestion import build_candidate_profile
from config import (
    COHERE_CHAT_MODEL,
    COHERE_API_KEY,
    EMBED_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)

# -------------------------------------------------------------------------
# 🔹 Helper Function — Flatten Nested Embeddings
# -------------------------------------------------------------------------
def flatten_embedding(e):
    """
    Recursively flatten any nested embedding into a 1D list of floats.
    This ensures compatibility with FAISS (which requires fixed-length vectors).
    """
    if isinstance(e, (float, int)):
        return [float(e)]
    elif isinstance(e, list):
        flattened = []
        for sub in e:
            flattened.extend(flatten_embedding(sub))
        return flattened
    else:
        return []


# -------------------------------------------------------------------------
# 🧠 Main Function — Build Vector Index from Folder
# -------------------------------------------------------------------------
def build_index_from_folder(folder: str):
    """
    Build the FAISS index from all resumes inside the specified folder.
    
    Steps:
      1. Test Cohere embeddings to detect dimensionality
      2. Read each file and extract a structured candidate profile
      3. Embed each resume’s text content
      4. Normalize and store embeddings with aligned metadata
    
    Args:
        folder (str): Path to the folder containing candidate resumes (PDF, DOCX, TXT)
    """
    # Initialize Cohere client
    co = CohereClient(COHERE_API_KEY)

    # Step 1: Detect embedding dimension
    print("🔹 Testing Cohere embedding endpoint...")
    sample = co.embed(["hello world"], model=EMBED_MODEL)
    if not sample or not sample[0]:
        raise ValueError("❌ Embedding model returned empty sample vector.")

    dim = len(sample[0])
    print(f"✅ Embedding dimension detected: {dim}")

    # Step 2: Initialize FAISS storage
    store = FaissStore(dim=dim, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)
    profiles = []

    # Step 3: Parse all resume files in folder
    for path in Path(folder).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".pdf", ".docx", ".txt", ".doc"):
            continue  # Skip unsupported file formats

        profile = build_candidate_profile(
            str(path),
            cohere=co,
            model_embed=EMBED_MODEL,
            model_chat=COHERE_CHAT_MODEL,
        )
        profiles.append(profile)

    if not profiles:
        print("⚠️ No candidate files found in folder.")
        return

    # Step 4: Batch-embed text content
    texts = [p["content_for_embedding"] for p in profiles]
    batch_size = 16  # You can increase this if you have a high API quota
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = co.embed(batch, model=EMBED_MODEL)

        # Validate embeddings
        clean_embs = []
        for e in embs:
            flat = flatten_embedding(e)
            if len(flat) == dim:
                clean_embs.append(flat)
            else:
                print(f"⚠️ Skipping malformed embedding of length {len(flat)} != {dim}")

        if not clean_embs:
            print(f"⚠️ No valid embeddings in batch {i // batch_size}")
            continue

        # Convert and normalize
        try:
            arr = np.array(clean_embs, dtype=np.float32)
        except Exception as err:
            print(f"❌ Shape mismatch in batch {i // batch_size}: {err}")
            continue

        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms  # Normalize for cosine similarity
        vectors.append(arr)

    # Step 5: Finalize vector matrix
    if not vectors:
        print("⚠️ No embeddings created — index not saved.")
        return

    all_vecs = np.vstack(vectors).astype("float32")

    # Step 6: Prepare metadata (aligned with embeddings)
    metadatas = []
    for p in profiles:
        md = {
            "file_name": p["file_name"],
            "file_path": p["file_path"],
            "name": p["name"],
            "email": p["email"],
            "phone": p["phone"],
            "skills": p["skills"],
            "summary": p["summary"],
        }
        metadatas.append(md)

    # Step 7: Add to FAISS index and save
    store.add(all_vecs, metadatas)
    store.save()

    print(f"✅ Index built and saved successfully! Total candidates indexed: {len(profiles)}")
