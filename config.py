"""
config.py — Central Configuration Module
----------------------------------------
This file manages all environment-based configurations for the 
AI Hiring Assistant project.

It loads API keys, model names, and paths dynamically from a `.env` file, 
allowing clean separation between code and secrets.

All key project parameters (e.g., models, FAISS paths, and top-k settings)
are centralized here for easy updates.
"""

from dotenv import load_dotenv
import os

# -------------------------------------------------------------------------
# 🔹 Load Environment Variables
# -------------------------------------------------------------------------
# Loads all variables from a `.env` file (if present in project root)
# Example `.env` file content:
# COHERE_API_KEY=your_api_key_here
# STORAGE_DIR=./storage
# -------------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------------
# 🔑 API KEYS
# -------------------------------------------------------------------------
# Main Cohere API key — used for both embeddings and chat completions.
# If missing, the system will fallback to a placeholder string.
# -------------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "YOUR_COHERE_API_KEY_HERE")

# -------------------------------------------------------------------------
# 🧠 Model Configuration
# -------------------------------------------------------------------------
# Embedding and chat model names (default values are stable Cohere releases).
# You can override them in your `.env` file to test new versions.
# -------------------------------------------------------------------------
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
COHERE_CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")

# -------------------------------------------------------------------------
# 💾 Storage Paths
# -------------------------------------------------------------------------
# Defines where the FAISS index and candidate metadata are stored locally.
# These files are created after the user builds the index from resumes.
# -------------------------------------------------------------------------
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
METADATA_PATH = os.path.join(STORAGE_DIR, "metadata.json")

# -------------------------------------------------------------------------
# ⚙️ General Parameters
# -------------------------------------------------------------------------
# TOP_K controls how many top candidates are retrieved per query.
# -------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "10"))
