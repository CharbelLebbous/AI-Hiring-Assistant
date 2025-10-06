"""
vector_store.py — FAISS Vector Database Wrapper
------------------------------------------------
This module provides a lightweight interface around Facebook's FAISS library.

It handles:
  ✅ Creating and persisting the FAISS index
  ✅ Storing & retrieving candidate embeddings
  ✅ Maintaining metadata (name, email, file_path, etc.)
  ✅ Performing similarity search efficiently
  ✅ Saving and loading both vectors + metadata together

Each embedding corresponds 1:1 with an entry in metadata.json.
"""

import os
import json
import numpy as np
import faiss


class FaissStore:
    """
    A simple FAISS-based vector store with persistent metadata.

    Attributes:
        dim (int): Dimensionality of the embedding vectors.
        index_path (str): Path to the saved FAISS index file.
        metadata_path (str): Path to the saved JSON metadata file.
        index (faiss.Index): FAISS index for efficient vector search.
        metadata (list[dict]): List of candidate metadata aligned with embeddings.
    """

    def __init__(self, dim: int, index_path: str, metadata_path: str):
        """
        Initialize the FAISS store, loading existing data if available.

        Args:
            dim (int): Dimension of embeddings (e.g., 1024 for Cohere models).
            index_path (str): Path where the FAISS index is stored.
            metadata_path (str): Path where JSON metadata is stored.
        """
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Ensure storage directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Load existing index + metadata if found
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # Create a new FAISS index (Inner Product → for cosine similarity on normalized vectors)
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []

    # -------------------------------------------------------------------------
    # 🧩 Add vectors and metadata
    # -------------------------------------------------------------------------
    def add(self, vectors: np.ndarray, metadatas: list):
        """
        Add vectors and their corresponding metadata entries.

        Args:
            vectors (np.ndarray): Array of shape (N, dim), dtype float32.
            metadatas (list[dict]): List of N metadata dictionaries aligned to vectors.

        Raises:
            AssertionError: If vector dimension doesn't match store dimension.
        """
        assert vectors.shape[1] == self.dim, "❌ Vector dimension mismatch."
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    # -------------------------------------------------------------------------
    # 🔍 Perform a similarity search
    # -------------------------------------------------------------------------
    def search(self, vector: np.ndarray, top_k: int = 10):
        """
        Search for the top-K most similar embeddings.

        Args:
            vector (np.ndarray): A normalized query vector of shape (dim,).
            top_k (int): Number of top matches to retrieve.

        Returns:
            list[tuple[dict, float]]: Pairs of (metadata, similarity_score)
        """
        # Query must be reshaped into 2D for FAISS
        D, I = self.index.search(vector.reshape(1, -1), top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.metadata):  # Ensure index is within valid range
                results.append((self.metadata[idx], float(score)))
        return results

    # -------------------------------------------------------------------------
    # 💾 Save index and metadata
    # -------------------------------------------------------------------------
    def save(self):
        """
        Persist both the FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
