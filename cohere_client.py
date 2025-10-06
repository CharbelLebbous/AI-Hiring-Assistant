"""
cohere_client.py — Cohere API Wrapper
-------------------------------------
This module defines a custom wrapper class around Cohere’s Python SDK (v2),
simplifying how embeddings and chat completions are called and parsed.

Functions include:
- Generating text embeddings
- Conversational chat interaction
- Prompt-based text extraction (skills, summaries, etc.)

It ensures robust parsing of Cohere responses and uniform handling of edge cases.
"""

import cohere
from typing import List, Dict


class CohereClient:
    """
    Lightweight wrapper around the Cohere V2 client.

    This class abstracts away Cohere's raw API responses, providing:
    - Clean list-based embeddings output
    - Simplified chat interface
    - A convenient helper for prompt-based extraction
    """

    def __init__(self, api_key: str):
        """
        Initialize the Cohere client using the provided API key.
        """
        self.client = cohere.ClientV2(api_key)

    # ----------------------------------------------------------------------
    # 🔹 Embedding Utility
    # ----------------------------------------------------------------------
    def embed(self, texts: List[str], model: str, input_type: str = "search_document") -> List[List[float]]:
        """
        Create vector embeddings for a list of text inputs.

        Args:
            texts (List[str]): Input strings to embed.
            model (str): Cohere embedding model name (e.g. "embed-english-v3.0").
            input_type (str): Embedding purpose (default = "search_document").

        Returns:
            List[List[float]]: List of embedding vectors (one per text).

        Notes:
            - Handles Cohere’s new `EmbedByTypeResponse` output.
            - Converts nested embeddings into clean float lists.
        """
        resp = self.client.embed(
            texts=texts,
            model=model,
            input_type=input_type
        )

        # ✅ Access float-based embeddings from the response
        embeddings = getattr(resp.embeddings, "float_", None)

        if not embeddings or not isinstance(embeddings, list):
            raise ValueError("❌ No valid embeddings found in response.")

        # Convert to Python float lists and handle malformed entries
        processed = []
        for e in embeddings:
            if e and isinstance(e, (list, tuple)):
                processed.append([float(x) for x in e])
            else:
                print("⚠️ Skipped invalid embedding:", e)

        if not processed:
            raise ValueError("❌ No valid embeddings found in response.")
        return processed

    # ----------------------------------------------------------------------
    # 🔹 Chat Utility
    # ----------------------------------------------------------------------
    def chat_text(self, messages: List[Dict[str, str]], model: str) -> str:
        """
        Send a list of chat messages to Cohere’s chat endpoint.

        Args:
            messages (List[Dict[str, str]]): Chat history including user/system turns.
            model (str): The Cohere chat model name (e.g. "command-a-03-2025").

        Returns:
            str: The concatenated text response from the model.
        """
        resp = self.client.chat(model=model, messages=messages)

        parts = []
        try:
            # Cohere V2 returns structured message parts (objects or dicts)
            for item in resp.message.content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
        except Exception as e:
            print(f"⚠️ Chat parse error: {e}")

        # Join all text parts into a single output string
        return " ".join(parts).strip() or "(no response)"

    # ----------------------------------------------------------------------
    # 🔹 Prompt-Based Extraction Utility
    # ----------------------------------------------------------------------
    def extract_with_prompt(self, text: str, prompt: str, model: str):
        """
        Quick one-shot extraction helper using a simple user prompt.

        Typically used for tasks such as:
        - Extracting skills from a resume
        - Summarizing candidate experience
        - Generating short insights

        Args:
            text (str): Input text content (e.g. a candidate CV segment).
            prompt (str): Instruction prompt for the model.
            model (str): Chat model to use for generation.

        Returns:
            str: Model-generated text output.
        """
        messages = [{"role": "user", "content": f"{prompt}\n\n{text}"}]
        return self.chat_text(messages=messages, model=model)
