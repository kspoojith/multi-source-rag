"""
Embedding Module
=================
Converts text chunks into dense vector representations using a
multilingual Sentence-Transformer model.

Model: paraphrase-multilingual-MiniLM-L12-v2
- 384-dimensional output vectors
- Supports 50+ languages including Hindi, Telugu, Tamil, Bengali
- Trained on parallel multilingual paraphrase data
- ~118M parameters — small enough for fast CPU inference

Why this model?
- LaBSE (Google): Better cross-lingual alignment but 2x slower
- XLM-RoBERTa: Larger (560M params), too slow for CPU
- MiniLM-L12-v2: Best latency-to-quality ratio for our use case

Research focus:
- We normalize vectors to unit length for cosine similarity via FAISS FlatIP
- We measure embedding quality across:
  (a) Pure English queries
  (b) Pure Hindi queries
  (c) Code-mixed queries (Hindi-English romanized)
- The "embedding degradation" on code-mixed queries is a key finding
"""

import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for consistent embedding generation.

    Usage:
        model = EmbeddingModel()
        vectors = model.embed(["Hello world", "नमस्ते दुनिया"])
        # vectors.shape = (2, 384)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Load the model on initialization.
        The model is ~500MB and downloads on first use.
        Subsequent loads are from cache (~/.cache/torch/sentence_transformers/).
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION
        logger.info(
            f"Model loaded. Dimension: {self.dimension}, "
            f"Max seq length: {self.model.max_seq_length}"
        )

    def embed(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        Args:
            texts:     A single string or list of strings
            normalize: If True, L2-normalize vectors (required for cosine sim)

        Returns:
            np.ndarray of shape (n_texts, 384) with float32 values

        How it works:
        1. The model tokenizes the text using a multilingual WordPiece tokenizer
        2. Passes tokens through 12 transformer layers
        3. Mean-pools the output to get a fixed-size vector
        4. If normalize=True, we L2-normalize so dot product = cosine similarity
        """
        if isinstance(texts, str):
            texts = [texts]

        # batch encoding for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,          # Process 32 texts at a time
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,  # L2 normalize for cosine sim
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query string. Returns a 1D vector of shape (384,).

        Separated from embed() for clarity — queries are always single strings
        and we may want different processing (e.g., query expansion) later.
        """
        return self.embed(query, normalize=normalize)[0]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        Useful for research experiments comparing embedding quality.

        Example:
            model.similarity("What is Modi's education?",
                            "Modi ji ka education kya hai?")
            # Returns ~0.75 (good cross-lingual alignment)
        """
        vec1 = self.embed_query(text1)
        vec2 = self.embed_query(text2)
        return float(np.dot(vec1, vec2))


# ─── Module-level convenience ────────────────────────────────────────────────
# Singleton pattern: reuse the same model across the application
_model_instance = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the singleton embedding model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel()
    return _model_instance
