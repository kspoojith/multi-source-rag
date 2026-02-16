"""
FAISS Indexer
==============
Builds, saves, and loads the vector index for similarity search.

Why FAISS?
- Written in C++ with Python bindings — extremely fast
- IndexFlatIP on normalized vectors = exact cosine similarity
- Persistent: save to disk, reload without re-embedding
- For <50k vectors, brute-force (FlatIP) is sufficient
  (search takes ~1ms for 10k vectors)

Architecture:
- FAISS stores only the vectors (float32 arrays)
- Metadata (chunk text, source, topic) is stored in a separate
  pickle file, keyed by integer position
- Both files must be in sync: metadata[i] corresponds to vector[i]

Research experiments enabled:
- Compare IndexFlatIP vs IndexIVFFlat for larger corpora
- Measure indexing time and search latency
- Evaluate the impact of deduplication on retrieval precision
"""

import pickle
import logging
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_DIMENSION,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """
    Manages the FAISS vector index and associated metadata.

    Workflow:
    1. build_index(vectors, metadata) — create index from embeddings
    2. save() — persist to disk
    3. load() — restore from disk
    4. search(query_vector, top_k) — find similar chunks
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = EMBEDDING_DIMENSION

    def build_index(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Build a FAISS index from pre-computed embedding vectors.

        Args:
            vectors:  np.ndarray of shape (n_chunks, 384), L2-normalized
            metadata: List of dicts, one per vector, containing:
                      - text: the chunk text
                      - source: source filename
                      - topic: detected topic
                      - chunk_id: unique chunk identifier

        How it works:
        1. Create an IndexFlatIP (Inner Product) index
        2. Add all vectors in one batch (very fast)
        3. Store metadata list in sync with vector positions

        IndexFlatIP computes dot product. Since our vectors are L2-normalized,
        dot product = cosine similarity. This avoids the overhead of
        IndexFlatL2 + post-processing.
        """
        n_vectors, dim = vectors.shape
        assert dim == self.dimension, f"Expected dim={self.dimension}, got {dim}"
        assert len(metadata) == n_vectors, "Vectors and metadata must match"

        start_time = time.time()

        # Create brute-force inner-product index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.metadata = metadata

        elapsed = time.time() - start_time
        logger.info(
            f"FAISS index built: {n_vectors} vectors, dim={dim}, "
            f"time={elapsed:.3f}s"
        )

    def save(
        self,
        index_path: str | Path = FAISS_INDEX_PATH,
        metadata_path: str | Path = FAISS_METADATA_PATH,
    ) -> None:
        """
        Save the FAISS index and metadata to disk.

        Two files are created:
        - faiss.index: The binary FAISS index (vectors only)
        - metadata.pkl: Python pickle of the metadata list
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        # Ensure directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(
            f"Index saved: {index_path} ({self.index.ntotal} vectors), "
            f"metadata: {metadata_path}"
        )

    def load(
        self,
        index_path: str | Path = FAISS_INDEX_PATH,
        metadata_path: str | Path = FAISS_METADATA_PATH,
    ) -> bool:
        """
        Load a previously saved FAISS index and metadata.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Index files not found at {index_path}")
            return False

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(
            f"Index loaded: {self.index.ntotal} vectors from {index_path}"
        )
        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Search the index for the most similar vectors to the query.

        Args:
            query_vector: 1D array of shape (384,), L2-normalized
            top_k:        Number of results to return

        Returns:
            List of dicts, each containing:
            - score: cosine similarity (0 to 1)
            - text: the chunk text
            - source: source filename
            - All other metadata fields

        How FAISS search works:
        1. Computes dot product between query and ALL stored vectors
        2. Returns the top_k highest-scoring indices and scores
        3. We map indices back to metadata
        """
        if self.index is None:
            raise ValueError("No index loaded. Call build_index() or load().")

        # FAISS expects a 2D array: (n_queries, dim)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        start_time = time.time()
        scores, indices = self.index.search(query_vector, top_k)
        search_time = time.time() - start_time

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            result = {
                **self.metadata[idx],
                "score": float(score),
            }
            results.append(result)

        logger.debug(
            f"Search completed: {len(results)} results in {search_time*1000:.1f}ms"
        )
        return results

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def is_loaded(self) -> bool:
        """Check if the index is ready for search."""
        return self.index is not None and self.index.ntotal > 0
