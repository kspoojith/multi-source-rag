"""
Hybrid Search Module
=====================
Combines semantic similarity (from FAISS) with keyword matching
to produce a hybrid retrieval score.

Why hybrid?
Pure semantic search fails on:
- Proper nouns: "Chandrayaan" might not be semantically close to "moon mission"
  in the embedding space, but keyword matching catches it instantly.
- Transliterated terms: "Mangalyaan" in Romanized Hindi might have weak
  embedding similarity with the English chunk about Mars mission, but
  the keyword appears literally in the text.
- Numbers and dates: "2023", "450 crore" — embeddings are poor at numbers.

Hybrid Scoring Formula:
    FinalScore = α × SemanticScore + β × KeywordBoost

Where:
- SemanticScore = cosine similarity from FAISS (0 to 1)
- KeywordBoost = fraction of query keywords found in chunk text (0 to 1)
- α = SEMANTIC_WEIGHT_ALPHA (default 0.7)
- β = KEYWORD_WEIGHT_BETA (default 0.3)

Research experiments:
- Ablation: α=1.0, β=0.0 (pure semantic) vs α=0.0, β=1.0 (pure keyword)
- Tuning: grid search over α/β values
- Metrics: Recall@5, MRR, Precision@k
"""

import re
import logging
import time
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import (
    RETRIEVAL_TOP_K,
    SEMANTIC_WEIGHT_ALPHA,
    KEYWORD_WEIGHT_BETA,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


def extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from text for keyword matching.

    We extract all words with 3+ characters (to skip Hindi particles
    like "ka", "ki" which are noise for keyword matching).

    Args:
        text: Input text (query or chunk)

    Returns:
        List of lowercased keywords
    """
    # Extract alphanumeric words (including Unicode for Hindi/Telugu)
    words = re.findall(r'[\w]+', text.lower(), re.UNICODE)
    # Filter short noise words
    return [w for w in words if len(w) >= 3]


def compute_keyword_score(query_keywords: List[str], chunk_text: str) -> float:
    """
    Compute the keyword overlap score between query and a chunk.

    Score = (number of query keywords found in chunk) / (total query keywords)

    Example:
        query_keywords = ["modi", "education"]
        chunk_text = "Modi completed his education in Gujarat..."
        → 2/2 = 1.0

    Args:
        query_keywords: Keywords extracted from the query
        chunk_text:     The text of the candidate chunk

    Returns:
        Float between 0.0 and 1.0
    """
    if not query_keywords:
        return 0.0

    chunk_lower = chunk_text.lower()
    matches = sum(1 for kw in query_keywords if kw in chunk_lower)
    return matches / len(query_keywords)


def hybrid_search(
    semantic_results: List[Dict[str, Any]],
    query: str,
    alpha: float = SEMANTIC_WEIGHT_ALPHA,
    beta: float = KEYWORD_WEIGHT_BETA,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Re-score semantic search results with keyword matching.

    This is the core hybrid retrieval function.

    Args:
        semantic_results: Results from FAISS search, each dict has:
                         - score: cosine similarity
                         - text: chunk text
                         - source, topic, etc.
        query:           The original user query (for keyword extraction)
        alpha:           Weight for semantic score (default 0.7)
        beta:            Weight for keyword boost (default 0.3)
        threshold:       Minimum final score to include (default 0.25)

    Returns:
        Re-ranked list of results with updated scores, sorted by final_score.
        Each result gets additional fields:
        - semantic_score: Original cosine similarity
        - keyword_score: Keyword overlap (0-1)
        - final_score: Weighted combination

    How it works:
    1. Extract keywords from the query
    2. For each semantic result:
       a. Keep the original semantic_score
       b. Compute keyword_score against chunk text
       c. final_score = α * semantic_score + β * keyword_score
    3. Filter by threshold
    4. Sort by final_score descending
    """
    start_time = time.time()
    query_keywords = extract_keywords(query)

    reranked = []
    for result in semantic_results:
        semantic_score = result.get("score", 0.0)
        chunk_text = result.get("text", "")

        # Compute keyword overlap
        keyword_score = compute_keyword_score(query_keywords, chunk_text)

        # Hybrid score
        final_score = alpha * semantic_score + beta * keyword_score

        if final_score >= threshold:
            reranked.append({
                **result,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "final_score": final_score,
            })

    # Sort by final score (highest first)
    reranked.sort(key=lambda x: x["final_score"], reverse=True)

    elapsed = time.time() - start_time
    logger.debug(
        f"Hybrid reranking: {len(semantic_results)} → {len(reranked)} results "
        f"(α={alpha}, β={beta}, time={elapsed*1000:.1f}ms)"
    )

    return reranked


def deduplicate_results(
    results: List[Dict[str, Any]],
    similarity_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks from results.

    Adjacent chunks from the same document often overlap significantly.
    This removes chunks that are >95% similar to a higher-ranked result.

    Research note:
    - We measure precision@k improvement from deduplication
    - Without dedup, top-5 might contain 3 overlapping chunks from
      the same paragraph, wasting context window slots

    Algorithm:
    - Compare each result against already-selected results
    - If >95% of its keywords appear in a selected result, skip it
    """
    if not results:
        return results

    selected = [results[0]]

    for result in results[1:]:
        is_duplicate = False
        result_text = result.get("text", "").lower()

        for sel in selected:
            sel_text = sel.get("text", "").lower()
            # Simple overlap check: if one text is mostly contained in another
            if len(result_text) > 0:
                # Check character overlap
                shorter = min(result_text, sel_text, key=len)
                longer = max(result_text, sel_text, key=len)
                if shorter in longer:
                    is_duplicate = True
                    break
                # Check word overlap
                result_words = set(result_text.split())
                sel_words = set(sel_text.split())
                if result_words and sel_words:
                    overlap = len(result_words & sel_words) / len(result_words)
                    if overlap > similarity_threshold:
                        is_duplicate = True
                        break

        if not is_duplicate:
            selected.append(result)

    if len(selected) < len(results):
        logger.debug(
            f"Deduplication: {len(results)} → {len(selected)} results"
        )

    return selected
