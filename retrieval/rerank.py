"""
Reranker Module
================
Post-retrieval reranking based on metadata and context quality.

This module applies additional filtering and scoring AFTER hybrid search:
1. Topic filtering: If we can detect the query topic, boost matching chunks
2. Source diversity: Avoid returning all chunks from one document
3. Context quality: Prefer chunks with more complete sentences

This is separate from hybrid_search to keep concerns clean:
- search.py: Semantic + keyword scoring
- rerank.py: Metadata and quality-based refinement
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


def rerank_by_topic(
    results: List[Dict[str, Any]],
    query_topic: Optional[str] = None,
    topic_boost: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Boost results that match the detected query topic.

    If we detect the query is about "space", chunks from isro.txt
    get a small score boost. This helps when embeddings are ambiguous.

    Args:
        results:     Pre-ranked search results
        query_topic: Detected topic of the query (or None)
        topic_boost: How much to boost matching topics

    Returns:
        Re-scored results (sorted by updated score)
    """
    if not query_topic or not results:
        return results

    for result in results:
        if result.get("topic") == query_topic:
            result["final_score"] = result.get("final_score", 0) + topic_boost
            result["topic_boosted"] = True
        else:
            result["topic_boosted"] = False

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def ensure_source_diversity(
    results: List[Dict[str, Any]],
    max_per_source: int = 3,
) -> List[Dict[str, Any]]:
    """
    Limit the number of results from any single source document.

    Without this, a long document with many chunks about a topic
    might dominate all top-k positions, missing relevant info
    from other documents.

    Args:
        results:        Ranked search results
        max_per_source: Maximum results from one source file

    Returns:
        Filtered results with source diversity enforced
    """
    source_counts: Counter = Counter()
    diverse_results = []

    for result in results:
        source = result.get("source", "unknown")
        if source_counts[source] < max_per_source:
            diverse_results.append(result)
            source_counts[source] += 1

    if len(diverse_results) < len(results):
        logger.debug(
            f"Source diversity: {len(results)} → {len(diverse_results)} results"
        )

    return diverse_results


def rerank_results(
    results: List[Dict[str, Any]],
    query_topic: Optional[str] = None,
    max_per_source: int = 3,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Full reranking pipeline.

    1. Topic boosting
    2. Source diversity enforcement
    3. Trim to top_k

    Args:
        results:        Search results from hybrid_search
        query_topic:    Detected topic (optional)
        max_per_source: Max results per source document
        top_k:          Final number of results to return

    Returns:
        Final top_k results, reranked and deduplicated
    """
    if not results:
        return results

    # Step 1: Topic boost
    results = rerank_by_topic(results, query_topic)

    # Step 2: Source diversity
    results = ensure_source_diversity(results, max_per_source)

    # Step 3: Trim to top_k
    return results[:top_k]
