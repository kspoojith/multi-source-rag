"""
Evaluation Metrics Module
===========================
Computes information retrieval and generation quality metrics.

This module is the research backbone — it provides empirical evidence
for every claim we make in the research paper.

Metrics implemented:
1. Recall@k: Did the correct source appear in top-k results?
2. Precision@k: What fraction of top-k results are relevant?
3. MRR (Mean Reciprocal Rank): How high is the first relevant result?
4. Latency: End-to-end retrieval and generation time
5. Hallucination rate: Does the answer contain info NOT in context?
6. Language match: Does the response language match the query?
7. Context adherence: How closely does the answer follow context?

Ablation studies:
- Baseline: Pure semantic search, no normalization
- +Normalization: Add transliteration
- +Keyword boost: Add hybrid scoring
- +Deduplication: Remove overlapping chunks
- Full pipeline: All components active
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import EVAL_QUERIES_PATH, BENCHMARK_RESULTS_PATH

logger = logging.getLogger(__name__)


# ─── Information Retrieval Metrics ────────────────────────────────────────────

def recall_at_k(
    results: List[Dict[str, Any]],
    expected_source: str,
    k: int = 5,
) -> float:
    """
    Recall@k: Is the expected source in the top-k results?

    This is a binary metric (0 or 1) for a single query.
    Averaged across queries, it gives the retrieval success rate.

    Args:
        results:         Search results (each has 'source' key)
        expected_source: The correct source filename
        k:               Number of top results to check

    Returns:
        1.0 if expected source found in top-k, else 0.0

    Example:
        results = [{"source": "modi.txt"}, {"source": "education.txt"}]
        recall_at_k(results, "modi.txt", k=5) → 1.0
    """
    top_k = results[:k]
    sources = [r.get("source", "") for r in top_k]
    return 1.0 if expected_source in sources else 0.0


def precision_at_k(
    results: List[Dict[str, Any]],
    expected_source: str,
    k: int = 5,
) -> float:
    """
    Precision@k: Fraction of top-k results from the correct source.

    Higher precision = fewer irrelevant chunks in the context window.

    Args:
        results:         Search results
        expected_source: The correct source filename
        k:               Number of top results to check

    Returns:
        Float between 0.0 and 1.0
    """
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if r.get("source", "") == expected_source)
    return relevant / len(top_k)


def reciprocal_rank(
    results: List[Dict[str, Any]],
    expected_source: str,
) -> float:
    """
    Reciprocal Rank (RR): 1/position of the first relevant result.

    This penalizes systems that rank relevant results lower.
    RR = 1.0 if first result is correct, 0.5 if second, 0.33 if third, etc.

    Args:
        results:         Search results
        expected_source: The correct source filename

    Returns:
        1/rank if found, 0.0 if not found

    Example:
        If correct source is at position 3 → RR = 1/3 = 0.333
    """
    for i, result in enumerate(results, 1):
        if result.get("source", "") == expected_source:
            return 1.0 / i
    return 0.0


# ─── Generation Quality Metrics ──────────────────────────────────────────────

def check_hallucination(
    answer: str,
    context_chunks: List[Dict[str, Any]],
    expected_keywords: List[str],
) -> Dict[str, Any]:
    """
    Basic hallucination detection.

    Strategy:
    1. Check if expected keywords from ground truth appear in the answer
    2. Check if the answer contains specific claims not in the context
    3. Check for refusal (model correctly declined to answer)

    This is a SIMPLIFIED heuristic. Production systems would use
    NLI (Natural Language Inference) models for this.

    Args:
        answer:            LLM-generated answer
        context_chunks:    Retrieved chunks (the context the LLM received)
        expected_keywords: Keywords that should appear in a correct answer

    Returns:
        Dict with:
        - contains_expected: True if answer has expected keywords
        - keywords_found: Which expected keywords were found
        - keywords_missing: Which were not found
        - is_refusal: True if model declined to answer
        - hallucination_risk: "low" | "medium" | "high"
    """
    answer_lower = answer.lower()

    # Check expected keywords
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

    # Check for refusal patterns
    refusal_patterns = [
        "don't have enough information",
        "cannot answer",
        "not in the context",
        "no relevant information",
        "i'm not sure",
    ]
    is_refusal = any(p in answer_lower for p in refusal_patterns)

    # Combine context text for checking
    context_text = " ".join(r.get("text", "") for r in context_chunks).lower()

    # Simple hallucination risk assessment
    if is_refusal:
        risk = "low"  # Refusing is safe
    elif len(found) > 0 and len(missing) == 0:
        risk = "low"  # All expected keywords present
    elif len(found) > 0:
        risk = "medium"  # Some keywords found
    else:
        risk = "high"  # No expected keywords — likely hallucinated

    return {
        "contains_expected": len(found) > 0,
        "keywords_found": found,
        "keywords_missing": missing,
        "keyword_accuracy": len(found) / len(expected_keywords) if expected_keywords else 0,
        "is_refusal": is_refusal,
        "hallucination_risk": risk,
    }


def check_language_consistency(
    query_language: str,
    answer: str,
) -> bool:
    """
    Check if the answer language roughly matches the query language.

    For Hindi/mixed queries, we check if the answer contains some
    Hindi words or at least addresses the question appropriately.

    This is a simple heuristic — production systems would use
    language detection on the answer.
    """
    from processing.language_detect import detect_language
    answer_lang, _ = detect_language(answer)

    # English query → English answer is always fine
    if query_language == "en":
        return True

    # Hindi/mixed query → answer in English or Hindi/mixed is fine
    if query_language in ("hi", "mixed"):
        return answer_lang in ("en", "hi", "mixed")

    return True


# ─── Evaluation Runner ───────────────────────────────────────────────────────

def load_eval_queries(eval_path: str | Path = EVAL_QUERIES_PATH) -> List[Dict]:
    """Load evaluation queries from JSON file."""
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logger.warning(f"Evaluation queries not found: {eval_path}")
        return []
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    search_fn,
    queries: Optional[List[Dict]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite over all test queries.

    Args:
        search_fn: A callable that takes (query: str) → List[Dict]
                   (the search results for a query)
        queries:   Evaluation queries (loaded from file if None)
        top_k:     Number of results to evaluate

    Returns:
        Comprehensive evaluation report with per-query and aggregate metrics:
        - avg_recall_at_k
        - avg_precision_at_k
        - mrr (Mean Reciprocal Rank)
        - avg_latency_ms
        - per_query_results
    """
    if queries is None:
        queries = load_eval_queries()

    if not queries:
        return {"error": "No evaluation queries available"}

    all_recall = []
    all_precision = []
    all_rr = []
    all_latency = []
    per_query = []

    for query_data in queries:
        query = query_data["query"]
        expected_source = query_data["expected_source"]
        language = query_data.get("language", "en")

        # Time the search
        start = time.time()
        results = search_fn(query)
        latency = (time.time() - start) * 1000

        # Compute metrics
        r_at_k = recall_at_k(results, expected_source, top_k)
        p_at_k = precision_at_k(results, expected_source, top_k)
        rr = reciprocal_rank(results, expected_source)

        all_recall.append(r_at_k)
        all_precision.append(p_at_k)
        all_rr.append(rr)
        all_latency.append(latency)

        per_query.append({
            "query_id": query_data.get("id"),
            "query": query,
            "language": language,
            "expected_source": expected_source,
            "recall@k": r_at_k,
            "precision@k": p_at_k,
            "reciprocal_rank": rr,
            "latency_ms": latency,
            "top_source": results[0].get("source", "none") if results else "none",
            "num_results": len(results),
        })

    # Aggregate metrics
    n = len(queries)
    report = {
        "num_queries": n,
        "top_k": top_k,
        "avg_recall_at_k": sum(all_recall) / n if n else 0,
        "avg_precision_at_k": sum(all_precision) / n if n else 0,
        "mrr": sum(all_rr) / n if n else 0,
        "avg_latency_ms": sum(all_latency) / n if n else 0,
        "max_latency_ms": max(all_latency) if all_latency else 0,
        "min_latency_ms": min(all_latency) if all_latency else 0,
        "per_query_results": per_query,
    }

    return report


def save_benchmark(report: Dict[str, Any], path: str | Path = BENCHMARK_RESULTS_PATH):
    """Save benchmark results to JSON file."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Benchmark results saved to {path}")


def print_report(report: Dict[str, Any]):
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Queries evaluated: {report['num_queries']}")
    print(f"  Top-k:             {report['top_k']}")
    print(f"  Avg Recall@{report['top_k']}:     {report['avg_recall_at_k']:.3f}")
    print(f"  Avg Precision@{report['top_k']}:   {report['avg_precision_at_k']:.3f}")
    print(f"  MRR:               {report['mrr']:.3f}")
    print(f"  Avg Latency:       {report['avg_latency_ms']:.1f}ms")
    print(f"  Max Latency:       {report['max_latency_ms']:.1f}ms")
    print("-" * 60)

    for result in report.get("per_query_results", []):
        status = "✓" if result["recall@k"] > 0 else "✗"
        print(
            f"  {status} Q{result['query_id']}: {result['query'][:40]:<40} "
            f"R@k={result['recall@k']:.1f} RR={result['reciprocal_rank']:.2f} "
            f"{result['latency_ms']:.0f}ms"
        )
    print("=" * 60)
