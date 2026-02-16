"""
Benchmarking Module — Ablation Studies
========================================
Runs controlled experiments to measure the impact of each component.

Ablation Study Design:
We compare 5 configurations to isolate each component's contribution:

1. BASELINE: Pure semantic search, no normalization, no keyword boost
   - α=1.0, β=0.0, no transliteration, no stopword removal
   - This is the "naive RAG" baseline

2. +NORMALIZATION: Add transliteration + stopword removal
   - α=1.0, β=0.0, WITH transliteration, WITH stopword removal
   - Measures: Does query normalization improve retrieval?

3. +KEYWORD BOOST: Add hybrid scoring
   - α=0.7, β=0.3, WITH transliteration, WITH stopword removal
   - Measures: Does keyword matching help?

4. +DEDUPLICATION: Add deduplication
   - Full pipeline with deduplication enabled
   - Measures: Does removing duplicates improve precision?

5. FULL PIPELINE: All components active
   - α=0.7, β=0.3, normalization, keyword boost, deduplication, reranking
   - This is our proposed system

Each experiment is run on the same set of evaluation queries and
the same index, ensuring fair comparison.
"""

import time
import logging
from typing import Dict, Any, List, Callable
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import run_evaluation, print_report, save_benchmark

logger = logging.getLogger(__name__)


def run_ablation_study(
    pipeline_configs: Dict[str, Callable],
    queries: List[Dict],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run ablation studies across multiple pipeline configurations.

    Args:
        pipeline_configs: Dict mapping config name to search function.
                         Each search function takes (query: str) → List[Dict]
        queries:          Evaluation queries
        top_k:           Number of results to evaluate

    Returns:
        Dict with results for each configuration, enabling comparison.

    Example usage:
        configs = {
            "baseline": baseline_search_fn,
            "+normalization": norm_search_fn,
            "+keyword": keyword_search_fn,
            "full_pipeline": full_search_fn,
        }
        results = run_ablation_study(configs, queries)
    """
    all_results = {}

    for config_name, search_fn in pipeline_configs.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Running experiment: {config_name}")
        logger.info(f"{'='*40}")

        report = run_evaluation(search_fn, queries, top_k)
        report["config_name"] = config_name
        all_results[config_name] = report

        print(f"\n--- {config_name} ---")
        print(f"  Recall@{top_k}: {report['avg_recall_at_k']:.3f}")
        print(f"  Precision@{top_k}: {report['avg_precision_at_k']:.3f}")
        print(f"  MRR: {report['mrr']:.3f}")
        print(f"  Avg Latency: {report['avg_latency_ms']:.1f}ms")

    # Print comparison table
    print_comparison_table(all_results, top_k)

    return all_results


def print_comparison_table(results: Dict[str, Any], top_k: int = 5):
    """
    Print a comparison table across all configurations.

    Output looks like:
    ┌─────────────────┬──────────┬───────────┬───────┬─────────┐
    │ Configuration   │ Recall@5 │ Precision │ MRR   │ Latency │
    ├─────────────────┼──────────┼───────────┼───────┼─────────┤
    │ baseline        │ 0.667    │ 0.333     │ 0.556 │ 45ms    │
    │ +normalization  │ 0.833    │ 0.417     │ 0.694 │ 48ms    │
    │ full_pipeline   │ 0.917    │ 0.500     │ 0.833 │ 52ms    │
    └─────────────────┴──────────┴───────────┴───────┴─────────┘
    """
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Configuration':<20} {'Recall@'+str(top_k):<10} {'Prec@'+str(top_k):<10} {'MRR':<8} {'Latency':<10}"
    print(header)
    print("-" * 70)

    for config_name, report in results.items():
        row = (
            f"{config_name:<20} "
            f"{report['avg_recall_at_k']:<10.3f} "
            f"{report['avg_precision_at_k']:<10.3f} "
            f"{report['mrr']:<8.3f} "
            f"{report['avg_latency_ms']:<10.1f}ms"
        )
        print(row)

    print("=" * 70)


def run_embedding_quality_experiment(embedding_model) -> Dict[str, Any]:
    """
    Measure embedding quality across different language modes.

    Compares cosine similarity between semantically equivalent
    queries in different forms:
    - Pure English
    - Pure Hindi (Devanagari)
    - Code-Mixed (Romanized Hindi + English)

    This demonstrates the "embedding degradation" on code-mixed text.
    """
    # Pairs of semantically equivalent queries
    query_pairs = [
        {
            "topic": "Modi's education",
            "english": "What is Narendra Modi's educational qualification?",
            "hindi": "नरेंद्र मोदी की शैक्षिक योग्यता क्या है?",
            "code_mixed": "Modi ji ka education kya hai?",
        },
        {
            "topic": "Constitution",
            "english": "Who is the father of Indian constitution?",
            "hindi": "भारतीय संविधान के जनक कौन हैं?",
            "code_mixed": "Samvidhan ke father kaun hain?",
        },
        {
            "topic": "ISRO",
            "english": "Where is ISRO headquarters located?",
            "hindi": "इसरो का मुख्यालय कहाँ है?",
            "code_mixed": "ISRO ka headquarter kahan hai?",
        },
        {
            "topic": "Chandrayaan",
            "english": "When did Chandrayaan-3 land on the moon?",
            "hindi": "चंद्रयान-3 चाँद पर कब उतरा?",
            "code_mixed": "Chandrayaan-3 kab launch hua tha?",
        },
    ]

    results = []
    for pair in query_pairs:
        # Compute all pairwise similarities
        en_hi = embedding_model.similarity(pair["english"], pair["hindi"])
        en_cm = embedding_model.similarity(pair["english"], pair["code_mixed"])
        hi_cm = embedding_model.similarity(pair["hindi"], pair["code_mixed"])

        results.append({
            "topic": pair["topic"],
            "en_hi_similarity": en_hi,
            "en_codemixed_similarity": en_cm,
            "hi_codemixed_similarity": hi_cm,
            # Degradation = how much worse code-mixed is vs pure language
            "degradation_en_cm": en_hi - en_cm,
        })

    # Aggregate
    avg_en_hi = sum(r["en_hi_similarity"] for r in results) / len(results)
    avg_en_cm = sum(r["en_codemixed_similarity"] for r in results) / len(results)
    avg_degradation = sum(r["degradation_en_cm"] for r in results) / len(results)

    report = {
        "per_query": results,
        "avg_english_hindi_similarity": avg_en_hi,
        "avg_english_codemixed_similarity": avg_en_cm,
        "avg_embedding_degradation": avg_degradation,
    }

    # Print results
    print("\n" + "=" * 60)
    print("  EMBEDDING QUALITY EXPERIMENT")
    print("=" * 60)
    print(f"{'Topic':<20} {'EN↔HI':<10} {'EN↔CM':<10} {'HI↔CM':<10} {'Degrad.':<10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['topic']:<20} "
            f"{r['en_hi_similarity']:<10.3f} "
            f"{r['en_codemixed_similarity']:<10.3f} "
            f"{r['hi_codemixed_similarity']:<10.3f} "
            f"{r['degradation_en_cm']:<10.3f}"
        )
    print("-" * 60)
    print(f"{'AVERAGE':<20} {avg_en_hi:<10.3f} {avg_en_cm:<10.3f} {'':10} {avg_degradation:<10.3f}")
    print("=" * 60)

    return report
