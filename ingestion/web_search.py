"""
Web Search Module
==================
Fetches real-time information from the internet using DuckDuckGo,
so the system can answer ANY question — not just from local files.

Why DuckDuckGo?
- Completely free — no API keys required
- No rate-limit issues for moderate usage
- Returns clean text snippets we can feed into the RAG pipeline

How it fits into the system:
1. User asks: "Elon Musk ki net worth kitni hai?"
2. We translate the query to English for better search results
3. DuckDuckGo returns top web results with snippets
4. We chunk + embed these snippets → build a temporary FAISS index
5. Same hybrid retrieval + LLM pipeline generates the answer

This replaces the need for thousands of pre-created text files.
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed. Web search disabled.")


@dataclass
class WebResult:
    """A single web search result."""
    title: str
    snippet: str
    url: str
    source: str  # domain name for citation


def extract_domain(url: str) -> str:
    """Extract a clean domain name from a URL for citation."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain
    except Exception:
        return url[:50]


def search_web(
    query: str,
    max_results: int = 8,
    region: str = "in-en",
) -> List[WebResult]:
    """
    Search the web using DuckDuckGo and return structured results.

    Args:
        query:       The search query (best in English for quality results)
        max_results: Number of results to fetch (default 8)
        region:      Region for search (default "in-en" for India-English)

    Returns:
        List of WebResult with title, snippet, url, source
    """
    if not DDGS_AVAILABLE:
        logger.error("DuckDuckGo search library not available")
        return []

    start_time = time.time()
    results = []

    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                keywords=query,
                region=region,
                max_results=max_results,
                safesearch="moderate",
            ))

        for r in search_results:
            title = r.get("title", "")
            snippet = r.get("body", "")
            url = r.get("href", r.get("link", ""))

            if snippet and len(snippet) > 20:
                results.append(WebResult(
                    title=title,
                    snippet=snippet,
                    url=url,
                    source=extract_domain(url),
                ))

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Web search: '{query[:50]}...' → {len(results)} results in {elapsed:.0f}ms")

    except Exception as e:
        logger.error(f"Web search failed: {e}")

    return results


def web_results_to_chunks(
    results: List[WebResult],
) -> List[Dict[str, Any]]:
    """
    Convert web search results into chunk-like dicts compatible
    with the existing retrieval pipeline.

    Each web result becomes a "chunk" with:
    - text: title + snippet combined
    - source: domain name (for citation)
    - url: full URL
    - topic: "web"
    """
    chunks = []
    for i, r in enumerate(results):
        text = f"{r.title}\n{r.snippet}"
        chunks.append({
            "text": text,
            "source": r.source,
            "url": r.url,
            "topic": "web",
            "chunk_id": f"web_{i}",
        })
    return chunks


def search_and_prepare(
    query: str,
    english_query: Optional[str] = None,
    max_results: int = 8,
) -> List[Dict[str, Any]]:
    """
    High-level function: Search the web and return RAG-ready chunks.

    If an English translation is provided, we search with that
    (DuckDuckGo returns better results for English queries),
    but include the original query context.

    Args:
        query:          Original user query (any language)
        english_query:  English translation for better search results
        max_results:    Number of web results to fetch

    Returns:
        List of chunk dicts ready for embedding + retrieval
    """
    # Use English query for search if available, otherwise original
    search_query = english_query if english_query else query

    results = search_web(search_query, max_results=max_results)

    if not results:
        # Fallback: try with original query if translation didn't work
        if english_query and english_query != query:
            logger.info("Retrying web search with original query...")
            results = search_web(query, max_results=max_results)

    return web_results_to_chunks(results)
