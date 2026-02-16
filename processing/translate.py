"""
Query Translator Module
========================
Translates Hindi / Telugu / Code-Mixed queries to English
for better web search results.

Why translate?
- DuckDuckGo, Google, Bing all return *much* better results for English queries
- "Elon Musk ki net worth kitni hai?" → "Elon Musk net worth" gets great results
- "చంద్రయాన్ 3 ఎప్పుడు launch అయింది?" → "Chandrayaan 3 launch date"

Translation Strategy:
We use the local Ollama LLM itself for translation — no external API needed.
This keeps the system fully offline and free.

Fallback: If Ollama is unavailable, we do a simple keyword extraction
(remove Hindi/Telugu stopwords, keep English words and proper nouns).
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Telugu Romanized Words (common ones for detection) ───────────────────────
TELUGU_STOPWORDS = {
    "endi", "ela", "eppudu", "ekkada", "evaru", "entha",
    "lo", "ki", "ni", "ga", "tho", "lekha",
    "undi", "unnadi", "chesindi", "cheppandi",
    "adi", "idi", "avi", "ivi",
    "mariyu", "kani", "ledhaa", "ante",
    "okka", "anni", "konni",
}

HINDI_STOPWORDS = {
    "ka", "ki", "ke", "ko", "se", "mein", "par", "tak",
    "hai", "hain", "tha", "thi", "the",
    "hua", "hui", "hue", "hota", "hoti",
    "kya", "kab", "kahan", "kaun", "kaise", "kitna", "kitni",
    "kyun", "kyunki",
    "aur", "ya", "lekin", "bhi", "toh", "phir",
    "agar", "jab", "tab",
    "ji", "sahab", "wala", "wali",
    "sabse", "bahut", "zyada", "kam",
    "ek", "yeh", "woh", "jo",
    "batao", "bataye", "bataiye",
}

ALL_STOPWORDS = HINDI_STOPWORDS | TELUGU_STOPWORDS | {
    "a", "an", "the", "is", "are", "was", "were",
    "of", "in", "to", "for", "with", "on", "at", "by",
    "and", "or", "but", "not", "no",
    "do", "does", "did", "has", "have", "had",
    "it", "its", "this", "that",
}


def extract_keywords_for_search(query: str) -> str:
    """
    Simple keyword extraction — remove stopwords, keep content words.
    This is the fast fallback when Ollama is not available.

    Example:
        "Elon Musk ki net worth kitni hai?" → "Elon Musk net worth"
        "Chandrayaan-3 kab launch hua tha?" → "Chandrayaan-3 launch"
        "భారత రాజధాని ఏమిటి?" → "భారత రాజధాని"
    """
    # Split into tokens
    tokens = query.split()
    # Remove punctuation from each token for matching, but keep original
    keywords = []
    for token in tokens:
        clean = re.sub(r'[^\w\-]', '', token.lower())
        if clean and clean not in ALL_STOPWORDS and len(clean) > 1:
            keywords.append(token.rstrip("?.!,;:"))
    return " ".join(keywords) if keywords else query


def translate_with_ollama(
    query: str,
    source_lang: str = "auto",
) -> Optional[str]:
    """
    Use the local Ollama LLM to translate a query to English.

    This is a lightweight translation — we only need the search intent,
    not a perfect literary translation.

    Args:
        query:       The original query in any language
        source_lang: Detected language code (hi, te, mixed, etc.)

    Returns:
        English translation string, or None if Ollama is unavailable
    """
    try:
        import requests
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

        # Quick check if Ollama is running
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code != 200:
                return None
        except Exception:
            return None

        prompt = (
            f"Translate the following query to English. "
            f"Only output the English translation, nothing else. "
            f"Keep proper nouns (names, places, organizations) as-is.\n\n"
            f"Query: {query}\n\n"
            f"English translation:"
        )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 100},
            },
            timeout=min(OLLAMA_TIMEOUT, 60),  # Cap at 60s for translation
        )
        response.raise_for_status()
        translated = response.json().get("response", "").strip()

        # Clean up — remove quotes, extra whitespace
        translated = translated.strip('"\'').strip()
        # Remove "Translation:" prefix if LLM added it
        for prefix in ["Translation:", "English:", "English translation:"]:
            if translated.lower().startswith(prefix.lower()):
                translated = translated[len(prefix):].strip()

        if translated and len(translated) > 3:
            logger.info(f"Translated: '{query[:40]}...' → '{translated[:40]}...'")
            return translated
        return None

    except Exception as e:
        logger.debug(f"Translation failed: {e}")
        return None


def get_search_query(query: str, language: str) -> str:
    """
    Get the best query for web search based on detected language.

    Strategy:
    1. If query is already English → use as-is
    2. If Hindi/Telugu/Mixed → try Ollama translation
    3. If Ollama unavailable → extract keywords (remove stopwords)

    Args:
        query:    Original user query
        language: Detected language code from language_detect.py

    Returns:
        English (or cleaned) query optimized for web search
    """
    if language == "en":
        return query

    # Try Ollama translation first
    translated = translate_with_ollama(query, language)
    if translated:
        return translated

    # Fallback: keyword extraction
    logger.info("Using keyword extraction fallback for search query")
    return extract_keywords_for_search(query)
