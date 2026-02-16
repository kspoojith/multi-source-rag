"""
Query Normalization Module
===========================
Preprocesses queries before embedding to improve retrieval quality.

The core insight: Our corpus contains text in BOTH Devanagari ("शिक्षा")
AND Romanized Hindi ("shiksha"). A user query might use either form.
Normalization ensures we search effectively regardless of input form.

Three key operations:
1. Transliteration: Romanized Hindi → Devanagari
   "kya" → "क्या", "hai" → "है"
   This helps because the embedding model produces better vectors
   for native-script text than transliterated text.

2. Query Expansion: Append both original + transliterated forms
   "Modi ji ka education kya hai" →
   "Modi ji ka education kya hai Modi जी का education क्या है"
   Now the embedding captures both scripts.

3. Stopword Removal: Remove noise words that don't carry meaning
   "ka", "ki", "ke", "the", "is", "a" → removed

Research contribution:
- We measure Recall@5 improvement with and without normalization
- We quantify the cosine similarity boost from transliteration
"""

import re
import logging
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import ENABLE_TRANSLITERATION, ENABLE_STOPWORD_REMOVAL
from processing.language_detect import detect_language

logger = logging.getLogger(__name__)

# ─── Romanized Hindi → Devanagari Mapping ────────────────────────────────────
# This is a simplified mapping for common words.
# A production system would use the 'indic-transliteration' library
# or Google's transliteration API, but for research reproducibility
# we use a controlled dictionary.
TRANSLITERATION_MAP = {
    # Question words
    "kya": "क्या", "kab": "कब", "kahan": "कहाँ", "kaun": "कौन",
    "kaise": "कैसे", "kitna": "कितना", "kitni": "कितनी", "kitne": "कितने",
    "kyun": "क्यों", "kyunki": "क्योंकि",
    # Pronouns & postpositions
    "ka": "का", "ki": "की", "ke": "के", "ko": "को",
    "se": "से", "mein": "में", "par": "पर", "tak": "तक",
    # Verbs
    "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी",
    "the": "थे", "hua": "हुआ", "hui": "हुई", "hue": "हुए",
    "hota": "होता", "hoti": "होती", "karna": "करना",
    "kiya": "किया", "gaya": "गया", "gayi": "गई",
    "bana": "बना", "bani": "बनी", "diya": "दिया",
    "batao": "बताओ", "bataye": "बताये",
    # Common nouns
    "ji": "जी", "sahab": "साहब",
    "desh": "देश", "bharat": "भारत",
    "shiksha": "शिक्षा", "vidyalaya": "विद्यालय",
    "sarkar": "सरकार", "pradhan": "प्रधान", "mantri": "मंत्री",
    "samvidhan": "संविधान", "kanoon": "कानून",
    "duniya": "दुनिया", "paisa": "पैसा",
    # Adjectives
    "sabse": "सबसे", "bahut": "बहुत", "zyada": "ज़्यादा",
    "pehla": "पहला", "pehli": "पहली",
    "naya": "नया", "purana": "पुराना",
    # Connectors
    "aur": "और", "ya": "या", "lekin": "लेकिन",
    "phir": "फिर", "toh": "तो", "bhi": "भी",
    "agar": "अगर", "jab": "जब", "tab": "तब",
    # Space-related (for ISRO corpus)
    "chand": "चाँद", "mission": "मिशन",
    # common names that appear transliterated
    "wala": "वाला", "wali": "वाली",
}

# ─── Stopwords ────────────────────────────────────────────────────────────────
# Combined English + Hindi stopwords that don't carry semantic meaning.
STOPWORDS = {
    # English stopwords (minimal set — we keep content words)
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "has", "have", "had",
    "i", "me", "my", "we", "our",
    "it", "its", "this", "that",
    "of", "in", "to", "for", "with", "on", "at", "by",
    "and", "or", "but", "not", "no",
    # Hindi stopwords (Romanized)
    "ka", "ki", "ke", "ko", "se", "mein", "par",
    "hai", "hain", "tha", "thi", "the",
    "aur", "ya", "bhi", "toh",
    "ek", "yeh", "woh", "jo",
}


def transliterate_query(query: str) -> str:
    """
    Convert Romanized Hindi words in the query to Devanagari.

    Only converts words found in our dictionary — unknown words are kept
    as-is (they might be English words or proper nouns like "Modi").

    Example:
        "Modi ji ka education kya hai?"
        → "Modi जी का education क्या है?"
    """
    tokens = query.split()
    transliterated = []

    for token in tokens:
        # Clean punctuation for lookup but preserve it in output
        clean = re.sub(r'[^\w]', '', token.lower())
        if clean in TRANSLITERATION_MAP:
            # Replace the clean part with Devanagari, keep punctuation
            replacement = TRANSLITERATION_MAP[clean]
            # Preserve trailing punctuation
            trailing = token[len(clean):] if len(token) > len(clean) else ""
            transliterated.append(replacement + trailing)
        else:
            transliterated.append(token)

    return " ".join(transliterated)


def remove_stopwords(query: str) -> str:
    """
    Remove stopwords from the query to reduce noise.

    We keep the query meaning-bearing words. Stopwords like "ka", "ki",
    "the", "is" don't help with embedding similarity.

    Example:
        "Modi ji ka education kya hai" → "Modi ji education kya"
        (keep "ji" and "kya" as they may carry meaning)
    """
    tokens = query.split()
    filtered = [t for t in tokens if t.lower().strip("?.!,") not in STOPWORDS]
    return " ".join(filtered) if filtered else query  # fallback to original if all removed


def normalize_query(query: str) -> dict:
    """
    Full normalization pipeline for a query.

    Args:
        query: Raw user query (any language/mix)

    Returns:
        Dict with:
        - original: The raw query
        - language: Detected language code
        - confidence: Detection confidence
        - normalized: The processed query for embedding
        - transliterated: Devanagari version (if applicable)
        - expanded: Combined original + transliterated for embedding
        - stopwords_removed: Query with stopwords removed

    The 'expanded' field is what gets embedded — it contains both
    the original tokens AND their Devanagari equivalents, maximizing
    the chance of matching against either script in the corpus.
    """
    # Step 1: Detect language
    lang, confidence = detect_language(query)
    logger.info(f"Language detected: {lang} (confidence={confidence:.2f})")

    result = {
        "original": query,
        "language": lang,
        "confidence": confidence,
        "normalized": query,
        "transliterated": None,
        "expanded": query,
        "stopwords_removed": query,
    }

    # Step 2: Transliterate if code-mixed and enabled
    if ENABLE_TRANSLITERATION and lang in ("mixed", "hi"):
        transliterated = transliterate_query(query)
        result["transliterated"] = transliterated

        # Expand: combine original + transliterated for richer embedding
        if transliterated != query:
            result["expanded"] = f"{query} {transliterated}"

    # Step 3: Remove stopwords if enabled
    if ENABLE_STOPWORD_REMOVAL:
        result["stopwords_removed"] = remove_stopwords(query)

    # Step 4: Set the final normalized form
    # For embedding, we use the expanded form (best recall)
    # For keyword matching, we use stopwords_removed form
    result["normalized"] = result["expanded"]

    logger.debug(f"Normalization result: {result}")
    return result
