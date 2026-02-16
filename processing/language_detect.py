"""
Language Detection Module
==========================
Detects the language of a query: English, Hindi, Telugu, or Code-Mixed.

Why build our own instead of using langdetect/fasttext?
- Off-the-shelf detectors fail on code-mixed text
  ("Modi ji ka education kya hai?" is classified as English by langdetect)
- We need fine-grained detection: "mixed" is a valid category for us
- Our heuristic approach is transparent and debuggable

Detection Strategy:
1. Check for Devanagari Unicode range → Hindi
2. Check for Telugu Unicode range → Telugu
3. Check for Romanized Hindi keywords in Latin text → Mixed
4. Otherwise → English

Research contribution:
- We measure detection accuracy on our test corpus
- We show that code-mixed detection improves retrieval by enabling
  targeted normalization
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ─── Unicode Range Patterns ──────────────────────────────────────────────────
# Devanagari: U+0900 to U+097F (Hindi, Sanskrit, Marathi)
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

# Telugu: U+0C00 to U+0C7F
TELUGU_PATTERN = re.compile(r'[\u0C00-\u0C7F]')

# Tamil: U+0B80 to U+0BFF
TAMIL_PATTERN = re.compile(r'[\u0B80-\u0BFF]')

# Bengali: U+0980 to U+09FF
BENGALI_PATTERN = re.compile(r'[\u0980-\u09FF]')

# ─── Romanized Hindi Indicators ──────────────────────────────────────────────
# Common Hindi words written in Roman script (transliterated).
# These appear in code-mixed queries like "Modi ji ka education kya hai?"
ROMANIZED_HINDI_WORDS = {
    # Question words
    "kya", "kab", "kahan", "kaun", "kaise", "kitna", "kitni", "kitne",
    "kyun", "kyunki",
    # Pronouns & particles
    "ka", "ki", "ke", "ko", "se", "mein", "hai", "hain", "tha", "thi",
    "the", "ho", "hota", "hoti", "hote",
    # Verbs
    "karna", "kiya", "hua", "hue", "hui", "gaya", "gayi", "gaye",
    "bana", "bani", "bane", "diya", "diye", "diyi",
    "batao", "bataye", "bataiye",
    # Common nouns/adjectives
    "ji", "sahab", "sahib", "wala", "wali", "wale",
    "sabse", "bahut", "thoda", "zyada", "kam",
    "pehla", "pehli", "pehle", "dusra", "dusri",
    "naya", "nayi", "naye", "purana", "purani",
    # Connectors
    "aur", "ya", "lekin", "par", "phir", "toh", "bhi",
    "agar", "jab", "tab",
}

# Minimum ratio of Hindi tokens to trigger "mixed" classification
MIXED_THRESHOLD = 0.15  # At least 15% of tokens look Hindi


def count_script_chars(text: str) -> dict:
    """
    Count characters belonging to different Unicode scripts.

    Returns a dict like:
    {"devanagari": 5, "telugu": 0, "latin": 20, "other": 2}
    """
    counts = {"devanagari": 0, "telugu": 0, "tamil": 0, "bengali": 0, "latin": 0, "other": 0}

    for char in text:
        if DEVANAGARI_PATTERN.match(char):
            counts["devanagari"] += 1
        elif TELUGU_PATTERN.match(char):
            counts["telugu"] += 1
        elif TAMIL_PATTERN.match(char):
            counts["tamil"] += 1
        elif BENGALI_PATTERN.match(char):
            counts["bengali"] += 1
        elif char.isascii() and char.isalpha():
            counts["latin"] += 1
        elif not char.isspace() and not char.isdigit():
            counts["other"] += 1

    return counts


def detect_romanized_hindi(text: str) -> float:
    """
    Detect the proportion of Romanized Hindi words in Latin-script text.

    Splits text into tokens and checks each against our Hindi word list.
    Returns the ratio of matched tokens.

    Example:
        "Modi ji ka education kya hai" → 4/6 = 0.67 (ji, ka, kya, hai)
    """
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    if not tokens:
        return 0.0

    hindi_count = sum(1 for token in tokens if token in ROMANIZED_HINDI_WORDS)
    return hindi_count / len(tokens)


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the primary language of the input text.

    Args:
        text: Input query or text

    Returns:
        Tuple of (language_code, confidence):
        - language_code: "hi" | "te" | "ta" | "bn" | "en" | "mixed"
        - confidence: 0.0 to 1.0

    Algorithm:
    1. Count characters per script (Devanagari, Telugu, Latin, etc.)
    2. If >30% Devanagari → "hi" (Hindi in native script)
    3. If >30% Telugu → "te"
    4. If mostly Latin, check for Romanized Hindi words
    5. If Romanized Hindi ratio > 15% → "mixed" (code-mixed)
    6. Otherwise → "en" (English)
    """
    if not text or not text.strip():
        return "en", 0.0

    counts = count_script_chars(text)
    total_alpha = sum(counts.values()) - counts["other"]

    if total_alpha == 0:
        return "en", 0.0

    # Check for native script languages
    devanagari_ratio = counts["devanagari"] / total_alpha if total_alpha else 0
    telugu_ratio = counts["telugu"] / total_alpha if total_alpha else 0
    tamil_ratio = counts["tamil"] / total_alpha if total_alpha else 0
    bengali_ratio = counts["bengali"] / total_alpha if total_alpha else 0

    # Pure Hindi (Devanagari script)
    if devanagari_ratio > 0.3:
        if counts["latin"] > 0 and counts["latin"] / total_alpha > 0.1:
            return "mixed", 0.8  # Mix of Devanagari + Latin
        return "hi", min(0.5 + devanagari_ratio, 1.0)

    # Pure Telugu
    if telugu_ratio > 0.3:
        return "te", min(0.5 + telugu_ratio, 1.0)

    # Pure Tamil
    if tamil_ratio > 0.3:
        return "ta", min(0.5 + tamil_ratio, 1.0)

    # Pure Bengali
    if bengali_ratio > 0.3:
        return "bn", min(0.5 + bengali_ratio, 1.0)

    # Mostly Latin script — check for Romanized Hindi (code-mixed)
    hindi_ratio = detect_romanized_hindi(text)
    if hindi_ratio >= MIXED_THRESHOLD:
        return "mixed", min(0.5 + hindi_ratio, 1.0)

    # Default: English
    return "en", 0.9


def get_language_label(code: str) -> str:
    """Human-readable language name from code."""
    labels = {
        "en": "English",
        "hi": "Hindi",
        "te": "Telugu",
        "ta": "Tamil",
        "bn": "Bengali",
        "mixed": "Code-Mixed (Hindi-English)",
    }
    return labels.get(code, "Unknown")
