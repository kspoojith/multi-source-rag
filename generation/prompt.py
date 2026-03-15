"""
Prompt Engineering Module
==========================
Constructs carefully designed prompts for the LLM to minimize hallucination.

Hallucination Control Strategy:
1. STRICT GROUNDING: The prompt explicitly tells the LLM to ONLY use
   the provided context. No external knowledge.
2. SOURCE CITATION: Every answer must reference which source file(s)
   the information comes from.
3. REFUSAL PROTOCOL: If the context doesn't contain enough information,
   the LLM must say so rather than making up an answer.
4. SHORT ANSWERS: We instruct the LLM to be concise — longer answers
   have more room for hallucinated details.

Prompt Template Design:
- System prompt: Sets the role and rules
- Context block: Retrieved chunks with source labels
- User query: The actual question
- Format instructions: How to structure the answer

Research metrics enabled:
- Hallucination rate: % of answers containing info NOT in context
- Context adherence score: How closely the answer matches context
- Language consistency: Does the answer match the query language?
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────
# Optimized for CPU: Clear but concise to reduce generation time

SYSTEM_PROMPT = """You are a helpful Q&A assistant. 

Rules:
1. Answer based ONLY on the context provided below
2. If the context has the answer, provide it clearly in 2-3 sentences
3. If the context doesn't have enough info, say "I don't have enough information"
4. Cite the sources mentioned in the context
5. Answer in the same language style as the question (Hindi/English mix is OK)"""


def build_context_block(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a labeled context block for the prompt.

    Each chunk is labeled with its source file so the LLM can cite it.
    For web results, the source is a domain name + URL.

    Example output (local):
        [Source: modi.txt]
        Modi completed his higher secondary education in Vadnagar...

    Example output (web):
        [Source: wikipedia.org | URL: https://en.wikipedia.org/...]
        Elon Musk is the CEO of Tesla and SpaceX...
    """
    if not results:
        return "[No relevant context found]"

    blocks = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "unknown")
        text = result.get("text", "")[:400]  # Increased to 400 chars for more context
        url = result.get("url", "")

        if url:
            blocks.append(f"[{source}] {text}")
        else:
            blocks.append(f"[{source}] {text}")

    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    results: List[Dict[str, Any]],
    language: str = "en",
) -> str:
    """
    Build the complete prompt for the LLM.

    Args:
        query:    The user's question
        results:  Retrieved and reranked chunks
        language: Detected language of the query

    Returns:
        Complete prompt string ready for LLM inference

    Structure:
        [System Prompt]
        [Context Block with source labels]
        [User Question]
        [Response Format Instructions]
    """
    context_block = build_context_block(results)

    # Clear, concise prompt for fast generation
    prompt = f"""{SYSTEM_PROMPT}

Context:
{context_block}

Question: {query}

Answer (2-3 sentences, cite sources):"""

    logger.debug(f"Prompt built: {len(prompt)} chars, {len(results)} context chunks")
    return prompt


def build_prompt_for_ollama(
    query: str,
    results: List[Dict[str, Any]],
    language: str = "en",
) -> List[Dict[str, str]]:
    """
    Build a chat-format prompt for Ollama's chat API.

    Ollama expects messages in the format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]

    We put the system instructions in the system message,
    and the context + question in the user message.
    """
    context_block = build_context_block(results)

    if language in ("hi", "mixed"):
        lang_instruction = (
            "The user's query is in Hindi/code-mixed language. "
            "You may respond in a similar style."
        )
    else:
        lang_instruction = "Respond in clear English."

    user_message = f"""{lang_instruction}

CONTEXT:
{context_block}

QUESTION: {query}

Answer concisely using ONLY the context above. Cite source files."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# ─── Web Search Prompts ──────────────────────────────────────────────────────
# For open-domain QA, we use web search results as context.
# The key difference: sources are websites with URLs, not local files.

WEB_SYSTEM_PROMPT = """You are a helpful multilingual question-answering assistant that answers questions using web search results.

STRICT RULES (you MUST follow these):
1. ONLY use the information provided in the WEB SEARCH RESULTS below to answer.
2. Do NOT add information beyond what is in the search results.
3. If the search results do not contain enough information, say: "The web search results don't have enough information to fully answer this question."
4. Always mention which website(s) your answer is based on (cite the domain names).
5. Keep your answer concise and factual (3-5 sentences maximum).
6. If the question is in Hindi, Telugu, or code-mixed, you may answer in the same language style.
7. Never make up facts, dates, numbers, or names not in the search results.

You are grounded. You are factual. You cite web sources."""


def build_web_prompt_for_ollama(
    query: str,
    results: List[Dict[str, Any]],
    language: str = "en",
) -> List[Dict[str, str]]:
    """
    Build a chat-format prompt for web search results.

    Uses WEB_SYSTEM_PROMPT which is tuned for web sources
    instead of local file context.
    """
    context_block = build_context_block(results)

    if language in ("hi", "mixed"):
        lang_instruction = (
            "The user's query is in Hindi/code-mixed language. "
            "You may respond in a similar style (Hindi-English mix)."
        )
    elif language == "te":
        lang_instruction = (
            "The user's query is in Telugu or Telugu-English mix. "
            "You may respond in English with transliterated Telugu terms."
        )
    else:
        lang_instruction = "Respond in clear English."

    user_message = f"""{lang_instruction}

WEB SEARCH RESULTS:
{context_block}

QUESTION: {query}

Answer concisely using ONLY the web search results above. Cite website names."""

    return [
        {"role": "system", "content": WEB_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
