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
# This is the "meta-instruction" that controls the LLM's behavior.
# Every word is intentional — tested through prompt engineering iterations.

SYSTEM_PROMPT = """You are a helpful multilingual question-answering assistant for Indian public information.

STRICT RULES (you MUST follow these):
1. ONLY use the information provided in the CONTEXT below to answer.
2. Do NOT use any external knowledge or information not in the context.
3. If the context does not contain enough information to answer, say: "I don't have enough information in the provided context to answer this question."
4. Always mention which source file(s) your answer is based on.
5. Keep your answer concise and factual (2-4 sentences maximum).
6. If the question is in Hindi or code-mixed (Hindi-English), you may answer in the same language style.
7. Never make up facts, dates, numbers, or names that are not in the context.
8. If asked about something completely unrelated to the context, politely decline.

You are grounded. You are factual. You cite sources."""


def build_context_block(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a labeled context block for the prompt.

    Each chunk is labeled with its source file so the LLM can cite it.

    Example output:
        [Source: modi.txt]
        Modi completed his higher secondary education in Vadnagar...

        [Source: indian_education.txt]
        Right to Education Act 2009 makes education free...
    """
    if not results:
        return "[No relevant context found]"

    blocks = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "unknown")
        text = result.get("text", "")
        score = result.get("final_score", result.get("score", 0))
        blocks.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}")

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

    # Language-aware instruction
    if language in ("hi", "mixed"):
        lang_instruction = (
            "The user's query is in Hindi/code-mixed language. "
            "You may respond in a similar language style (Hindi-English mix) "
            "if it helps the user understand better."
        )
    else:
        lang_instruction = "Respond in clear English."

    prompt = f"""{SYSTEM_PROMPT}

{lang_instruction}

CONTEXT:
{context_block}

USER QUESTION: {query}

ANSWER (cite sources, be concise, ONLY use context above):"""

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
