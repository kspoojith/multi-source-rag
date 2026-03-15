"""
LLM Module — Ollama Integration
=================================
Handles communication with the local Ollama LLM server.

Why Ollama?
- Runs models locally — no API keys, no cloud dependency
- Supports quantized models — fits on 8GB RAM
- REST API compatible — simple HTTP calls
- Models: mistral:7b-instruct, phi3:mini, llama3, etc.

Prerequisites:
- Install Ollama: https://ollama.ai
- Pull a model: `ollama pull mistral:7b-instruct`
- Ollama runs on http://localhost:11434 by default

Fallback strategy:
If Ollama is not available, the system returns the retrieved context
with a note that LLM generation is unavailable. This ensures the
retrieval pipeline is always functional for evaluation.

Research focus:
- Compare latency: mistral:7b-instruct vs phi3:mini
- Measure hallucination rate with strict vs weak prompting
- Track language consistency in responses
"""

import json
import time
import logging
import requests
from typing import List, Dict, Optional, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_TEMPERATURE,
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
)

logger = logging.getLogger(__name__)


class OllamaLLM:
    """
    Client for the Ollama LLM server.

    Usage:
        llm = OllamaLLM()
        if llm.is_available():
            response = llm.generate(messages=[...])
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
        temperature: float = OLLAMA_TEMPERATURE,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def is_available(self) -> bool:
        """
        Check if Ollama server is running and the model is available.

        Makes a GET request to /api/tags to list available models.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if our model (or a variant) is available
                # Handle both "mistral" and "mistral:latest" formats
                model_base = self.model.split(":")[0]
                for name in model_names:
                    name_base = name.split(":")[0]
                    if model_base == name_base or name == self.model:
                        logger.info(f"Ollama model '{self.model}' is available (found: {name})")
                        return True
                
                logger.warning(
                    f"Ollama is running but model '{self.model}' not found. "
                    f"Available: {model_names}. "
                    f"Run: ollama pull {self.model}"
                )
                return False
            return False
        except requests.ConnectionError:
            logger.warning(
                f"Ollama server not reachable at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
            return False
        except Exception as e:
            logger.warning(f"Error checking Ollama: {e}")
            return False

    def generate(
        self,
        prompt: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Supports two modes:
        1. Simple prompt (generate API): prompt="your text"
        2. Chat messages (chat API): messages=[{"role": "user", ...}]

        Args:
            prompt:   Simple text prompt (used if messages is None)
            messages: Chat-format messages (preferred)

        Returns:
            Dict with:
            - response: The generated text
            - model: Model name used
            - latency_ms: Time taken for generation
            - success: True/False
            - error: Error message (if failed)
        """
        start_time = time.time()

        try:
            if messages:
                # Use the chat API (preferred for instruction-tuned models)
                result = self._chat(messages)
            else:
                # Use the generate API (simpler)
                result = self._generate(prompt)

            latency = (time.time() - start_time) * 1000
            result["latency_ms"] = latency
            result["success"] = True
            logger.info(f"LLM response generated in {latency:.0f}ms")
            return result

        except requests.Timeout:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return {
                "response": "The LLM request timed out. Please try again.",
                "model": self.model,
                "latency_ms": latency,
                "success": False,
                "error": "timeout",
            }
        except requests.ConnectionError:
            return {
                "response": "Cannot connect to Ollama. Make sure it's running.",
                "model": self.model,
                "latency_ms": 0,
                "success": False,
                "error": "connection_error",
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"LLM generation failed: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "model": self.model,
                "latency_ms": latency,
                "success": False,
                "error": str(e),
            }

    def _chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call Ollama's /api/chat endpoint.

        This is the preferred method for instruction-tuned models
        because it properly formats system/user/assistant roles.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return {
            "response": data.get("message", {}).get("content", ""),
            "model": data.get("model", self.model),
        }

    def _generate(self, prompt: str) -> Dict[str, Any]:
        """
        Call Ollama's /api/generate endpoint.

        Simpler endpoint for raw text-in, text-out generation.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return {
            "response": data.get("response", ""),
            "model": data.get("model", self.model),
        }


def generate_answer(
    query: str,
    results: List[Dict[str, Any]],
    language: str = "en",
) -> Dict[str, Any]:
    """
    High-level function: Generate an answer for a query using retrieved context.

    This is the main entry point called by the API.
    Supports both Ollama (local) and Groq API (cloud) based on LLM_PROVIDER config.

    Args:
        query:    User's question
        results:  Retrieved and reranked chunks
        language: Detected query language

    Returns:
        Dict with answer, sources, latency, and metadata
    """
    # If no relevant results, return a "no context" response
    if not results:
        return {
            "answer": "I don't have enough information to answer this question. Please try a different query.",
            "sources": [],
            "model": "none",
            "latency_ms": 0,
            "success": True,
            "llm_used": False,
        }

    # Collect unique sources
    sources = list(set(r.get("source", "unknown") for r in results))

    # Choose LLM provider based on config
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        # Use Groq API (cloud deployment)
        try:
            from generation.llm_groq import GroqLLM
            from generation.prompt import build_web_prompt_for_ollama
            
            logger.info("Using Groq API for LLM generation")
            groq_llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
            
            # Use web search prompt (better for web results)
            messages = build_web_prompt_for_ollama(query, results, language)
            groq_result = groq_llm.generate(
                messages=messages,
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS
            )

            if groq_result.get("error"):
                logger.error(f"Groq API error: {groq_result.get('response')}")
                # Fall through to fallback
            else:
                return {
                    "answer": groq_result["response"],
                    "sources": sources,
                    "model": groq_result.get("model", GROQ_MODEL),
                    "latency_ms": groq_result.get("generation_ms", 0),
                    "success": True,
                    "llm_used": True,
                }
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Fall through to fallback

    # Try Ollama (local deployment) - default or fallback
    llm = OllamaLLM()
    if llm.is_available():
        from generation.prompt import build_web_prompt_for_ollama
        logger.info("Using Ollama for LLM generation")
        # Use web search prompt (better for web results)
        messages = build_web_prompt_for_ollama(query, results, language)
        llm_result = llm.generate(messages=messages)

        return {
            "answer": llm_result["response"],
            "sources": sources,
            "model": llm_result.get("model", OLLAMA_MODEL),
            "latency_ms": llm_result.get("latency_ms", 0),
            "success": llm_result.get("success", False),
            "llm_used": True,
        }
    else:
        # Fallback: Return retrieved context directly
        logger.warning("No LLM available. Returning raw context.")
        context_summary = "\n\n".join(
            f"[{r.get('source', 'unknown')}]: {r.get('text', '')[:200]}..."
            for r in results[:3]
        )
        return {
            "answer": f"(LLM unavailable — showing retrieved context)\n\n{context_summary}",
            "sources": sources,
            "model": "fallback-context-only",
            "latency_ms": 0,
            "success": True,
            "llm_used": False,
        }
