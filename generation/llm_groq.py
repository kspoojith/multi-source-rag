"""
Groq API Integration for LLM Generation
========================================
Fast, free-tier cloud LLM inference using Groq API.

Models available:
- llama3-70b-8192 (Best quality, recommended)
- llama3-8b-8192 (Faster)
- mixtral-8x7b-32768 (Good balance)
- gemma-7b-it (Fast)

Free Tier: 14,400 requests/day
"""

import os
import logging
from typing import Dict, Any, Optional
import requests
import time

logger = logging.getLogger(__name__)


class GroqLLM:
    """Groq API client for LLM generation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        """
        Initialize Groq API client.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Model name (default: llama3-70b-8192)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set environment variable or pass as argument.")
        
        # Debug: Check API key
        logger.info(f"GroqLLM initialized with key length: {len(self.api_key)} chars, starts: {self.api_key[:10]}...")
        
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.timeout = 60  # Groq is fast, 60s is plenty
        
        logger.info(f"Initialized Groq API with model: {model}")
    
    def is_available(self) -> bool:
        """Check if Groq API is accessible."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            available = response.status_code == 200
            if available:
                logger.info(f"Groq API is available")
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
            return available
        except Exception as e:
            logger.error(f"Groq API health check failed: {e}")
            return False
    
    def generate(
        self, 
        messages: list,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using Groq API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     [{"role": "system", "content": "..."}, ...]
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
        
        Returns:
            Dict with:
                - response: Generated text
                - generation_ms: Time taken in milliseconds
                - model: Model used
                - cached: Always False (no caching in Groq)
        """
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_msg = f"Groq API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "response": f"Error: {error_msg}",
                    "generation_ms": int((time.time() - start_time) * 1000),
                    "model": self.model,
                    "error": True
                }
            
            data = response.json()
            generated_text = data["choices"][0]["message"]["content"]
            generation_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"Groq response generated in {generation_ms}ms")
            
            return {
                "response": generated_text,
                "generation_ms": generation_ms,
                "model": self.model,
                "cached": False,
                "tokens_used": data.get("usage", {}).get("total_tokens", 0)
            }
            
        except requests.Timeout:
            error_msg = f"Groq API timeout after {self.timeout}s"
            logger.error(error_msg)
            return {
                "response": f"Error: {error_msg}",
                "generation_ms": int((time.time() - start_time) * 1000),
                "model": self.model,
                "error": True
            }
        except Exception as e:
            error_msg = f"Groq API error: {str(e)}"
            logger.error(error_msg)
            return {
                "response": f"Error: {error_msg}",
                "generation_ms": int((time.time() - start_time) * 1000),
                "model": self.model,
                "error": True
            }


def generate_answer(
    prompt_messages: list,
    model: str = "llama3-70b-8192",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate answer using Groq API.
    
    Args:
        prompt_messages: Chat messages for the LLM
        model: Groq model name
        api_key: API key (optional, reads from env)
    
    Returns:
        Dict with response and metadata
    """
    llm = GroqLLM(api_key=api_key, model=model)
    return llm.generate(prompt_messages)


# Health check function
def is_available(model: str = "llama3-70b-8192", api_key: Optional[str] = None) -> bool:
    """Check if Groq API is available."""
    try:
        llm = GroqLLM(api_key=api_key, model=model)
        return llm.is_available()
    except Exception as e:
        logger.error(f"Groq availability check failed: {e}")
        return False
