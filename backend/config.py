"""
Central Configuration for Multilingual RAG System
===================================================
Every tunable parameter lives here. This makes experiments reproducible
and lets us run ablation studies by simply changing one value.

Key design decisions documented inline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load Environment Variables ──────────────────────────────────────────────
# Load .env file if it exists (for local development)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from: {env_path}")
else:
    print(f"ℹ️  No .env file found at: {env_path} (using system environment variables)")

# ─── Paths ────────────────────────────────────────────────────────────────────
# All paths are relative to the project root so the system is portable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Embedding Model ─────────────────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2 supports 50+ languages including
# Hindi, Telugu, Tamil, Bengali, etc. It produces 384-dim vectors.
# Research note: We chose this over larger models (XLM-R, LaBSE) because
# it offers the best latency-quality tradeoff for CPU-only deployment.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384  # Output vector size of the chosen model

# ─── Chunking ────────────────────────────────────────────────────────────────
# CPU-optimized chunking: smaller chunks = less text for LLM
CHUNK_SIZE = 300          # Reduced from 500 for speed
CHUNK_OVERLAP = 40        # Reduced from 50
MIN_CHUNK_LENGTH = 30     # discard very short chunks (headings, noise)

# ─── FAISS Index ──────────────────────────────────────────────────────────────
# IndexFlatIP = brute-force inner product (cosine sim on normalized vectors).
# For <50k docs this is fast enough on CPU. Beyond that, switch to IVF.
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_METADATA_PATH = INDEX_DIR / "metadata.pkl"
FAISS_INDEX_TYPE = "FlatIP"  # Options: FlatIP, IVFFlat

# ─── Retrieval ────────────────────────────────────────────────────────────────
# Hybrid scoring: Final = α * semantic_score + β * keyword_boost
# Optimized for phi3:mini on CPU
RETRIEVAL_TOP_K = 3                # 3 chunks for better context
SEMANTIC_WEIGHT_ALPHA = 0.7        # Weight for cosine similarity
KEYWORD_WEIGHT_BETA = 0.3          # Weight for keyword match boost
SIMILARITY_THRESHOLD = 0.25        # Minimum score to consider relevant

# ─── LLM Configuration ──────────────────────────────────────────────────────
# Supports both Ollama (local) and Groq API (cloud)
# LLM_PROVIDER: "ollama" or "groq"
# - Use "ollama" for local deployment (default)
# - Use "groq" for Streamlit Cloud deployment (free tier available)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "groq"

# ─── Ollama Settings (Local) ─────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT = 300  # seconds (5 minutes for CPU - increased from 180s)
OLLAMA_TEMPERATURE = 0.1  # Low temp = more factual, less creative

# ─── Groq API Settings (Cloud) ───────────────────────────────────────────────
# Get free API key from: https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Set in Streamlit secrets or .env
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Best free tier model
# Options: llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768, gemma-7b-it
GROQ_TEMPERATURE = 0.3  # Slightly higher for better responses
GROQ_MAX_TOKENS = 1024  # Maximum response length

# ─── Language Processing ─────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = ["en", "hi", "te", "ta", "bn", "mixed"]
# Transliteration mappings for common Romanized Hindi patterns
ENABLE_TRANSLITERATION = True
ENABLE_STOPWORD_REMOVAL = True

# ─── Web Search (Open-Domain QA) ─────────────────────────────────────────────
# DuckDuckGo-based web search for answering ANY question in the world.
# AGGRESSIVE CPU OPTIMIZATION: Minimal results
WEB_SEARCH_MAX_RESULTS = 4     # MINIMUM: 4 results (was 6)
WEB_SEARCH_REGION = "wt-wt"    # Worldwide
WEB_SEARCH_ENABLED = True      # Master toggle
WEB_SEARCH_TIMEOUT = 10        # seconds

# ─── API Server ───────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_QUERIES_PATH = DATA_DIR / "eval_queries.json"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.json"
