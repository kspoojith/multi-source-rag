"""
FastAPI Application — Web-Only Multilingual RAG API
====================================================
Production-grade system that answers ANY question using real-time web search.

Architecture:
  User Query → Language Detection → Translation → DuckDuckGo Search
  → Embed Web Results → Hybrid Retrieval → LLM Generation → Answer

Endpoints:
  GET  /          — Interactive web UI
  POST /ask       — Ask any question, get answer from web search
  GET  /health    — System status
  GET  /cache     — Cache statistics

Key Features:
  ✅ Unlimited knowledge via web search (no local files needed)
  ✅ Smart query caching for popular questions (<1s responses)
  ✅ Multilingual: Hindi, English, Telugu, code-mixed
  ✅ Source citations with clickable URLs
  ✅ Production-ready with proper error handling
"""

import sys
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ─── Local imports ────────────────────────────────────────────────────────────
from backend.config import (
    API_HOST, API_PORT,
    RETRIEVAL_TOP_K, SEMANTIC_WEIGHT_ALPHA, KEYWORD_WEIGHT_BETA,
    WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_REGION,
)
from ingestion.embedder import EmbeddingModel, get_embedding_model
from ingestion.web_search import search_and_prepare
from processing.language_detect import detect_language, get_language_label
from processing.normalize import normalize_query
from processing.translate import get_search_query
from retrieval.search import hybrid_search, deduplicate_results
from retrieval.rerank import rerank_results
from generation.llm import generate_answer

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Simple In-Memory Cache ───────────────────────────────────────────────────
# For production, replace with Redis
class SimpleCache:
    """Simple in-memory cache with TTL for query results."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str) -> str:
        """Create cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        key = self._hash_query(query)
        
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires_at"]:
                self.hits += 1
                logger.info(f"Cache HIT for query: {query[:50]}...")
                return entry["data"]
            else:
                # Expired - remove it
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, query: str, data: Dict[str, Any]):
        """Cache a response."""
        key = self._hash_query(query)
        self.cache[key] = {
            "data": data,
            "expires_at": datetime.now() + timedelta(seconds=self.ttl),
            "cached_at": datetime.now().isoformat(),
        }
        logger.info(f"Cached response for: {query[:50]}...")
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "total_queries": total,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cached_entries": len(self.cache),
            "ttl_seconds": self.ttl,
        }

# ─── Global State ─────────────────────────────────────────────────────────────
embedding_model: Optional[EmbeddingModel] = None
query_cache = SimpleCache(ttl_seconds=3600)  # 1 hour cache


# ─── Lifespan Context Manager ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    global embedding_model
    logger.info("🚀 Starting Web-Only Multilingual RAG System...")
    logger.info("📝 Mode: Web Search Only (No local file storage)")
    
    # Load embedding model
    logger.info("📦 Loading embedding model...")
    embedding_model = get_embedding_model()
    logger.info("✅ System ready - answer ANY question from the web!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")
    logger.info(f"📊 Cache stats: {query_cache.stats()}")


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multilingual RAG System (Web-Only)",
    description=(
        "Production-grade RAG system that answers ANY question using real-time web search. "
        "Supports code-mixed Indian languages (Hindi-English, Telugu, etc.)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request model for web search QA."""
    query: str = Field(..., description="The question to ask (any language)", min_length=1)
    top_k: int = Field(default=RETRIEVAL_TOP_K, description="Number of results to use")
    max_web_results: int = Field(
        default=WEB_SEARCH_MAX_RESULTS,
        description="Number of web results to fetch"
    )
    use_cache: bool = Field(default=True, description="Use cached responses if available")


class AskResponse(BaseModel):
    """Response model for web search QA."""
    answer: str
    sources: list
    web_urls: list
    query_info: dict
    search_info: dict
    generation_info: dict
    from_cache: bool = False


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_loaded: bool
    mode: str
    cache_stats: dict


# ───────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ───────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive web UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌐 Multilingual Q&A - Web Search</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px; margin: 0 auto;
            background: white; border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 {
            text-align: center; font-size: 2rem; margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center; color: #64748b; font-size: 0.95rem;
            margin-bottom: 30px;
        }
        .status-bar {
            display: flex; gap: 12px; justify-content: center;
            margin-bottom: 24px; flex-wrap: wrap;
        }
        .badge {
            padding: 6px 14px; border-radius: 20px; font-size: 0.8rem;
            background: #f1f5f9; border: 1px solid #e2e8f0; color: #475569;
        }
        .badge.ok { background: #dcfce7; border-color: #86efac; color: #166534; }
        .search-box {
            display: flex; gap: 10px; margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1; padding: 16px 20px; border-radius: 12px;
            border: 2px solid #e2e8f0; font-size: 1rem;
            outline: none; transition: border-color 0.2s;
        }
        input[type="text"]:focus { border-color: #667eea; }
        button {
            padding: 16px 32px; border-radius: 12px; border: none;
            font-size: 1rem; font-weight: 600; cursor: pointer;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .samples {
            display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px;
        }
        .sample {
            padding: 8px 14px; border-radius: 8px; font-size: 0.85rem;
            background: #f1f5f9; border: 1px solid #e2e8f0; color: #475569;
            cursor: pointer; transition: all 0.2s;
        }
        .sample:hover {
            background: #667eea; color: white; border-color: #667eea;
        }
        .result-card {
            background: #f8fafc; border-radius: 12px;
            padding: 24px; margin-bottom: 16px;
        }
        .result-card h3 {
            color: #667eea; font-size: 0.9rem; margin-bottom: 12px;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .answer-text {
            font-size: 1.1rem; line-height: 1.8; color: #1e293b;
            white-space: pre-wrap; margin-bottom: 16px;
        }
        .cache-tag {
            display: inline-block; padding: 4px 10px; border-radius: 6px;
            font-size: 0.75rem; background: #dbeafe; color: #1e40af;
            margin-bottom: 12px;
        }
        .web-url {
            display: inline-block; padding: 6px 12px; border-radius: 6px;
            font-size: 0.8rem; background: #e0e7ff; color: #4338ca;
            text-decoration: none; margin: 4px; transition: all 0.2s;
        }
        .web-url:hover { background: #667eea; color: white; }
        .source-tag {
            display: inline-block; padding: 4px 10px; border-radius: 6px;
            font-size: 0.8rem; background: #dcfce7; color: #166534;
            margin: 4px;
        }
        .meta-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; margin-top: 12px;
        }
        .meta-item {
            padding: 10px; background: white; border-radius: 8px;
            font-size: 0.85rem;
        }
        .meta-label { color: #64748b; font-size: 0.75rem; }
        .meta-value { color: #1e293b; font-weight: 600; margin-top: 4px; }
        .spinner {
            display: inline-block; width: 18px; height: 18px;
            border: 2px solid #e2e8f0; border-top-color: #667eea;
            border-radius: 50%; animation: spin 0.8s linear infinite;
            vertical-align: middle; margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        #results { min-height: 40px; }
        .error {
            color: #dc2626; padding: 16px; background: #fee2e2;
            border-radius: 12px; border-left: 4px solid #dc2626;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🌐 Ask Anything</h1>
    <p class="subtitle">
        Multilingual Q&A powered by Web Search • Hindi • English • తెలుగు • Code-Mixed
    </p>

    <div class="status-bar" id="statusBar">
        <span class="badge" id="badgeModel">⏳ Loading...</span>
        <span class="badge" id="badgeCache">📊 Cache: 0 hits</span>
    </div>

    <div class="search-box">
        <input
            type="text"
            id="queryInput"
            placeholder="Ask anything... e.g. Elon Musk ki net worth kitni hai?"
            onkeypress="if(event.key==='Enter') ask()"
        />
        <button id="askBtn" onclick="ask()">🔍 Search</button>
    </div>

    <div class="samples">
        <span class="sample" onclick="setQuery(this)">Elon Musk ki net worth kitni hai?</span>
        <span class="sample" onclick="setQuery(this)">IPL 2024 winner kaun hai?</span>
        <span class="sample" onclick="setQuery(this)">Latest AI news kya hai?</span>
        <span class="sample" onclick="setQuery(this)">Python vs JavaScript comparison</span>
        <span class="sample" onclick="setQuery(this)">India ka GDP 2026 mein kitna hai?</span>
        <span class="sample" onclick="setQuery(this)">Climate change kya hai?</span>
    </div>

    <div id="results"></div>
</div>

<script>
const API = '';

function setQuery(el) {
    document.getElementById('queryInput').value = el.textContent;
    ask();
}

async function checkHealth() {
    try {
        const r = await fetch(API + '/health');
        const d = await r.json();
        const bm = document.getElementById('badgeModel');
        const bc = document.getElementById('badgeCache');
        
        bm.textContent = d.model_loaded ? '✅ Ready' : '❌ Model Error';
        bm.className = 'badge ' + (d.model_loaded ? 'ok' : '');
        
        const stats = d.cache_stats || {};
        bc.textContent = `📊 Cache: ${stats.cache_hits || 0} hits (${stats.hit_rate_percent || 0}%)`;
        bc.className = 'badge ok';
    } catch(e) {
        document.getElementById('badgeModel').textContent = '❌ Server error';
    }
}

async function ask() {
    const q = document.getElementById('queryInput').value.trim();
    if (!q) return;
    
    const btn = document.getElementById('askBtn');
    btn.disabled = true;
    
    document.getElementById('results').innerHTML =
        '<div class="result-card"><span class="spinner"></span> Searching the web &amp; generating answer...</div>';
    
    try {
        const r = await fetch(API + '/ask', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({query: q})
        });
        const d = await r.json();
        
        if (!r.ok) throw new Error(d.detail || 'Request failed');
        
        renderAnswer(d);
        checkHealth(); // Update cache stats
    } catch(e) {
        document.getElementById('results').innerHTML =
            '<div class="error">❌ ' + e.message + '</div>';
    }
    
    btn.disabled = false;
}

function renderAnswer(d) {
    const gi = d.generation_info || {};
    const qi = d.query_info || {};
    const si = d.search_info || {};
    
    const cacheTag = d.from_cache ?
        '<div class="cache-tag">⚡ Served from cache (&lt;1s)</div>' : '';
    
    const sources = (d.sources||[]).map(s =>
        '<span class="source-tag">🌐 ' + escHtml(s) + '</span>'
    ).join('');
    
    const urls = (d.web_urls||[]).map(u =>
        '<a class="web-url" href="' + u + '" target="_blank" rel="noopener">' +
        u.substring(0,50) + (u.length>50?'...':'') + '</a>'
    ).join('');
    
    document.getElementById('results').innerHTML =
        '<div class="result-card">' +
        cacheTag +
        '<h3>Answer</h3>' +
        '<div class="answer-text">' + escHtml(d.answer) + '</div>' +
        (sources ? '<h3>Sources</h3><div>' + sources + '</div>' : '') +
        (urls ? '<h3>References</h3><div>' + urls + '</div>' : '') +
        '</div>' +
        '<div class="result-card">' +
        '<h3>Query Analysis</h3>' +
        '<div class="meta-grid">' +
            meta('Language', qi.language_label || 'Unknown') +
            meta('Original', (qi.original||'').substring(0,40)) +
            meta('English Query', qi.english_query || '-') +
        '</div>' +
        '</div>' +
        '<div class="result-card">' +
        '<h3>Performance</h3>' +
        '<div class="meta-grid">' +
            meta('Total Time', (gi.total_ms||0).toFixed(0) + 'ms') +
            meta('Web Search', (gi.web_search_ms||0).toFixed(0) + 'ms') +
            meta('LLM Generation', (gi.generation_ms||0).toFixed(0) + 'ms') +
            meta('Results Found', si.web_results_fetched || 0) +
        '</div>' +
        '</div>';
}

function meta(label, value) {
    return '<div class="meta-item">' +
           '<div class="meta-label">' + label + '</div>' +
           '<div class="meta-value">' + (value ?? '-') + '</div>' +
           '</div>';
}

function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// Check health on load
checkHealth();
setInterval(checkHealth, 30000); // Update every 30s
</script>
</body>
</html>
"""


@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    return HealthResponse(
        status="ok",
        model_loaded=embedding_model is not None,
        mode="web_search_only",
        cache_stats=query_cache.stats(),
    )


@app.get("/cache")
async def cache_stats():
    """Get cache statistics and management."""
    return {
        "stats": query_cache.stats(),
        "note": "For production, replace with Redis for distributed caching",
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear the entire cache."""
    query_cache.clear()
    return {"status": "cache_cleared"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Answer any question using real-time web search.
    
    Flow:
    1. Check cache for this exact query
    2. If cached → return instantly
    3. If not cached:
       - Detect language & normalize query
       - Translate to English for better search
       - Search DuckDuckGo
       - Embed web results
       - Hybrid retrieval
       - LLM generation
       - Cache the result
    4. Return answer with sources
    """
    start_time = time.time()
    timing = {}
    
    # Check cache first
    if request.use_cache:
        cached = query_cache.get(request.query)
        if cached:
            logger.info(f"✅ Returning cached response for: {request.query[:50]}...")
            cached["from_cache"] = True
            cached["generation_info"]["total_ms"] = (time.time() - start_time) * 1000
            return AskResponse(**cached)
    
    try:
        # Step 1: Language detection & normalization
        t0 = time.time()
        lang_info = detect_language(request.query)
        lang_code = lang_info["language"]
        lang_label = get_language_label(lang_code)
        confidence = lang_info["confidence"]
        
        normalized_query = normalize_query(request.query, lang_code)
        timing["normalization_ms"] = (time.time() - t0) * 1000
        
        logger.info(
            f"[Web] Query: '{request.query[:50]}...' → "
            f"lang={lang_code}, normalized='{normalized_query[:50]}...'"
        )
        
        # Step 2: Translate to English for search
        t0 = time.time()
        english_query = get_search_query(normalized_query, lang_code)
        timing["translation_ms"] = (time.time() - t0) * 1000
        
        logger.info(f"[Web] Search query: '{english_query}'")
        
        # Step 3: Web search
        t0 = time.time()
        web_results, web_chunks, web_urls = search_and_prepare(
            query=english_query,
            max_results=request.max_web_results,
            embedding_model=embedding_model,
        )
        timing["web_search_ms"] = (time.time() - t0) * 1000
        
        if not web_chunks:
            raise HTTPException(
                status_code=500,
                detail="Web search failed or returned no results"
            )
        
        logger.info(f"[Web] Got {len(web_chunks)} web chunks")
        
        # Step 4: Hybrid search within web results
        t0 = time.time()
        hybrid_results = hybrid_search(
            semantic_results=web_chunks,
            query=normalized_query,
            alpha=request.alpha if hasattr(request, 'alpha') else SEMANTIC_WEIGHT_ALPHA,
            beta=request.beta if hasattr(request, 'beta') else KEYWORD_WEIGHT_BETA,
        )
        
        # Rerank
        reranked_results = rerank_results(
            results=hybrid_results,
            query=normalized_query,
            language=lang_code,
        )
        
        # Take top-K
        final_results = reranked_results[:request.top_k]
        timing["embedding_ms"] = (time.time() - t0) * 1000  # Includes search+rerank
        
        # Step 5: LLM generation
        t0 = time.time()
        answer, gen_info = generate_answer(
            query=request.query,
            results=final_results,
            language=lang_code,
        )
        timing["generation_ms"] = gen_info.get("latency_ms", 0)
        
        # Total timing
        timing["total_ms"] = (time.time() - start_time) * 1000
        
        # Extract sources
        sources = list(set(r.get("source", "unknown") for r in final_results))
        
        # Build response
        response_data = {
            "answer": answer,
            "sources": sources,
            "web_urls": web_urls,
            "query_info": {
                "original": request.query,
                "normalized": normalized_query,
                "english_query": english_query,
                "language_label": lang_label,
                "confidence": confidence,
            },
            "search_info": {
                "web_results_fetched": len(web_results),
                "semantic_results": len(web_chunks),
                "hybrid_results": len(hybrid_results),
                "final_results": len(final_results),
            },
            "generation_info": {
                **gen_info,
                **timing,
            },
            "from_cache": False,
        }
        
        # Cache successful responses
        if request.use_cache and gen_info.get("success") and not gen_info.get("error"):
            query_cache.set(request.query, response_data)
        
        return AskResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
