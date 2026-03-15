"""
Streamlit Cloud - Standalone Multilingual RAG System
====================================================
Single-process multilingual Q&A system for Streamlit Cloud deployment.
Runs RAG pipeline in-process (no separate backend required).

Features:
- Real-time web search with DuckDuckGo
- Multilingual: Hindi, English, Telugu, code-mixed
- Smart caching for faster responses
- Source citations with clickable URLs
- Groq API for fast LLM generation (1-3s)
"""

import streamlit as st
import time
import hashlib
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import RAG pipeline components
from ingestion.embedder import get_embedding_model
from ingestion.web_search import search_and_prepare
from processing.language_detect import detect_language, get_language_label
from processing.normalize import normalize_query
from processing.translate import get_search_query
from retrieval.search import hybrid_search, deduplicate_results
from retrieval.rerank import rerank_results
from generation.llm import generate_answer
from backend.config import (
    RETRIEVAL_TOP_K, SEMANTIC_WEIGHT_ALPHA, KEYWORD_WEIGHT_BETA,
    WEB_SEARCH_MAX_RESULTS
)

# ─── Simple Cache ──────────────────────────────────────────────────────────────
class SimpleCache:
    """In-memory cache with TTL for query results."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._hash_query(query)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires_at"]:
                self.hits += 1
                return entry["data"]
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, query: str, data: Dict[str, Any]):
        key = self._hash_query(query)
        self.cache[key] = {
            "data": data,
            "expires_at": datetime.now() + timedelta(seconds=self.ttl),
            "cached_at": datetime.now()
        }
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "total_queries": total,
            "hit_rate_percent": hit_rate,
            "cached_entries": len(self.cache)
        }


# ─── Initialize Global Resources ──────────────────────────────────────────────
@st.cache_resource
def initialize_system():
    """Initialize embedding model and cache (cached across reruns)."""
    try:
        model = get_embedding_model()
        cache = SimpleCache(ttl_seconds=3600)
        return {
            "model": model,
            "cache": cache,
            "status": "ready",
            "error": None
        }
    except Exception as e:
        return {
            "model": None,
            "cache": SimpleCache(ttl_seconds=3600),
            "status": "error",
            "error": str(e)
        }


# ─── RAG Pipeline ──────────────────────────────────────────────────────────────
def process_query(system, query: str, use_cache: bool = True) -> Dict[str, Any]:
    """Process query through RAG pipeline."""
    start_time = time.time()
    timings = {}
    
    # Check cache first
    if use_cache:
        cached = system["cache"].get(query)
        if cached:
            cached["was_cached"] = True
            cached["total_time_seconds"] = time.time() - start_time
            return cached
    
    try:
        # 1. Query Normalization & Language Detection
        t0 = time.time()
        norm_result = normalize_query(query)
        normalized_query = norm_result["normalized"]
        lang_code = norm_result["language"]
        lang_label = get_language_label(lang_code)
        timings["normalization"] = time.time() - t0
        
        # 2. Translation for Search
        t0 = time.time()
        english_query = get_search_query(normalized_query, lang_code)
        timings["translation"] = time.time() - t0
        
        # 3. Web Search
        t0 = time.time()
        web_chunks = search_and_prepare(
            query=normalized_query,
            english_query=english_query,
            max_results=WEB_SEARCH_MAX_RESULTS
        )
        timings["web_search"] = time.time() - t0
        
        if not web_chunks:
            return {
                "error": True,
                "message": "No web results found for your query.",
                "query": query
            }
        
        # 4. Embedding
        t0 = time.time()
        # Embed query
        query_embedding = system["model"].embed([normalized_query])[0]
        
        # Embed chunks
        chunk_texts = [chunk["text"] for chunk in web_chunks]
        chunk_embeddings = system["model"].embed(chunk_texts)
        
        # Add similarity scores to chunks
        for chunk, chunk_embedding in zip(web_chunks, chunk_embeddings):
            similarity = float(np.dot(query_embedding, chunk_embedding))
            chunk["score"] = similarity
        
        timings["embedding"] = time.time() - t0
        
        # 5. Hybrid Search
        t0 = time.time()
        hybrid_results = hybrid_search(
            semantic_results=web_chunks,
            query=normalized_query,
            alpha=SEMANTIC_WEIGHT_ALPHA,
            beta=KEYWORD_WEIGHT_BETA,
        )
        
        # 6. Reranking
        reranked_results = rerank_results(
            results=hybrid_results,
            query_topic=None,
            max_per_source=3,
            top_k=RETRIEVAL_TOP_K
        )
        
        timings["retrieval_rerank"] = time.time() - t0
        
        # 7. LLM Generation
        t0 = time.time()
        gen_result = generate_answer(
            query=query,
            results=reranked_results,
            language=lang_code
        )
        timings["llm_generation"] = time.time() - t0
        
        # Extract web URLs
        web_urls = [chunk.get("url", "") for chunk in web_chunks if chunk.get("url")]
        
        # Prepare response
        total_time = time.time() - start_time
        response = {
            "answer": gen_result["answer"],
            "sources": [
                {
                    "title": result.get("source", "Web"),
                    "url": result.get("url", ""),
                    "relevance_score": result.get("final_score", 0.0)
                }
                for result in reranked_results
            ],
            "query_analysis": {
                "original_query": query,
                "normalized_query": normalized_query,
                "detected_language": lang_label,
                "search_query": english_query,
                "total_web_results": len(web_chunks),
                "retrieved_docs": len(reranked_results)
            },
            "performance": {
                "total_time_seconds": total_time,
                **timings
            },
            "was_cached": False,
            "error": False
        }
        
        # Cache the result
        if use_cache:
            system["cache"].set(query, response)
        
        return response
        
    except Exception as e:
        return {
            "error": True,
            "message": f"Error processing query: {str(e)}",
            "query": query,
            "total_time_seconds": time.time() - start_time
        }


# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌐 Multilingual Q&A",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        padding: 2.5rem;
        border-radius: 16px;
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-left: 6px solid #10b981;
        font-size: 1.2rem;
        line-height: 2;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        color: #1f2937;
        font-weight: 400;
    }
    .source-tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background: #dcfce7;
        color: #166534;
        margin: 0.3rem;
        font-size: 0.95rem;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .cache-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .status-online {
        color: #15803d;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: #dcfce7;
        border-radius: 8px;
        display: inline-block;
    }
    .performance-stage {
        padding: 1rem;
        border-radius: 8px;
        background: #f9fafb;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize System ─────────────────────────────────────────────────────────
system = initialize_system()

# ─── Initialize Session State ──────────────────────────────────────────────────
if "use_cache" not in st.session_state:
    st.session_state.use_cache = True

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎛️ Control Panel")
    st.markdown("---")
    
    # System Status
    st.markdown("### ⚙️ System Status")
    if system["status"] == "ready":
        st.markdown('<div class="status-online">✅ System Ready</div>', unsafe_allow_html=True)
        st.caption("**Mode:** Groq Cloud + Web Search")
    else:
        st.error(f"❌ Error: {system['error']}")
    
    st.markdown("---")
    
    # Cache Statistics
    st.markdown("### 📊 Cache Statistics")
    cache_stats = system["cache"].stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cache Hits", cache_stats["cache_hits"])
    with col2:
        st.metric("Hit Rate", f"{cache_stats['hit_rate_percent']:.1f}%")
    
    st.caption(f"📝 **Total Queries:** {cache_stats['total_queries']}")
    st.caption(f"💾 **Cached Entries:** {cache_stats['cached_entries']}")
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        system["cache"].clear()
        st.success("✅ Cache cleared!")
        time.sleep(0.5)
        st.rerun()
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings", expanded=True):
        st.session_state.use_cache = st.checkbox(
            "Enable Caching",
            value=st.session_state.use_cache,
            help="Cache responses for 1 hour"
        )
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    **Tech Stack:**
    - 🤖 Groq API (llama-3.3-70b)
    - 🔍 DuckDuckGo Search
    - 🧠 Sentence Transformers
    - ⚡ FAISS Vector Search
    """)

# ─── Main Interface ────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🌐 Ask Anything</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multilingual Q&A powered by Web Search • Hindi • English • తెలుగు • Code-Mixed</p>',
    unsafe_allow_html=True
)

# Example Queries
st.markdown("### 💡 Try These Example Queries")
example_cols = st.columns(3)

examples = [
    "Elon Musk ki net worth kitni hai?",
    "IPL 2024 winner kaun hai?",
    "Latest AI news kya hai?",
    "India ka prime minister kon hai?",
    "Climate change kya hai?",
    "Python vs JavaScript comparison"
]

for idx, example in enumerate(examples):
    with example_cols[idx % 3]:
        if st.button(example, use_container_width=True, key=f"ex_{idx}"):
            st.session_state.query_input = example

# Query Input
query = st.text_input(
    "🔍 Your Question:",
    placeholder="Ask anything in English, Hindi, Telugu, or code-mixed...",
    key="query_input",
    label_visibility="collapsed"
)

# Process Query
if query:
    if system["status"] != "ready":
        st.error("❌ System not ready. Check error in sidebar.")
    else:
        with st.spinner("🔍 Searching the web and generating answer..."):
            result = process_query(system, query, st.session_state.use_cache)
        
        if result.get("error"):
            st.error(f"❌ {result.get('message', 'Unknown error')}")
        else:
            # Show cached badge
            if result.get("was_cached"):
                st.markdown(
                    '<div class="cache-badge">⚡ Cached Response (Instant)</div>',
                    unsafe_allow_html=True
                )
            
            # Display Answer
            st.markdown("### 💬 Answer")
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True
            )
            
            # Sources
            st.markdown("### 📚 Sources")
            for idx, source in enumerate(result["sources"], 1):
                st.markdown(
                    f'<span class="source-tag">{idx}. <a href="{source["url"]}" target="_blank">{source["title"]}</a></span>',
                    unsafe_allow_html=True
                )
            
            # Performance Metrics
            with st.expander("⏱️ Performance Metrics"):
                perf = result["performance"]
                st.markdown(f"**Total Time:** {perf['total_time_seconds']:.2f}s")
                
                stages = [
                    ("Query Normalization & Language Detection", perf.get("normalization", 0)),
                    ("Translation", perf.get("translation", 0)),
                    ("Web Search", perf.get("web_search", 0)),
                    ("Embedding", perf.get("embedding", 0)),
                    ("Retrieval", perf.get("retrieval", 0)),
                    ("Reranking", perf.get("reranking", 0)),
                    ("LLM Generation", perf.get("llm_generation", 0))
                ]
                
                for stage, timing in stages:
                    st.markdown(
                        f'<div class="performance-stage"><strong>{stage}:</strong> {timing:.3f}s</div>',
                        unsafe_allow_html=True
                    )
            
            # Query Analysis
            with st.expander("📊 Query Analysis"):
                analysis = result["query_analysis"]
                st.json(analysis)
