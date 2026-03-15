"""
Streamlit Cloud - Standalone Multilingual RAG System
====================================================
Beautiful web interface running in-process (no backend required).
Optimized for Streamlit Community Cloud deployment.

Features:
- Real-time web search with DuckDuckGo
- Multilingual: Hindi, English, Telugu, code-mixed
- Groq API for fast LLM generation (1-3s)
- Smart caching for faster responses
- Source citations with clickable URLs
- Detailed performance metrics and analysis
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
from retrieval.search import hybrid_search
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
    timing = {}
    
    # Check cache first
    if use_cache:
        cached = system["cache"].get(query)
        if cached:
            cached["from_cache"] = True
            cached["total_time_seconds"] = time.time() - start_time
            return cached
    
    try:
        # 1. Query Normalization & Language Detection
        t0 = time.time()
        norm_result = normalize_query(query)
        normalized_query = norm_result["normalized"]
        lang_code = norm_result["language"]
        lang_label = get_language_label(lang_code)
        timing["normalization_ms"] = (time.time() - t0) * 1000
        
        # 2. Translation for Search
        t0 = time.time()
        english_query = get_search_query(normalized_query, lang_code)
        timing["translation_ms"] = (time.time() - t0) * 1000
        
        # 3. Web Search
        t0 = time.time()
        web_chunks = search_and_prepare(
            query=normalized_query,
            english_query=english_query,
            max_results=WEB_SEARCH_MAX_RESULTS
        )
        timing["web_search_ms"] = (time.time() - t0) * 1000
        
        if not web_chunks:
            return {
                "error": True,
                "message": "No web results found for your query.",
                "query": query
            }
        
        # 4. Embedding
        t0 = time.time()
        query_embedding = system["model"].embed([normalized_query])[0]
        chunk_texts = [chunk["text"] for chunk in web_chunks]
        chunk_embeddings = system["model"].embed(chunk_texts)
        
        # Add similarity scores
        for chunk, chunk_embedding in zip(web_chunks, chunk_embeddings):
            similarity = float(np.dot(query_embedding, chunk_embedding))
            chunk["score"] = similarity
        
        timing["embedding_ms"] = (time.time() - t0) * 1000
        
        # 5. Hybrid Search & Reranking
        t0 = time.time()
        hybrid_results = hybrid_search(
            semantic_results=web_chunks,
            query=normalized_query,
            alpha=SEMANTIC_WEIGHT_ALPHA,
            beta=KEYWORD_WEIGHT_BETA,
        )
        
        reranked_results = rerank_results(
            results=hybrid_results,
            query_topic=None,
            max_per_source=3,
            top_k=RETRIEVAL_TOP_K
        )
        timing["search_rerank_ms"] = (time.time() - t0) * 1000
        
        # 6. LLM Generation
        t0 = time.time()
        gen_result = generate_answer(
            query=query,
            results=reranked_results,
            language=lang_code
        )
        timing["generation_ms"] = gen_result.get("latency_ms", 0)
        
        # Extract web URLs
        web_urls = [chunk.get("url", "") for chunk in web_chunks if chunk.get("url")]
        sources = list(set(result.get("source", "unknown") for result in reranked_results))
        
        # Total timing
        timing["total_ms"] = (time.time() - start_time) * 1000
        
        # Build response matching backend format
        response = {
            "answer": gen_result["answer"],
            "sources": sources,
            "web_urls": web_urls,
            "query_info": {
                "original": query,
                "normalized": normalized_query,
                "transliterated": norm_result.get("transliterated"),
                "english_query": english_query,
                "language_label": lang_label,
                "confidence": norm_result.get("confidence", 1.0),
            },
            "search_info": {
                "web_results_fetched": len(web_chunks),
                "semantic_results": len(web_chunks),
                "hybrid_results": len(hybrid_results),
                "final_results": len(reranked_results),
            },
            "generation_info": {
                **gen_result,
                **timing,
            },
            "from_cache": False,
            "error": False,
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


# ═══════════════════════════════════════════════════════════════════════════════
# UI CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🌐 Multilingual Q&A",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .info-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 1rem 0;
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
    .answer-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #10b981;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f3f4f6;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .answer-content {
        color: #374151;
        font-size: 1.15rem;
        line-height: 1.9;
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
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #374151;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .status-online {
        color: #15803d;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: #dcfce7;
        border-radius: 8px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize System ─────────────────────────────────────────────────────────
system = initialize_system()

# ─── Initialize Session State ──────────────────────────────────────────────────
if "use_cache" not in st.session_state:
    st.session_state.use_cache = True
if "show_advanced" not in st.session_state:
    st.session_state.show_advanced = True
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🎛️ Control Panel")
    st.markdown("---")
    
    # System Status
    st.markdown("### ⚙️ System Status")
    if system["status"] == "ready":
        st.markdown('<div class="status-online">✅ System Ready</div>', unsafe_allow_html=True)
        st.caption("**Mode:** Groq Cloud + Web Search")
        st.caption("**Deployment:** Streamlit Cloud")
    else:
        st.error(f"❌ Error: {system['error']}")
    
    st.markdown("---")
    
    # Cache Statistics
    st.markdown("### 📊 Cache Statistics")
    cache_stats = system["cache"].stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Cache Hits",
            cache_stats["cache_hits"],
            help="Queries served from cache"
        )
    with col2:
        hit_rate = cache_stats["hit_rate_percent"]
        st.metric(
            "Hit Rate",
            f"{hit_rate:.1f}%",
            delta="Good" if hit_rate > 50 else "Low"
        )
    
    st.caption(f"📝 **Total Queries:** {cache_stats['total_queries']}")
    st.caption(f"💾 **Cached Entries:** {cache_stats['cached_entries']}")
    st.caption(f"⏱️ **Cache TTL:** 1 hour")
    
    if st.button("🗑️ Clear Cache", use_container_width=True, type="secondary"):
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
        st.session_state.show_debug = st.checkbox(
            "Debug Mode",
            value=st.session_state.show_debug,
            help="Show detailed JSON response"
        )
        st.session_state.show_advanced = st.checkbox(
            "Advanced Metrics",
            value=st.session_state.show_advanced,
            help="Display detailed performance breakdown"
        )
    
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About System"):
        st.markdown("**Multilingual RAG v2.1**")
        st.caption("🌐 Cloud Deployment")
        st.caption("🔍 Real-time DuckDuckGo Search")
        st.caption("🌏 Hindi • English • తెలుగు")
        st.caption("💬 Code-Mixed Queries")
        st.markdown("")
        st.markdown("**Tech Stack:**")
        st.caption("• Streamlit Cloud")
        st.caption("• Groq API (llama-3.3-70b)")
        st.caption("• Sentence Transformers")
        st.caption("• FAISS Vector Search")
        st.caption("• DuckDuckGo Search")
    
    st.markdown("---")
    st.caption("💡 **Tip:** Use example queries to get started!")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🌐 Ask Anything</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multilingual Q&A powered by Web Search • '
    'Hindi • English • తెలుగు • Code-Mixed</p>',
    unsafe_allow_html=True
)

# Example queries
st.markdown('<div class="section-header">💡 Try These Example Queries</div>', unsafe_allow_html=True)

example_queries = [
    "Elon Musk ki net worth kitni hai?",
    "IPL 2024 winner kaun hai?",
    "Latest AI news kya hai?",
    "India ka prime minister kon hai?",
    "Climate change kya hai?",
    "Python vs JavaScript comparison"
]

# Display in 3 columns
example_cols = st.columns(3)
for idx, example in enumerate(example_queries):
    with example_cols[idx % 3]:
        if st.button(example, use_container_width=True, key=f"ex_{idx}"):
            st.session_state.query_input = example
            st.rerun()

st.markdown("")

# Query Input
query = st.text_input(
    "🔍 Ask your question:",
    placeholder="Type your question in English, Hindi, Telugu, or code-mixed...",
    key="query_input"
)

# Process Query
if query:
    if system["status"] != "ready":
        st.error("❌ System not ready. Check error in sidebar.")
    else:
        start_time = time.time()
        
        with st.spinner("🔍 Searching the web and generating answer..."):
            result = process_query(system, query, st.session_state.use_cache)
        
        elapsed = time.time() - start_time
        
        if result.get("error"):
            st.error(f"❌ **Error:** {result.get('message', 'Unknown error')}")
        else:
            # Cache indicator
            if result.get("from_cache"):
                st.markdown(
                    '<div class="cache-badge">⚡ Lightning Fast - Served from Cache (<1s)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info(f"⏱️ Generated in {elapsed:.1f}s (Groq API)")
            
            # Organize results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "💬 Answer",
                "📊 Analysis",
                "📚 Sources & References",
                "⚡ Performance"
            ])
            
            # TAB 1: Answer
            with tab1:
                answer = result.get("answer", "No answer generated")
                
                st.markdown(
                    f'''
                    <div class="answer-box">
                        <div class="answer-header">
                            <span>💬</span>
                            <span>AI-Generated Answer</span>
                        </div>
                        <div class="answer-content">
                            {answer}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                
                st.markdown("")
                
                # Generation details
                gen_info = result.get("generation_info", {})
                st.markdown('<div class="section-header">📊 Generation Details</div>', unsafe_allow_html=True)
                
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric(
                        "🤖 Model Used",
                        gen_info.get("model", "N/A"),
                        help="LLM model used for generation"
                    )
                with info_col2:
                    llm_used = gen_info.get("llm_used", False)
                    llm_status = "✅ Yes" if llm_used else "❌ No"
                    st.metric(
                        "🧠 LLM Generated",
                        llm_status,
                        help="Whether LLM was used"
                    )
                with info_col3:
                    gen_time = gen_info.get("generation_ms", 0) / 1000
                    st.metric(
                        "⏱️ Generation Time",
                        f"{gen_time:.1f}s",
                        help="LLM generation duration"
                    )
            
            # TAB 2: Query Analysis
            with tab2:
                st.markdown('<div class="section-header">🔍 Query Processing</div>', unsafe_allow_html=True)
                
                query_info = result.get("query_info", {})
                search_info = result.get("search_info", {})
                
                # Metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    lang_label = query_info.get("language_label", "Unknown")
                    st.metric("🌐 Language", lang_label)
                with metric_col2:
                    confidence = query_info.get("confidence", 0) * 100
                    st.metric(
                        "✓ Confidence",
                        f"{confidence:.0f}%",
                        delta="High" if confidence > 80 else "Medium"
                    )
                with metric_col3:
                    web_results = search_info.get("web_results_fetched", 0)
                    st.metric("🔍 Web Results", web_results)
                with metric_col4:
                    final_results = search_info.get("final_results", 0)
                    st.metric("✨ Final Results", final_results)
                
                st.markdown("")
                
                # Query transformations
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("**📝 Query Transformations:**")
                st.markdown(f"• **Original Query:** `{query_info.get('original', query)}`")
                
                if query_info.get("normalized"):
                    st.markdown(f"• **Normalized:** `{query_info.get('normalized', '')}`")
                if query_info.get("transliterated"):
                    st.markdown(f"• **Transliterated:** `{query_info.get('transliterated', '')}`")
                
                st.markdown(f"• **English Query:** `{query_info.get('english_query', '')}`")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # TAB 3: Sources and References
            with tab3:
                source_col1, source_col2 = st.columns(2)
                
                with source_col1:
                    st.markdown('<div class="section-header">📚 Source Domains</div>', unsafe_allow_html=True)
                    sources = result.get("sources", [])
                    if sources:
                        sources_html = "".join([
                            f'<span class="source-tag">🌐 {s}</span>'
                            for s in sources
                        ])
                        st.markdown(sources_html, unsafe_allow_html=True)
                    else:
                        st.info("No source domains available")
                
                with source_col2:
                    st.markdown('<div class="section-header">🔗 Reference URLs</div>', unsafe_allow_html=True)
                    urls = result.get("web_urls", [])
                    if urls:
                        for idx, url in enumerate(urls[:4], 1):
                            display_url = url.replace("https://", "").replace("http://", "")
                            if len(display_url) > 60:
                                display_url = display_url[:57] + "..."
                            st.markdown(f"**{idx}.** [{display_url}]({url})")
                    else:
                        st.info("No reference URLs available")
            
            # TAB 4: Performance Metrics
            with tab4:
                if st.session_state.show_advanced:
                    st.markdown('<div class="section-header">⚡ Performance Breakdown</div>', unsafe_allow_html=True)
                    
                    gen_info = result.get("generation_info", {})
                    total_ms = gen_info.get("total_ms", 0)
                    
                    stages = [
                        ("🔄 Normalization", gen_info.get("normalization_ms", 0), "#22d3ee"),
                        ("🌐 Translation", gen_info.get("translation_ms", 0), "#a78bfa"),
                        ("🔍 Web Search", gen_info.get("web_search_ms", 0), "#f97316"),
                        ("🧮 Embedding", gen_info.get("embedding_ms", 0), "#6366f1"),
                        ("🎯 Search & Rerank", gen_info.get("search_rerank_ms", 0), "#22c55e"),
                        ("🤖 LLM Generation", gen_info.get("generation_ms", 0), "#ef4444"),
                    ]
                    
                    # Display metrics
                    perf_cols = st.columns(3)
                    for idx, (stage, time_ms, color) in enumerate(stages):
                        with perf_cols[idx % 3]:
                            percentage = (time_ms / total_ms * 100) if total_ms > 0 else 0
                            st.metric(
                                stage,
                                f"{time_ms:.0f}ms",
                                delta=f"{percentage:.1f}%"
                            )
                    
                    st.markdown("")
                    
                    # Total summary
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    with summary_col1:
                        st.metric(
                            "📊 **Total Processing**",
                            f"{total_ms/1000:.2f}s"
                        )
                    with summary_col2:
                        st.metric(
                            "⏱️ **Actual Time**",
                            f"{elapsed:.2f}s"
                        )
                    with summary_col3:
                        overhead = ((elapsed * 1000) - total_ms)
                        st.metric(
                            "🔧 **Overhead**",
                            f"{overhead:.0f}ms"
                        )
                    
                    # Progress bars
                    st.markdown("")
                    st.markdown("**⏱️ Time Distribution:**")
                    
                    total_stages_ms = sum(time_ms for _, time_ms, _ in stages)
                    
                    for stage, time_ms, color in stages:
                        if time_ms > 0:
                            time_s = time_ms / 1000
                            percentage = (time_ms / total_stages_ms) if total_stages_ms > 0 else 0
                            percentage = min(max(percentage, 0.0), 1.0)
                            st.progress(percentage, text=f"{stage}: {time_s:.2f}s ({percentage*100:.1f}%)")
                else:
                    st.info("Enable 'Advanced Metrics' in sidebar to see detailed breakdown")
            
            # Debug information
            if st.session_state.show_debug:
                st.markdown("---")
                with st.expander("🔧 Debug Information (Full JSON Response)"):
                    st.json(result)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Multilingual RAG System v2.1 | Streamlit Cloud + Groq API</p>
        <p>Zero-Cost Deployment | Real-Time Web Search | Code-Mixed Support</p>
    </div>
    """,
    unsafe_allow_html=True
)
