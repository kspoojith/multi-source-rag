"""
Streamlit Frontend for Multilingual RAG System
===============================================
Beautiful web interface for the web-only RAG system.
Connects to the FastAPI backend running on http://localhost:8000

Features:
- Clean, modern UI with Streamlit
- Example queries (clickable)
- Real-time search with loading indicators
- Formatted answer display
- Source citations with clickable URLs
- Query analysis visualization
- Performance metrics breakdown
- Cache statistics
- Dark/light mode support
"""

import streamlit as st
import requests
import time
from typing import Dict, Any, List

# ─── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

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
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .example-query {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.3rem;
        cursor: pointer;
        background: #f8f9fa;
        transition: all 0.2s;
    }
    .example-query:hover {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
    }
    .source-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        background: #dcfce7;
        color: #166534;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .url-link {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        background: #e0e7ff;
        color: #4338ca;
        margin: 0.2rem;
        font-size: 0.85rem;
        text-decoration: none;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #667eea;
        font-size: 1.1rem;
        line-height: 1.8;
        margin: 1rem 0;
    }
    .cache-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        background: #dbeafe;
        color: #1e40af;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────────────

def check_backend_health() -> Dict[str, Any]:
    """Check if backend is running and get status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "model_loaded": False}
    except Exception as e:
        return {"status": "offline", "error": str(e)}


def ask_question(query: str, use_cache: bool = True) -> Dict[str, Any]:
    """Send question to backend API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": query, "use_cache": use_cache},
            timeout=180,  # 3 minute timeout for LLM generation
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": True,
                "message": f"API Error: {response.status_code}",
                "detail": response.text
            }
    except requests.Timeout:
        return {
            "error": True,
            "message": "Request timed out (LLM generation taking too long)",
        }
    except Exception as e:
        return {"error": True, "message": str(e)}


def clear_cache():
    """Clear backend cache."""
    try:
        response = requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/cache", timeout=5)
        if response.status_code == 200:
            return response.json().get("stats", {})
        return {}
    except:
        return {}


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    
    # Check backend health
    health = check_backend_health()
    
    if health.get("status") == "offline":
        st.error("❌ Backend Offline")
        st.caption("Start the backend: `python -m backend.app`")
    elif health.get("model_loaded"):
        st.success("✅ System Ready")
        st.caption(f"Mode: {health.get('mode', 'N/A')}")
    else:
        st.warning("⚠️ Model Loading...")
    
    st.markdown("---")
    
    # Cache statistics
    st.markdown("### 📊 Cache Stats")
    cache_stats = get_cache_stats()
    
    if cache_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hits", cache_stats.get("cache_hits", 0))
        with col2:
            st.metric("Hit Rate", f"{cache_stats.get('hit_rate_percent', 0):.1f}%")
        
        st.caption(f"Total Queries: {cache_stats.get('total_queries', 0)}")
        st.caption(f"Cached Entries: {cache_stats.get('cached_entries', 0)}")
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            if clear_cache():
                st.success("Cache cleared!")
                st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    use_cache = st.checkbox("Use Caching", value=True, help="Cache responses for faster repeated queries")
    show_debug = st.checkbox("Show Debug Info", value=False, help="Show detailed timing breakdown")
    
    st.markdown("---")
    
    # About
    st.markdown("### ℹ️ About")
    st.caption("**Multilingual RAG System v2.0**")
    st.caption("Web-only mode with real-time DuckDuckGo search")
    st.caption("Supports Hindi, English, Telugu & code-mixed queries")
    st.caption("")
    st.caption("Built with FastAPI, Streamlit, phi3:mini LLM")


# ─── Main Content ──────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-header">🌐 Ask Anything</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multilingual Q&A powered by Web Search • '
    'Hindi • English • తెలుగు • Code-Mixed</p>',
    unsafe_allow_html=True
)

# Example queries
st.markdown("#### 💡 Try these examples:")

example_queries = [
    "Elon Musk ki net worth kitni hai?",
    "IPL 2024 winner kaun hai?",
    "Latest AI news kya hai?",
    "India ka prime minister kon hai?",
    "Climate change kya hai?",
    "Python vs JavaScript comparison",
]

# Display examples in columns
cols = st.columns(3)
for idx, query in enumerate(example_queries):
    with cols[idx % 3]:
        if st.button(query, key=f"ex_{idx}", use_container_width=True):
            st.session_state.query = query
            st.session_state.submit = True

st.markdown("---")

# Search input
query = st.text_input(
    "🔍 Ask your question:",
    placeholder="Type anything... e.g., India ka GDP 2026 mein kitna hai?",
    key="query" if "query" not in st.session_state else None,
    value=st.session_state.get("query", "")
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_btn = st.button("🌐 Search", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 New Query", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Process query
if (search_btn or st.session_state.get("submit")) and query:
    st.session_state.submit = False  # Reset submit flag
    
    with st.spinner("🔎 Searching the web and generating answer..."):
        start_time = time.time()
        result = ask_question(query, use_cache=use_cache)
        elapsed = time.time() - start_time
    
    if result.get("error"):
        st.error(f"❌ {result.get('message', 'Unknown error')}")
        if result.get("detail"):
            with st.expander("Error Details"):
                st.code(result["detail"])
    else:
        # Cache indicator
        if result.get("from_cache"):
            st.markdown('<div class="cache-badge">⚡ Served from cache (<1s)</div>', unsafe_allow_html=True)
        
        # Answer section
        st.markdown("### 💬 Answer")
        answer = result.get("answer", "No answer generated")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        
        # Sources and References
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 Sources")
            sources = result.get("sources", [])
            if sources:
                sources_html = "".join([f'<span class="source-tag">🌐 {s}</span>' for s in sources])
                st.markdown(sources_html, unsafe_allow_html=True)
            else:
                st.caption("No sources available")
        
        with col2:
            st.markdown("### 🔗 References")
            urls = result.get("web_urls", [])
            if urls:
                for url in urls[:4]:  # Show first 4 URLs
                    st.markdown(f"[{url[:50]}...]({url})")
            else:
                st.caption("No references available")
        
        st.markdown("---")
        
        # Query Analysis
        st.markdown("### 📊 Query Analysis")
        query_info = result.get("query_info", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Language", query_info.get("language_label", "Unknown"))
        with col2:
            confidence = query_info.get("confidence", 0) * 100
            st.metric("Confidence", f"{confidence:.0f}%")
        with col3:
            search_info = result.get("search_info", {})
            st.metric("Results Found", search_info.get("web_results_fetched", 0))
        with col4:
            st.metric("Final Results", search_info.get("final_results", 0))
        
        # Show normalized query and English translation
        with st.expander("🔍 Query Processing Details"):
            st.markdown(f"**Original:** {query_info.get('original', query)}")
            if query_info.get("normalized"):
                st.markdown(f"**Normalized:** {query_info.get('normalized', '')}")
            if query_info.get("transliterated"):
                st.markdown(f"**Transliterated:** {query_info.get('transliterated', '')}")
            st.markdown(f"**English Query:** {query_info.get('english_query', '')}")
        
        # Performance Metrics
        st.markdown("### ⚡ Performance")
        gen_info = result.get("generation_info", {})
        
        total_ms = gen_info.get("total_ms", 0)
        
        # Progress bar visualization
        stages = [
            ("Normalization", gen_info.get("normalization_ms", 0), "#22d3ee"),
            ("Translation", gen_info.get("translation_ms", 0), "#a78bfa"),
            ("Web Search", gen_info.get("web_search_ms", 0), "#f97316"),
            ("Embedding", gen_info.get("embedding_ms", 0), "#6366f1"),
            ("Search & Rerank", gen_info.get("search_rerank_ms", 0), "#22c55e"),
            ("LLM Generation", gen_info.get("generation_ms", 0), "#ef4444"),
        ]
        
        # Performance metrics in columns
        cols = st.columns(3)
        for idx, (stage, time_ms, color) in enumerate(stages):
            with cols[idx % 3]:
                st.metric(stage, f"{time_ms:.0f}ms")
        
        # Total time
        st.metric("**Total Time**", f"{total_ms/1000:.1f}s", delta=f"{elapsed:.1f}s actual")
        
        # LLM Info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", gen_info.get("model", "N/A"))
        with col2:
            llm_used = "Yes ✅" if gen_info.get("llm_used") else "No (Context Only)"
            st.metric("LLM Used", llm_used)
        
        # Debug information
        if show_debug:
            with st.expander("🔧 Debug Information"):
                st.json(result)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Multilingual RAG System v2.0 | Web-Only Mode with Real-Time Search</p>
        <p>Powered by FastAPI + Streamlit + phi3:mini + DuckDuckGo</p>
    </div>
    """,
    unsafe_allow_html=True
)
