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
    .metric-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 0.8rem 0;
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
    .performance-stage {
        padding: 1rem;
        border-radius: 8px;
        background: #f9fafb;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .status-online {
        color: #15803d;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: #dcfce7;
        border-radius: 8px;
        display: inline-block;
    }
    .status-offline {
        color: #991b1b;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: #fee2e2;
        border-radius: 8px;
        display: inline-block;
    }
    div.stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    div.stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────────────

def check_backend_health() -> Dict[str, Any]:
    """Check if backend is running and get status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Ensure status is set
            if "status" not in data:
                data["status"] = "ready"
            return data
        return {"status": "error", "model_loaded": False, "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError as e:
        return {
            "status": "offline", 
            "error": f"Connection refused. Is backend running on port 8000?\n\nDetails: {str(e)[:200]}"
        }
    except requests.exceptions.Timeout:
        return {"status": "offline", "error": "Backend health check timed out (>5s)"}
    except Exception as e:
        return {
            "status": "offline", 
            "error": f"Unexpected error: {type(e).__name__}: {str(e)}"
        }


def ask_question(query: str, use_cache: bool = True) -> Dict[str, Any]:
    """Send question to backend API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": query, "use_cache": use_cache},
            timeout=300,  # 5 minute timeout for LLM generation (phi3:mini on CPU can be slow)
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
            "message": "Request timed out (LLM generation taking too long). Try a shorter query or enable caching.",
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


# ─── Initialize Session State ──────────────────────────────────────────────────
if "show_advanced" not in st.session_state:
    st.session_state.show_advanced = True
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
if "use_cache" not in st.session_state:
    st.session_state.use_cache = True


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🎛️ Control Panel")
    st.markdown("---")
    
    # System Status Section
    with st.container():
        st.markdown("### ⚙️ System Status")
        
        # Check backend health
        health = check_backend_health()
        
        # Debug: Show raw health response (temporary)
        if st.session_state.get("show_debug", False):
            with st.expander("🔍 Health Check Response"):
                st.json(health)
        
        # Backend offline (connection error)
        if health.get("status") == "offline":
            st.markdown('<div class="status-offline">❌ Backend Offline</div>', unsafe_allow_html=True)
            st.caption("Start backend: `python -m backend.app`")
            if health.get("error"):
                with st.expander("Error Details"):
                    st.code(health['error'])
        # Backend error (HTTP error but connected)
        elif health.get("status") == "error":
            st.markdown('<div class="status-offline">⚠️ Backend Error</div>', unsafe_allow_html=True)
            if health.get("message"):
                st.caption(f"{health['message']}")
        # Backend OK and model loaded
        elif health.get("status") == "ok" and health.get("model_loaded"):
            st.markdown('<div class="status-online">✅ System Ready</div>', unsafe_allow_html=True)
            st.caption(f"**Mode:** {health.get('mode', 'web_search_only')}")
            st.caption(f"**Endpoint:** http://localhost:8000")
        # Backend OK but model still loading
        elif health.get("status") == "ok":
            st.warning("🔄 Model Loading...")
            st.caption("Backend is up, embedding model loading...")
        # Unknown status
        else:
            st.error("❓ Unknown Status")
            with st.expander("Debug Info"):
                st.json(health)
        
        # Refresh button
        st.markdown("")
        if st.button("🔄 Refresh Status", use_container_width=True, type="secondary"):
            st.rerun()
    
    st.markdown("---")
    
    # Cache Statistics Section
    with st.container():
        st.markdown("### 📊 Cache Statistics")
        cache_stats = get_cache_stats()
        
        if cache_stats:
            # Key metrics in columns
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    "Cache Hits", 
                    cache_stats.get("cache_hits", 0),
                    delta=None,
                    help="Number of queries served from cache"
                )
            with metric_col2:
                hit_rate = cache_stats.get('hit_rate_percent', 0)
                st.metric(
                    "Hit Rate", 
                    f"{hit_rate:.1f}%",
                    delta=f"{hit_rate-50:.1f}%" if hit_rate > 50 else None,
                    help="Percentage of cached responses"
                )
            
            # Additional stats
            st.caption(f"📝 **Total Queries:** {cache_stats.get('total_queries', 0)}")
            st.caption(f"💾 **Cached Entries:** {cache_stats.get('cached_entries', 0)}")
            st.caption(f"⏱️ **Cache TTL:** 1 hour")
            
            # Clear cache button
            st.markdown("")
            if st.button("🗑️ Clear Cache", use_container_width=True, type="secondary"):
                if clear_cache():
                    st.success("✅ Cache cleared!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Failed to clear cache")
        else:
            st.info("No cache data available")
    
    st.markdown("---")
    
    # Settings Section
    with st.expander("⚙️ Settings", expanded=True):
        st.session_state.use_cache = st.checkbox(
            "Enable Caching", 
            value=st.session_state.use_cache, 
            help="Cache responses for faster repeated queries (1hr TTL)"
        )
        st.session_state.show_debug = st.checkbox(
            "Debug Mode", 
            value=st.session_state.show_debug, 
            help="Show detailed JSON response and timing breakdown"
        )
        st.session_state.show_advanced = st.checkbox(
            "Advanced Metrics", 
            value=st.session_state.show_advanced, 
            help="Display detailed performance breakdown"
        )
    
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About System"):
        st.markdown("**Multilingual RAG v2.0**")
        st.caption("🌐 Web-Only Mode")
        st.caption("🔍 Real-time DuckDuckGo Search")
        st.caption("🌏 Hindi • English • తెలుగు")
        st.caption("💬 Code-Mixed Queries")
        st.markdown("")
        st.markdown("**Tech Stack:**")
        st.caption("• FastAPI Backend")
        st.caption("• Streamlit Frontend")
        st.caption("• phi3:mini 3B LLM")
        st.caption("• Sentence Transformers")
        st.caption("• FAISS Vector Search")
        st.caption("• DuckDuckGo API")
    
    st.markdown("---")
    st.caption("💡 **Tip:** Use example queries below to get started!")


# ─── Main Content ──────────────────────────────────────────────────────────────

# Header
st.markdown('<h1 class="main-header">🌐 Ask Anything</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multilingual Q&A powered by Web Search • '
    'Hindi • English • తెలుగు • Code-Mixed</p>',
    unsafe_allow_html=True
)

# Main container for search interface
with st.container():
    # Example queries section
    st.markdown('<div class="section-header">💡 Try These Example Queries</div>', unsafe_allow_html=True)
    
    example_queries = [
        "Elon Musk ki net worth kitni hai?",
        "IPL 2024 winner kaun hai?",
        "Latest AI news kya hai?",
        "India ka prime minister kon hai?",
        "Climate change kya hai?",
        "Python vs JavaScript comparison",
    ]
    
    # Display examples in 3 columns
    cols = st.columns(3)
    for idx, query in enumerate(example_queries):
        with cols[idx % 3]:
            if st.button(
                query, 
                key=f"ex_{idx}", 
                use_container_width=True,
                help=f"Click to search: {query}"
            ):
                st.session_state.query = query
                st.session_state.submit = True
    
    st.markdown("")  # Spacing
    
    # Search input section
    st.markdown('<div class="section-header">🔍 Ask Your Question</div>', unsafe_allow_html=True)
    
    query = st.text_input(
        "Type anything in any language:",
        placeholder="e.g., भारत की राजधानी क्या है? / What is India's GDP? / Latest tech news",
        label_visibility="collapsed",
        key="query" if "query" not in st.session_state else None,
        value=st.session_state.get("query", "")
    )
    
    # Action buttons
    button_col1, button_col2, button_col3 = st.columns([3, 1, 1])
    with button_col1:
        search_btn = st.button("🌐 Search & Get Answer", type="primary", use_container_width=True)
    with button_col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

st.markdown("---")

# Results section
if (search_btn or st.session_state.get("submit")) and query:
    st.session_state.submit = False  # Reset
    
    with st.spinner("🔎 Searching the web and generating intelligent answer..."):
        start_time = time.time()
        result = ask_question(query, use_cache=st.session_state.use_cache)
        elapsed = time.time() - start_time
    
    if result.get("error"):
        st.error(f"❌ **Error:** {result.get('message', 'Unknown error')}")
        if result.get("detail"):
            with st.expander("🔍 Error Details"):
                st.code(result["detail"])
    else:
        # Cache indicator
        if result.get("from_cache"):
            st.markdown(
                '<div class="cache-badge">⚡ Lightning Fast - Served from Cache (<1s)</div>', 
                unsafe_allow_html=True
            )
        else:
            st.info(f"⏱️ Generated in {elapsed:.1f}s (phi3:mini LLM)")
        
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
            
            # Display answer with better formatting
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
            
            # LLM info under answer with better styling
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
                    help="Whether LLM was used for this answer"
                )
            with info_col3:
                gen_time = gen_info.get("generation_ms", 0) / 1000
                st.metric(
                    "⏱️ Generation Time", 
                    f"{gen_time:.1f}s",
                    delta=f"{gen_time - 30:.1f}s" if gen_time > 30 else None,
                    help="Time taken by LLM to generate answer"
                )
        
        # TAB 2: Query Analysis
        with tab2:
            st.markdown('<div class="section-header">🔍 Query Processing</div>', unsafe_allow_html=True)
            
            query_info = result.get("query_info", {})
            search_info = result.get("search_info", {})
            
            # Key metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                lang_label = query_info.get("language_label", "Unknown")
                st.metric("🌐 Language", lang_label, help="Detected query language")
            with metric_col2:
                confidence = query_info.get("confidence", 0) * 100
                st.metric(
                    "✓ Confidence", 
                    f"{confidence:.0f}%",
                    delta="High" if confidence > 80 else "Medium" if confidence > 50 else "Low"
                )
            with metric_col3:
                web_results = search_info.get("web_results_fetched", 0)
                st.metric("🔍 Web Results", web_results, help="Results fetched from web")
            with metric_col4:
                final_results = search_info.get("final_results", 0)
                st.metric("✨ Final Results", final_results, help="Results after filtering")
            
            st.markdown("")
            
            # Query processing details
            with st.container():
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
                        # Extract domain for display
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
                
                # Performance stages with visual indicators
                stages = [
                    ("🔄 Normalization", gen_info.get("normalization_ms", 0), "#22d3ee"),
                    ("🌐 Translation", gen_info.get("translation_ms", 0), "#a78bfa"),
                    ("🔍 Web Search", gen_info.get("web_search_ms", 0), "#f97316"),
                    ("🧮 Embedding", gen_info.get("embedding_ms", 0), "#6366f1"),
                    ("🎯 Search & Rerank", gen_info.get("search_rerank_ms", 0), "#22c55e"),
                    ("🤖 LLM Generation", gen_info.get("generation_ms", 0), "#ef4444"),
                ]
                
                # Display in 3 columns
                perf_cols = st.columns(3)
                for idx, (stage, time_ms, color) in enumerate(stages):
                    with perf_cols[idx % 3]:
                        percentage = (time_ms / total_ms * 100) if total_ms > 0 else 0
                        st.metric(
                            stage,
                            f"{time_ms:.0f}ms",
                            delta=f"{percentage:.1f}% of total"
                        )
                
                st.markdown("")
                
                # Total time summary
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric(
                        "📊 **Total Processing**", 
                        f"{total_ms/1000:.2f}s",
                        help="Sum of all processing stages"
                    )
                with summary_col2:
                    st.metric(
                        "⏱️ **Actual Time**", 
                        f"{elapsed:.2f}s",
                        help="Real wall-clock time"
                    )
                with summary_col3:
                    overhead = ((elapsed * 1000) - total_ms)
                    st.metric(
                        "🔧 **Overhead**", 
                        f"{overhead:.0f}ms",
                        help="Network + serialization time"
                    )
                
                # Performance visualization (progress bar)
                st.markdown("")
                st.markdown("**⏱️ Time Distribution:**")
                
                # Calculate total of all stages for accurate percentage
                total_stages_ms = sum(time_ms for _, time_ms, _ in stages)
                
                for stage, time_ms, color in stages:
                    if time_ms > 0:
                        time_s = time_ms / 1000
                        # Use stage total for more accurate percentage
                        percentage = (time_ms / total_stages_ms) if total_stages_ms > 0 else 0
                        # Clamp percentage between 0 and 1 for progress bar
                        percentage = min(max(percentage, 0.0), 1.0)
                        st.progress(percentage, text=f"{stage}: {time_s:.2f}s ({percentage*100:.1f}%)")
            else:
                st.info("Enable 'Advanced Metrics' in sidebar to see detailed breakdown")
        
        # Debug information (if enabled)
        if st.session_state.show_debug:
            st.markdown("---")
            with st.expander("🔧 Debug Information (Full JSON Response)"):
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
