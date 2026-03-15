"""
FastAPI Application — Multilingual RAG API
=============================================
The main entry point for the system. Orchestrates all modules:
ingestion, retrieval, generation, and evaluation.

Endpoints:
  POST /ingest   — Index documents from data/raw/
  POST /ask      — Ask a question, get a grounded answer
  GET  /health   — System status
  POST /evaluate — Run evaluation suite
  GET  /stats    — Index and system statistics

How the /ask pipeline works (step by step):
1. User sends query: "Modi ji ka education kya hai?"
2. Language detection → "mixed" (code-mixed Hindi-English)
3. Query normalization → transliterate + expand
4. Embed the normalized query → 384-dim vector
5. FAISS search → top-k semantically similar chunks
6. Hybrid reranking → boost keyword matches
7. Deduplication → remove near-duplicate chunks
8. Prompt construction → grounded prompt with context
9. LLM generation → factual answer with source citations
10. Return response with answer + sources + metadata
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ─── Local imports ────────────────────────────────────────────────────────────
from backend.config import (
    RAW_DATA_DIR, API_HOST, API_PORT,
    RETRIEVAL_TOP_K, SEMANTIC_WEIGHT_ALPHA, KEYWORD_WEIGHT_BETA,
    WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_REGION, WEB_SEARCH_ENABLED,
)
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents, Chunk
from ingestion.embedder import EmbeddingModel, get_embedding_model
from ingestion.indexer import FAISSIndexer
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

# ─── Lifespan Context Manager ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events using lifespan context manager."""
    # Startup
    global embedding_model
    logger.info("Starting Multilingual RAG System...")
    
    # Load embedding model
    logger.info("Loading embedding model...")
    embedding_model = get_embedding_model()
    
    # Try to load existing FAISS index
    if indexer.load():
        logger.info(f"Loaded existing index with {indexer.total_vectors} vectors")
    else:
        logger.info("No existing index found. Use POST /ingest to create one.")
    
    yield
    
    # Shutdown (if needed in future)
    logger.info("Shutting down...")

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multilingual RAG System",
    description=(
        "A research-driven, production-grade Retrieval-Augmented Generation "
        "system for code-mixed Indian language queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
# These are initialized during /ingest and reused for /ask
indexer = FAISSIndexer()
embedding_model: Optional[EmbeddingModel] = None


# ─── Request/Response Models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    data_dir: str = Field(
        default=str(RAW_DATA_DIR),
        description="Path to directory containing documents to ingest"
    )
    chunk_size: int = Field(default=500, description="Characters per chunk")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    vectors_indexed: int
    time_seconds: float


class AskRequest(BaseModel):
    query: str = Field(..., description="The question to ask", min_length=1)
    top_k: int = Field(default=RETRIEVAL_TOP_K, description="Number of results")
    alpha: float = Field(default=SEMANTIC_WEIGHT_ALPHA, description="Semantic weight")
    beta: float = Field(default=KEYWORD_WEIGHT_BETA, description="Keyword weight")


class AskResponse(BaseModel):
    answer: str
    sources: list
    query_info: dict
    retrieval_info: dict
    generation_info: dict


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    index_loaded: bool
    total_vectors: int
    model_loaded: bool


class AskWebRequest(BaseModel):
    """Request model for open-domain web search QA."""
    query: str = Field(..., description="The question to ask (any language)", min_length=1)
    top_k: int = Field(default=5, description="Number of results to use")
    max_web_results: int = Field(
        default=WEB_SEARCH_MAX_RESULTS,
        description="Number of web results to fetch"
    )


class AskWebResponse(BaseModel):
    """Response model for web search QA."""
    answer: str
    sources: list
    web_urls: list
    query_info: dict
    search_info: dict
    generation_info: dict


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive web UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual RAG System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0f172a; color: #e2e8f0;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 {
            text-align: center; font-size: 1.8rem; margin: 30px 0 8px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center; color: #94a3b8; font-size: 0.95rem; margin-bottom: 30px;
        }
        .status-bar {
            display: flex; gap: 12px; justify-content: center;
            margin-bottom: 24px; flex-wrap: wrap;
        }
        .badge {
            padding: 6px 14px; border-radius: 20px; font-size: 0.8rem;
            background: #1e293b; border: 1px solid #334155;
        }
        .badge.ok { border-color: #22c55e; color: #4ade80; }
        .badge.warn { border-color: #f59e0b; color: #fbbf24; }
        .search-box {
            display: flex; gap: 10px; margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1; padding: 14px 18px; border-radius: 12px;
            border: 1px solid #334155; background: #1e293b; color: #f1f5f9;
            font-size: 1rem; outline: none; transition: border-color 0.2s;
        }
        input[type="text"]:focus { border-color: #818cf8; }
        input::placeholder { color: #64748b; }
        button {
            padding: 14px 28px; border-radius: 12px; border: none;
            font-size: 1rem; font-weight: 600; cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #6366f1, #818cf8);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(99,102,241,0.4); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-secondary {
            background: #1e293b; color: #94a3b8; border: 1px solid #334155;
        }
        .btn-secondary:hover { border-color: #818cf8; color: #c7d2fe; }
        .actions { display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap; }
        .sample-queries {
            display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px;
        }
        .sample-q {
            padding: 6px 12px; border-radius: 8px; font-size: 0.82rem;
            background: #1e293b; border: 1px solid #334155; color: #94a3b8;
            cursor: pointer; transition: all 0.2s;
        }
        .sample-q:hover { border-color: #818cf8; color: #c7d2fe; }
        .mode-toggle {
            display: flex; justify-content: center; gap: 4px;
            margin-bottom: 20px; background: #1e293b; border-radius: 12px;
            padding: 4px; width: fit-content; margin-left: auto; margin-right: auto;
        }
        .mode-btn {
            padding: 10px 24px; border-radius: 10px; border: none;
            font-size: 0.9rem; font-weight: 600; cursor: pointer;
            background: transparent; color: #94a3b8; transition: all 0.2s;
        }
        .mode-btn.active {
            background: linear-gradient(135deg, #6366f1, #818cf8);
            color: white; box-shadow: 0 2px 10px rgba(99,102,241,0.3);
        }
        .mode-btn:hover:not(.active) { color: #c7d2fe; }
        .web-url {
            padding: 4px 10px; border-radius: 6px; font-size: 0.78rem;
            background: rgba(56,189,248,0.15); color: #38bdf8;
            border: 1px solid rgba(56,189,248,0.3);
            text-decoration: none; display: inline-block; margin: 3px;
        }
        .web-url:hover { background: rgba(56,189,248,0.25); }
        .result-card {
            background: #1e293b; border: 1px solid #334155; border-radius: 12px;
            padding: 20px; margin-bottom: 16px;
        }
        .result-card h3 { color: #38bdf8; font-size: 0.85rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
        .answer-text { font-size: 1.05rem; line-height: 1.7; color: #f1f5f9; white-space: pre-wrap; }
        .meta-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px; margin-top: 12px;
        }
        .meta-item { padding: 8px 12px; background: #0f172a; border-radius: 8px; font-size: 0.82rem; }
        .meta-label { color: #64748b; }
        .meta-value { color: #c7d2fe; font-weight: 600; }
        .sources { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
        .source-tag {
            padding: 4px 10px; border-radius: 6px; font-size: 0.8rem;
            background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3);
        }
        .timing-bar {
            display: flex; gap: 2px; height: 6px; border-radius: 3px;
            overflow: hidden; margin-top: 8px;
        }
        .timing-seg { height: 100%; min-width: 3px; }
        .spinner {
            display: inline-block; width: 18px; height: 18px;
            border: 2px solid #334155; border-top-color: #818cf8;
            border-radius: 50%; animation: spin 0.8s linear infinite;
            vertical-align: middle; margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        #results { min-height: 40px; }
        .hidden { display: none; }
        .error { color: #f87171; padding: 16px; background: rgba(248,113,113,0.1); border-radius: 12px; }
    </style>
</head>
<body>
<div class="container">
    <h1>&#x1F1EE;&#x1F1F3; Multilingual RAG System</h1>
    <p class="subtitle">Code-Mixed Indian Language Question Answering &mdash; Research + Production</p>

    <div class="status-bar" id="statusBar">
        <span class="badge" id="badgeModel">⏳ Model loading...</span>
        <span class="badge" id="badgeIndex">⏳ Checking index...</span>
    </div>

    <div class="mode-toggle">
        <button class="mode-btn" id="modeLocal" onclick="setMode('local')">&#128194; Local Knowledge</button>
        <button class="mode-btn active" id="modeWeb" onclick="setMode('web')">&#127760; Web Search</button>
    </div>

    <div class="search-box">
        <input type="text" id="queryInput" placeholder="Ask anything in the world... e.g. Elon Musk ki net worth kitni hai?" />
        <button class="btn-primary" id="askBtn" onclick="ask()">🌐 Search</button>
    </div>

    <div class="sample-queries hidden" id="localSamples">
        <span class="sample-q" onclick="setQuery(this)">Modi ji ka education kya hai?</span>
        <span class="sample-q" onclick="setQuery(this)">Chandrayaan-3 kab launch hua tha?</span>
        <span class="sample-q" onclick="setQuery(this)">What are the fundamental rights?</span>
        <span class="sample-q" onclick="setQuery(this)">ISRO ka headquarter kahan hai?</span>
        <span class="sample-q" onclick="setQuery(this)">Telangana kab bana tha?</span>
        <span class="sample-q" onclick="setQuery(this)">NEP 2020 mein kya changes hain?</span>
    </div>

    <div class="sample-queries" id="webSamples">
        <span class="sample-q" onclick="setQuery(this)">Elon Musk ki net worth kitni hai?</span>
        <span class="sample-q" onclick="setQuery(this)">IPL mein sabse zyada runs kisne banaye?</span>
        <span class="sample-q" onclick="setQuery(this)">ChatGPT kya hai aur kaise kaam karta hai?</span>
        <span class="sample-q" onclick="setQuery(this)">Latest Mars mission ka status kya hai?</span>
        <span class="sample-q" onclick="setQuery(this)">Python ya JavaScript mein kya difference hai?</span>
        <span class="sample-q" onclick="setQuery(this)">India mein best engineering colleges kaun se hain?</span>
    </div>

    <div class="actions">
        <button class="btn-secondary" onclick="ingest()">&#128229; Ingest Documents</button>
        <button class="btn-secondary" onclick="evaluate()">&#128202; Run Evaluation</button>
    </div>

    <div id="results"></div>
</div>

<script>
const API = '';
let currentMode = 'web';  // 'web' is now the default for production

function setMode(mode) {
    currentMode = mode;
    document.getElementById('modeLocal').className = 'mode-btn' + (mode === 'local' ? ' active' : '');
    document.getElementById('modeWeb').className = 'mode-btn' + (mode === 'web' ? ' active' : '');
    document.getElementById('localSamples').className = 'sample-queries' + (mode === 'local' ? '' : ' hidden');
    document.getElementById('webSamples').className = 'sample-queries' + (mode === 'web' ? '' : ' hidden');
    const input = document.getElementById('queryInput');
    input.placeholder = mode === 'local'
        ? 'Ask about local knowledge... e.g. Modi ji ka education kya hai?'
        : 'Ask anything in the world... e.g. Elon Musk ki net worth kitni hai?';
    document.getElementById('askBtn').textContent = mode === 'local' ? 'Ask' : '🌐 Search';
}

function setQuery(el) {
    document.getElementById('queryInput').value = el.textContent;
    ask();
}

async function checkHealth() {
    try {
        const r = await fetch(API + '/health');
        const d = await r.json();
        const bm = document.getElementById('badgeModel');
        const bi = document.getElementById('badgeIndex');
        bm.textContent = d.model_loaded ? '✅ Model Loaded' : '❌ Model Error';
        bm.className = 'badge ' + (d.model_loaded ? 'ok' : 'warn');
        bi.textContent = d.index_loaded ? '✅ Index: ' + d.total_vectors + ' vectors' : '⚠️ No Index — click Ingest';
        bi.className = 'badge ' + (d.index_loaded ? 'ok' : 'warn');
    } catch(e) {
        document.getElementById('badgeModel').textContent = '❌ Server error';
    }
}

async function ask() {
    const q = document.getElementById('queryInput').value.trim();
    if (!q) return;
    const btn = document.getElementById('askBtn');
    btn.disabled = true;
    const modeLabel = currentMode === 'web' ? 'Searching the web' : 'Searching local knowledge';
    document.getElementById('results').innerHTML = '<div class="result-card"><span class="spinner"></span> ' + modeLabel + ' &amp; generating...</div>';
    try {
        const endpoint = currentMode === 'web' ? '/ask-web' : '/ask';
        const r = await fetch(API + endpoint, {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({query: q})
        });
        const d = await r.json();
        if (!r.ok) { throw new Error(d.detail || 'Request failed'); }
        if (currentMode === 'web') { renderWebAnswer(d); } else { renderAnswer(d); }
    } catch(e) {
        document.getElementById('results').innerHTML = '<div class="error">❌ ' + e.message + '</div>';
    }
    btn.disabled = false;
}

function renderAnswer(d) {
    const gi = d.generation_info || {};
    const qi = d.query_info || {};
    const ri = d.retrieval_info || {};
    const total = gi.total_ms || 1;
    const segments = [
        {name:'Normalize', ms: gi.normalization_ms||0, color:'#22d3ee'},
        {name:'Embed', ms: gi.embedding_ms||0, color:'#6366f1'},
        {name:'Search', ms: gi.search_ms||0, color:'#22c55e'},
        {name:'Rerank', ms: gi.rerank_ms||0, color:'#f59e0b'},
        {name:'Generate', ms: gi.generation_ms||0, color:'#ef4444'},
    ];
    const bar = segments.map(s =>
        '<div class="timing-seg" style="width:' + Math.max((s.ms/total)*100, 2) + '%;background:' + s.color + '" title="' + s.name + ': ' + s.ms.toFixed(0) + 'ms"></div>'
    ).join('');
    const timingLabels = segments.map(s =>
        '<span style="color:' + s.color + '">' + s.name + ' ' + s.ms.toFixed(0) + 'ms</span>'
    ).join(' &middot; ');

    const sources = (d.sources||[]).map(s => '<span class="source-tag">' + s + '</span>').join('');

    document.getElementById('results').innerHTML =
        '<div class="result-card"><h3>Answer</h3><div class="answer-text">' + escHtml(d.answer) + '</div>' +
        (sources ? '<h3 style="margin-top:12px">Sources</h3><div class="sources">' + sources + '</div>' : '') +
        '</div>' +
        '<div class="result-card"><h3>Query Analysis</h3><div class="meta-grid">' +
            meta('Language', (qi.language_label||'') + ' (' + ((qi.confidence||0)*100).toFixed(0) + '%)') +
            meta('Normalized', qi.normalized ? qi.normalized.substring(0,60) + (qi.normalized.length>60?'...':'') : '-') +
            meta('Transliterated', qi.transliterated || 'N/A') +
        '</div></div>' +
        '<div class="result-card"><h3>Retrieval Pipeline</h3><div class="meta-grid">' +
            meta('Semantic Hits', ri.semantic_results) +
            meta('After Hybrid', ri.hybrid_results) +
            meta('After Dedup', ri.deduped_results) +
            meta('Final Results', ri.final_results) +
            meta('&alpha; (semantic)', ri.alpha) +
            meta('&beta; (keyword)', ri.beta) +
        '</div></div>' +
        '<div class="result-card"><h3>Performance &mdash; ' + total.toFixed(0) + 'ms total</h3>' +
        '<div class="timing-bar">' + bar + '</div>' +
        '<div style="margin-top:8px;font-size:0.8rem">' + timingLabels + '</div>' +
        '<div class="meta-grid" style="margin-top:8px">' +
            meta('Model', gi.model || 'N/A') +
            meta('LLM Used', gi.llm_used ? 'Yes' : 'No (context only)') +
        '</div></div>';
}

function meta(label, value) {
    return '<div class="meta-item"><span class="meta-label">' + label + '</span><br><span class="meta-value">' + (value ?? '-') + '</span></div>';
}
function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function renderWebAnswer(d) {
    const gi = d.generation_info || {};
    const qi = d.query_info || {};
    const si = d.search_info || {};
    const total = gi.total_ms || 1;
    const segments = [
        {name:'Normalize', ms: gi.normalization_ms||0, color:'#22d3ee'},
        {name:'Translate', ms: gi.translation_ms||0, color:'#a78bfa'},
        {name:'Web Search', ms: gi.web_search_ms||0, color:'#f97316'},
        {name:'Embed+Search', ms: gi.embedding_ms||0, color:'#6366f1'},
        {name:'Rerank', ms: gi.rerank_ms||0, color:'#22c55e'},
        {name:'Generate', ms: gi.generation_ms||0, color:'#ef4444'},
    ];
    const bar = segments.map(s =>
        '<div class="timing-seg" style="width:' + Math.max((s.ms/total)*100, 2) + '%;background:' + s.color + '" title="' + s.name + ': ' + s.ms.toFixed(0) + 'ms"></div>'
    ).join('');
    const timingLabels = segments.map(s =>
        '<span style="color:' + s.color + '">' + s.name + ' ' + s.ms.toFixed(0) + 'ms</span>'
    ).join(' &middot; ');

    const sources = (d.sources||[]).map(s => '<span class="source-tag">🌐 ' + s + '</span>').join('');
    const urls = (d.web_urls||[]).map(u => '<a class="web-url" href="' + u + '" target="_blank" rel="noopener">' + u.substring(0,60) + (u.length>60?'...':'') + '</a>').join('');

    document.getElementById('results').innerHTML =
        '<div class="result-card"><h3>&#127760; Web Search Answer</h3><div class="answer-text">' + escHtml(d.answer) + '</div>' +
        (sources ? '<h3 style="margin-top:12px">Web Sources</h3><div class="sources">' + sources + '</div>' : '') +
        (urls ? '<h3 style="margin-top:12px">References</h3><div>' + urls + '</div>' : '') +
        '</div>' +
        '<div class="result-card"><h3>Query Analysis</h3><div class="meta-grid">' +
            meta('Language', (qi.language_label||'') + ' (' + ((qi.confidence||0)*100).toFixed(0) + '%)') +
            meta('Original', qi.original ? qi.original.substring(0,60) : '-') +
            meta('English Query', qi.english_query || '-') +
            meta('Transliterated', qi.transliterated || 'N/A') +
        '</div></div>' +
        '<div class="result-card"><h3>Web Search Pipeline</h3><div class="meta-grid">' +
            meta('Web Results', si.web_results_fetched) +
            meta('Semantic Hits', si.semantic_results) +
            meta('After Hybrid', si.hybrid_results) +
            meta('Final Results', si.final_results) +
        '</div></div>' +
        '<div class="result-card"><h3>Performance &mdash; ' + total.toFixed(0) + 'ms total</h3>' +
        '<div class="timing-bar">' + bar + '</div>' +
        '<div style="margin-top:8px;font-size:0.8rem">' + timingLabels + '</div>' +
        '<div class="meta-grid" style="margin-top:8px">' +
            meta('Model', gi.model || 'N/A') +
            meta('LLM Used', gi.llm_used ? 'Yes' : 'No') +
        '</div></div>';
}

async function ingest() {
    document.getElementById('results').innerHTML = '<div class="result-card"><span class="spinner"></span> Ingesting documents...</div>';
    try {
        const r = await fetch(API + '/ingest', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
        const d = await r.json();
        document.getElementById('results').innerHTML =
            '<div class="result-card"><h3>Ingestion Complete</h3><div class="meta-grid">' +
            meta('Documents', d.documents_loaded) + meta('Chunks', d.chunks_created) +
            meta('Vectors', d.vectors_indexed) + meta('Time', d.time_seconds + 's') +
            '</div></div>';
        checkHealth();
    } catch(e) {
        document.getElementById('results').innerHTML = '<div class="error">❌ ' + e.message + '</div>';
    }
}

async function evaluate() {
    document.getElementById('results').innerHTML = '<div class="result-card"><span class="spinner"></span> Running evaluation on 12 queries...</div>';
    try {
        const r = await fetch(API + '/evaluate', {method:'POST', headers:{'Content-Type':'application/json'}});
        const d = await r.json();
        const rows = (d.per_query_results||[]).map(q =>
            '<tr><td>' + q.query_id + '</td><td>' + escHtml(q.query) + '</td><td>' + q.language +
            '</td><td>' + (q['recall@k']>0?'✅':'❌') + '</td><td>' + q.reciprocal_rank.toFixed(2) +
            '</td><td>' + q.latency_ms.toFixed(0) + 'ms</td></tr>'
        ).join('');
        document.getElementById('results').innerHTML =
            '<div class="result-card"><h3>Evaluation Results</h3><div class="meta-grid">' +
            meta('Recall@' + d.top_k, (d.avg_recall_at_k*100).toFixed(1) + '%') +
            meta('Precision@' + d.top_k, (d.avg_precision_at_k*100).toFixed(1) + '%') +
            meta('MRR', d.mrr.toFixed(3)) +
            meta('Avg Latency', d.avg_latency_ms.toFixed(1) + 'ms') +
            '</div>' +
            '<table style="width:100%;margin-top:16px;font-size:0.85rem;border-collapse:collapse">' +
            '<tr style="color:#64748b;text-align:left"><th style="padding:6px">ID</th><th>Query</th><th>Lang</th><th>Hit</th><th>RR</th><th>Time</th></tr>' +
            rows + '</table></div>';
    } catch(e) {
        document.getElementById('results').innerHTML = '<div class="error">❌ ' + e.message + '</div>';
    }
}

document.getElementById('queryInput').addEventListener('keydown', e => { if(e.key==='Enter') ask(); });
checkHealth();
</script>
</body>
</html>
"""


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns system status: whether the model and index are loaded.
    """
    return HealthResponse(
        status="healthy",
        index_loaded=indexer.is_loaded(),
        total_vectors=indexer.total_vectors,
        model_loaded=embedding_model is not None,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest = IngestRequest()):
    """
    Ingest documents: load → chunk → embed → index.

    This is the offline pipeline. Run once, then use /ask repeatedly.

    Steps:
    1. Load all .txt/.json/.md files from data_dir
    2. Split each document into overlapping chunks
    3. Embed all chunks using the multilingual model
    4. Build and save a FAISS index

    The index is persisted to disk, so you don't need to re-ingest
    unless the documents change.
    """
    global embedding_model
    start_time = time.time()

    try:
        # Step 1: Load documents
        logger.info(f"Loading documents from {request.data_dir}...")
        documents = load_documents(request.data_dir)
        if not documents:
            raise HTTPException(status_code=400, detail="No documents found")
        logger.info(f"Loaded {len(documents)} documents")

        # Step 2: Chunk documents
        logger.info("Chunking documents...")
        chunks = chunk_documents(documents, request.chunk_size, request.chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Embed chunks
        logger.info("Embedding chunks...")
        if embedding_model is None:
            embedding_model = get_embedding_model()

        chunk_texts = [c.text for c in chunks]
        vectors = embedding_model.embed(chunk_texts)
        logger.info(f"Generated {vectors.shape[0]} vectors of dim {vectors.shape[1]}")

        # Step 4: Build metadata (stored alongside FAISS index)
        metadata = []
        for chunk in chunks:
            metadata.append({
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                **chunk.metadata,
            })

        # Step 5: Build and save FAISS index
        logger.info("Building FAISS index...")
        indexer.build_index(vectors, metadata)
        indexer.save()

        elapsed = time.time() - start_time
        logger.info(f"Ingestion complete in {elapsed:.2f}s")

        return IngestResponse(
            status="success",
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            vectors_indexed=indexer.total_vectors,
            time_seconds=round(elapsed, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get a grounded answer.

    The full pipeline:
    1. Detect query language
    2. Normalize query (transliterate, expand, remove stopwords)
    3. Embed normalized query
    4. Search FAISS index for similar chunks
    5. Apply hybrid reranking (semantic + keyword)
    6. Deduplicate overlapping chunks
    7. Generate answer with LLM (or return context if LLM unavailable)

    Example:
      POST /ask
      {"query": "Modi ji ka education kya hai?"}
      →
      {"answer": "Modi completed MA in Political Science from Gujarat University...",
       "sources": ["modi.txt"]}
    """
    if not indexer.is_loaded():
        raise HTTPException(
            status_code=400,
            detail="Index not loaded. Call POST /ingest first."
        )

    total_start = time.time()

    try:
        # ── Step 1: Normalize query ──
        norm_start = time.time()
        query_info = normalize_query(request.query)
        norm_time = (time.time() - norm_start) * 1000

        logger.info(
            f"Query: '{request.query}' → lang={query_info['language']}, "
            f"normalized='{query_info['normalized'][:60]}...'"
        )

        # ── Step 2: Embed normalized query ──
        embed_start = time.time()
        query_vector = embedding_model.embed_query(query_info["normalized"])
        embed_time = (time.time() - embed_start) * 1000

        # ── Step 3: FAISS search ──
        search_start = time.time()
        semantic_results = indexer.search(query_vector, top_k=request.top_k * 2)
        search_time = (time.time() - search_start) * 1000

        # ── Step 4: Hybrid reranking ──
        rerank_start = time.time()
        hybrid_results = hybrid_search(
            semantic_results,
            request.query,  # Use original query for keyword matching
            alpha=request.alpha,
            beta=request.beta,
        )

        # Deduplicate
        deduped_results = deduplicate_results(hybrid_results)

        # Final reranking (topic boost + source diversity)
        final_results = rerank_results(deduped_results, top_k=request.top_k)
        rerank_time = (time.time() - rerank_start) * 1000

        # ── Step 5: Generate answer ──
        gen_start = time.time()
        gen_result = generate_answer(
            request.query,
            final_results,
            query_info["language"],
        )
        gen_time = (time.time() - gen_start) * 1000

        total_time = (time.time() - total_start) * 1000

        return AskResponse(
            answer=gen_result["answer"],
            sources=gen_result["sources"],
            query_info={
                "original": query_info["original"],
                "language": query_info["language"],
                "language_label": get_language_label(query_info["language"]),
                "confidence": query_info["confidence"],
                "normalized": query_info["normalized"],
                "transliterated": query_info.get("transliterated"),
            },
            retrieval_info={
                "semantic_results": len(semantic_results),
                "hybrid_results": len(hybrid_results),
                "deduped_results": len(deduped_results),
                "final_results": len(final_results),
                "top_sources": [r.get("source") for r in final_results],
                "top_scores": [round(r.get("final_score", 0), 3) for r in final_results],
                "alpha": request.alpha,
                "beta": request.beta,
            },
            generation_info={
                "model": gen_result.get("model", "unknown"),
                "llm_used": gen_result.get("llm_used", False),
                "normalization_ms": round(norm_time, 1),
                "embedding_ms": round(embed_time, 1),
                "search_ms": round(search_time, 1),
                "rerank_ms": round(rerank_time, 1),
                "generation_ms": round(gen_time, 1),
                "total_ms": round(total_time, 1),
            },
        )

    except Exception as e:
        logger.error(f"Ask failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-web", response_model=AskWebResponse)
async def ask_web_question(request: AskWebRequest):
    """
    Ask ANY question — uses live web search for open-domain QA.

    Pipeline:
    1. Detect query language (hi / en / te / mixed)
    2. Translate query to English (for better web search results)
    3. Search DuckDuckGo with English query
    4. Convert web results → chunks → embed on-the-fly
    5. Hybrid search + rerank within web chunks
    6. Generate answer with LLM using web context

    This endpoint does NOT require /ingest — it works without any local data.

    Example:
      POST /ask-web
      {"query": "Elon Musk ki net worth kitni hai?"}
      →
      {"answer": "Elon Musk ki net worth approximately $230 billion hai...",
       "sources": ["forbes.com", "bloomberg.com"],
       "web_urls": ["https://www.forbes.com/..."]}
    """
    total_start = time.time()

    try:
        # ── Step 1: Detect language & normalize ──
        norm_start = time.time()
        query_info = normalize_query(request.query)
        language = query_info["language"]
        norm_time = (time.time() - norm_start) * 1000

        logger.info(
            f"[Web] Query: '{request.query}' → lang={language}, "
            f"normalized='{query_info['normalized'][:60]}...'"
        )

        # ── Step 2: Translate to English for web search ──
        translate_start = time.time()
        english_query = get_search_query(request.query, language)
        translate_time = (time.time() - translate_start) * 1000
        logger.info(f"[Web] Search query: '{english_query}'")

        # ── Step 3: Web search with DuckDuckGo ──
        web_start = time.time()
        web_chunks = search_and_prepare(
            query=request.query,
            english_query=english_query,
            max_results=request.max_web_results,
        )
        web_time = (time.time() - web_start) * 1000

        if not web_chunks:
            raise HTTPException(
                status_code=404,
                detail="No web results found. Try rephrasing your question."
            )
        logger.info(f"[Web] Got {len(web_chunks)} web chunks")

        # ── Step 4: Embed web chunks on-the-fly ──
        embed_start = time.time()
        chunk_texts = [c["text"] for c in web_chunks]
        vectors = embedding_model.embed(chunk_texts)

        # Build temporary FAISS index for web results
        import faiss
        import numpy as np
        temp_indexer = FAISSIndexer()
        temp_indexer.build_index(vectors, web_chunks)

        # Search within the web results
        query_vector = embedding_model.embed_query(query_info["normalized"])
        semantic_results = temp_indexer.search(
            query_vector, top_k=min(request.top_k * 2, len(web_chunks))
        )
        embed_time = (time.time() - embed_start) * 1000

        # ── Step 5: Hybrid reranking ──
        rerank_start = time.time()
        hybrid_results = hybrid_search(
            semantic_results,
            request.query,
            alpha=SEMANTIC_WEIGHT_ALPHA,
            beta=KEYWORD_WEIGHT_BETA,
        )
        deduped_results = deduplicate_results(hybrid_results)
        final_results = rerank_results(deduped_results, top_k=request.top_k)
        rerank_time = (time.time() - rerank_start) * 1000

        # ── Step 6: Generate answer with web-aware prompt ──
        gen_start = time.time()
        from generation.prompt import build_web_prompt_for_ollama
        from generation.llm import OllamaLLM

        sources = list(set(r.get("source", "unknown") for r in final_results))
        web_urls = [r.get("url", "") for r in final_results if r.get("url")]

        llm = OllamaLLM()
        if llm.is_available():
            messages = build_web_prompt_for_ollama(
                request.query, final_results, language
            )
            llm_result = llm.generate(messages=messages)
            answer = llm_result["response"]
            model_name = llm_result.get("model", "unknown")
            llm_used = True
        else:
            # Fallback: show web snippets directly
            context_summary = "\n\n".join(
                f"[{r.get('source', 'web')}]: {r.get('text', '')[:250]}..."
                for r in final_results[:3]
            )
            answer = f"(LLM unavailable — showing web search results)\n\n{context_summary}"
            model_name = "fallback-context-only"
            llm_used = False

        gen_time = (time.time() - gen_start) * 1000
        total_time = (time.time() - total_start) * 1000

        return AskWebResponse(
            answer=answer,
            sources=sources,
            web_urls=web_urls[:5],
            query_info={
                "original": query_info["original"],
                "language": query_info["language"],
                "language_label": get_language_label(query_info["language"]),
                "confidence": query_info["confidence"],
                "normalized": query_info["normalized"],
                "english_query": english_query,
                "transliterated": query_info.get("transliterated"),
            },
            search_info={
                "web_results_fetched": len(web_chunks),
                "semantic_results": len(semantic_results),
                "hybrid_results": len(hybrid_results),
                "deduped_results": len(deduped_results),
                "final_results": len(final_results),
                "top_sources": [r.get("source") for r in final_results],
                "top_scores": [
                    round(r.get("final_score", 0), 3) for r in final_results
                ],
            },
            generation_info={
                "model": model_name,
                "llm_used": llm_used,
                "normalization_ms": round(norm_time, 1),
                "translation_ms": round(translate_time, 1),
                "web_search_ms": round(web_time, 1),
                "embedding_ms": round(embed_time, 1),
                "rerank_ms": round(rerank_time, 1),
                "generation_ms": round(gen_time, 1),
                "total_ms": round(total_time, 1),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask-web failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def run_evaluation_endpoint():
    """
    Run the full evaluation suite on predefined test queries.

    Returns metrics: Recall@k, Precision@k, MRR, latency per query.
    Uses the saved eval_queries.json file.
    """
    if not indexer.is_loaded():
        raise HTTPException(
            status_code=400,
            detail="Index not loaded. Call POST /ingest first."
        )

    from evaluation.metrics import run_evaluation, load_eval_queries, print_report

    queries = load_eval_queries()
    if not queries:
        raise HTTPException(status_code=404, detail="No evaluation queries found")

    def search_fn(query: str):
        query_info = normalize_query(query)
        query_vector = embedding_model.embed_query(query_info["normalized"])
        semantic_results = indexer.search(query_vector, top_k=10)
        hybrid_results = hybrid_search(semantic_results, query)
        deduped = deduplicate_results(hybrid_results)
        return rerank_results(deduped, top_k=5)

    report = run_evaluation(search_fn, queries, top_k=5)
    print_report(report)

    return report


@app.get("/stats")
async def get_stats():
    """System statistics: index size, model info, etc."""
    return {
        "index": {
            "loaded": indexer.is_loaded(),
            "total_vectors": indexer.total_vectors,
        },
        "model": {
            "loaded": embedding_model is not None,
            "name": embedding_model.model.get_sentence_embedding_dimension()
            if embedding_model else None,
        },
        "config": {
            "top_k": RETRIEVAL_TOP_K,
            "alpha": SEMANTIC_WEIGHT_ALPHA,
            "beta": KEYWORD_WEIGHT_BETA,
        },
    }


# ─── Run with uvicorn ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
