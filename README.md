# 🇮🇳 Research-Driven Code-Mixed Multilingual RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system built for **code-mixed Indian language queries** — Hindi-English, Telugu-English, Romanized Hindi, and pure Devanagari.

Now with **🌐 Open-Domain Web Search** — ask anything in the world in Hindi, English, or Telugu!

Built with **FastAPI**, **FAISS**, **Sentence-Transformers**, **DuckDuckGo Search**, and **Ollama (Mistral 7B)**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [How It Works — Step by Step](#how-it-works--step-by-step)
- [Configuration](#configuration)
- [Evaluation Results](#evaluation-results)
- [Research Components](#research-components)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Problem Statement

Most RAG systems fail on real Indian user queries like:

| Query | Type | Challenge |
|-------|------|-----------|
| *"Modi ji ka education kya hai?"* | Code-mixed Hindi-English | Romanized Hindi words mixed with English |
| *"Chandrayaan-3 kab launch hua tha?"* | Romanized Hindi | No native script — embeddings degrade |
| *"संविधान में कितने अनुच्छेद हैं?"* | Pure Devanagari | Requires multilingual embedding model |
| *"Telangana kab bana tha?"* | Code-mixed | Needs both keyword + semantic matching |

### Core Challenges Addressed

1. **Embedding Degradation** — Multilingual models produce weaker vectors for transliterated text vs native script
2. **Semantic Drift** — Pure semantic search misses proper nouns, dates, and transliterated terms
3. **Hallucination** — LLMs fabricate answers when context is insufficient
4. **Script Variability** — Same query can arrive in Latin, Devanagari, or Telugu script

---

## Features

### Two Retrieval Modes

| Mode | Endpoint | Description |
|------|----------|-------------|
| **📁 Local Knowledge** | `POST /ask` | Answers from pre-ingested corpus files (offline, fast retrieval) |
| **🌐 Web Search** | `POST /ask-web` | Answers ANY question using live DuckDuckGo search (open-domain) |

### Core Capabilities

- **Code-Mixed Language Detection** — Identifies Hindi-English, Telugu, Tamil, Bengali, and mixed queries using Unicode analysis + Romanized Hindi word detection
- **Query Normalization** — Transliterates Romanized Hindi to Devanagari and expands queries to capture both scripts
- **Multilingual Embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, 384-dim vectors)
- **FAISS Vector Search** — Exact cosine similarity via IndexFlatIP on L2-normalized vectors
- **Hybrid Retrieval** — Combines semantic similarity (α=0.7) with keyword matching (β=0.3)
- **Post-Retrieval Reranking** — Topic boosting + source diversity enforcement + deduplication
- **Anti-Hallucination Prompting** — Strict grounding rules, source citation, refusal protocol
- **Local LLM via Ollama** — Mistral 7B runs entirely on your machine, no API keys needed

### Web Search (Open-Domain QA)

- **DuckDuckGo Integration** — Free, no API keys, no rate limits for moderate usage
- **Query Translation** — Automatically translates Hindi/Telugu queries to English for better search results (via Ollama or keyword extraction fallback)
- **On-The-Fly Embedding** — Web results are chunked, embedded, and indexed in a temporary FAISS index per request
- **Web-Aware Prompting** — Separate `WEB_SYSTEM_PROMPT` instructs the LLM to cite website names and URLs
- **Clickable References** — UI displays source URLs as clickable links

### UI & Evaluation

- **Interactive Web UI** — Built-in HTML interface with Local/Web mode toggle at `http://localhost:8000`
- **Evaluation Suite** — Recall@k, Precision@k, MRR, hallucination detection, latency benchmarks
- **Ablation Study Framework** — Compare baseline vs normalized vs hybrid vs full pipeline

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                  │
│              "Elon Musk ki net worth kitni hai?"                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Mode?      │
                    └──┬──────┬───┘
                 Local │      │ Web
                       │      │
          ┌────────────┘      └────────────┐
          ▼                                ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│  PROCESSING LAYER    │     │  PROCESSING LAYER                    │
│  language_detect.py  │     │  language_detect.py                  │
│  normalize.py        │     │  normalize.py                        │
│  → "mixed" (93%)     │     │  translate.py ← NEW                 │
│                      │     │  → English: "What is Elon Musk's     │
│                      │     │    net worth?"                       │
└──────────┬───────────┘     └──────────────┬───────────────────────┘
           │                                │
           ▼                                ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│  INGESTION LAYER     │     │  WEB SEARCH LAYER ← NEW             │
│  loader.py (files)   │     │  web_search.py                      │
│  chunker.py          │     │  DuckDuckGo → 8 web results         │
│  embedder.py         │     │  → chunk-like dicts with URLs        │
│  indexer.py (persist)│     │  embedder.py → embed on-the-fly     │
│                      │     │  indexer.py (TEMPORARY, per-request) │
└──────────┬───────────┘     └──────────────┬───────────────────────┘
           │                                │
           ▼                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER (shared by both modes)                              │
│  ┌─────────────────────────┐   ┌──────────────────────────────────┐ │
│  │ search.py               │   │ rerank.py                        │ │
│  │ Hybrid scoring:         │   │ Topic boost (+0.1)               │ │
│  │ α(0.7)×semantic         │   │ Source diversity (max 3/source)  │ │
│  │   + β(0.3)×keyword      │   │ Trim to top_k                   │ │
│  │ Deduplication (>95%)    │   │                                  │ │
│  └─────────────────────────┘   └──────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  GENERATION LAYER                                                    │
│  ┌─────────────────────┐   ┌──────────────────────────────────────┐ │
│  │ prompt.py            │   │ llm.py                              │ │
│  │ Local: SYSTEM_PROMPT │──▶│ OllamaLLM client                   │ │
│  │ Web: WEB_SYSTEM_PROMPT│  │ POST /api/chat → Mistral 7B        │ │
│  │ Anti-hallucination   │   │ Fallback to raw context if offline  │ │
│  │ Source/URL citation  │   │                                      │ │
│  └─────────────────────┘   └──────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│               Grounded Answer + Sources/URLs + Timings               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
multi-source-rag/
│
├── backend/                    # Core application layer
│   ├── __init__.py
│   ├── app.py                  # FastAPI server — /ask, /ask-web, /ingest, /health
│   └── config.py               # Central configuration (all tunable parameters)
│
├── ingestion/                  # Document processing pipeline
│   ├── __init__.py
│   ├── loader.py               # Reads .txt/.json/.md → Document objects
│   ├── chunker.py              # Sliding-window chunking (500 chars, 50 overlap)
│   ├── embedder.py             # Multilingual sentence embeddings (MiniLM-L12-v2)
│   ├── indexer.py              # FAISS IndexFlatIP — build, save, load, search
│   └── web_search.py           # 🌐 DuckDuckGo search → web chunks (NEW)
│
├── processing/                 # Query understanding
│   ├── __init__.py
│   ├── language_detect.py      # Detects EN / HI / TE / TA / BN / Mixed
│   ├── normalize.py            # Transliteration + query expansion + stopword removal
│   └── translate.py            # 🌐 Hindi/Telugu → English translation for web search (NEW)
│
├── retrieval/                  # Search and ranking (shared by both modes)
│   ├── __init__.py
│   ├── search.py               # Hybrid search (semantic + keyword) + deduplication
│   └── rerank.py               # Topic boosting + source diversity enforcement
│
├── generation/                 # Answer generation
│   ├── __init__.py
│   ├── prompt.py               # SYSTEM_PROMPT + WEB_SYSTEM_PROMPT + anti-hallucination
│   └── llm.py                  # Ollama LLM client with timeout + fallback
│
├── evaluation/                 # Research metrics
│   ├── __init__.py
│   ├── metrics.py              # Recall@k, MRR, Precision@k, hallucination check
│   └── benchmarks.py           # Ablation study framework
│
├── data/
│   ├── raw/                    # Source corpus (5 multilingual documents)
│   │   ├── modi.txt            # PM Modi — politics (EN + HI)
│   │   ├── indian_education.txt# Education system — NEP 2020 (EN + HI)
│   │   ├── constitution.txt    # Indian Constitution — law (EN + HI)
│   │   ├── isro.txt            # ISRO / Space — Chandrayaan (EN + HI)
│   │   └── telangana.txt       # Telangana state — geography (EN + TE)
│   ├── eval_queries.json       # 12 test queries with ground truth
│   └── index/                  # Generated at runtime (gitignored)
│       ├── faiss.index         # FAISS binary index
│       └── metadata.pkl        # Chunk metadata (text, source, topic)
│
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── FLOW_EXPLANATION.md         # Detailed step-by-step flow with examples
└── README.md                   # This file
```

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Python** | 3.10 or higher | Runtime |
| **pip** | Latest | Package manager |
| **Ollama** | Latest | Local LLM server |
| **RAM** | 8 GB minimum | Embedding model (~500MB) + Mistral 7B (~4.4GB) |
| **Disk** | ~6 GB free | Model downloads + index storage |

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/multi-source-rag.git
cd multi-source-rag
```

### Step 2: Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `fastapi==0.115.0` — Web framework
- `uvicorn==0.30.0` — ASGI server
- `sentence-transformers==3.0.0` — Multilingual embeddings
- `torch>=2.0.0` — PyTorch (CPU)
- `faiss-cpu>=1.9.0` — Vector similarity search
- `numpy>=1.24.0` — Numerical computing
- `requests>=2.31.0` — HTTP client for Ollama
- `duckduckgo-search==7.5.3` — Web search for open-domain QA (no API key needed)
- `pydantic==2.9.0` — Data validation
- `python-multipart==0.0.9` — Form data parsing

### Step 4: Install Ollama (for LLM Generation)

**Windows:**
Download and install from [https://ollama.ai/download](https://ollama.ai/download)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 5: Pull the Mistral Model

```bash
ollama pull mistral
```

This downloads the Mistral 7B model (~4.4 GB). Wait for it to complete.

> **Note:** If you have limited RAM, use a smaller model instead:
> ```bash
> ollama pull phi3:mini
> ```
> Then update `OLLAMA_MODEL` in `.env` or `backend/config.py`.

### Step 6: Set Up Environment Variables (Optional)

```bash
cp .env.example .env
```

Edit `.env` if you need to change the Ollama URL or model name. The defaults work out of the box.

### Step 7: Start the Server

```bash
python -m backend.app
```

You should see:
```
INFO:     Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2
INFO:     Model loaded. Dimension: 384
INFO:     No existing index found. Use POST /ingest to create one.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> **First run:** The embedding model (~471 MB) will be downloaded automatically. Subsequent runs use the cached version.

### Step 8: Open the Web UI

Open your browser and go to: **http://localhost:8000**

### Step 9: Ingest Documents

Click the **"Ingest Documents"** button in the UI, or run:

```bash
curl -X POST http://localhost:8000/ingest
```

This processes the 5 corpus files: load → chunk (21 chunks) → embed (384-dim vectors) → build FAISS index.

### Step 10: Ask Questions!

Type a question in the UI or use the API:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Modi ji ka education kya hai?\"}"
```

---

## Usage

### Web UI

The interactive web interface is available at `http://localhost:8000` with:

- **Mode Toggle** — Switch between **📁 Local Knowledge** and **🌐 Web Search** modes
- **Local mode** — Ask questions about your ingested documents
- **Web mode** — Ask anything in the world in Hindi, English, or Telugu (code-mixed too!)
- Sample query buttons for quick testing (different samples per mode)
- Real-time display of: answer, sources/URLs, language detection, retrieval pipeline stats, and performance timings
- Ingest and Evaluation buttons

### API (cURL / Postman)

**Ask from local knowledge:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Modi ji ka education kya hai?\", \"top_k\": 5}"
```

**Ask anything (web search):**
```bash
curl -X POST http://localhost:8000/ask-web \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Elon Musk ki net worth kitni hai?\"}"
```

**Ask in Telugu-English mix (web search):**
```bash
curl -X POST http://localhost:8000/ask-web \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"IPL mein sabse zyada runs kisne banaye?\"}"
```

**Ingest documents:**
```bash
curl -X POST http://localhost:8000/ingest
```

**Run evaluation:**
```bash
curl -X POST http://localhost:8000/evaluate
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive web UI with Local/Web mode toggle |
| `GET` | `/health` | System status — model loaded, index size |
| `POST` | `/ingest` | Load documents → chunk → embed → build FAISS index |
| `POST` | `/ask` | Ask from local knowledge → grounded answer with file sources |
| `POST` | `/ask-web` | 🌐 Ask anything → web search → answer with URL sources |
| `POST` | `/evaluate` | Run evaluation suite on 12 test queries |
| `GET` | `/stats` | Index and system statistics |

### POST /ask — Local Knowledge

Request:
```json
{
    "query": "Modi ji ka education kya hai?",
    "top_k": 5,
    "alpha": 0.7,
    "beta": 0.3
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *required* | The question (any language) |
| `top_k` | int | 5 | Number of results to retrieve |
| `alpha` | float | 0.7 | Semantic similarity weight |
| `beta` | float | 0.3 | Keyword match weight |

Response:
```json
{
    "answer": "Modi completed his higher secondary education in Vadnagar...",
    "sources": ["modi.txt"],
    "query_info": {
        "original": "Modi ji ka education kya hai?",
        "language": "mixed",
        "language_label": "Code-Mixed (Hindi-English)",
        "confidence": 1.0,
        "normalized": "Modi ji ka education kya hai? Modi जी का education क्या है?",
        "transliterated": "Modi जी का education क्या है?"
    },
    "retrieval_info": {
        "semantic_results": 10,
        "hybrid_results": 6,
        "deduped_results": 2,
        "final_results": 2,
        "alpha": 0.7,
        "beta": 0.3
    },
    "generation_info": {
        "model": "mistral",
        "llm_used": true,
        "normalization_ms": 2,
        "embedding_ms": 362,
        "search_ms": 3,
        "rerank_ms": 1,
        "generation_ms": 89739,
        "total_ms": 90107
    }
}
```

### POST /ask-web — Open-Domain Web Search

Request:
```json
{
    "query": "Elon Musk ki net worth kitni hai?",
    "top_k": 5,
    "max_web_results": 8
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *required* | Any question in Hindi/English/Telugu (mixed OK) |
| `top_k` | int | 5 | Number of web results to use for answer |
| `max_web_results` | int | 8 | Number of web results to fetch from DuckDuckGo |

Response:
```json
{
    "answer": "Elon Musk ki net worth approximately $638 billion hai, as per fortune.com.",
    "sources": ["forbes.com", "celebritynetworth.com", "businessinsider.com"],
    "web_urls": [
        "https://www.celebritynetworth.com/richest-businessmen/ceos/elon-musk-net-worth/",
        "https://www.businessinsider.com/elon-musk-net-worth",
        "https://fortune.com/2025/12/16/elon-musk-wealth-soared-past-600-billion..."
    ],
    "query_info": {
        "original": "Elon Musk ki net worth kitni hai?",
        "language": "mixed",
        "language_label": "Code-Mixed (Hindi-English)",
        "confidence": 0.93,
        "normalized": "Elon Musk ki net worth kitni hai? Elon Musk की net worth कितनी है?",
        "english_query": "What is Elon Musk's net worth?",
        "transliterated": "Elon Musk की net worth कितनी है?"
    },
    "search_info": {
        "web_results_fetched": 8,
        "semantic_results": 8,
        "hybrid_results": 7,
        "final_results": 5,
        "top_sources": ["celebritynetworth.com", "forbes.com", "businessinsider.com"]
    },
    "generation_info": {
        "model": "mistral",
        "llm_used": true,
        "normalization_ms": 2,
        "translation_ms": 15675,
        "web_search_ms": 1292,
        "embedding_ms": 522,
        "rerank_ms": 1,
        "generation_ms": 80720,
        "total_ms": 98212
    }
}
```

---

## How It Works — Step by Step

### Flow A — Local Knowledge (`POST /ask`)

Using the example query: **"Modi ji ka education kya hai?"**

#### 1. Language Detection (`processing/language_detect.py`)
- Scans Unicode ranges → all Latin, no Devanagari
- Checks tokens against Romanized Hindi word list → `ji`, `ka`, `kya`, `hai` match (4/6 = 67%)
- 67% > 15% threshold → classified as **"mixed"** (Code-Mixed Hindi-English)

#### 2. Query Normalization (`processing/normalize.py`)
- **Transliterate:** `ji→जी`, `ka→का`, `kya→क्या`, `hai→है` → `"Modi जी का education क्या है?"`
- **Expand:** Concatenate original + transliterated → embedding captures both scripts
- **Stopword removal:** Remove `ka`, `hai` for cleaner keyword matching

#### 3. Embedding (`ingestion/embedder.py`)
- Feed expanded query to `paraphrase-multilingual-MiniLM-L12-v2`
- 12-layer transformer → mean pooling → L2 normalization
- Output: 384-dimensional unit vector

#### 4. FAISS Search (`ingestion/indexer.py`)
- Compute dot product (= cosine similarity) against all 21 indexed vectors
- Return top-10 highest-scoring chunks with metadata

#### 5. Hybrid Reranking (`retrieval/search.py`)
- Extract keywords from query (words ≥ 3 chars)
- For each result: `final_score = 0.7 × semantic + 0.3 × keyword_overlap`
- Filter below threshold (0.25), sort by final score → 6 results

#### 6. Deduplication (`retrieval/search.py`)
- Compare token overlap between remaining chunks
- Remove chunks with >95% similarity to a higher-ranked result → 2 results

#### 7. Post-Retrieval Reranking (`retrieval/rerank.py`)
- Boost results matching detected topic (+0.1)
- Enforce source diversity (max 3 from same file)

#### 8. Prompt Construction (`generation/prompt.py`)
- Build context block with `[Source: filename.txt]` labels
- Inject `SYSTEM_PROMPT` with anti-hallucination rules (strict grounding, citation requirements, refusal protocol)

#### 9. LLM Generation (`generation/llm.py`)
- Send prompt to Ollama Mistral 7B via `POST /api/chat`
- Temperature = 0.1 (factual, low creativity)
- Response: grounded answer with source citations
- Fallback: if Ollama is unavailable, returns raw context chunks

#### 10. Response Assembly (`backend/app.py`)
- Combine answer + sources + query analysis + retrieval stats + per-stage timings
- Return as structured JSON → rendered in web UI

---

### Flow B — Web Search (`POST /ask-web`)

Using the example query: **"Elon Musk ki net worth kitni hai?"**

This question **cannot** be answered from local corpus files — it requires real-time web data.

#### 1. Language Detection & Normalization (same as local flow)
- Detect language → `"mixed"` (93% confidence)
- Transliterate + expand query → captures both Romanized and Devanagari forms

#### 2. Query Translation (`processing/translate.py`) ← NEW
- **Purpose:** DuckDuckGo returns MUCH better results for English queries
- **Primary method:** Uses Ollama to translate: `"Elon Musk ki net worth kitni hai?"` → `"What is Elon Musk's net worth?"`
- **Fallback:** If Ollama is unavailable, extracts keywords by removing Hindi/Telugu stopwords: → `"Elon Musk net worth"`
- Translation uses `temperature=0.0` for deterministic output, capped at 60s timeout

#### 3. DuckDuckGo Web Search (`ingestion/web_search.py`) ← NEW
- Searches DuckDuckGo with the English query
- Region: `"in-en"` (India-English) for regional relevance
- Fetches 8 web results with title, snippet, URL, and domain
- Converts to chunk-like dicts compatible with existing pipeline:
  ```python
  {"text": "title\nsnippet", "source": "forbes.com", "url": "https://...", "topic": "web"}
  ```
- If English translation returned no results, retries with original Hindi query

#### 4. On-The-Fly Embedding & Temporary Index
- Embed all 8 web snippets using the same multilingual model → shape `(8, 384)`
- Build a **temporary FAISS index** (not saved to disk) — ephemeral per request
- Search within the web results using the normalized query vector

#### 5. Hybrid Search + Rerank (same pipeline as local flow)
- Same `α=0.7 semantic + β=0.3 keyword` scoring
- Same deduplication and source diversity enforcement
- Web results are naturally diverse (different domains), so dedup rarely removes any

#### 6. Web-Aware Prompt (`generation/prompt.py`) ← NEW
- Uses `WEB_SYSTEM_PROMPT` (different from local `SYSTEM_PROMPT`):
  - Instructs LLM to cite **website names** instead of filenames
  - Context block includes `[Source: domain.com | URL: https://...]`
- Calls `build_web_prompt_for_ollama()` with language-aware instructions

#### 7. LLM Generation + Response Assembly
- Same Ollama Mistral 7B call as local flow
- Response includes `web_urls` list with clickable links to original web pages
- UI renders with 🌐 header, source URLs as clickable badges, and a timing bar showing: Normalize → Translate → Web Search → Embed+Search → Rerank → Generate

> **See [FLOW_EXPLANATION.md](FLOW_EXPLANATION.md) for a detailed walkthrough with exact function calls, data types, intermediate outputs, and real response examples.**

---

## Configuration

All settings are in `backend/config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embedding model |
| `EMBEDDING_DIMENSION` | `384` | Output vector dimensionality |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MIN_CHUNK_LENGTH` | `30` | Minimum chunk size (discard smaller) |
| `RETRIEVAL_TOP_K` | `5` | Number of results to return |
| `SEMANTIC_WEIGHT_ALPHA` | `0.7` | Weight for semantic similarity |
| `KEYWORD_WEIGHT_BETA` | `0.3` | Weight for keyword matching |
| `SIMILARITY_THRESHOLD` | `0.25` | Minimum score to include result |
| `OLLAMA_MODEL` | `mistral` | LLM model name |
| `OLLAMA_TIMEOUT` | `120` | LLM request timeout (seconds) |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM temperature (lower = more factual) |
| `WEB_SEARCH_MAX_RESULTS` | `8` | Number of web results to fetch per query |
| `WEB_SEARCH_REGION` | `in-en` | DuckDuckGo region (India-English) |
| `WEB_SEARCH_ENABLED` | `True` | Master toggle for web search feature |

Environment variables (`.env`) override `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.

---

## Evaluation Results

Tested on 12 multilingual queries (English, Hindi, Code-Mixed, Telugu):

| Metric | Score |
|--------|-------|
| **Recall@5** | 1.000 (100%) |
| **MRR** | 1.000 |
| **Precision@5** | 0.764 (76.4%) |
| **Avg Latency** | 21.4 ms (retrieval only) |

### Sample Queries Tested

| Query | Language | Source Hit |
|-------|----------|-----------|
| Modi ji ka education kya hai? | Code-Mixed | ✅ modi.txt |
| Chandrayaan-3 kab launch hua tha? | Romanized Hindi | ✅ isro.txt |
| What are the fundamental rights? | English | ✅ constitution.txt |
| Telangana kab bana tha? | Code-Mixed | ✅ telangana.txt |
| NEP 2020 mein kya changes hain? | Code-Mixed | ✅ indian_education.txt |
| ISRO ka headquarter kahan hai? | Code-Mixed | ✅ isro.txt |

---

## Research Components

| Component | What It Measures | Key Finding |
|-----------|------------------|-------------|
| **Embedding Quality** | Cosine similarity: EN vs HI vs Code-Mixed | Code-mixed queries show ~15-20% similarity degradation vs native script |
| **Hybrid Retrieval** | Recall@k with/without keyword boosting | Keywords recover proper nouns and dates missed by pure semantic search |
| **Query Normalization** | Precision gain from transliteration | Expanding to both scripts improves recall by capturing cross-script matches |
| **Hallucination Control** | Grounding rate with strict vs weak prompts | Anti-hallucination prompt reduces fabrication to near-zero |
| **Latency Benchmarks** | End-to-end timing per stage | Embedding dominates retrieval time; LLM dominates overall time |
| **Ablation Studies** | Isolate each component's contribution | Full pipeline (norm + hybrid + dedup + rerank) outperforms all partial configs |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Web Framework** | FastAPI 0.115 | Async, auto-docs, Pydantic validation |
| **Embedding Model** | paraphrase-multilingual-MiniLM-L12-v2 | Best latency-quality tradeoff for 50+ languages on CPU |
| **Vector Store** | FAISS (IndexFlatIP) | C++ speed, exact cosine similarity, persistent |
| **LLM** | Ollama + Mistral 7B | Local inference, no API keys, quantized for 8GB RAM |
| **Web Search** | DuckDuckGo (duckduckgo-search) | Free, no API keys, no rate limits, clean snippets |
| **Language** | Python 3.10+ | Ecosystem support for ML/NLP |
| **Deep Learning** | PyTorch (CPU) | Backend for sentence-transformers |

---

## Troubleshooting

### "No existing index found"
Run ingestion first: click **Ingest Documents** in the UI or `curl -X POST http://localhost:8000/ingest`

### Embedding model download is slow
The first run downloads ~471 MB. Subsequent runs use the cached model from `~/.cache/torch/sentence_transformers/`.

### Ollama connection error
Make sure Ollama is running:
```bash
ollama serve
```
Check if the model is pulled:
```bash
ollama list
```
If empty, pull the model:
```bash
ollama pull mistral
```

### LLM generation is slow (~90 seconds)
Mistral 7B on CPU is slow. Options:
1. Use a smaller model: `ollama pull phi3:mini` and set `OLLAMA_MODEL=phi3:mini`
2. Use GPU: if you have an NVIDIA GPU, Ollama automatically uses CUDA

### `faiss-cpu` installation fails
On Python 3.13+, use: `pip install faiss-cpu>=1.9.0`

### PowerShell script execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Web search returns no results
- Check your internet connection — DuckDuckGo requires network access
- Try rephrasing the query in English for best results
- If behind a corporate proxy, DuckDuckGo requests may be blocked

### Web search translation is slow
- Translation uses Ollama (same LLM), which takes ~15s on CPU
- If Ollama is unavailable, the system automatically falls back to keyword extraction (instant)
- The keyword fallback still produces good search queries (e.g., `"Elon Musk net worth"`)

---

## Adding Your Own Documents

1. Place `.txt`, `.json`, or `.md` files in `data/raw/`
2. Restart the server or call `POST /ingest`
3. The system automatically detects topics, chunks text, and rebuilds the index

Supported formats:
- **`.txt`** — Plain text (UTF-8)
- **`.json`** — JSON objects or arrays (converted to readable text)
- **`.md`** — Markdown files (treated as plain text)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
