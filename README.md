# 🇮🇳 Research-Driven Code-Mixed Multilingual RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system built for **code-mixed Indian language queries** — Hindi-English, Telugu-English, Romanized Hindi, and pure Devanagari.

Built with **FastAPI**, **FAISS**, **Sentence-Transformers**, and **Ollama (Mistral 7B)**.

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

- **Code-Mixed Language Detection** — Identifies Hindi-English, Telugu, Tamil, Bengali, and mixed queries using Unicode analysis + Romanized Hindi word detection
- **Query Normalization** — Transliterates Romanized Hindi to Devanagari and expands queries to capture both scripts
- **Multilingual Embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, 384-dim vectors)
- **FAISS Vector Search** — Exact cosine similarity via IndexFlatIP on L2-normalized vectors
- **Hybrid Retrieval** — Combines semantic similarity (α=0.7) with keyword matching (β=0.3)
- **Post-Retrieval Reranking** — Topic boosting + source diversity enforcement + deduplication
- **Anti-Hallucination Prompting** — Strict grounding rules, source citation, refusal protocol
- **Local LLM via Ollama** — Mistral 7B runs entirely on your machine, no API keys needed
- **Interactive Web UI** — Built-in HTML interface at `http://localhost:8000`
- **Evaluation Suite** — Recall@k, Precision@k, MRR, hallucination detection, latency benchmarks
- **Ablation Study Framework** — Compare baseline vs normalized vs hybrid vs full pipeline

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                  │
│              "Modi ji ka education kya hai?"                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PROCESSING LAYER                                                    │
│  ┌─────────────────────┐   ┌──────────────────────────────────────┐ │
│  │ language_detect.py   │──▶│ normalize.py                        │ │
│  │ Unicode analysis     │   │ Transliterate → Expand → Stopwords  │ │
│  │ Romanized Hindi det. │   │ "Modi जी का education क्या है?"      │ │
│  │ → "mixed" (100%)     │   │                                      │ │
│  └─────────────────────┘   └──────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  INGESTION / EMBEDDING LAYER                                         │
│  ┌───────────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────────┐ │
│  │ loader.py     │ │ chunker.py   │ │ embedder.py│ │ indexer.py  │ │
│  │ Read files    │ │ Sliding      │ │ MiniLM-L12 │ │ FAISS       │ │
│  │ .txt/.json/.md│ │ window       │ │ 384-dim    │ │ IndexFlatIP │ │
│  │ Topic detect  │ │ 500ch/50ovl  │ │ L2-norm    │ │ Save/Load   │ │
│  └───────────────┘ └──────────────┘ └────────────┘ └─────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER                                                     │
│  ┌─────────────────────────┐   ┌──────────────────────────────────┐ │
│  │ search.py               │   │ rerank.py                        │ │
│  │ Hybrid scoring:         │   │ Topic boost (+0.1)               │ │
│  │ α(0.7)×semantic         │   │ Source diversity (max 3/file)    │ │
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
│  │ SYSTEM_PROMPT with   │──▶│ OllamaLLM client                   │ │
│  │ anti-hallucination   │   │ POST /api/chat → Mistral 7B        │ │
│  │ Source citation rules│   │ Fallback to raw context if offline  │ │
│  └─────────────────────┘   └──────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Grounded Answer + Sources + Timings               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
multi-source-rag/
│
├── backend/                    # Core application layer
│   ├── __init__.py
│   ├── app.py                  # FastAPI server — orchestrates all modules
│   └── config.py               # Central configuration (all tunable parameters)
│
├── ingestion/                  # Document processing pipeline
│   ├── __init__.py
│   ├── loader.py               # Reads .txt/.json/.md → Document objects
│   ├── chunker.py              # Sliding-window chunking (500 chars, 50 overlap)
│   ├── embedder.py             # Multilingual sentence embeddings (MiniLM-L12-v2)
│   └── indexer.py              # FAISS IndexFlatIP — build, save, load, search
│
├── processing/                 # Query understanding
│   ├── __init__.py
│   ├── language_detect.py      # Detects EN / HI / TE / TA / BN / Mixed
│   └── normalize.py            # Transliteration + query expansion + stopword removal
│
├── retrieval/                  # Search and ranking
│   ├── __init__.py
│   ├── search.py               # Hybrid search (semantic + keyword) + deduplication
│   └── rerank.py               # Topic boosting + source diversity enforcement
│
├── generation/                 # Answer generation
│   ├── __init__.py
│   ├── prompt.py               # Anti-hallucination prompt engineering
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
├── FLOW_EXPLANATION.md         # Detailed step-by-step flow with example
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
- Text input for questions (any language)
- Sample query buttons for quick testing
- Real-time display of: answer, sources, language detection, retrieval pipeline stats, and performance timings
- Ingest and Evaluation buttons

### API (cURL / Postman)

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What are the fundamental rights?\", \"top_k\": 5}"
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
| `GET` | `/` | Interactive web UI (HTML) |
| `GET` | `/health` | System status — model loaded, index size |
| `POST` | `/ingest` | Load documents → chunk → embed → build FAISS index |
| `POST` | `/ask` | Ask a question → get grounded answer with sources |
| `POST` | `/evaluate` | Run evaluation suite on 12 test queries |
| `GET` | `/stats` | Index and system statistics |

### POST /ask — Request Body

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

### POST /ask — Response

```json
{
    "answer": "Modi completed his higher secondary education in Vadnagar and later earned an MA in Political Science from Gujarat University. (Source: modi.txt)",
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

---

## How It Works — Step by Step

Using the example query: **"Modi ji ka education kya hai?"**

### 1. Language Detection (`processing/language_detect.py`)
- Scans Unicode ranges → all Latin, no Devanagari
- Checks tokens against Romanized Hindi word list → `ji`, `ka`, `kya`, `hai` match (4/6 = 67%)
- 67% > 15% threshold → classified as **"mixed"** (Code-Mixed Hindi-English)

### 2. Query Normalization (`processing/normalize.py`)
- **Transliterate:** `ji→जी`, `ka→का`, `kya→क्या`, `hai→है` → `"Modi जी का education क्या है?"`
- **Expand:** Concatenate original + transliterated → embedding captures both scripts
- **Stopword removal:** Remove `ka`, `hai` for cleaner keyword matching

### 3. Embedding (`ingestion/embedder.py`)
- Feed expanded query to `paraphrase-multilingual-MiniLM-L12-v2`
- 12-layer transformer → mean pooling → L2 normalization
- Output: 384-dimensional unit vector

### 4. FAISS Search (`ingestion/indexer.py`)
- Compute dot product (= cosine similarity) against all 21 indexed vectors
- Return top-10 highest-scoring chunks with metadata

### 5. Hybrid Reranking (`retrieval/search.py`)
- Extract keywords from query (words ≥ 3 chars)
- For each result: `final_score = 0.7 × semantic + 0.3 × keyword_overlap`
- Filter below threshold (0.25), sort by final score → 6 results

### 6. Deduplication (`retrieval/search.py`)
- Compare token overlap between remaining chunks
- Remove chunks with >95% similarity to a higher-ranked result → 2 results

### 7. Post-Retrieval Reranking (`retrieval/rerank.py`)
- Boost results matching detected topic (+0.1)
- Enforce source diversity (max 3 from same file)

### 8. Prompt Construction (`generation/prompt.py`)
- Build context block with `[Source: filename]` labels
- Inject anti-hallucination system prompt (strict grounding rules, citation requirements, refusal protocol)

### 9. LLM Generation (`generation/llm.py`)
- Send prompt to Ollama Mistral 7B via `POST /api/chat`
- Temperature = 0.1 (factual, low creativity)
- Response: grounded answer with source citations
- Fallback: if Ollama is unavailable, returns raw context chunks

### 10. Response Assembly (`backend/app.py`)
- Combine answer + sources + query analysis + retrieval stats + per-stage timings
- Return as structured JSON → rendered in web UI

> **See [FLOW_EXPLANATION.md](FLOW_EXPLANATION.md) for a detailed walkthrough with exact function calls, data types, and intermediate outputs.**

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

## Hybrid Scoring Formula

```
FinalScore = α × SemanticScore + β × KeywordBoost
```
- `α = 0.7` (semantic weight)
- `β = 0.3` (keyword weight)
- Tuned via grid search in ablation studies

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System status |
| POST | `/ingest` | Index documents |
| POST | `/ask` | Ask a question |
| POST | `/evaluate` | Run evaluation suite |
| GET | `/stats` | Index statistics |

## LLM Setup (Optional)

For full answer generation, install [Ollama](https://ollama.ai):
```bash
# Install Ollama, then:
ollama pull mistral:7b-instruct
```
Without Ollama, the system returns retrieved context as the answer (retrieval still works fully).

## Technologies

- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, 384-dim)
- **Vector DB**: FAISS (IndexFlatIP, cosine similarity)
- **LLM**: Ollama (Mistral 7B / Phi-3 Mini)
- **API**: FastAPI + Uvicorn
- **Languages**: Python 3.10+
