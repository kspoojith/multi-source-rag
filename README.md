# 🌐 Multilingual RAG System v2.1

**Answer ANY question in the world** using real-time web search powered by AI.

No local files needed. No manual curation. Just ask anything in English, Hindi, Telugu, or code-mixed language!

⚡ **NEW:** Cloud deployment with **Groq API** for 10-30x faster responses (1-3s vs 40-90s)!

![Version](https://img.shields.io/badge/Version-2.1-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [User Interfaces](#-user-interfaces)
- [How It Works](#-how-it-works)
- [Configuration](#️-configuration)
- [API Documentation](#-api-documentation)
- [Performance Optimization](#-performance-optimization)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Tech Stack](#-tech-stack)

---

## ✨ Features

### 🌍 Multilingual Support
- **Pure Hindi:** "भारत की राजधानी क्या है?"
- **Pure English:** "What is the capital of India?"
- **Code-Mixed (Romanized):** "India ki capital kya hai?"
- **Telugu:** "భారతదేశ రాజధాని ఏమిటి?"
- **Automatic Language Detection** with 80%+ confidence
- **Tested:** English+Hindi+Telugu code-mixed queries work seamlessly

### 🔍 Web-Only Architecture
- **Real-time web search** via DuckDuckGo (no API key needed)
- **Answer ANYTHING** - not limited to pre-loaded documents
- **Automatic source citations** with clickable URLs
- **Domain authority** visible for each source

### ⚡ Smart Caching
- Popular queries served in <1 second
- 1-hour TTL (configurable)
- Cache hit rate tracking
- One-click cache clearing

### 🎨 Dual UI Options
1. **Streamlit Frontend** (Recommended)
   - Modern, tabbed interface
   - Real-time system monitoring
   - Debug mode & advanced metrics
   - Mobile-responsive design

2. **FastAPI HTML** (Embedded)
   - Simple, lightweight
   - No extra dependencies
   - Good for API-first deployments

### 🚀 Production-Ready
- **Dual LLM Support:**
  - 🏠 **Ollama** (local): phi3:mini, mistral, llama3
  - ☁️ **Groq API** (cloud): llama-3.3-70b-versatile (free tier: 14,400 req/day)
- **Performance:** 1-3s with Groq vs 40-90s with Ollama
- **Zero-cost deployment** on Streamlit Community Cloud
- CORS enabled
- Comprehensive logging
- Health check endpoints
- Error handling & retries with automatic fallback

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- Internet connection for web search

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose Your LLM Provider

**Option A: Groq API (Recommended - Fast & Free)**

```bash
# 1. Get free API key from: https://console.groq.com
# 2. Copy .env.example to .env
cp .env.example .env

# 3. Edit .env file and add your API key:
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

**Benefits:**
- ⚡ **1-3 seconds** per query (10-30x faster!)
- 🆓 **14,400 requests/day** free tier
- 🌐 Works anywhere (no GPU needed)
- 📱 Deploy to cloud easily

**Option B: Ollama (Local - Private)**

```bash
# Download Ollama from https://ollama.ai
# On Windows: Run installer
# On Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose based on your hardware):

# Option 1: phi3:mini (Recommended for CPU) - 2.2GB, fast
ollama pull phi3:mini

# Option 2: mistral (For GPU) - 4.4GB, higher quality
ollama pull mistral

# Option 3: llama3 (Best quality, needs GPU) - 4.7GB
ollama pull llama3

# Edit .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=phi3:mini
```

**Benefits:**
- 🔒 **100% private** (no data sent to cloud)
- 🏠 **Offline capable** (after model download)
- 💰 **Truly free** (no API limits)
- ⏱️ **40-90 seconds** per query on CPU

### 3. Start the Backend

**With Groq API:**
```bash
# Set environment variables (Windows PowerShell)
$env:LLM_PROVIDER="groq"
$env:GROQ_API_KEY="gsk_your_actual_key_here"
python -m backend.app

# Or on Linux/Mac:
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_your_actual_key_here
python -m backend.app
```

**With Ollama:**
```bash
python -m backend.app
```

You should see:
```
✅ Loaded environment variables from: .env
🚀 Starting Web-Only Multilingual RAG System...
✅ System ready - answer ANY question from the web!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Choose Your Interface

**Option A: Streamlit UI (Recommended)**

In a new terminal:
```bash
streamlit run streamlit_app.py
```

Access at: **http://localhost:8501**

**Option B: FastAPI HTML UI**

Access at: **http://localhost:8000**

Access at: **http://localhost:8000**

### 5. Try It Out!

**Single-language queries:**
- "Elon Musk ki net worth kitni hai?"
- "IPL 2024 winner kaun hai?"
- "Latest AI news kya hai?"
- "India ka prime minister kon hai?"
- "Climate change kya hai?"
- "Python vs JavaScript comparison"

**Code-mixed queries (English + Hindi + Telugu):**
- "AI technology ka future kya hai aur daani impact on jobs ela untundi?"
- "Elon Musk ki company Tesla eppudu start ayindi and unki net worth kitni hai?"
- "Modi government ki latest policies for education sector emiti mariyu student loans par kya changes hue?"

**Performance:**
- Groq API: **1-3 seconds** ⚡
- Ollama (CPU): **40-90 seconds** 🐢
- Cached queries: **<1 second** 🚀

---

## 🎨 User Interfaces

### Streamlit Frontend (Recommended)

**Start:**
```bash
streamlit run streamlit_app.py
```

**Features:**
- 💬 **4 Tabbed Sections:**
  - Answer: AI-generated response with model info
  - Analysis: Language detection, query processing
  - Sources & References: Clickable source URLs
  - Performance: Detailed timing breakdown with progress bars
  
- 📊 **Sidebar Controls:**
  - System Status: Real-time backend health
  - Cache Statistics: Hit rate, total queries
  - Settings: Caching, debug mode, advanced metrics
  - About: Tech stack details

- ✨ **Advanced Features:**
  - Debug mode with JSON responses
  - Refresh button for system status
  - Mobile-responsive design
  - Progress indicators during search
  - Auto-refreshing cache stats

### FastAPI HTML UI

Built-in at http://localhost:8000 when backend runs.

**Simpler but functional:**
- Query input with examples
- Answer display
- Source citations
- Basic performance metrics

---

## 🔄 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  User Query (any language)                                   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  1. Language Detection                                       │
│     → Hindi / English / Telugu / Code-Mixed (90% confidence) │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Query Normalization                                      │
│     → Transliterate Romanized Hindi to Devanagari           │
│     → Expand abbreviations                                   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Translation to English                                   │
│     → Google Translate API                                   │
│     → "India ki capital kya hai" → "What is India's capital" │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  4. DuckDuckGo Web Search                                    │
│     → Fetch 4-6 web results (no API key needed)              │
│     → Extract titles + snippets                              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Chunk & Embed                                            │
│     → Split into 300-char chunks                             │
│     → Embed using Sentence-Transformers (384-dim multilingual)│
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  6. Hybrid Retrieval                                         │
│     → Cosine similarity (semantic)                           │
│     → BM25 keyword matching                                  │
│     → Combined score ranking                                 │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  7. LLM Generation (Ollama)                                  │
│     → Top 3 chunks as context                                │
│     → Anti-hallucination prompt                              │
│     → Source-grounded generation                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Answer + Sources + Performance Metrics                      │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Environment Variables (.env file)

```bash
# ─── LLM Provider Selection ───────────────────────────────
LLM_PROVIDER=groq  # Options: "groq" (cloud) or "ollama" (local)

# ─── Groq API Configuration (Cloud LLM) ───────────────────
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.3
GROQ_MAX_TOKENS=1024

# ─── Ollama Configuration (Local LLM) ─────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

### Application Settings (backend/config.py)

```python
# ─── LLM Settings ─────────────────────────────────────────
# Groq (read from environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Ollama (read from environment)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT = 300
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Retrieval Settings ───────────────────────────────────
RETRIEVAL_TOP_K = 3             # Chunks sent to LLM (2-4 recommended)
WEB_SEARCH_MAX_RESULTS = 4      # Web results to fetch (4-10)

# ─── Performance Tuning ───────────────────────────────────
CHUNK_SIZE = 300                # Characters per chunk (200-500)
CHUNK_OVERLAP = 50              # Overlap between chunks

# ─── Search Weights ───────────────────────────────────────
ALPHA_SEMANTIC = 0.7            # Semantic similarity weight
ALPHA_KEYWORD = 0.3             # Keyword matching weight
```

### Recommended Configurations

| Provider | Model | Hardware | TOP_K | Expected Latency |
|----------|-------|----------|-------|------------------|
| **Groq API** ⭐ | llama-3.3-70b-versatile | Any | 3 | **1-3s** |
| **Groq API** | llama-3.1-8b-instant | Any | 3 | **0.5-1s** |
| **Ollama** | phi3:mini | CPU Only | 3 | 40-90s |
| **Ollama** | mistral | CPU Only | 2 | 120-180s |
| **Ollama** | phi3:mini | GPU (4GB) | 3 | 10-20s |
| **Ollama** | mistral | GPU (8GB+) | 3 | 5-10s |
| **Ollama** | llama3 | GPU (8GB+) | 4 | 6-12s |

⭐ **Recommended for production:** Groq API with automatic Ollama fallback

---

## 📡 API Documentation

### POST `/ask`

Ask any question and get an AI-generated answer from web search.

**Request:**
```json
{
  "query": "Elon Musk ki net worth kitni hai?",
  "top_k": 3,                    // Optional: Number of chunks (default: 3)
  "max_web_results": 4,          // Optional: Web results (default: 4)
  "use_cache": true              // Optional: Use caching (default: true)
}
```

**Response (200 OK):**
```json
{
  "answer": "As of 2024, Elon Musk's net worth is approximately $230 billion according to Forbes...",
  "sources": ["forbes.com", "bloomberg.com", "wikipedia.org"],
  "web_urls": [
    "https://www.forbes.com/real-time-billionaires/",
    "https://www.bloomberg.com/billionaires/",
    "https://en.wikipedia.org/wiki/Elon_Musk"
  ],
  "query_info": {
    "original": "Elon Musk ki net worth kitni hai?",
    "language_label": "Code-Mixed (Hindi-English)",
    "confidence": 0.95,
    "english_query": "What is Elon Musk's net worth?"
  },
  "search_info": {
    "web_results_fetched": 4,
    "final_results": 3
  },
  "generation_info": {
    "model": "phi3:mini",
    "llm_used": true,
    "total_ms": 45230,
    "normalization_ms": 2,
    "translation_ms": 8520,
    "web_search_ms": 1847,
    "embedding_ms": 1245,
    "search_rerank_ms": 12,
    "generation_ms": 33604
  },
  "from_cache": false
}
```

### GET `/health`

System health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "web_search_only",
  "cache_stats": {
    "total_queries": 156,
    "cache_hits": 42,
    "cache_misses": 114,
    "hit_rate_percent": 26.9,
    "cached_entries": 42,
    "ttl_seconds": 3600
  }
}
```

### GET `/cache`

Get cache statistics.

**Response:**
```json
{
  "stats": {
    "total_queries": 156,
    "cache_hits": 42,
    "hit_rate_percent": 26.9,
    "cached_entries": 42
  },
  "note": "For production, replace with Redis for distributed caching"
}
```

### POST `/cache/clear`

Clear the entire query cache.

**Response:**
```json
{
  "message": "Cache cleared",
  "entries_cleared": 42
}
```

---

## ⚡ Performance Optimization

### CPU Optimization (No GPU)

**1. Use phi3:mini Instead of Mistral**
```bash
ollama pull phi3:mini
```
Update `backend/config.py`:
```python
OLLAMA_MODEL = "phi3:mini"
```
**Result:** 3x faster (40-60s vs 120-180s)

**2. Reduce Context Chunks**
```python
RETRIEVAL_TOP_K = 2  # Instead of 3 or 4
```
**Result:** 20-30% faster LLM generation

**3. Reduce Web Results**
```python
WEB_SEARCH_MAX_RESULTS = 4  # Instead of 6-10
```
**Result:** Faster embedding and search

**4. Smaller Chunks**
```python
CHUNK_SIZE = 250  # Instead of 300-500
```
**Result:** Faster embedding, less context to process

### GPU Optimization

**1. Use Mistral or Llama3**
```bash
ollama pull mistral
# or
ollama pull llama3
```

**2. Increase Context for Better Answers**
```python
RETRIEVAL_TOP_K = 4
WEB_SEARCH_MAX_RESULTS = 6
```

**3. Reduce Timeout**
```python
OLLAMA_TIMEOUT = 60  # 1 minute is enough with GPU
```

### Caching Strategy

Enable caching to serve repeated queries instantly:
```python
# Automatically enabled
# Popular queries: <1s response time
# 1-hour TTL (configurable)
```

**Cache Hit Rate Improvements:**
- Deploy in production → More users → Higher hit rate
- Longer TTL for stable facts (e.g., historical events)
- Shorter TTL for news/current events

### Performance Benchmarks

#### Groq API (llama-3.3-70b-versatile):
| Stage | Time | % of Total |
|-------|------|-----------|
| Language Detection | 2ms | 0.1% |
| Translation | 8-20s | 60% |
| Web Search | 1-2s | 30% |
| Embedding | 200ms | 5% |
| Search & Rerank | 10ms | 0.3% |
| **LLM Generation** | **1-3s** | **7%** |
| **Total (uncached)** | **10-25s** | **100%** |

#### Ollama CPU (phi3:mini):
| Stage | Time | % of Total |
|-------|------|------------|
| Language Detection | 2ms | 0.004% |
| Translation | 8-20s | 18% |
| Web Search | 1-2s | 4% |
| Embedding | 1s | 2% |
| Search & Rerank | 10ms | 0.02% |
| **LLM Generation** | **30-60s** | **75%** |
| **Total (uncached)** | **40-90s** | **100%** |

#### Ollama GPU (mistral):
| Stage | Time | % of Total |
|-------|------|------------|
| Language Detection | 2ms | 0.03% |
| Translation | 8-20s | 65% |
| Web Search | 1-2s | 15% |
| Embedding | 200ms | 2% |
| Search & Rerank | 5ms | 0.06% |
| **LLM Generation** | **3-6s** | **18%** |
| **Total (uncached)** | **12-30s** | **100%** |

#### Cached Queries:
| Stage | Time |
|-------|------|
| Cache Lookup + Response | **<1s** ⚡ |

---

## 🚢 Deployment

### 🌟 Zero-Cost Cloud Deployment (Recommended)

**Deploy to Streamlit Community Cloud with Groq API - completely FREE!**

📖 **See detailed guide:** [DEPLOY.md](DEPLOY.md)

**Quick summary:**
1. Push code to GitHub
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Get free Groq API key from [console.groq.com](https://console.groq.com)
4. Deploy with one click
5. Add secrets in Streamlit dashboard

**Cost:** $0/month  
**Performance:** 1-3 second responses  
**Limits:** 14,400 requests/day (Groq free tier)

### Local Development
```bash
# With Groq API
$env:LLM_PROVIDER="groq"
$env:GROQ_API_KEY="gsk_your_key_here"
python -m backend.app

# With Ollama
python -m backend.app
```

### Production (Uvicorn with Workers)
```bash
uvicorn backend.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --timeout-keep-alive 300
```

### Docker

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 8501 11434

# Start Ollama and app
CMD ollama serve & \
    sleep 5 && \
    ollama pull phi3:mini && \
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Build & Run:**
```bash
docker build -t multilingual-rag .
docker run -p 8000:8000 -p 8501:8501 multilingual-rag
```

### Cloud Deployment Options Comparison

| Option | Cost/Month | Setup Time | Performance | Best For |
|--------|-----------|------------|-------------|----------|
| **Streamlit Cloud + Groq** ⭐ | **$0** | 15 min | 1-3s | Personal projects, MVPs |
| **AWS EC2 t3.medium + Groq** | $30 | 2 hours | 1-3s | Small business |
| **AWS EC2 + Ollama (CPU)** | $30-50 | 3 hours | 40-90s | Private/offline needs |
| **AWS EC2 + Ollama (GPU)** | $360+ | 4 hours | 5-15s | High privacy, high volume |

⭐ **Recommended:** Streamlit Cloud + Groq for 99% of use cases

### Self-Hosting Tips

**With Groq API (Recommended):**
1. Any cheap VM (1GB RAM sufficient)
2. Set GROQ_API_KEY environment variable
3. No GPU needed
4. Use Redis for distributed caching

**With Ollama (GPU required for speed):**
1. Use Redis for distributed caching
2. Set up CloudWatch/monitoring
3. Enable auto-scaling (if high traffic)
4. Use CloudFront CDN for frontend
5. Set up health check monitoring

---

## 🔧 Troubleshooting

### "Backend Offline" in Streamlit

**Symptoms:** Red "❌ Backend Offline" in sidebar

**Solutions:**
1. Check backend is running: `python -m backend.app`
2. Verify port 8000 is accessible: `curl http://localhost:8000/health`
3. Click "🔄 Refresh Status" button in sidebar
4. Hard refresh browser: Ctrl+Shift+R
5. Enable Debug Mode in settings to see error details

### "Request timed out"

**Cause:** LLM generation taking too long (>300s)

**Solutions:**

**Option 1: Switch to Groq API (Recommended)**
```bash
# Edit .env file:
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_api_key_here

# Restart backend
python -m backend.app
```
**Result:** 1-3 second responses! ⚡

**Option 2: Use faster Ollama model**
```bash
ollama pull phi3:mini
# Update .env: OLLAMA_MODEL=phi3:mini
```

**Option 3: Increase timeout**
```python
# backend/config.py
OLLAMA_TIMEOUT = 600  # 10 minutes
```

### "Groq API error: 401 Invalid API Key"

**Cause:** Wrong or missing Groq API key

**Solutions:**
1. Get valid API key from [console.groq.com](https://console.groq.com)
2. Check .env file has correct key (starts with `gsk_`)
3. Set environment variable before starting backend:
   ```bash
   # Windows PowerShell
   $env:GROQ_API_KEY="gsk_your_actual_key_here"
   python -m backend.app
   
   # Linux/Mac
   export GROQ_API_KEY=gsk_your_actual_key_here
   python -m backend.app
   ```
4. Verify key is 56 characters long
5. No quotes or spaces in the key

### "Groq API error: 400 Model decommissioned"

**Cause:** Using outdated model name

**Solution:**
```bash
# Update .env to use current model:
GROQ_MODEL=llama-3.3-70b-versatile

# Other options:
# GROQ_MODEL=llama-3.1-8b-instant (faster)
# GROQ_MODEL=qwen/qwen3-32b
```

Check available models: `python list_groq_models.py`

### "Web search failed"

**Possible Causes:**
- No internet connection
- DuckDuckGo blocked/rate-limited
- Network firewall blocking requests

**Solutions:**
1. Check internet: `curl https://duckduckgo.com`
2. Check logs for detailed error
3. Try again after a few minutes (rate limit)
4. Use VPN if DuckDuckGo is blocked

### "Ollama model not available"

**Error:** Model phi3:mini/mistral not found

**Solution:**
```bash
# List installed models
ollama list

# Pull the model
ollama pull phi3:mini

# Restart backend
python -m backend.app
```

### Slow Performance (>3 minutes)

**Solutions:**
1. Check which stage is slow in performance metrics
2. If translation is slow: Network issue, try again
3. If LLM is slow: Switch to smaller model (phi3:mini)
4. If web search is slow: Reduce MAX_WEB_RESULTS
5. Enable caching for repeat queries

### Progress Bar Error in Streamlit

**Error:** `Progress Value has invalid value`

**This is fixed** in the latest code. If you still see it:
1. Pull latest code
2. Restart Streamlit
3. The fix clamps percentage values between 0-1

---

## 🎨 Tech Stack

### Backend
- **Framework:** FastAPI 0.115
- **Server:** Uvicorn (ASGI)
- **LLM (Dual Support):**
  - ☁️ **Groq API** (cloud): llama-3.3-70b-versatile, llama-3.1-8b-instant
  - 🏠 **Ollama** (local): phi3:mini (3B), mistral (7B), llama3 (7B)
  - Automatic fallback: Groq → Ollama → Raw context
- **Embeddings:** Sentence-Transformers
  - Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
  - Supports 50+ languages
- **Vector Search:** FAISS (IndexFlatIP for cosine similarity)
- **Web Search:** DuckDuckGo (duckduckgo-search 7.5.3)
- **Translation:** GoogleTrans (free Google Translate API)
- **Environment:** python-dotenv for configuration

### Frontend
- **UI Framework:** Streamlit 1.44.1
- **HTTP Client:** Requests
- **Alternative:** FastAPI embedded HTML (Jinja2 templates)

### Language Processing
- **Detection:** langdetect (Google's language detection)
- **Transliteration:** indic-transliteration (Romanized Hindi → Devanagari)
- **Normalization:** Custom query expansion logic

### Infrastructure
- **Caching:** In-memory dictionary (production: Redis)
- **Logging:** Python standard logging
- **Monitoring:** Built-in performance metrics

### Development
- **Language:** Python 3.10+
- **Type Hints:** Full typing with mypy compliance
- **Code Style:** Black formatter, isort
- **Logging:** Structured JSON logs

---

## 📝 Project Structure

```
multi-source-rag/
├── backend/
│   ├── __init__.py
│   ├── app.py                  # Main FastAPI application
│   └── config.py               # Configuration settings (with .env loading)
├── generation/
│   ├── __init__.py
│   ├── llm.py                  # Dual LLM support (Groq + Ollama)
│   ├── llm_groq.py             # Groq API client
│   └── prompt.py               # Prompt templates
├── ingestion/
│   ├── __init__.py
│   ├── embedder.py             # Sentence-Transformers embeddings
│   └── web_search.py           # DuckDuckGo search
├── processing/
│   ├── __init__.py
│   ├── language_detect.py      # Language detection
│   ├── normalize.py            # Query normalization
│   └── translate.py            # Translation
├── retrieval/
│   ├── __init__.py
│   ├── rerank.py               # Result re-ranking
│   └── search.py               # Hybrid search (semantic + keyword)
├── .streamlit/
│   └── secrets.toml.example    # Streamlit Cloud secrets template
├── streamlit_app.py            # Streamlit frontend
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── DEPLOY.md                   # Deployment guide (Streamlit Cloud + Groq)
├── GROQ_LOCAL_SETUP.md         # Local testing with Groq API
├── test_codemixed_queries.txt  # Code-mixed testing queries
├── list_groq_models.py         # Script to list available Groq models
├── test_groq_key.py            # Script to validate Groq API key
└── debug_env.py                # Environment variable debugging tool
```

---

## 🆚 Version History

### v2.1 (Latest) - Cloud Deployment Ready

**Major Updates:**
- ✅ **Groq API Integration** - 10-30x faster responses (1-3s vs 40-90s)
- ✅ **Dual LLM Support** - Groq API (cloud) + Ollama (local) with automatic fallback
- ✅ **Zero-Cost Deployment** - Streamlit Community Cloud + Groq free tier
- ✅ **Environment Variables** - .env file support with python-dotenv
- ✅ **Code-Mixed Testing** - Validated English+Hindi+Telugu queries
- ✅ **Updated Models** - llama-3.3-70b-versatile (latest Groq model)
- ✅ **Deployment Guides** - DEPLOY.md and GROQ_LOCAL_SETUP.md
- ✅ **Debug Tools** - Model listing, API key validation, env debugging

**Performance:**
- Groq API: 1-3s per query ⚡
- Ollama fallback: 40-90s per query
- Free tier: 14,400 requests/day

### v2.0 - Web-Only Architecture

**Removed Features:**
- ❌ Local file knowledge base
- ❌ Document ingestion endpoint
- ❌ FAISS index persistence

**New Features:**
- ✅ 100% web-only architecture
- ✅ Smart query caching
- ✅ Streamlit frontend
- ✅ Performance metrics tracking

**Code Reduction:**
- 40% less code
- Simplified architecture
- Easier to maintain

---

## 🙏 Acknowledgments

- **Groq** for blazing-fast cloud LLM inference (1-3s responses!)
- **Ollama** for local LLM inference
- **HuggingFace** for multilingual embeddings
- **DuckDuckGo** for free web search API
- **FastAPI** for blazing-fast web framework
- **Streamlit** for beautiful UI framework

---

## 🔮 Future Enhancements

- [ ] Enhanced Telugu romanization support
- [ ] Redis caching for distributed systems
- [ ] Response streaming (show partial answers)
- [ ] Multi-search (Bing/Google as fallback)
- [ ] Query rewriting for better search results
- [ ] Domain authority scoring
- [ ] Answer confidence scores
- [ ] Analytics dashboard

---

## 📄 License

MIT License - feel free to use for personal or commercial projects!

---

## 📧 Support

For issues, questions, or contributions:
- Check the [Troubleshooting](#-troubleshooting) section above
- Review backend logs for errors
- Enable Debug Mode in Streamlit to see full responses
- See [DEPLOY.md](DEPLOY.md) for deployment help
- Check available Groq models: `python list_groq_models.py`

---

**Built with ❤️ for multilingual AI accessibility**

**TL;DR:** Web-only RAG system that answers ANY question using real-time search. Works with code-mixed Indian languages. Production-ready with caching and cloud deployment. Free tier available with Groq API. 🚀
