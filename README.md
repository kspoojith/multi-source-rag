# 🌐 Multilingual RAG System v2.0

**Answer ANY question in the world** using real-time web search powered by AI.

No local files needed. No manual curation. Just ask anything in English, Hindi, Telugu, or code-mixed language!

![Version](https://img.shields.io/badge/Version-2.0-blue)
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
- **Code-Mixed:** "India ki capital kya hai?"
- **Telugu:** "భారతదేశ రాజధాని ఏమిటి?"
- **Automatic Language Detection** with 80%+ confidence

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
- CORS enabled
- Comprehensive logging
- Health check endpoints
- Error handling & retries
- Performance metrics tracking

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

### 2. Install Ollama & LLM Model
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
```

### 3. Start the Backend
```bash
python -m backend.app
```

You should see:
```
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

Example queries:
- "Elon Musk ki net worth kitni hai?"
- "IPL 2024 winner kaun hai?"
- "Latest AI news kya hai?"
- "India ka prime minister kon hai?"
- "Climate change kya hai?"
- "Python vs JavaScript comparison"

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

Edit `backend/config.py` to customize:

```python
# ─── LLM Settings ─────────────────────────────────────────
OLLAMA_MODEL = "phi3:mini"      # Model name
OLLAMA_TIMEOUT = 300            # 5 minutes for CPU (increase if needed)
OLLAMA_BASE_URL = "http://localhost:11434"

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

| Hardware | Model | TOP_K | TIMEOUT | Expected Latency |
|----------|-------|-------|---------|------------------|
| **CPU Only** | phi3:mini | 3 | 300s | 40-90s |
| **CPU Only** | mistral | 2 | 300s | 120-180s |
| **GPU (4GB)** | phi3:mini | 3 | 120s | 10-20s |
| **GPU (8GB+)** | mistral | 3 | 60s | 5-10s |
| **GPU (8GB+)** | llama3 | 4 | 60s | 6-12s |

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

#### CPU (phi3:mini):
| Stage | Time | % of Total |
|-------|------|------------|
| Language Detection | 2ms | 0.004% |
| Translation | 8-20s | 18% |
| Web Search | 1-2s | 4% |
| Embedding | 1s | 2% |
| Search & Rerank | 10ms | 0.02% |
| **LLM Generation** | **30-60s** | **75%** |
| **Total (uncached)** | **40-90s** | **100%** |

#### GPU (mistral):
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

### Local Development
```bash
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

### Cloud Deployment (AWS Example)

**Small Scale (<100 queries/day):**
- **Instance:** EC2 t3.medium (2 vCPU, 4GB RAM)
- **Model:** phi3:mini
- **Cost:** ~$30/month
- **Latency:** 40-90s per query

**Medium Scale (100-1000 queries/day):**
- **Instance:** EC2 g4dn.xlarge (4 vCPU, 16GB RAM, 1 GPU)
- **Model:** mistral or llama3
- **Cost:** ~$360/month
- **Latency:** 5-15s per query

**Production Tips:**
1. Use Redis for distributed caching
2. Set up CloudWatch logging
3. Enable auto-scaling
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
1. Switch to phi3:mini:
   ```bash
   ollama pull phi3:mini
   # Update config: OLLAMA_MODEL = "phi3:mini"
   ```

2. Increase timeout in `backend/config.py`:
   ```python
   OLLAMA_TIMEOUT = 600  # 10 minutes
   ```

3. Increase timeout in `streamlit_app.py`:
   ```python
   timeout=600  # in ask_question() function
   ```

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
- **LLM:** Ollama (local inference)
  - Models: phi3:mini (3B), mistral (7B), llama3 (7B)
- **Embeddings:** Sentence-Transformers
  - Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
  - Supports 50+ languages
- **Vector Search:** FAISS (IndexFlatIP for cosine similarity)
- **Web Search:** DuckDuckGo (duckduckgo-search 7.5.3)
- **Translation:** GoogleTrans (free Google Translate API)

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
│   ├── app.py              # Main FastAPI application
│   └── config.py           # Configuration settings
├── generation/
│   ├── __init__.py
│   ├── llm.py              # Ollama LLM integration
│   └── prompt.py           # Prompt templates
├── ingestion/
│   ├── __init__.py
│   ├── embedder.py         # Sentence-Transformers embeddings
│   └── web_search.py       # DuckDuckGo search
├── processing/
│   ├── __init__.py
│   ├── language_detect.py  # Language detection
│   ├── normalize.py        # Query normalization
│   └── translate.py        # Translation
├── retrieval/
│   ├── __init__.py
│   ├── rerank.py           # Result re-ranking
│   └── search.py           # Hybrid search (semantic + keyword)
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🆚 v1 → v2 Migration

### What Changed

**Removed Features:**
- ❌ Local file knowledge base (ingestion/loader.py, chunker.py, indexer.py)
- ❌ Document ingestion endpoint (POST /ingest)
- ❌ FAISS index persistence to disk
- ❌ Local mode toggle in UI
- ❌ Evaluation suite for local documents
- ❌ data/raw/ directory

**New Features:**
- ✅ 100% web-only architecture
- ✅ Smart query caching with statistics
- ✅ Streamlit frontend with real-time monitoring
- ✅ Better error handling and logging
- ✅ Performance metrics tracking
- ✅ Health check and cache endpoints

**Code Reduction:**
- 40% less code (1200 → 680 lines in app.py)
- Simplified architecture
- Easier to maintain

---

## 📄 License

MIT License - Feel free to use for commercial or personal projects.

---

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **HuggingFace** for multilingual embeddings
- **DuckDuckGo** for free web search API
- **FastAPI** for blazing-fast web framework
- **Streamlit** for beautiful UI framework

---

## 📧 Support

For issues, questions, or contributions:
- Check the Troubleshooting section above
- Review backend logs for errors
- Enable Debug Mode in Streamlit to see full responses
- Check Ollama logs: `ollama logs`

---

**Built with ❤️ for multilingual AI accessibility**
- ❌ `loader.py`, `chunker.py`, `indexer.py`

### Why Removed:
- **Complexity:** 500+ lines of code for 5 demo files
- **Maintenance:** Need manual curation of thousands of files
- **Scalability:** Can't scale to "answer any question"
- **Redundancy:** Web search does everything better

**Result:** 40% less code, 100% more useful!

---

## 🔮 Future Enhancements

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

## 🙏 Credits

Built with:
- FastAPI
- Ollama
- Sentence-Transformers
- DuckDuckGo Search
- FAISS

---

## 📞 Support

**Need help?**
- Check `PRODUCTION.md` for deployment guide
- Check `CPU_OPTIMIZATION.md` for performance tuning
- Check logs for detailed error messages

**Upgrading from v1?**
- Old code backed up in `backend/app_old_backup.py`
- No migration needed - just start using v2!

---

**TL;DR:** Web-only RAG system that answers ANY question using real-time search. Works with code-mixed Indian languages. Production-ready with caching. No file management needed. 🚀
