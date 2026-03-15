# 🌐 Multilingual RAG System - Web Search Edition

**Answer ANY question in the world** using real-time web search powered by AI.

No local files needed. No manual curation. Just ask anything in English, Hindi, Telugu, or code-mixed language!

![Version](https://img.shields.io/badge/Version-2.0-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ What's New in v2.0

🎉 **Complete Refactor:** Web Search Only - Removed all local file storage complexity

### Key Changes:
- ✅ **Single Mode:** Web search only (no local knowledge toggle)
- ✅ **Smart Caching:** Popular queries answered in <1s
- ✅ **Cleaner Code:** 40% less code, easier to maintain
- ✅ **Better UX:** Simplified interface, no mode confusion
- ✅ **Production Ready:** Built for scale from day one

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama & Model
```bash
# Download Ollama from https://ollama.ai

# For CPU (faster):
ollama pull tinyllama

# For GPU (best quality):
ollama pull mistral
```

### 3. Start the Server
```bash
python -m backend.app
```

### 4. Open Your Browser
Navigate to: **http://localhost:8000**

Ask anything:
- "Elon Musk ki net worth kitni hai?"
- "IPL 2024 winner kaun hai?"
- "Latest AI breakthroughs kya hain?"

---

## 🎯 How It Works

```
User Query (any language)
    ↓
Language Detection (Hindi/English/Telugu/Mixed)
    ↓
Query Normalization & Translation
    ↓
DuckDuckGo Web Search (4-10 results)
    ↓
Chunk & Embed Web Content
    ↓
Hybrid Retrieval (Semantic + Keyword)
    ↓
LLM Generation (Ollama/Mistral/TinyLlama)
    ↓
Answer with Source Citations
```

---

## 📊 Features

### Multilingual Support
- **Pure Hindi:** "भारत की राजधानी क्या है?"
- **Pure English:** "What is the capital of India?"
- **Code-Mixed:** "India ki capital kya hai?"
- **Telugu:** "భారతదేశ రాజధాని ఏమిటి?"

### Smart Caching
- Popular queries cached for 1 hour
- <1s response time for cached queries
- Automatic cache invalidation

### Source Citations
- Every answer cites web sources
- Clickable URLs for verification
- Domain authority visible

### Production Features
- CORS enabled
- Error handling
- Logging & monitoring
- Health check endpoint
- Cache statistics

---

## 📚 API Endpoints

### `POST /ask`
Ask any question and get an answer from web search.

**Request:**
```json
{
  "query": "Elon Musk ki net worth kitni hai?",
  "top_k": 3,
  "max_web_results": 6,
  "use_cache": true
}
```

**Response:**
```json
{
  "answer": "According to Forbes, Elon Musk's net worth as of 2024 is approximately $230 billion...",
  "sources": ["forbes.com", "bloomberg.com"],
  "web_urls": ["https://forbes.com/...", "https://bloomberg.com/..."],
  "query_info": {
    "language_label": "Mixed (Hindi-English)",
    "english_query": "Elon Musk net worth"
  },
  "from_cache": false
}
```

### `GET /health`
System health check.

```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "web_search_only",
  "cache_stats": {
    "cache_hits": 42,
    "hit_rate_percent": 65.5
  }
}
```

### `GET /cache`
Cache statistics.

### `POST /cache/clear`
Clear the entire cache.

---

## ⚙️ Configuration

Edit `backend/config.py`:

```python
# LLM Settings
OLLAMA_MODEL = "tinyllama"  # or "mistral" for GPU
OLLAMA_TIMEOUT = 300        # 5 minutes for CPU

# Retrieval
RETRIEVAL_TOP_K = 2         # Number of chunks for LLM
WEB_SEARCH_MAX_RESULTS = 4  # Web results to fetch

# Performance
CHUNK_SIZE = 300            # Smaller = faster LLM
```

### Recommended Settings:

| Hardware | Model | RETRIEVAL_TOP_K | Expected Latency |
|----------|-------|-----------------|------------------|
| CPU Only | tinyllama | 2 | 10-20s |
| CPU Only | mistral | 2 | 60-180s (may timeout) |
| GPU | mistral | 3 | 3-8s |
| GPU | llama3 | 4 | 5-10s |

---

## 🔧 Troubleshooting

### "LLM timeout"
**Solution:** Switch to TinyLlama (10x faster on CPU)
```bash
ollama pull tinyllama
# Update backend/config.py: OLLAMA_MODEL = "tinyllama"
```

### "Web search failed"
**Check:**
1. Internet connection
2. DuckDuckGo is accessible (not blocked)
3. Check logs for detailed error

### "Slow responses"
**Solutions:**
1. Reduce `RETRIEVAL_TOP_K` to 2
2. Reduce `WEB_SEARCH_MAX_RESULTS` to 4
3. Switch to GPU instance
4. Use lighter model (tinyllama)

---

## 📈 Performance Benchmarks

### Latency Breakdown (CPU - TinyLlama):

| Stage | Time |
|-------|------|
| Language Detection | 5ms |
| Web Search | 2-5s |
| Embedding | 300ms |
| LLM Generation | 10-20s |
| **Total** | **12-25s** |

### Latency Breakdown (GPU - Mistral):

| Stage | Time |
|-------|------|
| Language Detection | 5ms |
| Web Search | 2-5s |
| Embedding | 100ms |
| LLM Generation | 2-4s |
| **Total** | **4-9s** |

### With Caching (Popular Queries):

| Stage | Time |
|-------|------|
| Cache Lookup | <1s |
| **Total** | **<1s** ✨ |

---

## 🚢 Deployment

### Local Development
```bash
python -m backend.app
```

### Production (Uvicorn)
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud (Recommended)

**For <100 queries/day:**
- AWS EC2 t3.medium (CPU)
- Use TinyLlama
- Cost: ~$1/day

**For >100 queries/day:**
- AWS g4dn.xlarge (GPU)
- Use Mistral or Llama3
- Cost: ~$12/day
- 5x faster responses

---

## 🎨 Tech Stack

- **Backend:** FastAPI
- **LLM:** Ollama (Local inference)
- **Embeddings:** Sentence-Transformers (Multilingual MiniLM)
- **Vector Search:** FAISS (temporary in-memory)
- **Web Search:** DuckDuckGo (Free, no API key)
- **Caching:** In-memory (production: Redis)
- **Languages:** Python 3.10+

---

## 📝 What Was Removed (v1 → v2)

### Removed Features:
- ❌ Local file knowledge base
- ❌ Document ingestion (`POST /ingest`)
- ❌ FAISS index persistence
- ❌ Local mode toggle in UI
- ❌ Evaluation suite (was for local docs)
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
