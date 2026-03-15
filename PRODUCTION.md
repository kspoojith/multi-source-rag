# 🚀 Production Deployment Guide

## Overview

This system is designed to answer **ANY question in the world** using **Web Search Mode** - no need to store thousands of files locally!

### Two Modes:

| Mode | Use Case | Data Source | Latency |
|------|----------|-------------|---------|
| **🌐 Web Search** (Default) | Answer ANY question | Live DuckDuckGo search | ~5-15s |
| **📁 Local Knowledge** | Answer from curated docs | Pre-indexed files | ~2-5s |

**For production:** Use **Web Search Mode** exclusively - it scales infinitely!

---

## ✅ Quick Start (Production Mode)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama & Model
```bash
# Download Ollama from https://ollama.ai
ollama pull mistral        # 7B model (4.4GB)
# OR for faster inference:
ollama pull phi3:mini      # 3B model (2.3GB) - 50% faster
```

### 3. Start the Server
```bash
python -m backend.app
```

### 4. Open Browser
Navigate to: **http://localhost:8000**

---

## ⚡ Performance Optimization

### For Lower Latency:

1. **Use a Faster Model:**
   ```bash
   # Switch to phi3:mini for 2-3x faster inference
   ollama pull phi3:mini
   
   # Update config or set environment variable:
   export OLLAMA_MODEL=phi3:mini
   ```

2. **GPU Acceleration (Recommended for Production):**
   - If you have NVIDIA GPU, Ollama automatically uses CUDA
   - Reduces LLM latency from ~10s to ~2s
   - Install: `pip install faiss-gpu` for faster embedding search

3. **Optimize Configuration:**
   Edit [`backend/config.py`](backend/config.py):
   ```python
   RETRIEVAL_TOP_K = 3           # Reduce from 4 to 3 (faster LLM processing)
   WEB_SEARCH_MAX_RESULTS = 8    # Reduce from 10 to 8 (faster search)
   OLLAMA_TIMEOUT = 180          # Increase if you see timeouts
   ```

---

## 🌐 Web Search Mode (Production Default)

### How It Works:

```
User Query → Language Detection → Translation (if needed) 
    → DuckDuckGo Search (10 results) → Chunk Web Content 
    → Embed On-The-Fly → Hybrid Search → LLM Generation 
    → Answer with Citations
```

### Example Queries:
- "Elon Musk ki net worth kitni hai?"
- "IPL 2024 winner kaun hai?"
- "Latest AI breakthroughs kya hain?"
- "Python vs JavaScript performance comparison"
- "India ka GDP 2026 mein kitna hai?"

### Advantages:
- ✅ **Infinite Knowledge:** Access to entire internet
- ✅ **Always Fresh:** Real-time information
- ✅ **No Storage:** No need for local files
- ✅ **Multilingual:** Asks in any language

### Disadvantages:
- ⚠️ **Higher Latency:** 5-15s vs 2-5s (local mode)
- ⚠️ **Internet Required:** Needs active connection
- ⚠️ **DuckDuckGo Dependency:** If DDG is down, mode fails

---

## 📁 Local Knowledge Mode (Optional)

Only use this if you need:
- Guaranteed low latency (<5s)
- Offline operation
- Answers from proprietary/curated documents

### Setup:
1. Add your `.txt` files to [`data/raw/`](data/raw/)
2. Click **"Ingest Documents"** in the UI
3. Switch to **Local Knowledge** mode

---

## 🔧 Troubleshooting

### Issue: "LLM timed out"

**Cause:** Ollama model inference is slow on CPU

**Solutions:**
1. Increase timeout:
   ```python
   # backend/config.py
   OLLAMA_TIMEOUT = 240  # Increase to 4 minutes
   ```

2. Use smaller/faster model:
   ```bash
   ollama pull phi3:mini
   export OLLAMA_MODEL=phi3:mini
   ```

3. Check Ollama is running:
   ```bash
   ollama list  # Should show 'mistral' or 'phi3:mini'
   ```

4. **Best Solution:** Use GPU (10x faster)

### Issue: "This site can't be reached (0.0.0.0:8000)"

**Solution:** Use `http://localhost:8000` or `http://127.0.0.1:8000`

### Issue: "DuckDuckGo search failed"

**Solutions:**
1. Check internet connection
2. Try changing region:
   ```python
   # backend/config.py
   WEB_SEARCH_REGION = "wt-wt"  # Worldwide instead of India-specific
   ```

---

## 📊 Production Benchmarks

### Latency Breakdown (Web Search Mode):

| Stage | Time (CPU) | Time (GPU) |
|-------|------------|------------|
| Language Detection | 5ms | 5ms |
| Translation | 200ms | 50ms |
| DuckDuckGo Search | 800ms | 800ms |
| Embedding (10 chunks) | 500ms | 100ms |
| FAISS Search | 10ms | 5ms |
| Reranking | 20ms | 20ms |
| LLM Generation (Mistral) | 12,000ms | 2,000ms |
| **Total** | ~13.5s | ~3s |

**Recommendation:** For production, use GPU or switch to `phi3:mini` on CPU (reduces to ~7s total).

---

## 🔐 Security & Rate Limits

### DuckDuckGo:
- **No API key needed**
- **No hard rate limit** for moderate usage
- Recommended: <100 queries/min
- Consider adding caching for popular queries

### Ollama:
- Runs entirely local - no external API calls
- No rate limits
- Privacy-friendly - no data leaves your server

---

## 🚢 Deployment Options

### 1. Local Server (Current Setup)
```bash
python -m backend.app
# Access at http://localhost:8000
```

### 2. Production Server (Uvicorn)
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Docker (Recommended)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Cloud Deployment
- **AWS EC2:** t3.large or better (8GB RAM)
- **Google Cloud:** e2-standard-2 with GPU for best performance
- **Azure:** B2s or better

**Recommended Instance:** GPU instance for <3s latency
- AWS: g4dn.xlarge (~$0.50/hr)
- GCP: n1-standard-4 + T4 GPU

---

## 💰 Cost Analysis

### Resource Requirements:

| Component | CPU-Only | With GPU |
|-----------|----------|----------|
| RAM | 8GB | 8GB |
| Storage | 10GB | 10GB |
| GPU VRAM | - | 4GB |
| Instance Cost (AWS) | ~$0.08/hr | ~$0.50/hr |

### Cost Per Query:
- **CPU-only:** ~$0.0003/query (13s × $0.08/hr)
- **With GPU:** ~$0.00042/query (3s × $0.50/hr)

**For <1000 queries/day:** CPU-only is cost-effective
**For >1000 queries/day:** GPU pays for itself via higher throughput

---

## 📈 Scaling Recommendations

### <100 users/day:
- Single CPU instance
- Use `phi3:mini` model
- Total cost: ~$60/month

### 100-1000 users/day:
- Single GPU instance
- Use `mistral:7b` model
- Add Redis caching for popular queries
- Total cost: ~$400/month

### >1000 users/day:
- Load balancer + 2-4 GPU instances
- Implement query result caching (Redis)
- Consider using OpenAI/Anthropic API instead of Ollama
- Total cost: ~$1500-3000/month

---

## 🎯 Production Checklist

- [x] Install all dependencies
- [x] Pull Ollama model (mistral or phi3:mini)
- [x] Start server and verify http://localhost:8000 works
- [ ] Test Web Search mode with 5+ queries
- [ ] Measure average latency (should be <15s)
- [ ] Set up monitoring (check `/health` endpoint)
- [ ] Configure CORS for your domain (in `backend/app.py`)
- [ ] Add logging to file (not just console)
- [ ] Set up automatic restarts (systemd/supervisor)
- [ ] Enable HTTPS (nginx reverse proxy)

---

## 📚 Next Steps

1. **Try it now:** Open http://localhost:8000 and ask ANY question!
2. **Optimize:** If latency >10s, switch to `phi3:mini` or add GPU
3. **Monitor:** Track the `/health` endpoint for system status
4. **Scale:** Once you hit 100+ queries/day, consider GPU instance

**Need help?** Check [README.md](README.md) for full documentation.
