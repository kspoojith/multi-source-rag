# Quick Setup for Groq API (Local Testing)
# =========================================

## 🎯 To Test Groq API Locally (Before Deploying)

### Step 1: Create .env File

```bash
# Copy the example file
cp .env.example .env
```

Or on Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

### Step 2: Edit .env File

Open `.env` and change these lines:

```bash
# Change from "ollama" to "groq"
LLM_PROVIDER=groq

# Add your actual API key (get from https://console.groq.com)
GROQ_API_KEY=gsk_YOUR_ACTUAL_KEY_HERE
```

### Step 3: Restart Backend

Stop your current backend (Ctrl+C) and restart:

```bash
python -m backend.app
```

You should see in the logs:
```
[INFO] generation.llm: Using Groq API for LLM generation
```

### Step 4: Test!

- Keep backend running (port 8000)
- Keep Streamlit running (port 8501)
- Try a query
- Should get response in **1-3 seconds** instead of 40-90 seconds!

---

## 🚨 Troubleshooting

### "GROQ_API_KEY not found"

Make sure:
1. `.env` file exists in project root
2. You added your actual API key
3. You restarted the backend

### Still using Ollama?

Check backend logs when you start it. Should say:
```
"LLM_PROVIDER = groq"
```

If it still says "ollama", your .env file isn't being loaded.

### Quick .env Content (Copy This)

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## ✅ Success Indicators

You'll know Groq is working when:
1. Backend logs show: "Using Groq API for LLM generation"
2. Responses come in 1-3 seconds
3. No Ollama logs appear
4. Streamlit shows model as "llama-3.3-70b-versatile"

That's it! Much faster than local Ollama ⚡
