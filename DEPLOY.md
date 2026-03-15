# 🚀 Zero-Cost Deployment Guide

Deploy your Multilingual RAG system for **FREE** using Streamlit Community Cloud + Groq API.

**No credit card required. No setup complexity. Ready in 10 minutes.**

---

## 📋 Prerequisites

- GitHub account (free)
- Groq API account (free - no credit card)
- Your code pushed to GitHub (you've already done this!)

---

## 🎯 Step-by-Step Deployment

### Step 1: Get Your Free Groq API Key (2 minutes)

1. **Visit:** https://console.groq.com
2. **Sign up** with your email (no credit card needed!)
3. **Go to:** API Keys section
4. **Click:** "Create API Key"
5. **Copy** the key (starts with `gsk_...`)

**✅ You now have:** 14,400 free requests per day!

---

### Step 2: Prepare Your Code (1 minute)

Your code is already cloud-ready! Just verify:

```bash
# Make sure you have all the latest changes
git status

# If there are changes, commit them:
git add .
git commit -m "Add Groq API support for cloud deployment"
git push origin main
```

---

### Step 3: Deploy on Streamlit Community Cloud (5 minutes)

1. **Visit:** https://share.streamlit.io

2. **Sign in** with GitHub

3. **Click:** "New app"

4. **Fill in the form:**
   - **Repository:** Select your `multi-source-rag` repo
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** Choose a custom name (e.g., `yourname-rag-app`)

5. **Click:** "Advanced settings" (before deploying)

6. **Go to:** "Secrets" tab

7. **Paste this** (replace with YOUR actual API key):
   ```toml
   GROQ_API_KEY = "gsk_YOUR_ACTUAL_API_KEY_HERE"
   LLM_PROVIDER = "groq"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```

8. **Click:** "Deploy!"

9. **Wait** 2-3 minutes for initial deployment

---

### Step 4: Test Your Live App! 🎉

Your app will be live at: `https://yourname-rag-app.streamlit.app`

**Try these queries:**
- "Elon Musk ki net worth kitni hai?"
- "Latest AI news kya hai?"
- "Climate change kya hai?"

**Expected performance:**
- ⚡ **1-3 seconds** per query (10x faster than local Ollama!)
- 🌍 **Works globally** - your friends can use it too
- 💰 **$0 cost** - completely free tier

---

## 🔧 Configuration Details

### Environment Variables (Streamlit Secrets)

Your app reads these from Streamlit's secure secrets management:

| Secret | Required | Description | Default |
|--------|----------|-------------|---------|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key | None |
| `LLM_PROVIDER` | ✅ Yes | Must be "groq" for cloud | "ollama" |
| `GROQ_MODEL` | No | Model to use | "llama-3.3-70b-versatile" |
| `GROQ_TEMPERATURE` | No | Response creativity (0-2) | "0.3" |
| `GROQ_MAX_TOKENS` | No | Max response length | "1024" |

### Available Models (Free Tier)

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| **llama-3.3-70b-versatile** | Medium | ⭐⭐⭐⭐⭐ | Most accurate answers (recommended) |
| **llama-3.1-8b-instant** | Fast | ⭐⭐⭐⭐ | Quick responses |
| **qwen/qwen3-32b** | Medium | ⭐⭐⭐⭐ | Good balance, 131K context |
| **openai/gpt-oss-120b** | Slower | ⭐⭐⭐⭐⭐ | Largest model (experimental) |

---

## 📊 Free Tier Limits

### Groq API:
- **14,400 requests/day** (600/hour)
- **30 requests/minute** rate limit
- **No credit card required**
- **No expiration**

### Streamlit Community Cloud:
- **Unlimited** visitors
- **1GB RAM** per app
- **1 CPU core**
- **Auto-sleep** after 7 days of inactivity (wakes on first visit)
- **GitHub required** (public or private repos)

---

## 🔄 Updating Your Deployed App

After deployment, any changes you push to GitHub will auto-deploy:

```bash
# Make changes to your code
git add .
git commit -m "Update app"
git push origin main

# Streamlit automatically redeploys in 2-3 minutes!
```

**Manual redeploy:**
1. Go to your app's dashboard
2. Click "☰" menu → "Reboot app"

---

## 🐛 Troubleshooting

### "Backend Offline" Error

**Cause:** Groq API key not set or invalid

**Fix:**
1. Go to your app dashboard: https://share.streamlit.io
2. Click your app → "⚙️ Settings"
3. Go to "Secrets" tab
4. Verify `GROQ_API_KEY` is correct (no extra spaces!)
5. Save and reboot app

### "Request timed out"

**Cause:** Groq API rate limit hit or network issue

**Fix:**
- Wait 1 minute and try again
- Check Groq API status: https://status.groq.com
- Reduce `GROQ_MAX_TOKENS` in secrets

### "Module not found"

**Cause:** Missing dependency in requirements.txt

**Fix:**
1. Check requirements.txt has all packages
2. Push updated requirements.txt
3. Streamlit will auto-redeploy

### App sleeps after 7 days

**Expected behavior!** Free tier apps sleep after 7 days of inactivity.

**Fix:** Just visit the URL - it wakes up in 30 seconds

---

## 🎨 Customization

### Change the Model

In Streamlit Secrets, change:
```toml
GROQ_MODEL = "llama3-8b-8192"  # Faster model
```

### Adjust Temperature

For more creative/varied responses:
```toml
GROQ_TEMPERATURE = "0.7"  # Higher = more creative (0-2)
```

For more factual/consistent responses:
```toml
GROQ_TEMPERATURE = "0.1"  # Lower = more factual
```

### Increase Response Length

```toml
GROQ_MAX_TOKENS = "2048"  # Longer answers
```

---

## 🚀 Advanced: Custom Domain (Optional)

Streamlit apps get a free `.streamlit.app` subdomain, but you can add a custom domain:

1. Get a domain from: Freenom (free) or Namecheap (~$10/year)
2. In Streamlit app settings → "Custom domain"
3. Add your domain
4. Update DNS records as instructed
5. Done! Your app at `yourapp.com`

---

## 💰 Cost Breakdown

| Service | Free Tier | Cost After Free | Our Usage |
|---------|-----------|-----------------|-----------|
| **Groq API** | 14,400 req/day | N/A (we stay within) | ~100-500 req/day |
| **Streamlit Cloud** | 1 app unlimited | $0 (community tier) | 1 app |
| **GitHub** | Unlimited repos | $0 (free tier) | 1 repo |
| **Total →** | **$0/month** | **$0/month** | **$0/month** |

---

## 📈 Scaling Beyond Free Tier

If you outgrow the free tier (>14,000 queries/day):

### Option 1: Groq Paid Tier
- **$0.27 per 1M tokens** (input)
- **$0.27 per 1M tokens** (output)
- Still cheaper than OpenAI!

### Option 2: Switch to Ollama + Oracle Cloud
- Deploy on Oracle Cloud Free Tier
- Run Ollama locally (no per-request cost)
- Only pay for bandwidth (10TB/month free)

### Option 3: Use Multiple API Keys
- Create multiple Groq accounts
- Rotate between them (14k × N accounts)
- Free but requires management

---

## ✅ Your Deployment Checklist

- [ ] Got Groq API key from console.groq.com
- [ ] Code pushed to GitHub
- [ ] Created Streamlit Cloud account
- [ ] Deployed app with secrets configured
- [ ] Tested with example queries
- [ ] Shared URL with friends!

---

## 🎉 Congratulations!

Your multilingual RAG system is now:
- ✅ **Live globally** at a custom URL
- ✅ **Free forever** (within limits)
- ✅ **10x faster** than local Ollama
- ✅ **Auto-updating** from GitHub
- ✅ **Zero maintenance** required

**Share your app:** `https://yourname-rag-app.streamlit.app`

---

## 📞 Need Help?

- **Groq API Docs:** https://console.groq.com/docs
- **Streamlit Docs:** https://docs.streamlit.io/deploy
- **GitHub Issues:** Create an issue in your repo

---

**Next Steps:**
1. Share your app with friends
2. Monitor usage in Groq dashboard
3. Customize the UI to your liking
4. Add more features!

**Happy deploying!** 🚀
