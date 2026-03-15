"""Debug script to see exactly what backend reads"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load exactly how backend does
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

import os

print("="*60)
print("🔍 BACKEND ENVIRONMENT DEBUG")
print("="*60)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

print(f"\nLLM_PROVIDER: '{LLM_PROVIDER}'")
print(f"GROQ_MODEL: '{GROQ_MODEL}'")
print(f"\nGROQ_API_KEY:")
print(f"  - Length: {len(GROQ_API_KEY)} chars")
print(f"  - Starts with: '{GROQ_API_KEY[:10]}'")
print(f"  - Ends with: '...{GROQ_API_KEY[-6:]}'")
print(f"  - Has whitespace? {GROQ_API_KEY != GROQ_API_KEY.strip()}")
has_quotes = GROQ_API_KEY.startswith('"') or GROQ_API_KEY.startswith("'")
print(f"  - Has quotes? {has_quotes}")

if GROQ_API_KEY and GROQ_API_KEY.strip():
    # Test it
    import requests
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"\n📡 Testing API with this key...")
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ KEY WORKS!")
        else:
            print(f"   ❌ KEY FAILED: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print("\n❌ GROQ_API_KEY is empty!")

print("\n" + "="*60)

# Now check what config.py sees
print("\n📦 CHECKING CONFIG.PY...")
try:
    from backend.config import LLM_PROVIDER as cfg_provider, GROQ_API_KEY as cfg_key
    print(f"Config LLM_PROVIDER: '{cfg_provider}'")
    print(f"Config GROQ_API_KEY length: {len(cfg_key)} chars")
    print(f"Config GROQ_API_KEY starts: '{cfg_key[:10]}'")
except Exception as e:
    print(f"❌ Error loading config: {e}")

print("="*60)
