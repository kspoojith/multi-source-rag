"""Quick test to verify Groq API key works"""
import os
import requests

# Read from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# If .env not loaded, read file directly
if not GROQ_API_KEY or GROQ_API_KEY == "":
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    GROQ_API_KEY = line.split("=", 1)[1].strip()
                    break
    except:
        pass

print(f"\n🔑 Testing API Key...")
print(f"Key starts with: {GROQ_API_KEY[:10]}...")
print(f"Key length: {len(GROQ_API_KEY)} chars")

# Test API call
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

print(f"\n🧪 Testing Groq API...\n")

try:
    # Test 1: Get models (simple auth check)
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers=headers,
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS! Your API key is valid!")
        models = response.json()
        print(f"\nAvailable models: {len(models.get('data', []))}")
        for model in models.get('data', [])[:3]:
            print(f"  - {model.get('id')}")
    elif response.status_code == 401:
        print("❌ INVALID API KEY!")
        print(f"\nError: {response.text}")
        print("\n💡 Solutions:")
        print("1. Go to https://console.groq.com/keys")
        print("2. Check if your key is active")
        print("3. Create a NEW key if needed")
        print("4. Copy the FULL key (including 'gsk_' prefix)")
        print("5. Paste it in .env file")
    else:
        print(f"⚠️ Unexpected response: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50)
