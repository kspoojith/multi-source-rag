"""Quick script to list all available Groq models"""
import os
from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

print(f"Using API key: {api_key[:15]}..." if api_key else "No API key found")
print("\n" + "="*80)
print("AVAILABLE GROQ MODELS")
print("="*80 + "\n")

try:
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10
    )
    
    if response.status_code == 200:
        models = response.json()["data"]
        print(f"✅ Found {len(models)} models:\n")
        
        for model in sorted(models, key=lambda x: x["id"]):
            model_id = model["id"]
            owned_by = model.get("owned_by", "unknown")
            print(f"  • {model_id}")
            if "context_window" in model:
                print(f"    Context: {model['context_window']} tokens")
        
        print("\n" + "="*80)
        print("RECOMMENDED MODELS FOR YOUR APP:")
        print("="*80)
        print("\nLLM Models (for text generation):")
        llm_models = [m for m in models if any(x in m["id"].lower() for x in ["llama", "mixtral", "gemma"])]
        for model in sorted(llm_models, key=lambda x: x["id"]):
            print(f"  • {model['id']}")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")
