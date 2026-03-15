#!/usr/bin/env python3
"""
Production Optimization Script
================================
Checks system capabilities and recommends optimal configuration
for lowest latency in production.
"""

import subprocess
import sys
import platform
import requests
from pathlib import Path

def check_ollama():
    """Check if Ollama is running and which models are available."""
    print("🔍 Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"✅ Ollama is running")
            print(f"📦 Available models: {', '.join(model_names)}")
            
            # Check for recommended models
            has_mistral = any("mistral" in m for m in model_names)
            has_phi3 = any("phi3" in m for m in model_names)
            
            if not has_mistral and not has_phi3:
                print("\n⚠️  No recommended models found!")
                print("Run one of these commands:")
                print("  ollama pull mistral        # 7B model, best quality")
                print("  ollama pull phi3:mini      # 3B model, 2-3x faster")
                return False
            
            return True
    except Exception as e:
        print(f"❌ Ollama not running: {e}")
        print("\n💡 Start Ollama:")
        print("  1. Download from https://ollama.ai")
        print("  2. Run: ollama serve")
        return False

def check_gpu():
    """Check if GPU is available for PyTorch."""
    print("\n🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ No GPU detected (CPU-only mode)")
            print("\n💡 For 10x faster inference:")
            print("  - Install NVIDIA GPU with CUDA")
            print("  - Or switch to phi3:mini model (2-3x faster on CPU)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def recommend_config():
    """Provide configuration recommendations."""
    print("\n" + "="*60)
    print("📋 PRODUCTION CONFIGURATION RECOMMENDATIONS")
    print("="*60)
    
    has_ollama = check_ollama()
    has_gpu = check_gpu()
    
    print("\n🎯 Recommended Settings:")
    print("-" * 60)
    
    if has_gpu:
        print("✅ GPU Mode:")
        print("   - Model: mistral (best quality)")
        print("   - Expected latency: 2-4 seconds")
        print("   - Update backend/config.py:")
        print("     OLLAMA_MODEL = 'mistral'")
        print("     RETRIEVAL_TOP_K = 4")
        print("     WEB_SEARCH_MAX_RESULTS = 10")
    else:
        print("⚡ CPU Optimization Mode:")
        print("   - Model: phi3:mini (faster on CPU)")
        print("   - Expected latency: 6-10 seconds")
        print("   - Update backend/config.py:")
        print("     OLLAMA_MODEL = 'phi3:mini'")
        print("     RETRIEVAL_TOP_K = 3")
        print("     WEB_SEARCH_MAX_RESULTS = 8")
        print("\n   💡 To switch to phi3:mini:")
        print("      ollama pull phi3:mini")
    
    print("\n📊 Timeout Settings:")
    print("   - OLLAMA_TIMEOUT = 180  # Current setting (3 minutes)")
    if not has_gpu:
        print("   - Consider increasing to 240 for CPU-only mode")
    
    print("\n🌐 Web Search Settings (Production):")
    print("   - WEB_SEARCH_REGION = 'wt-wt'  # Worldwide")
    print("   - WEB_SEARCH_MAX_RESULTS = 8-10")
    print("   - WEB_SEARCH_TIMEOUT = 10")
    
    print("\n" + "="*60)
    print("📝 NEXT STEPS:")
    print("="*60)
    
    if not has_ollama:
        print("1. ❌ Install and start Ollama")
        print("2. ❌ Pull a model (mistral or phi3:mini)")
    else:
        print("1. ✅ Ollama is ready")
    
    print("2. 🌐 Use Web Search Mode (default) for unlimited questions")
    print("3. 🔧 If latency >10s, switch to phi3:mini model")
    print("4. 🚀 For production: Deploy on GPU instance")
    
    print("\n💡 Test your setup:")
    print("   python -m backend.app")
    print("   Open http://localhost:8000")
    print("   Try: 'Elon Musk ki net worth kitni hai?'")
    print("="*60)

def benchmark_embedding():
    """Quick embedding speed test."""
    print("\n⏱️  Running quick benchmark...")
    try:
        from ingestion.embedder import get_embedding_model
        import time
        
        model = get_embedding_model()
        test_texts = [
            "Modi ji ka education kya hai?",
            "Elon Musk ki net worth kitni hai?",
            "What is the meaning of life?",
        ] * 3  # 9 texts total
        
        start = time.time()
        vectors = model.embed(test_texts)
        elapsed = (time.time() - start) * 1000
        
        print(f"✅ Embedding 9 texts: {elapsed:.0f}ms ({elapsed/9:.0f}ms per text)")
        
        if elapsed > 1000:
            print("⚠️  Slow embedding. Consider GPU or smaller batch sizes.")
        else:
            print("✅ Embedding speed is good!")
            
    except Exception as e:
        print(f"❌ Could not benchmark: {e}")

if __name__ == "__main__":
    print("🚀 Multi-Source RAG - Production Optimizer")
    print("="*60 + "\n")
    
    recommend_config()
    
    # Optional: Run benchmark if model is available
    try_benchmark = input("\n❓ Run embedding benchmark? (y/N): ").strip().lower()
    if try_benchmark == 'y':
        benchmark_embedding()
    
    print("\n✅ Optimization check complete!")
    print("📚 For detailed guide, see: PRODUCTION.md")
