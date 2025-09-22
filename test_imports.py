"""
Test script to verify all dependencies are working
"""

print("🧪 Testing imports...")

try:
    import numpy
    print("✅ numpy:", numpy.__version__)
except ImportError as e:
    print("❌ numpy:", e)

try:
    import faiss
    print("✅ faiss: imported successfully")
except ImportError as e:
    print("❌ faiss:", e)

try:
    import torch
    print("✅ torch:", torch.__version__)
except ImportError as e:
    print("❌ torch:", e)

try:
    import transformers
    print("✅ transformers:", transformers.__version__)
except ImportError as e:
    print("❌ transformers:", e)

try:
    import huggingface_hub
    print("✅ huggingface_hub:", huggingface_hub.__version__)
except ImportError as e:
    print("❌ huggingface_hub:", e)

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence_transformers: imported successfully")
    
    # Test model loading
    print("🧪 Testing model loading...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded successfully!")
    
    # Test encoding
    test_embedding = model.encode(["test sentence"])
    print(f"✅ Encoding test successful! Shape: {test_embedding.shape}")
    
except ImportError as e:
    print("❌ sentence_transformers:", e)
except Exception as e:
    print("❌ Model loading error:", e)

print("\n🎯 If all tests pass, you can run setup_roadmap.py")
print("🔧 If any tests fail, follow the installation instructions above")