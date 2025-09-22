"""
Setup script to create FAISS index for career roadmap generation.
Run this once to prepare the vector search index.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def create_roadmap_index():
    """Create FAISS index from career roadmap data."""
    
    print("🚀 Starting Career Roadmap Setup...")
    print("=" * 50)
    
    # Check if required file exists
    if not os.path.exists("career_roadmaps_full.json"):
        print("❌ Error: career_roadmaps_full.json not found!")
        print("Please make sure you've created this file in the project root.")
        return None, None
    
    print("✅ Found career_roadmaps_full.json")
    
    print("\n📦 Loading sentence transformer model...")
    print("(This might take a minute on first run - downloading ~90MB)")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have internet connection and sentence-transformers installed")
        return None, None
    
    print("\n📄 Loading career roadmap data...")
    try:
        with open("career_roadmaps_full.json", "r") as f:
            roadmap_data = json.load(f)
        print(f"✅ Loaded {len(roadmap_data)} career roadmaps")
    except Exception as e:
        print(f"❌ Error loading roadmap data: {e}")
        return None, None
    
    print("\n🔤 Creating role descriptions for embedding...")
    role_descriptions = []
    for i, career in enumerate(roadmap_data):
        # Create a comprehensive description for better matching
        description = f"{career['role']}. Skills: {', '.join(career['skills'])}. Tools: {', '.join(career['tools'])}"
        role_descriptions.append(description)
        print(f"  {i+1:2d}. {career['role']}")
    
    print(f"\n🧠 Generating embeddings for {len(role_descriptions)} careers...")
    print("(This will take 30-60 seconds)")
    try:
        embeddings = model.encode(role_descriptions, convert_to_numpy=True, show_progress_bar=True).astype("float32")
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None, None
    
    print("\n🔍 Creating FAISS index...")
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        print(f"✅ Created FAISS index with {index.ntotal} vectors, dimension {dimension}")
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")
        return None, None
    
    print("\n💾 Saving FAISS index...")
    try:
        faiss.write_index(index, "roadmap_index_local.faiss")
        print("✅ Saved: roadmap_index_local.faiss")
    except Exception as e:
        print(f"❌ Error saving FAISS index: {e}")
        return None, None
    
    print("\n📋 Creating metadata file...")
    try:
        with open("roles_metadata_local.json", "w") as f:
            json.dump(role_descriptions, f, indent=2)
        print("✅ Saved: roles_metadata_local.json")
    except Exception as e:
        print(f"❌ Error saving metadata: {e}")
        return None, None
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("Files created:")
    print("  📁 roadmap_index_local.faiss (Vector search index)")
    print("  📁 roles_metadata_local.json (Role descriptions)")
    print("=" * 50)
    
    return index, roadmap_data, model

def test_search(index, roadmap_data, model, query_text="Python machine learning data science"):
    """Test the search functionality."""
    print(f"\n🧪 Testing search functionality...")
    print(f"Query: '{query_text}'")
    print("-" * 30)
    
    try:
        # Generate query embedding
        query_embedding = model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        
        # Search
        D, I = index.search(query_embedding, k=3)
        
        print("Top 3 career matches:")
        for i, (idx, score) in enumerate(zip(I[0], D[0])):
            career = roadmap_data[idx]
            confidence = min(100, max(0, 100 - (score * 10)))
            print(f"\n{i+1}. 🎯 {career['role']}")
            print(f"   📊 Confidence: {confidence:.0f}%")
            print(f"   💡 Top Skills: {', '.join(career['skills'][:4])}")
        
        print("\n✅ Search test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def verify_files():
    """Verify all required files exist."""
    print("\n🔍 Verifying generated files...")
    
    files_to_check = [
        "roadmap_index_local.faiss",
        "roles_metadata_local.json",
        "career_roadmaps_full.json"
    ]
    
    all_good = True
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            if size > 0:
                print(f"  ✅ {file} ({size:,} bytes)")
            else:
                print(f"  ⚠️  {file} (empty file)")
                all_good = False
        else:
            print(f"  ❌ {file} (missing)")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🚀 Career Navigator - Roadmap Generation Setup")
    print("This will create the necessary files for AI-powered career matching.\n")
    
    try:
        # Main setup
        result = create_roadmap_index()
        if result[0] is None:
            print("\n❌ Setup failed. Please check the errors above.")
            exit(1)
        
        index, roadmap_data, model = result
        
        # Test the setup
        if test_search(index, roadmap_data, model):
            print("\n🧪 Running additional tests...")
            
            # Test with different queries
            test_queries = [
                "web development frontend React",
                "cybersecurity penetration testing",
                "artificial intelligence neural networks"
            ]
            
            for query in test_queries:
                test_search(index, roadmap_data, model, query)
        
        # Verify files
        if verify_files():
            print("\n" + "=" * 60)
            print("🎉 SETUP SUCCESSFUL!")
            print("Your Career Navigator now has AI-powered roadmap generation!")
            print("\nNext steps:")
            print("1. ✅ Start your FastAPI backend")
            print("2. ✅ Start your Streamlit frontend") 
            print("3. ✅ Try the new 'Roadmap Generator' page")
            print("4. ✅ Upload a resume and get personalized career suggestions!")
            print("=" * 60)
        else:
            print("\n⚠️  Some files may be missing or corrupted.")
            print("Please run the setup again.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        print("Please check your environment and try again.")
        import traceback
        traceback.print_exc()