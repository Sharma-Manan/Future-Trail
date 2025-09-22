from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# These would be in separate files in a real app
from app.resume_parser import parse_resume
from app.prompts import build_ats_prompt
from app.gemini_handler import get_gemini_response

app = FastAPI(title="Career Navigator API")

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals for ML Models ---
career_model = None
label_encoder = None
sentence_model = None
roadmap_index = None
roadmap_data = None

# --- Pydantic Models for Validation ---
class CareerFeatures(BaseModel):
    pass

class ATSRequest(BaseModel):
    resume_text: str
    job_role: str

class RoadmapRequest(BaseModel):
    resume_text: str

# --- App Events ---
@app.on_event("startup")
def load_models():
    """Load all models on startup to avoid loading them on every request."""
    global career_model, label_encoder, sentence_model, roadmap_index, roadmap_data
    
    # Load existing career prediction models
    career_model = joblib.load(r"trained-models/careermodel.pkl")
    label_encoder = joblib.load(r"trained-models/labelencoder.pkl")
    
    # Load sentence transformer for roadmap generation
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load FAISS index and roadmap data
    roadmap_index = faiss.read_index("roadmap_index_local.faiss")
    with open("career_roadmaps_full.json", "r") as f:
        roadmap_data = json.load(f)

# --- Helper Classes ---
class _SyncUploadWrapper:
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data
    def read(self):
        return self._data

# --- Roadmap Generation Functions ---
def get_resume_embedding(text):
    """Generate embedding for resume text using sentence transformer."""
    return sentence_model.encode([text], convert_to_numpy=True).astype("float32")

def search_career_match(query_embedding, k=3):
    """Search for best career matches using FAISS."""
    D, I = roadmap_index.search(query_embedding, k=k)
    results = []
    for i, score in zip(I[0], D[0]):
        career_info = roadmap_data[i].copy()
        career_info['similarity_score'] = float(score)
        results.append(career_info)
    return results

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Career Navigator FastAPI with Roadmap Generation is running!"}

@app.post("/parse-resume/")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Extracts text from an uploaded resume file (PDF/DOCX)."""
    try:
        raw = await file.read()
        wrapper = _SyncUploadWrapper(file.filename, raw)
        text = parse_resume(wrapper)
        return {"resume_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")

@app.post("/ats-score/")
async def ats_score_endpoint(request: ATSRequest):
    """Get ATS score for resume against job role."""
    try:
        prompt = build_ats_prompt(request.resume_text, request.job_role)
        result = get_gemini_response(prompt)
        return {"ats_result": result}
    except Exception as e:
        print("--- ERROR IN /ats-score/ ---")
        traceback.print_exc()
        print("-----------------------------")
        raise HTTPException(status_code=500, detail=f"Error getting ATS score: {str(e)}")

@app.post("/predict-career/")
def predict_career_endpoint(features: dict):
    """Predicts a career based on a JSON object of user features."""
    if not career_model:
        raise HTTPException(status_code=500, detail="Career model is not loaded.")
    try:
        df = pd.DataFrame([features])
        model_cols = career_model.feature_names_in_
        
        # Add any missing columns with a value of 0
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder to match model's expected input
        df = df[model_cols]
        
        prediction = career_model.predict(df)
        label = label_encoder.inverse_transform(prediction)[0]
        return {"recommended_career": label}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error in predict_career: {tb}") 
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/generate-roadmap/")
async def generate_roadmap_endpoint(request: RoadmapRequest):
    """Generate career roadmap based on resume analysis."""
    try:
        if not sentence_model or roadmap_index is None or roadmap_data is None:
            raise HTTPException(status_code=500, detail="Roadmap models are not loaded.")
        
        # Generate embedding from resume text
        embedding = get_resume_embedding(request.resume_text)
        
        # Search for best career matches
        career_matches = search_career_match(embedding, k=3)
        
        return {
            "roadmap_suggestions": career_matches,
            "primary_match": career_matches[0] if career_matches else None
        }
    except Exception as e:
        print("--- ERROR IN /generate-roadmap/ ---")
        traceback.print_exc()
        print("-----------------------------")
        raise HTTPException(status_code=500, detail=f"Error generating roadmap: {str(e)}")

@app.post("/comprehensive-analysis/")
async def comprehensive_analysis_endpoint(file: UploadFile = File(...)):
    """
    Comprehensive analysis combining resume parsing, career prediction via roadmap,
    and learning path generation in a single endpoint.
    """
    try:
        # Parse resume
        raw = await file.read()
        wrapper = _SyncUploadWrapper(file.filename, raw)
        resume_text = parse_resume(wrapper)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from resume")
        
        # Generate roadmap based on resume
        embedding = get_resume_embedding(resume_text)
        career_matches = search_career_match(embedding, k=5)
        
        # Return comprehensive analysis
        return {
            "resume_text": resume_text,
            "career_suggestions": career_matches,
            "top_recommendation": career_matches[0] if career_matches else None,
            "analysis_summary": {
                "total_matches": len(career_matches),
                "confidence_score": career_matches[0]['similarity_score'] if career_matches else 0
            }
        }
    except Exception as e:
        print("--- ERROR IN /comprehensive-analysis/ ---")
        traceback.print_exc()
        print("-----------------------------")
        raise HTTPException(status_code=500, detail=f"Error in comprehensive analysis: {str(e)}")