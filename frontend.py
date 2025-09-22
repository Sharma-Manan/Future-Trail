import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Future Trail | Career Navigator", layout="wide", page_icon="🚀")

import joblib
import pandas as pd
import numpy as np
import requests
import os
import plotly.graph_objects as go
import plotly.express as px
import re
import psycopg2
import streamlit_authenticator as stauth

# --- DATABASE FUNCTIONS ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using secrets."""
    try:
        conn = psycopg2.connect(st.secrets["DB_CONNECTION_STRING"])
        return conn
    except:
        return None

def fetch_users():
    """Fetches user data from the database for the authenticator."""
    try:
        conn = get_db_connection()
        if conn is None:
            return {"usernames": {
                "demo": {
                    "name": "Demo User",
                    "password": "$2b$12$demo_hashed_password",
                    "email": "demo@example.com"
                }
            }}
        
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name, username, password, email FROM users")
                users = cur.fetchall()

        credentials = {"usernames": {}}
        for name, username, password, email in users:
            credentials["usernames"][username] = {
                "name": name, 
                "password": password,
                "email": email
            }
        return credentials
    except Exception as error:
        st.error(f"Error fetching users from database: {error}")
        return {"usernames": {}}

def save_new_user(username, name, hashed_password, email):
    """Saves a new registered user to the database."""
    try:
        conn = get_db_connection()
        if conn is None:
            st.warning("Database not connected. User registration disabled.")
            return False
            
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (name, username, password, email) VALUES (%s, %s, %s, %s)",
                    (name, username, hashed_password, email)
                )
                conn.commit()
        return True
    except Exception as error:
        st.error(f"Error saving new user: {error}")
        return False

def update_user_career(username, career):
    """Updates the user's predicted career in the database."""
    try:
        conn = get_db_connection()
        if conn is None:
            st.info("Database not connected. Career not saved.")
            return False
            
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET career = %s WHERE username = %s",
                    (career, username)
                )
                conn.commit()
        return True
    except Exception as error:
        st.error(f"Error updating career: {error}")
        return False

# --- SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initialize session state variables for resume management"""
    if "uploaded_resume" not in st.session_state:
        st.session_state.uploaded_resume = None
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "resume_filename" not in st.session_state:
        st.session_state.resume_filename = ""
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    if "registration_success" not in st.session_state:
        st.session_state.registration_success = False
    if "show_preview" not in st.session_state:
        st.session_state.show_preview = False

# --- USER AUTHENTICATION ---
user_credentials = fetch_users()

config = {
    'credentials': user_credentials,
    'cookie': {
        'name': st.secrets.get("COOKIE_NAME", "career_navigator_cookie"),
        'key': st.secrets.get("COOKIE_KEY", "some_default_secret_key"),
        'expiry_days': 30
    },
    'preauthorized': []
}

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Initialize session state
initialize_session_state()

# Fixed: Change to local backend URL
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Enhanced CSS
st.markdown(
    """
    <style>
    body {background-color: #f5f7fa;}
    .main-title {
        font-size: 2.6em;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.3em;
        color: #3949ab;
        font-weight: 600;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }
    .roadmap-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .skill-tag {
        background: rgba(255,255,255,0.2);
        padding: 4px 12px;
        border-radius: 20px;
        margin: 2px;
        display: inline-block;
        font-size: 0.9em;
    }
    .learning-step {
        background: #f8f9ff;
        border-left: 4px solid #3949ab;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .confidence-badge {
        background: #4caf50;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8em;
    }
    .analysis-option {
        background: #f8f9ff;
        border: 1px solid #e0e7ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- RESUME MANAGEMENT FUNCTIONS ---
def run_resume_manager():
    """Centralized resume upload and management"""
    st.title("📄 Resume Manager")
    st.markdown("Upload your resume once and access all analysis features!")
    
    # Resume upload section
    uploaded_file = st.file_uploader(
        "Upload your Resume", 
        type=["pdf", "docx"],
        help="Upload your resume to unlock all analysis features"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        if (st.session_state.uploaded_resume is None or 
            uploaded_file.name != st.session_state.resume_filename):
            
            with st.spinner("Processing your resume..."):
                try:
                    # Store file info
                    st.session_state.uploaded_resume = uploaded_file.getvalue()
                    st.session_state.resume_filename = uploaded_file.name
                    
                    # Extract text
                    files = {"file": (uploaded_file.name, st.session_state.uploaded_resume, uploaded_file.type)}
                    resp = requests.post(f"{API_BASE}/parse-resume/", files=files, timeout=30)
                    resp.raise_for_status()
                    st.session_state.resume_text = resp.json().get("resume_text", "")
                    
                    # Clear previous analysis cache
                    st.session_state.analysis_cache = {}
                    
                    st.success(f"✅ Resume '{uploaded_file.name}' processed successfully!")
                    
                except Exception as e:
                    st.error(f"Failed to process resume: {e}")
                    return
    
    # Show current resume status
    if st.session_state.uploaded_resume:
        st.markdown("### 📋 Current Resume")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"📄 **{st.session_state.resume_filename}**")
        with col2:
            if st.button("🔍 Preview Text"):
                st.session_state.show_preview = not st.session_state.show_preview
        with col3:
            if st.button("🗑️ Remove Resume"):
                st.session_state.uploaded_resume = None
                st.session_state.resume_text = ""
                st.session_state.resume_filename = ""
                st.session_state.analysis_cache = {}
                st.session_state.show_preview = False
                st.rerun()
        
        # Show preview if requested
        if st.session_state.show_preview:
            with st.expander("📃 Resume Text Preview", expanded=True):
                st.text_area("Extracted Text", st.session_state.resume_text, height=200, disabled=True)
        
        # Available analysis options
        st.markdown("### 🚀 Available Analysis")
        
        analysis_options = [
            ("📄 ATS Score Analysis", "Evaluate how well your resume matches specific job roles", "ats"),
            ("🛣️ Career Roadmap", "Get personalized career suggestions and learning paths", "roadmap"),
            ("🔍 Comprehensive Report", "Complete analysis with career matching and ATS evaluation", "comprehensive")
        ]
        
        for option, description, key in analysis_options:
            with st.container():
                st.markdown(f"""
                    <div class="analysis-option">
                        <h4>{option}</h4>
                        <p style="margin: 5px 0; color: #666;">{description}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Analyze", key=f"btn_{key}", use_container_width=True):
                    st.session_state.active_analysis = key
                    st.rerun()
    
    else:
        st.info("👆 Upload your resume to get started with career analysis!")
        
        # Show sample features
        st.markdown("### 🌟 What You'll Get")
        
        features = [
            "📊 **ATS Score Analysis** - See how your resume performs against applicant tracking systems",
            "🎯 **Career Matching** - Get AI-powered career suggestions based on your experience",
            "📚 **Learning Roadmaps** - Personalized learning paths for your target career",
            "🔍 **Comprehensive Reports** - Complete analysis with actionable insights"
        ]
        
        for feature in features:
            st.markdown(feature)

def run_analysis_hub():
    """Main analysis hub with resume-based features"""
    
    # Check if resume is uploaded
    if not st.session_state.get("uploaded_resume"):
        st.warning("📄 Please upload your resume in the Resume Manager first!")
        if st.button("Go to Resume Manager"):
            st.session_state.active_analysis = None
            st.rerun()
        return
    
    analysis_type = st.session_state.get("active_analysis", "ats")
    
    # Back button
    if st.button("← Back to Resume Manager"):
        st.session_state.active_analysis = None
        st.rerun()
    
    # Analysis type selector
    if analysis_type == "ats":
        run_ats_analysis()
    elif analysis_type == "roadmap":
        run_roadmap_analysis()
    elif analysis_type == "comprehensive":
        run_comprehensive_analysis_cached()

def run_ats_analysis():
    """ATS analysis using cached resume"""
    st.markdown("### 📄 ATS Score Analysis")
    st.markdown(f"**Resume:** {st.session_state.resume_filename}")
    
    # Job role input
    job_role = st.text_input(
        "Target Job Role", 
        value="Data Science", 
        help="Enter the specific job role you want to optimize for"
    )
    
    # Analysis button
    if st.button("🎯 Analyze ATS Score", use_container_width=True):
        cache_key = f"ats_{job_role}"
        
        # Check cache first
        if cache_key in st.session_state.analysis_cache:
            result = st.session_state.analysis_cache[cache_key]
            st.success("📊 Analysis retrieved from cache!")
        else:
            # Perform analysis
            with st.spinner("Analyzing resume for ATS compatibility..."):
                try:
                    ats_payload = {
                        "resume_text": st.session_state.resume_text,
                        "job_role": job_role
                    }
                    resp = requests.post(f"{API_BASE}/ats-score/", json=ats_payload, timeout=60)
                    resp.raise_for_status()
                    result = resp.json().get("ats_result", "")
                    
                    # Cache the result
                    st.session_state.analysis_cache[cache_key] = result
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return
        
        # Display results
        st.markdown("---")
        st.markdown(f"### 📊 ATS Analysis Results for {job_role}")
        
        # Parse the result string and display with gauge
        summary_match = re.search(r"-\s*(?:\*\*)?Summary(?:\*\*)?:\s*(.*)", result, re.DOTALL | re.IGNORECASE)
        score_match = re.search(r"-\s*(?:\*\*)?Score \(out of 10\)(?:\*\*)?:\s*(\d+\.?\d*)", result, re.IGNORECASE)
        suggestions_match = re.search(r"-\s*(?:\*\*)?Suggestions(?:\*\*)?:\s*(.*)", result, re.DOTALL | re.IGNORECASE)

        if score_match:
            score = float(score_match.group(1))
            summary = summary_match.group(1).strip() if summary_match else ""
            suggestions_str = suggestions_match.group(1).strip() if suggestions_match else ""

            col1, col2 = st.columns([2, 3])

            with col1:
                # RESTORED: ATS Score Gauge
                gauge_color = "#4caf50" if score >= 8 else ("#ff9800" if score >= 5 else "#f44336")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "<b>ATS Match Score</b>", 'font': {'size': 20, 'color': '#1a237e'}},
                    gauge={
                        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#d1d1d1",
                        'steps': [
                            {'range': [0, 5], 'color': '#ffebee'},
                            {'range': [5, 8], 'color': '#fff3e0'},
                            {'range': [8, 10], 'color': '#e8f5e9'},
                        ],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#3949ab", 'family': "Arial, sans-serif"},
                    height=280,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("<h3 style='color: #1a237e;'>📝 Evaluation Summary</h3>", unsafe_allow_html=True)
                if summary:
                    st.info(summary)
                else:
                    st.warning("Could not generate a summary for this resume.")

                if suggestions_str:
                    st.markdown("<h3 style='color: #1a237e;'>💡 Improvement Suggestions</h3>", unsafe_allow_html=True)
                    suggestions_list = [s.strip('*-. ') for s in suggestions_str.split('\n') if s.strip()]
                    for suggestion in suggestions_list:
                        if suggestion:
                            st.markdown(f"""
                                <div style="background-color: #f0f2f6; border-left: 5px solid #3949ab; padding: 12px; margin-bottom: 10px; border-radius: 5px;">
                                    <p style="margin: 0; font-size: 1.05em; color: #333;">{suggestion}</p>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            # Fallback if parsing fails
            st.write(result)

def run_roadmap_analysis():
    """Career roadmap analysis using cached resume"""
    st.markdown("### 🛣️ Career Roadmap Generator")
    st.markdown(f"**Resume:** {st.session_state.resume_filename}")
    
    # Analysis button
    if st.button("🚀 Generate Career Roadmap", use_container_width=True):
        cache_key = "roadmap_analysis"
        
        # Check cache first
        if cache_key in st.session_state.analysis_cache:
            result = st.session_state.analysis_cache[cache_key]
            st.success("🛣️ Roadmap retrieved from cache!")
        else:
            # Perform analysis
            with st.spinner("Generating personalized career roadmap..."):
                try:
                    files = {"file": (st.session_state.resume_filename, st.session_state.uploaded_resume, "application/pdf")}
                    resp = requests.post(f"{API_BASE}/comprehensive-analysis/", files=files, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    # Cache the result
                    st.session_state.analysis_cache[cache_key] = result
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return
        
        # Display results
        career_suggestions = result.get("career_suggestions", [])
        if career_suggestions:
            st.markdown("---")
            st.markdown("### 🛣️ Your Personalized Career Roadmap")
            
            # Display top recommendation
            top_career = career_suggestions[0]
            st.markdown(f"#### 🎯 Top Recommendation: {top_career['role']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Required Skills:**")
                for skill in top_career.get('skills', []):
                    st.markdown(f"• {skill}")
            
            with col2:
                st.markdown("**Essential Tools:**")
                for tool in top_career.get('tools', []):
                    st.markdown(f"• {tool}")
            
            # Learning path
            if 'learning_path' in top_career:
                st.markdown("#### 📚 Learning Path")
                for i, step in enumerate(top_career['learning_path'], 1):
                    st.markdown(f"{i}. {step}")

def run_comprehensive_analysis_cached():
    """Comprehensive analysis with ATS component always included"""
    st.markdown("### 🔍 Comprehensive Career Report")
    st.markdown(f"**Resume:** {st.session_state.resume_filename}")
    
    # Target role input (always included now)
    target_role = st.text_input(
        "Target Role for ATS Analysis", 
        value="Data Scientist",
        placeholder="e.g., Data Scientist, Software Engineer"
    )
    
    # Analysis button
    if st.button("📊 Generate Comprehensive Report", use_container_width=True):
        cache_key = f"comprehensive_{target_role}"
        
        # Check cache first
        if cache_key in st.session_state.analysis_cache:
            result = st.session_state.analysis_cache[cache_key]
            st.success("📊 Report retrieved from cache!")
        else:
            # Perform analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Career roadmap analysis
                status_text.text("🛣️ Analyzing career matches...")
                progress_bar.progress(30)
                
                files = {"file": (st.session_state.resume_filename, st.session_state.uploaded_resume, "application/pdf")}
                resp = requests.post(f"{API_BASE}/comprehensive-analysis/", files=files, timeout=60)
                resp.raise_for_status()
                roadmap_result = resp.json()
                
                progress_bar.progress(60)
                
                # Step 2: ATS analysis (always included)
                status_text.text("📄 Running ATS analysis...")
                ats_payload = {
                    "resume_text": st.session_state.resume_text,
                    "job_role": target_role
                }
                resp2 = requests.post(f"{API_BASE}/ats-score/", json=ats_payload, timeout=60)
                resp2.raise_for_status()
                ats_result = resp2.json().get("ats_result", "")
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Combine results
                result = {
                    "roadmap": roadmap_result,
                    "ats": ats_result,
                    "target_role": target_role
                }
                
                # Cache the result
                st.session_state.analysis_cache[cache_key] = result
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                progress_bar.progress(0)
                status_text.text("")
                return
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Comprehensive Career Analysis Report")
        
        # Roadmap section
        roadmap_data = result.get("roadmap", {})
        career_suggestions = roadmap_data.get("career_suggestions", [])
        
        if career_suggestions:
            st.markdown("#### 🎯 Career Recommendations")
            for i, career in enumerate(career_suggestions[:3], 1):
                with st.expander(f"{i}. {career['role']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Skills:**")
                        st.write(", ".join(career.get('skills', [])))
                    with col2:
                        st.markdown("**Tools:**")
                        st.write(", ".join(career.get('tools', [])))
        
        # ATS section with restored gauge
        ats_result = result.get("ats")
        target_role = result.get("target_role")
        
        if ats_result and target_role:
            st.markdown("---")
            st.markdown(f"### 📄 ATS Analysis for {target_role}")
            
            # Parse ATS result with the gauge visualization restored
            summary_match = re.search(r"-\s*(?:\*\*)?Summary(?:\*\*)?:\s*(.*)", ats_result, re.DOTALL | re.IGNORECASE)
            score_match = re.search(r"-\s*(?:\*\*)?Score \(out of 10\)(?:\*\*)?:\s*(\d+\.?\d*)", ats_result, re.IGNORECASE)
            suggestions_match = re.search(r"-\s*(?:\*\*)?Suggestions(?:\*\*)?:\s*(.*)", ats_result, re.DOTALL | re.IGNORECASE)
            
            if score_match:
                score = float(score_match.group(1))
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # RESTORED: ATS Score Gauge
                    gauge_color = "#4caf50" if score >= 8 else ("#ff9800" if score >= 5 else "#f44336")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "<b>ATS Match Score</b>", 'font': {'size': 20, 'color': '#1a237e'}},
                        gauge={
                            'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': gauge_color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#d1d1d1",
                            'steps': [
                                {'range': [0, 5], 'color': '#ffebee'},
                                {'range': [5, 8], 'color': '#fff3e0'},
                                {'range': [8, 10], 'color': '#e8f5e9'},
                            ],
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#3949ab", 'family': "Arial, sans-serif"},
                        height=280,
                        margin=dict(l=10, r=10, t=50, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if summary_match:
                        summary = summary_match.group(1).strip()
                        st.markdown("**📝 Summary:**")
                        st.info(summary)
                    
                    if suggestions_match:
                        suggestions_str = suggestions_match.group(1).strip()
                        st.markdown("**💡 Improvement Suggestions:**")
                        if suggestions_str:
                            suggestions_list = [s.strip('*-. ') for s in suggestions_str.split('\n') if s.strip()]
                            for suggestion in suggestions_list[:3]:  # Show top 3 suggestions
                                st.markdown(f"""
                                    <div style="background-color: #f0f2f6; border-left: 5px solid #3949ab; padding: 12px; margin-bottom: 10px; border-radius: 5px;">
                                        <p style="margin: 0; font-size: 1.05em; color: #333;">{suggestion}</p>
                                    </div>
                                """, unsafe_allow_html=True)
            else:
                # Fallback if score parsing fails
                st.write(ats_result)

def run_career_predictor():
    """Original career predictor with manual input"""
    st.title("🎯 Career Predictor")
    st.markdown("Fill out your profile to get AI-powered career recommendations!")
    
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "trained-models")
        mlb_dict = joblib.load(os.path.join(MODEL_DIR, "mlbdict.pkl"))
        ohe = joblib.load(os.path.join(MODEL_DIR, "ohencoder.pkl"))

        st.markdown("<div class='section-header'>🔍 Profile Information</div>", unsafe_allow_html=True)
        
        left_spacer, main_col, right_spacer = st.columns([1,3,1])
        with main_col:
            multi_label_inputs = {}
            for col in mlb_dict.keys():
                options = mlb_dict[col].classes_
                selected = st.multiselect(f"{col.replace('_', ' ')}", options, help=f"Select your {col.replace('_', ' ')}")
                multi_label_inputs[col] = selected
            preferred_style = st.selectbox("Preferred Work Style", ohe.categories_[0], help="Where do you prefer to work?")
            problem_style = st.multiselect("Problem Solving Style", mlb_dict['Problem_Solving_Style'].classes_, help="How do you approach problems?")
            masters = st.radio("Do you want to go for Masters?", ["Yes", "No"], horizontal=True)
            research = st.radio("Interested in Research?", ["Yes", "No"], horizontal=True)
            cgpa = st.slider("Current CGPA", 2.0, 10.0, 7.5, 0.1, help="Your latest CGPA")
            projects = st.number_input("Current Projects Count", min_value=0, step=1, help="How many projects have you done?")
            internships = st.number_input("Internship Duration (in months)", min_value=0, step=1, help="Total months of internship experience")

        def prepare_input():
            feature_parts = []
            for col, mlb in mlb_dict.items():
                selected_values = multi_label_inputs[col]
                encoded = mlb.transform([selected_values])
                df = pd.DataFrame(encoded, columns=[f"{col}_{c}" for c in mlb.classes_])
                feature_parts.append(df)
            other_features = pd.DataFrame([{
                **{f"Preferred_Work_Style_{cls}": 1 if cls == preferred_style else 0 for cls in ohe.categories_[0]},
                **dict(zip([f"Problem_Solving_Style_{cls}" for cls in mlb_dict['Problem_Solving_Style'].classes_], 
                             mlb_dict['Problem_Solving_Style'].transform([problem_style])[0])),
                "Wants_to_Go_for_Masters": 1 if masters.lower() == "yes" else 0,
                "Interested_in_Research": 1 if research.lower() == "yes" else 0,
                "CGPA": cgpa,
                "Current_Projects_Count": projects,
                "Internship_Experience": internships
            }])
            feature_parts.append(other_features)
            final_input = pd.concat(feature_parts, axis=1)
            return final_input

        if st.button("🎯 Predict My Career", use_container_width=True):
            input_df = prepare_input()
            features_dict = input_df.iloc[0].to_dict()
            with st.spinner("Getting recommendation..."):
                try:
                    resp = requests.post(f"{API_BASE}/predict-career/", json=features_dict, timeout=30)
                    resp.raise_for_status()
                    career = resp.json().get("recommended_career", "Unknown")
                    st.success(f"🎯 We recommend: **{career}**")
                    
                    if update_user_career(st.session_state["username"], career):
                        st.info("Your predicted career has been saved successfully!")
                        
                except Exception as err:
                    st.error(f"Failed to get recommendation: {err}")
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please make sure your models are trained and available in the trained-models directory.")

# --- MAIN APPLICATION ---
if st.session_state.get("authentication_status"):
    # --- LOGGED-IN VIEW ---
    # st.title(f'Welcome Back {st.session_state["name"]}!')
    authenticator.logout('Logout', 'sidebar')
    
    st.markdown('<div class="main-title">🚀 Future Trail | Career Navigator</div>', unsafe_allow_html=True)
    
    # Check if user is accessing analysis without uploading resume
    active_analysis = st.session_state.get("active_analysis")
    if active_analysis:
        run_analysis_hub()
    else:
        # Main navigation
        page = st.sidebar.radio("Navigate", [
            "📄 Resume Manager",
            "🎯 Career Predictor"
        ])
        
        if page == "📄 Resume Manager":
            run_resume_manager()
        elif page == "🎯 Career Predictor":
            run_career_predictor()

else:
    # --- LOGIN/REGISTER VIEW (FINAL ROBUST VERSION) ---
    st.title("Welcome to Future Trail 🚀")
    st.markdown("Please log in or register to continue.")
    
    try:
        login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

        with login_tab:
            # Fix the login method call - parameters should be in correct order
            try:
                name, authentication_status, username = authenticator.login('Login Form', 'main')
                
                # This logic runs ONLY after the user has submitted the form
                if authentication_status:
                    st.session_state["name"] = name
                    st.session_state["username"] = username
                    st.session_state["authentication_status"] = authentication_status
                    st.rerun()
                
                elif authentication_status is False:
                    st.error('Username/password is incorrect')
                elif authentication_status is None:
                    st.warning('Please enter your username and password')
                    
            except Exception as e:
                st.error(f"Authentication error: {str(e)}")
                st.info("If you're having trouble with authentication, you can continue without login for testing purposes.")
                
                # Emergency bypass for testing
                if st.button("Continue Without Login (Demo Mode)"):
                    st.session_state["authentication_status"] = True
                    st.session_state["name"] = "Demo User"
                    st.session_state["username"] = "demo"
                    st.rerun()

        with register_tab:
            if st.session_state.registration_success:
                st.balloons()
                st.success('Registration successful! Please go to the "Login" tab to sign in.')
                st.session_state.registration_success = False
                
            st.subheader("📝 Register New User")
            
            with st.form("register_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Full Name*", placeholder="Enter your full name")
                    username = st.text_input("Username*", placeholder="Choose a username")
                    email = st.text_input("Email*", placeholder="Enter your email address")
                
                with col2:
                    password = st.text_input("Password*", type="password", placeholder="Enter password")
                    confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Re-enter password")
                
                submitted = st.form_submit_button("Register User", use_container_width=True)
                
                if submitted:
                    if not all([name, username, email, password, confirm_password]):
                        st.error("⚠️ Please fill in all required fields")
                    elif len(password) < 6:
                        st.error("⚠️ Password must be at least 6 characters long")
                    elif password != confirm_password:
                        st.error("⚠️ Passwords do not match")
                    elif '@' not in email or '.' not in email.split('@')[-1]:
                        st.error("⚠️ Please enter a valid email address")
                    elif username in config['credentials']['usernames']:
                        st.error("⚠️ Username already exists. Please choose a different username.")
                    else:
                        try:
                            hasher = stauth.Hasher()
                            hashed_password = hasher.hash(password)

                            if save_new_user(username, name, hashed_password, email):
                                config['credentials']['usernames'][username] = {
                                    'name': name,
                                    'password': hashed_password,
                                    'email': email
                                }
                                st.session_state.registration_success = True
                                st.rerun()
                            else:
                                st.error("⚠️ Registration failed. Please try again.")

                        except Exception as e:
                            st.error(f"⚠️ Registration error: {str(e)}")

    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.info("If you're having trouble with authentication, you can continue without login for testing purposes.")
        
        # Emergency bypass for testing
        if st.button("Continue Without Login (Demo Mode)"):
            st.session_state["authentication_status"] = True
            st.session_state["name"] = "Demo User"
            st.session_state["username"] = "demo"
            st.rerun()