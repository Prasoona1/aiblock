import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# ==================== CONFIGURATION ====================
# Auto-create necessary folders
FOLDERS = ['blueprints', 'test_certs', 'database', 'resumes']
for folder in FOLDERS:
    Path(folder).mkdir(exist_ok=True)

DB_PATH = 'database/verification_history.db'

# ==================== DATABASE SETUP ====================
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for organization blueprints
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blueprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_name TEXT NOT NULL,
            template_name TEXT NOT NULL,
            template_hash TEXT NOT NULL,
            upload_timestamp TEXT NOT NULL,
            file_path TEXT NOT NULL
        )
    ''')
    
    # Table for verification history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cert_name TEXT NOT NULL,
            org_name TEXT,
            match_score REAL,
            result TEXT NOT NULL,
            verification_mode TEXT NOT NULL,
            cert_hash TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            blockchain_entry TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# ==================== BLOCKCHAIN SIMULATION ====================
def create_blockchain_hash(data):
    """Simulate blockchain entry with SHA-256 hashing"""
    timestamp = datetime.now().isoformat()
    block_data = f"{data}|{timestamp}"
    block_hash = hashlib.sha256(block_data.encode()).hexdigest()
    
    blockchain_entry = {
        'hash': block_hash,
        'timestamp': timestamp,
        'data': data
    }
    return blockchain_entry

def hash_image(image_bytes):
    """Generate SHA-256 hash for an image"""
    return hashlib.sha256(image_bytes).hexdigest()

# ==================== IMAGE SIMILARITY (AI-BASED VERIFICATION) ====================
def calculate_ssim(img1_bytes, img2_bytes):
    """Calculate Structural Similarity Index (SSIM) between two images"""
    try:
        # Load images
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))
        
        # Resize to same dimensions
        size = (800, 600)
        img1 = img1.resize(size)
        img2 = img2.resize(size)
        
        # Convert to grayscale numpy arrays
        img1_gray = np.array(img1.convert('L'))
        img2_gray = np.array(img2.convert('L'))
        
        # Calculate SSIM using OpenCV
        # Normalize to 0-1 range
        score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Alternative: Use simple correlation
        correlation = np.corrcoef(img1_gray.flatten(), img2_gray.flatten())[0, 1]
        
        # Return average of both methods as percentage
        final_score = ((score + correlation) / 2) * 100
        return max(0, min(100, final_score))  # Clamp between 0-100
        
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

def compare_images_advanced(img1_bytes, img2_bytes):
    """Advanced image comparison using multiple techniques"""
    try:
        # Method 1: Histogram comparison
        img1 = Image.open(io.BytesIO(img1_bytes)).convert('RGB')
        img2 = Image.open(io.BytesIO(img2_bytes)).convert('RGB')
        
        img1 = img1.resize((300, 300))
        img2 = img2.resize((300, 300))
        
        # Convert to numpy
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate histogram correlation for each channel
        hist_scores = []
        for i in range(3):  # RGB channels
            hist1 = cv2.calcHist([arr1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([arr2], [i], None, [256], [0, 256])
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            hist_scores.append(score)
        
        hist_similarity = np.mean(hist_scores) * 100
        
        # Method 2: Pixel-wise comparison
        mse = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
        pixel_similarity = max(0, 100 - (mse / 100))
        
        # Combined score
        final_score = (hist_similarity * 0.6) + (pixel_similarity * 0.4)
        return final_score
        
    except Exception as e:
        st.error(f"Error in advanced comparison: {e}")
        return 0.0

# ==================== ORGANIZATION VIEW FUNCTIONS ====================
def save_blueprint(org_name, template_file):
    """Save organization blueprint to database and file system"""
    try:
        # Read file bytes
        file_bytes = template_file.read()
        template_hash = hash_image(file_bytes)
        
        # Save file
        file_path = f"blueprints/{org_name}_{template_file.name}"
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO blueprints (org_name, template_name, template_hash, upload_timestamp, file_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (org_name, template_file.name, template_hash, datetime.now().isoformat(), file_path))
        
        conn.commit()
        conn.close()
        
        return True, template_hash
    except Exception as e:
        return False, str(e)

def get_all_blueprints():
    """Fetch all blueprints from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM blueprints ORDER BY upload_timestamp DESC')
    blueprints = cursor.fetchall()
    
    conn.close()
    return blueprints

# ==================== USER VERIFICATION FUNCTIONS ====================
def verify_algorithmic(cert_bytes, cert_name):
    """AI-Based verification using image similarity"""
    blueprints = get_all_blueprints()
    
    if not blueprints:
        return None, 0.0, "No blueprints available"
    
    best_match_score = 0.0
    best_match_org = "Unknown"
    
    # Compare with all blueprints
    for blueprint in blueprints:
        org_name = blueprint[1]
        file_path = blueprint[5]
        
        try:
            with open(file_path, 'rb') as f:
                blueprint_bytes = f.read()
            
            # Calculate similarity
            similarity = compare_images_advanced(cert_bytes, blueprint_bytes)
            
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_org = org_name
                
        except Exception as e:
            continue
    
    # Determine result
    if best_match_score >= 75:
        result = "✅ Verified"
    elif best_match_score >= 50:
        result = "⚠️ Suspicious"
    else:
        result = "❌ Possible Forgery"
    
    return best_match_org, best_match_score, result

def verify_organization(cert_bytes, selected_org):
    """Organization-based verification using hash matching"""
    cert_hash = hash_image(cert_bytes)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT template_hash, org_name FROM blueprints WHERE org_name = ?', (selected_org,))
    org_blueprints = cursor.fetchall()
    conn.close()
    
    if not org_blueprints:
        return selected_org, 0.0, "❌ Organization not found"
    
    # Check hash match
    for template_hash, org_name in org_blueprints:
        if cert_hash == template_hash:
            return org_name, 100.0, "✅ Verified (Exact Match)"
    
    # If no exact match, use image similarity
    best_score = 0.0
    for blueprint in get_all_blueprints():
        if blueprint[1] == selected_org:
            try:
                with open(blueprint[5], 'rb') as f:
                    blueprint_bytes = f.read()
                score = compare_images_advanced(cert_bytes, blueprint_bytes)
                best_score = max(best_score, score)
            except:
                continue
    
    if best_score >= 75:
        result = "✅ Verified"
    elif best_score >= 50:
        result = "⚠️ Suspicious"
    else:
        result = "❌ Possible Forgery"
    
    return selected_org, best_score, result

def save_verification(cert_name, org_name, match_score, result, verification_mode, cert_hash, blockchain_entry):
    """Save verification result to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO verification_history 
        (cert_name, org_name, match_score, result, verification_mode, cert_hash, timestamp, blockchain_entry)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (cert_name, org_name, match_score, result, verification_mode, cert_hash, 
          datetime.now().isoformat(), json.dumps(blockchain_entry)))
    
    conn.commit()
    conn.close()

# ==================== RESUME ANALYZER ====================
def analyze_resume(resume_text, job_description):
    """Analyze resume fit with job description using NLP"""
    try:
        # Clean texts
        resume_clean = resume_text.lower().strip()
        job_clean = job_description.lower().strip()
        
        # Extract keywords
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Fit and transform
        vectors = vectorizer.fit_transform([resume_clean, job_clean])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        fit_score = similarity * 100
        
        # Get top keywords from job description
        feature_names = vectorizer.get_feature_names_out()
        job_vector = vectors[1].toarray()[0]
        top_indices = job_vector.argsort()[-10:][::-1]
        job_keywords = [feature_names[i] for i in top_indices if job_vector[i] > 0]
        
        # Check which keywords are missing in resume
        resume_vector = vectors[0].toarray()[0]
        missing_keywords = [kw for i, kw in enumerate(feature_names) 
                          if job_vector[i] > 0.1 and resume_vector[i] == 0][:5]
        
        # Generate suggestion
        if fit_score >= 75:
            suggestion = "🎉 Excellent fit! You're highly qualified for this position."
        elif fit_score >= 50:
            suggestion = f"👍 Good fit! Consider highlighting: {', '.join(missing_keywords[:3])}"
        else:
            suggestion = f"⚠️ Skills gap detected. Focus on: {', '.join(missing_keywords[:5])}"
        
        return fit_score, suggestion, job_keywords, missing_keywords
        
    except Exception as e:
        return 0.0, f"Error analyzing: {str(e)}", [], []

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="AI & Blockchain Certificate Verifier",
        page_icon="🔐",
        layout="wide"
    )
    
    # Initialize database
    init_database()
    
    # Sidebar navigation
    st.sidebar.title("🔐 Certificate Verifier")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏢 Organization View", "👤 User View", "🧠 Resume Analyzer", "📊 Verification History"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("**AI + Blockchain-Powered**\nSkill & Certificate Verification System")
    
    # ==================== ORGANIZATION VIEW ====================
    if page == "🏢 Organization View":
        st.title("🏢 Organization View")
        st.markdown("Upload genuine certificate blueprints for verification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload Certificate Blueprint")
            
            org_name = st.text_input("Organization Name", placeholder="e.g., Google, Microsoft")
            template_file = st.file_uploader("Upload Certificate Template", type=['png', 'jpg', 'jpeg', 'pdf'])
            
            if st.button("💾 Save Blueprint", type="primary"):
                if org_name and template_file:
                    success, result = save_blueprint(org_name, template_file)
                    if success:
                        st.success(f"✅ Blueprint saved successfully!")
                        st.info(f"**Blockchain Hash:** `{result}`")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {result}")
                else:
                    st.warning("⚠️ Please provide organization name and template")
        
        with col2:
            st.subheader("📊 Statistics")
            blueprints = get_all_blueprints()
            st.metric("Total Blueprints", len(blueprints))
            st.metric("Organizations", len(set([b[1] for b in blueprints])))
        
        # Display all blueprints
        st.markdown("---")
        st.subheader("📋 All Registered Blueprints")
        
        if blueprints:
            import pandas as pd
            df = pd.DataFrame(blueprints, columns=['ID', 'Organization', 'Template Name', 'Hash', 'Timestamp', 'File Path'])
            df['Hash'] = df['Hash'].apply(lambda x: f"{x[:16]}...")
            st.dataframe(df[['Organization', 'Template Name', 'Hash', 'Timestamp']], use_container_width=True)
        else:
            st.info("No blueprints uploaded yet")
    
    # ==================== USER VIEW ====================
    elif page == "👤 User View":
        st.title("👤 User View")
        st.markdown("Verify your certificates using AI and Blockchain technology")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📤 Upload Certificate")
            cert_file = st.file_uploader("Upload Certificate to Verify", type=['png', 'jpg', 'jpeg'])
            
            if cert_file:
                st.image(cert_file, caption="Uploaded Certificate", use_container_width=True)
        
        with col2:
            st.subheader("⚙️ Verification Settings")
            
            verification_type = st.radio(
                "Select Verification Type",
                ["🤖 Algorithmic Verification (AI-Based)", "🏢 Organization Verification"]
            )
            
            selected_org = None
            if "Organization" in verification_type:
                orgs = list(set([b[1] for b in get_all_blueprints()]))
                if orgs:
                    selected_org = st.selectbox("Select Organization", orgs)
                else:
                    st.warning("No organizations registered yet")
            
            verify_btn = st.button("🔍 Verify Certificate", type="primary", use_container_width=True)
        
        # Verification Process
        if verify_btn and cert_file:
            with st.spinner("🔄 Verifying certificate..."):
                cert_bytes = cert_file.read()
                cert_hash = hash_image(cert_bytes)
                
                # Perform verification
                if "Algorithmic" in verification_type:
                    org_name, match_score, result = verify_algorithmic(cert_bytes, cert_file.name)
                    verification_mode = "AI-Based"
                else:
                    if selected_org:
                        org_name, match_score, result = verify_organization(cert_bytes, selected_org)
                        verification_mode = "Organization"
                    else:
                        st.error("Please select an organization")
                        st.stop()
                
                # Create blockchain entry
                blockchain_entry = create_blockchain_hash(f"{cert_file.name}|{org_name}|{match_score}")
                
                # Save verification
                save_verification(
                    cert_file.name, org_name, match_score, result,
                    verification_mode, cert_hash, blockchain_entry
                )
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Verification Results")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                
                with col_r1:
                    st.metric("Match Score", f"{match_score:.2f}%")
                
                with col_r2:
                    st.metric("Organization", org_name)
                
                with col_r3:
                    if "Verified" in result:
                        st.success(result)
                    elif "Suspicious" in result:
                        st.warning(result)
                    else:
                        st.error(result)
                
                # Blockchain info
                st.markdown("---")
                st.subheader("⛓️ Blockchain Entry")
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.code(f"Hash: {blockchain_entry['hash']}", language="text")
                with col_b2:
                    st.code(f"Timestamp: {blockchain_entry['timestamp']}", language="text")
                
                # Show notification
                if "Verified" in result:
                    st.toast("✅ Certificate verified successfully!", icon="✅")
                else:
                    st.toast("⚠️ Certificate verification failed!", icon="⚠️")
    
    # ==================== RESUME ANALYZER ====================
    elif page == "🧠 Resume Analyzer":
        st.title("🧠 AI-Powered Resume Analyzer")
        st.markdown("Analyze how well your resume matches a job description")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Resume")
            resume_file = st.file_uploader("Upload Resume (TXT/PDF)", type=['txt', 'pdf'])
            resume_text = st.text_area("Or paste resume text", height=300)
            
            if resume_file:
                if resume_file.type == "text/plain":
                    resume_text = resume_file.read().decode('utf-8')
                    st.success("Resume loaded!")
        
        with col2:
            st.subheader("💼 Job Description")
            job_desc = st.text_area("Paste job description", height=300, 
                                    placeholder="Paste the job requirements, skills, and qualifications...")
        
        if st.button("🎯 Analyze Fit", type="primary"):
            if resume_text and job_desc:
                with st.spinner("🔄 Analyzing..."):
                    fit_score, suggestion, job_keywords, missing_keywords = analyze_resume(resume_text, job_desc)
                    
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Score display
                    col_s1, col_s2 = st.columns([1, 2])
                    
                    with col_s1:
                        if fit_score >= 75:
                            st.success(f"### {fit_score:.1f}%")
                            st.success("Excellent Fit!")
                        elif fit_score >= 50:
                            st.warning(f"### {fit_score:.1f}%")
                            st.warning("Good Fit")
                        else:
                            st.error(f"### {fit_score:.1f}%")
                            st.error("Needs Improvement")
                    
                    with col_s2:
                        st.info(suggestion)
                    
                    # Keywords analysis
                    col_k1, col_k2 = st.columns(2)
                    
                    with col_k1:
                        st.markdown("**🎯 Key Job Requirements:**")
                        for kw in job_keywords[:5]:
                            st.markdown(f"- {kw}")
                    
                    with col_k2:
                        st.markdown("**⚠️ Missing Keywords:**")
                        if missing_keywords:
                            for kw in missing_keywords[:5]:
                                st.markdown(f"- {kw}")
                        else:
                            st.markdown("*None - Great match!*")
                    
                    st.balloons()
            else:
                st.warning("⚠️ Please provide both resume and job description")
    
    # ==================== VERIFICATION HISTORY ====================
    elif page == "📊 Verification History":
        st.title("📊 Verification History Dashboard")
        st.markdown("View all past certificate verifications")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM verification_history ORDER BY timestamp DESC')
        history = cursor.fetchall()
        conn.close()
        
        if history:
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(history, columns=[
                'ID', 'Certificate Name', 'Organization', 'Match Score', 
                'Result', 'Verification Mode', 'Cert Hash', 'Timestamp', 'Blockchain Entry'
            ])
            
            # Filters
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                org_filter = st.multiselect("Filter by Organization", 
                                           options=df['Organization'].unique())
            
            with col_f2:
                result_filter = st.multiselect("Filter by Result",
                                              options=df['Result'].unique())
            
            with col_f3:
                mode_filter = st.multiselect("Filter by Verification Mode",
                                            options=df['Verification Mode'].unique())
            
            # Apply filters
            filtered_df = df.copy()
            if org_filter:
                filtered_df = filtered_df[filtered_df['Organization'].isin(org_filter)]
            if result_filter:
                filtered_df = filtered_df[filtered_df['Result'].isin(result_filter)]
            if mode_filter:
                filtered_df = filtered_df[filtered_df['Verification Mode'].isin(mode_filter)]
            
            # Statistics
            st.markdown("---")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("Total Verifications", len(filtered_df))
            
            with col_s2:
                verified_count = len(filtered_df[filtered_df['Result'].str.contains('Verified')])
                st.metric("Verified", verified_count)
            
            with col_s3:
                avg_score = filtered_df['Match Score'].mean()
                st.metric("Avg Match Score", f"{avg_score:.1f}%")
            
            with col_s4:
                unique_orgs = filtered_df['Organization'].nunique()
                st.metric("Organizations", unique_orgs)
            
            # Display table
            st.markdown("---")
            display_df = filtered_df[['Certificate Name', 'Organization', 'Match Score', 
                                     'Result', 'Verification Mode', 'Timestamp']]
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name="verification_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No verification history yet. Start verifying certificates!")

if __name__ == "__main__":
    main()
