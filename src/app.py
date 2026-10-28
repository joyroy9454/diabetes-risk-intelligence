import streamlit as st
import pandas as pd
import joblib
import time
import os

# 1. Page Configuration
st.set_page_config(
    page_title="CardioMetabolic & Diabetes Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling for Portfolio-grade UI/UX & Animations
st.markdown("""
<style>
    /* Main Background & Font */
    .main {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    .hero-card h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .hero-card p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Section Containers */
    .stCard {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    /* Custom Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model Pipeline with Caching
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../model/diabetes_model.pkl')
    return joblib.load(model_path)
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ **Model Loading Failed:** Make sure `diabetes_pipeline_model.pkl` is in the same directory.")
    st.stop()

# 4. Hero Header Section
st.markdown("""
<div class="hero-card">
    <h1>🩺 CardioMetabolic Risk AI</h1>
    <p>Advanced machine learning pipeline predicting diabetes risk based on clinical biomarkers, physiological vitals, and lifestyle factors.</p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Navigation & Quick Guide
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=70)
    st.markdown("### **Portfolio Project**")
    st.info("This application uses an optimized scikit-learn pipeline combined with Gradient Boosting to compute predictive clinical risk scores.")
    st.markdown("---")
    st.markdown("#### **Navigation Guide**")
    st.markdown("1. Fill in patient demographics.")
    st.markdown("2. Input clinical vital signs.")
    st.markdown("3. Provide lab lipid panels & history.")
    st.markdown("4. Run instant AI assessment.")
    st.markdown("---")
    st.caption("Developed for Data Science & ML Portfolio.")

# 6. Multi-Column Input Layout
st.markdown("### 📋 Patient Assessment Form")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container():
        st.markdown("#### 👤 Demographics & Lifestyle")
        age = st.number_input("Age (Years)", min_value=1.0, max_value=120.0, value=45.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Hispanic", "Asian", "Other"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
        income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired", "Student"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

with col2:
    with st.container():
        st.markdown("#### ⚡ Vitals & Physical Metrics")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
        waist_to_hip_ratio = st.number_input("Waist-to-Hip Ratio", min_value=0.5, max_value=1.5, value=0.85)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=200.0, value=120.0)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50.0, max_value=130.0, value=80.0)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=180.0, value=72.0)
        alcohol_consumption_per_week = st.number_input("Alcohol (drinks/week)", min_value=0.0, max_value=50.0, value=2.0)
        physical_activity_minutes_per_week = st.number_input("Physical Activity (min/week)", min_value=0.0, max_value=1000.0, value=150.0)

with col3:
    with st.container():
        st.markdown("#### 🧬 Lab Results & History")
        diet_score = st.slider("Diet Score (1-10)", min_value=1.0, max_value=10.0, value=6.0)
        sleep_hours_per_day = st.number_input("Sleep Hours / Day", min_value=1.0, max_value=16.0, value=7.0)
        screen_time_hours_per_day = st.number_input("Screen Time (hours/day)", min_value=0.0, max_value=24.0, value=4.0)
        cholesterol_total = st.number_input("Total Cholesterol", min_value=100.0, max_value=400.0, value=200.0)
        hdl_cholesterol = st.number_input("HDL Cholesterol", min_value=20.0, max_value=120.0, value=50.0)
        ldl_cholesterol = st.number_input("LDL Cholesterol", min_value=50.0, max_value=250.0, value=120.0)
        triglycerides = st.number_input("Triglycerides", min_value=50.0, max_value=500.0, value=150.0)
        
        family_history_diabetes = st.selectbox("Family History of Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hypertension_history = st.selectbox("Hypertension History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cardiovascular_history = st.selectbox("Cardiovascular History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# 7. Construct Input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'alcohol_consumption_per_week': [alcohol_consumption_per_week],
    'physical_activity_minutes_per_week': [physical_activity_minutes_per_week],
    'diet_score': [diet_score],
    'sleep_hours_per_day': [sleep_hours_per_day],
    'screen_time_hours_per_day': [screen_time_hours_per_day],
    'bmi': [bmi],
    'waist_to_hip_ratio': [waist_to_hip_ratio],
    'systolic_bp': [systolic_bp],
    'diastolic_bp': [diastolic_bp],
    'heart_rate': [heart_rate],
    'cholesterol_total': [cholesterol_total],
    'hdl_cholesterol': [hdl_cholesterol],
    'ldl_cholesterol': [ldl_cholesterol],
    'triglycerides': [triglycerides],
    'gender': [gender],
    'ethnicity': [ethnicity],
    'education_level': [education_level],
    'income_level': [income_level],
    'smoking_status': [smoking_status],
    'employment_status': [employment_status],
    'family_history_diabetes': [family_history_diabetes],
    'hypertension_history': [hypertension_history],
    'cardiovascular_history': [cardiovascular_history]
})

st.markdown("---")

# 8. Prediction Action with Animated Loading Spinner
if st.button("🚀 Run AI Risk Assessment", use_container_width=True):
    # Simulated smooth loading effect for portfolio application feel
    with st.spinner("Analyzing clinical markers and running inference..."):
        time.sleep(1.0)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    st.markdown("### 📊 Assessment Results")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk of Diabetes Detected**\n\nThe patient's clinical profile exhibits biomarkers strongly correlated with insulin resistance and diabetes risk. Clinical follow-up is recommended.")
        else:
            st.success("✅ **Low Risk of Diabetes**\n\nThe patient's clinical markers and metabolic metrics currently fall within normal healthy thresholds.")
            
    with res_col2:
        if probability is not None:
            risk_score = probability * 100
            st.metric(label="Calculated Risk Probability", value=f"{risk_score:.1f}%", delta="Elevated" if risk_score > 50 else "Normal", delta_color="inverse" if risk_score > 50 else "normal")
            st.progress(float(probability))

    # Detailed Breakdown expander
    with st.expander("🔍 View Raw Model Payload Sent to Pipeline"):
        st.json(input_data.to_dict(orient="records")[0])