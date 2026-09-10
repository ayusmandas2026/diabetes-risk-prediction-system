from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Robust Base Directory Resolution
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="DiaGuard AI — Diabetes Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Advanced Custom Styling (CSS & Glassmorphism)
# --------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Radiant Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: 
            radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 45%),
            radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
            linear-gradient(145deg, #070a12 0%, #0d1322 45%, #070a12 100%) !important;
        color: #f1f5f9;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13, 19, 34, 0.95) 0%, rgba(7, 10, 18, 0.98) 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px);
    }

    /* Main Form Card */
    [data-testid="stForm"] {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 2.25rem !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6), 0 0 30px rgba(99, 102, 241, 0.08) !important;
        backdrop-filter: blur(20px) !important;
    }

    /* Input Fields */
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background: rgba(30, 41, 59, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        color: #f8fafc !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        background: rgba(30, 41, 59, 0.9) !important;
    }

    input {
        color: #f8fafc !important;
        font-weight: 500 !important;
    }

    /* Primary Action Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.85rem 2rem !important;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4), 0 0 20px rgba(6, 182, 212, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.008) !important;
        box-shadow: 0 15px 35px -5px rgba(79, 70, 229, 0.6), 0 0 30px rgba(6, 182, 212, 0.4) !important;
    }

    /* Secondary Preset Buttons */
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        transition: all 0.25s ease !important;
    }

    button[kind="secondary"]:hover {
        background: rgba(51, 65, 85, 0.8) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 1.5rem 1rem 2rem 1rem;
        margin-bottom: 0.5rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 1.1rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 20%, #93c5fd 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 700px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.5;
    }

    .disclaimer-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        font-size: 0.82rem;
        color: #fcd34d;
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    /* Assessment Output Cards */
    .assessment-card {
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1.75rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        animation: fadeIn 0.4s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .assessment-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.25);
    }

    .assessment-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%);
        border-color: rgba(245, 158, 11, 0.4);
        box-shadow: 0 15px 35px -10px rgba(245, 158, 11, 0.25);
    }

    .assessment-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.06) 100%);
        border-color: rgba(239, 68, 68, 0.45);
        box-shadow: 0 15px 35px -10px rgba(239, 68, 68, 0.3);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .chip-warning {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #fca5a5;
        font-size: 0.88rem;
        font-weight: 500;
        margin: 0.3rem 0.3rem 0.3rem 0;
    }

    .chip-info {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #fde047;
        font-size: 0.88rem;
        font-weight: 500;
        margin: 0.3rem 0.3rem 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Trained Model & Preprocessing Artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "final_mlp_model.keras", compile=False)
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
    median_imp = joblib.load(BASE_DIR / "median_imputer.joblib")
    knn_imp = joblib.load(BASE_DIR / "knn_imputer.joblib")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.joblib")
    return model, scaler, median_imp, knn_imp, feature_columns

# --------------------------------------------------
# Feature Engineering (Exact Match with Pipeline)
# --------------------------------------------------
def engineer_features(data):
    data = data.copy()

    data["Age_Group"] = pd.cut(
        data["Age"],
        bins=[20, 30, 40, 50, 100],
        labels=["21-30", "31-40", "41-50", "51+"]
    )

    data["BMI_Category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

    data["Glucose_Category"] = pd.cut(
        data["Glucose"],
        bins=[0, 99, 125, 300],
        labels=["Normal", "Prediabetic", "Diabetic"]
    )

    data["Glucose_BMI_Interaction"] = data["Glucose"] * data["BMI"]
    data["Age_Pregnancies_Interaction"] = data["Age"] * data["Pregnancies"]
    data["Log_Insulin"] = np.log1p(data["Insulin"])
    data["Log_DiabetesPedigreeFunction"] = np.log1p(data["DiabetesPedigreeFunction"])

    data = pd.get_dummies(
        data,
        columns=["Age_Group", "BMI_Category", "Glucose_Category"],
        drop_first=True
    )
    return data

model, scaler, median_imp, knn_imp, feature_columns = load_artifacts()

# --------------------------------------------------
# Clinical Benchmark Presets
# --------------------------------------------------
PRESET_PROFILES = {
    "🟢 Healthy / Low Risk": {
        "pregnancies": 1,
        "glucose": 88.0,
        "blood_pressure": 68.0,
        "skin_thickness": 20.0,
        "insulin": 75.0,
        "bmi": 22.4,
        "dpf": 0.25,
        "age": 25
    },
    "🟡 Borderline / Pre-diabetic": {
        "pregnancies": 2,
        "glucose": 118.0,
        "blood_pressure": 76.0,
        "skin_thickness": 28.0,
        "insulin": 120.0,
        "bmi": 28.5,
        "dpf": 0.45,
        "age": 42
    },
    "🔴 High Risk Diabetic": {
        "pregnancies": 6,
        "glucose": 178.0,
        "blood_pressure": 86.0,
        "skin_thickness": 36.0,
        "insulin": 220.0,
        "bmi": 36.5,
        "dpf": 0.85,
        "age": 52
    }
}

# Session state initialization
default_profile = PRESET_PROFILES["🟡 Borderline / Pre-diabetic"]
for key, val in default_profile.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Clinical Presets")
    st.caption("One-click benchmark test profiles:")

    for p_name, p_data in PRESET_PROFILES.items():
        if st.button(p_name, use_container_width=True):
            for k, v in p_data.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Diagnostic Sensitivity")

    threshold = st.slider(
        "Decision Cutoff",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Adjust the classification boundary. Lower thresholds (e.g. 0.35-0.40) maximize sensitivity/recall for screening."
    )

    if threshold < 0.50:
        st.markdown("<div style='color: #38bdf8; font-size: 0.85rem;'>🩺 <b>High Sensitivity Mode</b>: Prioritizes catching potential cases early.</div>", unsafe_allow_html=True)
    elif threshold > 0.50:
        st.markdown("<div style='color: #a78bfa; font-size: 0.85rem;'>🎯 <b>High Specificity Mode</b>: Minimizes false alarms.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #94a3b8; font-size: 0.85rem;'>⚖️ <b>Standard Balanced Mode</b> (0.50 cutoff).</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Medical Reference Ranges", expanded=False):
        st.markdown("""
        **Fasting Glucose:**
        - Normal: `< 100 mg/dL`
        - Prediabetes: `100 - 125 mg/dL`
        - Diabetes: `≥ 126 mg/dL`

        **Body Mass Index (BMI):**
        - Normal: `18.5 - 24.9`
        - Overweight: `25.0 - 29.9`
        - Obese: `≥ 30.0`

        **Blood Pressure (Diastolic):**
        - Normal: `< 80 mm Hg`
        - Stage 1: `80 - 89 mm Hg`
        - Stage 2: `≥ 90 mm Hg`
        """)

# --------------------------------------------------
# Hero Header Banner
# --------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ Clinical Neural Intelligence</div>
    <div class="hero-title">Diabetes Risk Assessment</div>
    <div class="hero-subtitle">
        Powered by an Optuna-tuned Multilayer Perceptron (MLP) trained on the Pima Clinical Diabetes dataset.
    </div>
    <div class="disclaimer-pill">
        ⚠️ Academic & Demonstration Prototype • Not a Certified Diagnostic Device
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Patient Input Form
# --------------------------------------------------
with st.form("clinical_form"):
    st.markdown("#### 🩺 Patient Biomarkers")
    st.caption("Enter patient measurements below or select a preset from the sidebar.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        pregnancies = st.number_input(
            "Pregnancies (count)",
            min_value=0,
            max_value=20,
            step=1,
            key="pregnancies",
            help="Total number of pregnancies"
        )

        glucose = st.number_input(
            "Fasting Glucose (mg/dL)",
            min_value=1.0,
            max_value=300.0,
            key="glucose",
            help="Plasma glucose concentration at 2 hours in an oral glucose tolerance test"
        )

        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=1.0,
            max_value=200.0,
            key="blood_pressure",
            help="Diastolic blood pressure"
        )

        skin_thickness = st.number_input(
            "Triceps Skinfold Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            key="skin_thickness",
            help="Triceps skinfold thickness"
        )

    with col2:
        insulin = st.number_input(
            "2-Hour Serum Insulin (μU/mL)",
            min_value=0.0,
            max_value=900.0,
            key="insulin",
            help="2-Hour serum insulin"
        )

        bmi = st.number_input(
            "Body Mass Index — BMI (kg/m²)",
            min_value=1.0,
            max_value=70.0,
            key="bmi",
            help="Body mass index (weight in kg / height in m²)"
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function (family history)",
            min_value=0.01,
            max_value=3.0,
            key="dpf",
            help="Genetic risk factor score based on family history"
        )

        age = st.number_input(
            "Age (years)",
            min_value=21,
            max_value=100,
            step=1,
            key="age",
            help="Patient age (minimum 21 in Pima dataset)"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ Compute Clinical Risk Assessment", type="primary", use_container_width=True)

# --------------------------------------------------
# Inference & Visual Presentation
# --------------------------------------------------
if submitted:
    raw = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    # Preprocessing
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    raw[zero_invalid] = raw[zero_invalid].replace(0, np.nan)
    raw[["Glucose", "BloodPressure", "BMI"]] = median_imp.transform(raw[["Glucose", "BloodPressure", "BMI"]])
    raw[["SkinThickness", "Insulin"]] = knn_imp.transform(raw[["SkinThickness", "Insulin"]])

    processed = engineer_features(raw).reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(processed)

    probability = float(model.predict(scaled, verbose=0).ravel()[0])
    prediction = int(probability >= threshold)

    # Risk Tier Logic
    if probability < 0.35:
        tier_class = "assessment-low"
        tier_icon = "🟢"
        tier_title = "Low Clinical Risk"
        tier_desc = "Biomarkers indicate low probability of diabetic pathology under current thresholds."
        tier_color = "#34d399"
    elif probability < 0.60:
        tier_class = "assessment-moderate"
        tier_icon = "🟡"
        tier_title = "Moderate / Pre-Diabetic Risk"
        tier_desc = "Biomarkers indicate borderline risk factors. Monitoring lifestyle and dietary measures is recommended."
        tier_color = "#fbbf24"
    else:
        tier_class = "assessment-high"
        tier_icon = "🔴"
        tier_title = "High Clinical Risk"
        tier_desc = "Biomarkers suggest elevated probability of diabetes. Formal medical consultation is strongly advised."
        tier_color = "#f87171"

    # Assessment Card HTML
    st.markdown(f"""
    <div class="assessment-card {tier_class}">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.8rem;">{tier_icon}</span>
                <span style="font-size: 1.5rem; font-weight: 800; color: {tier_color}; letter-spacing: -0.01em;">
                    {tier_title}
                </span>
            </div>
            <span style="font-size: 0.85rem; padding: 0.3rem 0.8rem; border-radius: 9999px; background: rgba(255,255,255,0.08); color: #cbd5e1;">
                Cutoff: {threshold * 100:.0f}%
            </span>
        </div>
        <p style="color: #cbd5e1; margin: 0; font-size: 1rem; line-height: 1.5;">{tier_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # Visual Risk Progress Bar
    clamped_prob = min(max(probability, 0.0), 1.0)
    st.progress(clamped_prob, text=f"Estimated Diabetes Risk Probability: {clamped_prob * 100:.1f}%")

    # 3-Metric Summary Grid
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Risk</div>
            <div class="metric-number" style="color: {tier_color};">{probability * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Screening Threshold</div>
            <div class="metric-number" style="color: #94a3b8;">{threshold * 100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        status_label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        status_color = "#f87171" if prediction == 1 else "#34d399"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Diagnostic Call</div>
            <div class="metric-number" style="color: {status_color};">{status_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Clinical Biomarker Alert Chips
    alerts = []
    if glucose >= 126.0:
        alerts.append(("warning", f"🚨 Fasting Glucose ({glucose:.0f} mg/dL) ≥ 126 mg/dL — Diabetic Range"))
    elif glucose >= 100.0:
        alerts.append(("info", f"⚠️ Fasting Glucose ({glucose:.0f} mg/dL) 100–125 mg/dL — Prediabetic Range"))

    if bmi >= 30.0:
        alerts.append(("warning", f"🚨 BMI ({bmi:.1f} kg/m²) ≥ 30.0 — Clinical Obesity"))
    elif bmi >= 25.0:
        alerts.append(("info", f"⚠️ BMI ({bmi:.1f} kg/m²) 25.0–29.9 — Overweight Range"))

    if blood_pressure >= 90.0:
        alerts.append(("warning", f"🚨 Diastolic BP ({blood_pressure:.0f} mm Hg) ≥ 90 mm Hg — Stage 2 Hypertension"))
    elif blood_pressure >= 80.0:
        alerts.append(("info", f"⚠️ Diastolic BP ({blood_pressure:.0f} mm Hg) 80–89 mm Hg — Stage 1 Hypertension"))

    if alerts:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Biomarker Highlight Signals")
        chip_html = ""
        for level, msg in alerts:
            css_class = "chip-warning" if level == "warning" else "chip-info"
            chip_html += f'<div class="{css_class}">{msg}</div>'
        st.markdown(chip_html, unsafe_allow_html=True)
