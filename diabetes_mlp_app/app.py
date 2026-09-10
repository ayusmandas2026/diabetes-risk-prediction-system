from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Base Directory Resolution
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Clinical Diabetes Risk Assessment Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Professional Healthcare UI Styling (Clean & Clinical)
# --------------------------------------------------
st.markdown("""
<style>
    /* Clean System Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0f172a;
    }

    /* Subtle, Elegant Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: 
            radial-gradient(at 0% 0%, rgba(224, 242, 254, 0.7) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(238, 242, 255, 0.7) 0px, transparent 50%),
            linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
        color: #0f172a;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* Main Clinical Card Form */
    [data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 2rem !important;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05) !important;
    }

    /* Clean Input Fields */
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15) !important;
    }

    input {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    /* Primary Submit Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.75rem !important;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2) !important;
        transition: background-color 0.15s ease, transform 0.1s ease !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary Preset Buttons */
    button[kind="secondary"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
    }

    button[kind="secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }

    /* Header Banner */
    .portal-header {
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    .portal-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .portal-subtitle {
        font-size: 0.98rem;
        color: #64748b;
        margin: 0;
        line-height: 1.5;
    }

    .notice-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }

    /* Clinical Outcome Cards */
    .outcome-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .outcome-low {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    .outcome-moderate {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
    }

    .outcome-high {
        background-color: #fff1f2;
        border: 1px solid #fecdd3;
        color: #9f1239;
    }

    /* Metric Summary Boxes */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }

    .stat-value {
        font-size: 1.85rem;
        font-weight: 700;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 500;
        border-radius: 6px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Model & Pipeline Artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "final_mlp_model.keras", compile=False)
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
    median_imp = joblib.load(BASE_DIR / "median_imputer.joblib")
    knn_imp = joblib.load(BASE_DIR / "knn_imputer.joblib")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.joblib")
    return model, scaler, median_imp, knn_imp, feature_columns

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
# Standardized Patient Profiles
# --------------------------------------------------
PRESET_PROFILES = {
    "Profile 1: Normal Findings": {
        "pregnancies": 1,
        "glucose": 88.0,
        "blood_pressure": 68.0,
        "skin_thickness": 20.0,
        "insulin": 75.0,
        "bmi": 22.4,
        "dpf": 0.25,
        "age": 25
    },
    "Profile 2: Borderline / Prediabetic": {
        "pregnancies": 2,
        "glucose": 118.0,
        "blood_pressure": 76.0,
        "skin_thickness": 28.0,
        "insulin": 120.0,
        "bmi": 28.5,
        "dpf": 0.45,
        "age": 42
    },
    "Profile 3: Elevated Risk / Diabetic": {
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

# State initialization
default_profile = PRESET_PROFILES["Profile 2: Borderline / Prediabetic"]
for key, val in default_profile.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------
# Sidebar (Controls & References)
# --------------------------------------------------
with st.sidebar:
    st.subheader("Patient Profiles (Test Cases)")
    st.caption("Load benchmark clinical data:")

    for p_name, p_data in PRESET_PROFILES.items():
        if st.button(p_name, use_container_width=True):
            for k, v in p_data.items():
                st.session_state[k] = v
            st.rerun()

    st.divider()

    st.subheader("Decision Threshold")
    threshold = st.slider(
        "Classification Cutoff",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Standard cutoff is 0.50. Lower thresholds (0.35 - 0.40) increase sensitivity for preliminary screening."
    )

    if threshold < 0.50:
        st.caption("🩺 **High Sensitivity Setting**: Maximizes early detection rate.")
    elif threshold > 0.50:
        st.caption("🎯 **High Specificity Setting**: Reduces false-positive indications.")
    else:
        st.caption("⚖️ **Standard Setting** (0.50 balanced threshold).")

    st.divider()

    with st.expander("Reference Standards (ADA / WHO)", expanded=True):
        st.markdown("""
        **Fasting Blood Glucose:**
        - Normal: `< 100 mg/dL`
        - Prediabetes: `100 - 125 mg/dL`
        - Diabetes: `≥ 126 mg/dL`

        **Body Mass Index (BMI):**
        - Normal weight: `18.5 - 24.9`
        - Overweight: `25.0 - 29.9`
        - Obese: `≥ 30.0`

        **Diastolic Blood Pressure:**
        - Normal: `< 80 mm Hg`
        - Pre-hypertension: `80 - 89 mm Hg`
        - Hypertension: `≥ 90 mm Hg`
        """)

# --------------------------------------------------
# Header & Medical Notice
# --------------------------------------------------
st.markdown("""
<div class="portal-header">
    <div class="portal-title">
        <span>🩺</span> Diabetes Risk Assessment Tool
    </div>
    <p class="portal-subtitle">
        Clinical risk stratification system using a Multilayer Perceptron (MLP) trained on the Pima Indians Diabetes benchmark dataset.
    </p>
</div>
<div class="notice-box">
    <strong>Educational & Demonstration Prototype:</strong> This tool is developed for academic evaluation. Predictions provide statistical risk estimates and are not a substitute for clinical laboratory evaluation.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Input Form
# --------------------------------------------------
with st.form("patient_form"):
    st.markdown("#### Patient Biomarkers")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            step=1,
            key="pregnancies",
            help="Total number of pregnancies recorded"
        )

        glucose = st.number_input(
            "Fasting Plasma Glucose (mg/dL)",
            min_value=1.0,
            max_value=300.0,
            key="glucose",
            help="Fasting plasma glucose level (mg/dL)"
        )

        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=1.0,
            max_value=200.0,
            key="blood_pressure",
            help="Diastolic blood pressure (mm Hg)"
        )

        skin_thickness = st.number_input(
            "Triceps Skinfold Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            key="skin_thickness",
            help="Triceps skinfold measurement (mm)"
        )

    with col2:
        insulin = st.number_input(
            "2-Hour Serum Insulin (μU/mL)",
            min_value=0.0,
            max_value=900.0,
            key="insulin",
            help="Serum insulin after 2 hours (μU/mL)"
        )

        bmi = st.number_input(
            "Body Mass Index (BMI, kg/m²)",
            min_value=1.0,
            max_value=70.0,
            key="bmi",
            help="Body Mass Index in kg/m²"
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function (DPF)",
            min_value=0.01,
            max_value=3.0,
            key="dpf",
            help="Calculated score reflecting hereditary risk factor"
        )

        age = st.number_input(
            "Age (years)",
            min_value=21,
            max_value=100,
            step=1,
            key="age",
            help="Patient age (minimum 21 years)"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Calculate Risk Assessment", type="primary", use_container_width=True)

# --------------------------------------------------
# Results Presentation
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

    # Risk Tiers
    if probability < 0.35:
        card_class = "outcome-low"
        title = "Low Risk Indication (Negative)"
        desc = "Current clinical biomarkers indicate low statistical likelihood of diabetes under the active cutoff threshold."
        val_color = "#059669"
        status_label = "Negative"
    elif probability < 0.60:
        card_class = "outcome-moderate"
        title = "Borderline / Pre-Diabetic Indication"
        desc = "Patient exhibits intermediate risk factors. Follow-up fasting plasma or oral glucose testing is suggested."
        val_color = "#d97706"
        status_label = "Borderline"
    else:
        card_class = "outcome-high"
        title = "Elevated Risk Indication (Positive)"
        desc = "Biomarkers reflect substantial risk characteristics. Formal laboratory evaluation and clinical consultation advised."
        val_color = "#dc2626"
        status_label = "Positive"

    st.markdown(f"""
    <div class="outcome-card {card_class}">
        <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.35rem;">
            {title}
        </div>
        <div style="font-size: 0.95rem; line-height: 1.5;">
            {desc}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Meter
    clamped_prob = min(max(probability, 0.0), 1.0)
    st.progress(clamped_prob, text=f"Estimated Risk Probability: {clamped_prob * 100:.1f}%")

    # Metrics Summary
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Risk Probability</div>
            <div class="stat-value" style="color: {val_color};">{probability * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Decision Threshold</div>
            <div class="stat-value" style="color: #475569;">{threshold * 100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Screening Call</div>
            <div class="stat-value" style="color: {val_color};">{status_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Clinical Biomarker Observations
    alerts = []
    if glucose >= 126.0:
        alerts.append(("alert", f"Fasting Glucose ({glucose:.0f} mg/dL) meets diabetic range criteria (≥ 126 mg/dL)"))
    elif glucose >= 100.0:
        alerts.append(("warn", f"Fasting Glucose ({glucose:.0f} mg/dL) falls within pre-diabetic range (100–125 mg/dL)"))

    if bmi >= 30.0:
        alerts.append(("alert", f"BMI ({bmi:.1f} kg/m²) indicates clinical obesity (threshold: ≥ 30.0)"))
    elif bmi >= 25.0:
        alerts.append(("warn", f"BMI ({bmi:.1f} kg/m²) falls in the overweight range (25.0–29.9)"))

    if blood_pressure >= 90.0:
        alerts.append(("alert", f"Diastolic Blood Pressure ({blood_pressure:.0f} mm Hg) indicates Stage 2 Hypertension"))
    elif blood_pressure >= 80.0:
        alerts.append(("warn", f"Diastolic Blood Pressure ({blood_pressure:.0f} mm Hg) indicates Stage 1 Hypertension"))

    if alerts:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Clinical Observations")
        for level, msg in alerts:
            bg = "#fef2f2" if level == "alert" else "#fffbeb"
            border = "#fecaca" if level == "alert" else "#fef3c7"
            color = "#991b1b" if level == "alert" else "#92400e"
            icon = "🔴" if level == "alert" else "🟡"
            st.markdown(f"""
            <div style="background-color: {bg}; border: 1px solid {border}; color: {color}; padding: 0.6rem 1rem; border-radius: 6px; font-size: 0.88rem; margin-bottom: 0.4rem;">
                {icon} {msg}
            </div>
            """, unsafe_allow_html=True)
