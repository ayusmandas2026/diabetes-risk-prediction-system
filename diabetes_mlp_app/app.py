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
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# Load trained model and preprocessing artifacts
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "final_mlp_model.keras")
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
    median_imp = joblib.load(BASE_DIR / "median_imputer.joblib")
    knn_imp = joblib.load(BASE_DIR / "knn_imputer.joblib")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.joblib")

    return model, scaler, median_imp, knn_imp, feature_columns

# --------------------------------------------------
# Feature engineering (Identical to training pipeline)
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

    data["Glucose_BMI_Interaction"] = (
        data["Glucose"] * data["BMI"]
    )

    data["Age_Pregnancies_Interaction"] = (
        data["Age"] * data["Pregnancies"]
    )

    data["Log_Insulin"] = np.log1p(data["Insulin"])

    data["Log_DiabetesPedigreeFunction"] = np.log1p(
        data["DiabetesPedigreeFunction"]
    )

    data = pd.get_dummies(
        data,
        columns=[
            "Age_Group",
            "BMI_Category",
            "Glucose_Category"
        ],
        drop_first=True
    )

    return data

# Load artifacts
model, scaler, median_imp, knn_imp, feature_columns = load_artifacts()

# --------------------------------------------------
# Preset Clinical Profiles for Testing
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

# Initialize session state defaults if absent
default_values = PRESET_PROFILES["🟡 Borderline / Pre-diabetic"]
for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------
# Sidebar: Presets, Thresholds & Clinical Guide
# --------------------------------------------------
with st.sidebar:
    st.header("⚡ Sample Clinical Profiles")
    st.caption("Quickly populate the form with verified test cases:")

    for profile_name, profile_data in PRESET_PROFILES.items():
        if st.button(profile_name, use_container_width=True):
            for k, v in profile_data.items():
                st.session_state[k] = v
            st.rerun()

    st.divider()

    st.header("⚙️ Diagnostic Sensitivity")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Standard threshold is 0.50. In medical screening, lower thresholds (0.35 - 0.40) improve sensitivity (recall) to reduce false negatives."
    )

    if threshold < 0.50:
        st.info("🩺 **High Sensitivity Screening**: Prioritizes detecting true positive cases.")
    elif threshold > 0.50:
        st.info("🎯 **High Specificity Screening**: Prioritizes reducing false alarms.")
    else:
        st.info("⚖️ **Balanced Standard Threshold** (0.50).")

    st.divider()

    st.header("📋 Clinical Reference")
    st.markdown("""
    **Fasting Glucose:**
    - Normal: `< 100 mg/dL`
    - Prediabetes: `100 - 125 mg/dL`
    - Diabetes: `≥ 126 mg/dL`

    **Body Mass Index (BMI):**
    - Normal: `18.5 - 24.9`
    - Overweight: `25.0 - 29.9`
    - Obese: `≥ 30.0`
    """)

# --------------------------------------------------
# Application UI & Main Input Form
# --------------------------------------------------

st.title("🩺 Diabetes Risk Prediction System")
st.caption("Multilayer Perceptron (MLP) Neural Network with Optuna Hyperparameter Optimization")

st.warning(
    "⚠️ **Educational / Demonstration Project Only.** This system is trained on the Pima Indians Diabetes dataset and is not a certified medical diagnostic device."
)

with st.form("prediction_form"):
    st.subheader("Patient Clinical Attributes")

    c1, c2 = st.columns(2)

    with c1:
        pregnancies = st.number_input(
            "Pregnancies (count)",
            min_value=0,
            max_value=20,
            step=1,
            key="pregnancies"
        )

        glucose = st.number_input(
            "Fasting Glucose (mg/dL)",
            min_value=1.0,
            max_value=300.0,
            key="glucose"
        )

        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=1.0,
            max_value=200.0,
            key="blood_pressure"
        )

        skin_thickness = st.number_input(
            "Triceps Skinfold Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            key="skin_thickness"
        )

    with c2:
        insulin = st.number_input(
            "2-Hour Serum Insulin (μU/mL)",
            min_value=0.0,
            max_value=900.0,
            key="insulin"
        )

        bmi = st.number_input(
            "Body Mass Index - BMI (kg/m²)",
            min_value=1.0,
            max_value=70.0,
            key="bmi"
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function (family history score)",
            min_value=0.01,
            max_value=3.0,
            key="dpf"
        )

        age = st.number_input(
            "Age (years)",
            min_value=21,
            max_value=100,
            step=1,
            key="age"
        )

    submitted = st.form_submit_button("🔍 Compute Diabetes Risk Assessment", use_container_width=True)

# --------------------------------------------------
# Prediction & Clinical Interpretation
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

    # Treat physiologically invalid zeros as missing (same as training pipeline)
    zero_invalid = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]
    raw[zero_invalid] = raw[zero_invalid].replace(0, np.nan)

    # Median imputation for low-missingness features
    median_cols = ["Glucose", "BloodPressure", "BMI"]
    raw[median_cols] = median_imp.transform(raw[median_cols])

    # KNN imputation for high-missingness features
    knn_cols = ["SkinThickness", "Insulin"]
    raw[knn_cols] = knn_imp.transform(raw[knn_cols])

    # Feature engineering (20 total columns)
    processed = engineer_features(raw)

    # Ensure identical column alignment to training
    processed = processed.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Standard scaling
    scaled = scaler.transform(processed)

    # Model inference
    probability = float(
        model.predict(
            scaled,
            verbose=0
        ).ravel()[0]
    )

    prediction = int(probability >= threshold)

    # --------------------------------------------------
    # Display Results & Clinical Tiers
    # --------------------------------------------------
    st.divider()
    st.subheader("Diagnostic Assessment Result")

    if probability < 0.35:
        st.success("✅ **Assessment: Low Risk (Non-Diabetic)**")
    elif probability < 0.60:
        st.warning("⚠️ **Assessment: Moderate / Pre-Diabetic Risk**")
    else:
        st.error("🚨 **Assessment: High Risk (Diabetic)**")

    # Visual risk gauge
    clamped_prob = min(max(probability, 0.0), 1.0)
    st.progress(clamped_prob, text=f"Estimated Diabetes Risk: {clamped_prob * 100:.1f}%")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Probability", f"{probability * 100:.2f}%")
    m2.metric("Screening Threshold", f"{threshold * 100:.0f}%")
    m3.metric("Binary Classification", "Positive (Diabetic)" if prediction == 1 else "Negative (Non-Diabetic)")

    # Clinical Flagging
    flags = []
    if glucose >= 126.0:
        flags.append(f"• **Fasting Glucose ({glucose:.1f} mg/dL)** is in the diabetic range (≥ 126 mg/dL).")
    elif glucose >= 100.0:
        flags.append(f"• **Fasting Glucose ({glucose:.1f} mg/dL)** is in the pre-diabetic range (100–125 mg/dL).")

    if bmi >= 30.0:
        flags.append(f"• **BMI ({bmi:.1f})** indicates obesity (≥ 30.0).")
    elif bmi >= 25.0:
        flags.append(f"• **BMI ({bmi:.1f})** indicates overweight range (25.0–29.9).")

    if blood_pressure >= 90.0:
        flags.append(f"• **Diastolic Blood Pressure ({blood_pressure:.0f} mm Hg)** indicates stage 2 hypertension.")
    elif blood_pressure >= 80.0:
        flags.append(f"• **Diastolic Blood Pressure ({blood_pressure:.0f} mm Hg)** indicates stage 1 hypertension.")

    if flags:
        st.markdown("#### 🔍 Clinical Biomarker Highlights")
        for flag in flags:
            st.markdown(flag)