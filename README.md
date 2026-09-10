# 🩺 Diabetes Prediction System (MLP & Classical Machine Learning)

An end-to-end Machine Learning and Deep Learning system for clinical diabetes risk prediction using the Pima Indians Diabetes dataset. This project includes comprehensive exploratory data analysis, data hygiene and hybrid imputation, extensive feature engineering, classical model benchmarking, hyperparameter-optimized Multilayer Perceptrons (MLP) via Optuna, and an interactive Streamlit web application.

---

## 📌 Project Overview & Highlights

- **Data Hygiene & Zero Recoding**: Biologically impossible values of zero in clinical measurements (*Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI*) are detected and treated as missing data (NaN).
- **Leakage-Free Hybrid Imputation**: Train/test splitting (80/20 stratified) is performed strictly prior to imputation:
  - **Median Imputation** for low-missingness features (*Glucose, Blood Pressure, BMI*).
  - **KNN Imputation (=5$)** for multivariate, high-missingness features (*Skin Thickness, Insulin*).
- **Domain-Specific Feature Engineering**: Expands 8 raw predictors into **20 engineered features** through clinical binning, log transformations, and interaction variables.
- **Model Benchmarking**: 6 classical algorithms evaluated side-by-side with deep Multilayer Perceptrons.
- **Optuna Hyperparameter Optimization**: Automated Bayesian optimization (Tree-structured Parzen Estimator) tuning layer depth, hidden unit sizes, dropout rates, learning rates, batch sizes, and optimizers.
- **Clinical Web Application**: Interactive Streamlit application featuring pre-loaded patient profiles, adjustable sensitivity thresholding, colored risk tiers, and clinical biomarker warning flags.

---

## 📊 Model Evaluation & Comparison

Performance evaluated on the held-out test set (stratified 20%):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **77.92%** | **70.00%** | **64.81%** | **0.6731** | **0.8389** |
| **Support Vector Machine (SVM)** | 77.92% | 72.73% | 59.26% | 0.6531 | 0.8213 |
| **K-Nearest Neighbors (KNN)** | 76.62% | 69.57% | 59.26% | 0.6400 | 0.8334 |
| **Random Forest** | 74.03% | 64.00% | 59.26% | 0.6154 | 0.8314 |
| **MLP (Optuna Tuned)** | 73.38% | 62.75% | 59.26% | 0.6095 | 0.8306 |
| **Logistic Regression** | 71.43% | 60.42% | 53.70% | 0.5686 | 0.8367 |
| **Decision Tree** | 71.43% | 60.42% | 53.70% | 0.5686 | 0.7888 |

---

## 🧬 Feature Engineering Pipeline (8 → 20 Features)

1. **Clinical Category Bins**:
   - Age_Group: 21-30, 31-40, 41-50, 51+
   - BMI_Category: Underweight, Normal, Overweight, Obese
   - Glucose_Category: Normal (< 100), Prediabetic (100–125), Diabetic (≥ 126)
2. **Physiological Interactions**:
   - Glucose_BMI_Interaction: Captures compound metabolic risk ( \times BMI$).
   - Age_Pregnancies_Interaction: Captures gestational and chronological risk ( \times Pregnancies$).
3. **Distribution Normalization**:
   - Log_Insulin: $\log(1 + Insulin)$
   - Log_DiabetesPedigreeFunction: $\log(1 + DPF)$
4. **Encoding & Scaling**: One-hot encoding with first category dropped (drop_first=True) followed by StandardScaler.

---

## 📁 Repository Structure

`
Ayusman SDP PROJECT/
│
├── .gitignore                           # Excludes virtual environments and build cache
├── README.md                            # Comprehensive project documentation
├── diabetes.csv                         # Raw Pima Indians Diabetes dataset
├── MLPWP_LAB_1_final.ipynb              # Complete training, tuning & evaluation notebook
│
├── [Training Pipeline Artifacts - Root]
│   ├── final_mlp_model.keras            # Baseline Keras MLP model
│   ├── scaler.joblib                    # Fitted StandardScaler
│   ├── median_imputer.joblib            # Fitted SimpleImputer (median)
│   ├── knn_imputer.joblib               # Fitted KNNImputer
│   └── feature_columns.joblib           # Ordered 20-feature column schema
│
└── diabetes_mlp_app/                    # Standalone Web Application Package
    ├── app.py                           # Enhanced Streamlit web application
    ├── requirements.txt                 # Deployment dependencies
    ├── final_mlp_model.keras            # Tuned Keras MLP model (Optuna best)
    ├── scaler.joblib                    # Preprocessing scaler
    ├── median_imputer.joblib            # Preprocessing median imputer
    ├── knn_imputer.joblib               # Preprocessing KNN imputer
    └── feature_columns.joblib           # Preprocessing feature schema
`

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone or open the project folder in your terminal, then create and activate a Python virtual environment:

`powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux / macOS
source .venv/bin/activate
`

### 2. Install Dependencies

`powershell
pip install -r diabetes_mlp_app/requirements.txt
`

### 3. Launch the Web Application

The application can be launched from anywhere within the repository:

`powershell
streamlit run diabetes_mlp_app/app.py
`

Open http://localhost:8501 in your browser to interact with the application.

---

## 🩺 Web Application Features

- **⚡ Quick-Load Sample Profiles**: One-click preset buttons (*Healthy / Low Risk*, *Borderline / Pre-diabetic*, *High Risk Diabetic*) for quick demonstrations.
- **⚙️ Diagnostic Sensitivity Slider**: Allows clinicians or evaluators to adjust the decision threshold (0.20 to 0.80) to optimize for higher sensitivity (fewer false negatives) during preliminary screening.
- **📊 Visual Risk Gauge**: Color-coded progress bar and risk tier classifications (*Low Risk*, *Moderate / Pre-diabetic Risk*, *High Risk*).
- **📋 Clinical Biomarker Highlighting**: Contextual callouts when input values fall into pre-diabetic, diabetic, or hypertensive ranges.

---

## ⚠️ Disclaimer

This software is an academic demonstration developed for educational and research purposes. It is not an FDA-approved or clinically validated diagnostic system. Clinical decisions should always be made by licensed healthcare professionals.
