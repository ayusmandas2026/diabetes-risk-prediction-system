# 🩺 Diabetes Risk Prediction System
### *Multilayer Perceptron (MLP) Neural Network & Machine Learning Benchmark*

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://diabetes-risk-prediction-system-mlp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](#)

An end-to-end Machine Learning and Deep Learning project for diabetes screening using the Pima Indians Diabetes dataset. The repository features rigorous exploratory data analysis, physiological zero handling, leakage-free hybrid imputation, feature engineering (expanding 8 clinical features to 20), classical ML benchmarking, Optuna-driven hyperparameter optimization for Multilayer Perceptrons (MLP), and an interactive Streamlit web application.

---

## 📌 Key Highlights

- **Data Hygiene & Zero Recoding**: Identifies biologically impossible measurements of `0` in *Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI* and treats them as missing (`NaN`).
- **Leakage-Free Hybrid Imputation**: An 80/20 stratified train/test split is applied **strictly before** imputation:
  - **Median Imputation** for low-missingness features (*Glucose, Blood Pressure, BMI*).
  - **KNN Imputation (k=5)** for multivariate, high-missingness features (*Skin Thickness, Insulin*).
- **Domain-Specific Feature Engineering**: Expands 8 raw predictors into **20 engineered features** through clinical binning, log transformations, and interaction variables.
- **Model Benchmarking**: 6 classical algorithms evaluated side-by-side with deep Multilayer Perceptrons.
- **Optuna Hyperparameter Optimization**: Automated Bayesian optimization (Tree-structured Parzen Estimator) tuning layer depth, hidden unit sizes, dropout rates, learning rates, batch sizes, and optimizers.
- **Clinical Web Application**: Interactive Streamlit web app featuring quick-load patient presets, adjustable diagnostic sensitivity thresholding, colored risk tiers, and real-time clinical biomarker flags.

---

## 📊 Model Evaluation & Comparison

All models were evaluated on the held-out test set (stratified 20%):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Gradient Boosting** | **77.92%** | **70.00%** | **64.81%** | **0.6731** | **0.8389** |
| 🥈 **Support Vector Machine (SVM)** | 77.92% | 72.73% | 59.26% | 0.6531 | 0.8213 |
| 🥉 **K-Nearest Neighbors (KNN)** | 76.62% | 69.57% | 59.26% | 0.6400 | 0.8334 |
| **Random Forest** | 74.03% | 64.00% | 59.26% | 0.6154 | 0.8314 |
| **MLP (Optuna Tuned)** | 73.38% | 62.75% | 59.26% | 0.6095 | 0.8306 |
| **Logistic Regression** | 71.43% | 60.42% | 53.70% | 0.5686 | 0.8367 |
| **Decision Tree** | 71.43% | 60.42% | 53.70% | 0.5686 | 0.7888 |

---

## 🧬 Feature Engineering Pipeline (8 → 20 Features)

```text
8 Raw Features
  ├── Categorical Binning ───> Age_Group, BMI_Category, Glucose_Category
  ├── Interaction Terms ─────> Glucose × BMI, Age × Pregnancies
  ├── Log Normalization ─────> log1p(Insulin), log1p(DPF)
  └── One-Hot Encoding ──────> 20 Final Standardized Predictors
```

1. **Clinical Bins**:
   - `Age_Group`: `21-30`, `31-40`, `41-50`, `51+`
   - `BMI_Category`: `Underweight`, `Normal`, `Overweight`, `Obese`
   - `Glucose_Category`: `Normal` (< 100 mg/dL), `Prediabetic` (100–125 mg/dL), `Diabetic` (≥ 126 mg/dL)
2. **Physiological Interactions**:
   - `Glucose_BMI_Interaction`: Compound metabolic risk factor (Glucose × BMI).
   - `Age_Pregnancies_Interaction`: Gestational and chronological risk factor (Age × Pregnancies).
3. **Distribution Normalization**:
   - `Log_Insulin`: log(1 + Insulin)
   - `Log_DiabetesPedigreeFunction`: log(1 + DPF)
4. **Encoding & Scaling**: Dummy encoding with first category dropped (`drop_first=True`) followed by `StandardScaler`.

---

## 📁 Repository Structure

```text
diabetes-risk-prediction-system/
│
├── .gitignore                           # Excludes .venv and Python/Jupyter caches
├── .python-version                      # Specifies Python 3.11 for Streamlit Cloud
├── README.md                            # Comprehensive project documentation
├── diabetes.csv                         # Pima Indians Diabetes Dataset (768 rows)
├── MLPWP_LAB_1_final.ipynb              # Complete Jupyter Notebook (EDA to Optuna)
├── requirements.txt                     # Root-level requirements for deployment
│
├── [Training Pipeline Artifacts]
│   ├── final_mlp_model.keras            # Baseline Keras MLP model
│   ├── scaler.joblib                    # Fitted StandardScaler
│   ├── median_imputer.joblib            # Fitted SimpleImputer (median)
│   ├── knn_imputer.joblib               # Fitted KNNImputer
│   └── feature_columns.joblib           # Ordered 20-feature column schema
│
└── diabetes_mlp_app/                    # Standalone Web Application Package
    ├── app.py                           # Enhanced Streamlit web application
    ├── requirements.txt                 # Application deployment requirements
    ├── final_mlp_model.keras            # Tuned Keras MLP model (Optuna best)
    ├── scaler.joblib                    # Scaler artifact
    ├── median_imputer.joblib            # Median imputer artifact
    ├── knn_imputer.joblib               # KNN imputer artifact
    └── feature_columns.joblib           # Feature schema artifact
```

---

## 🚀 Getting Started

### 1. Clone & Environment Setup

```bash
git clone https://github.com/ayusmandas2026/diabetes-risk-prediction-system.git
cd diabetes-risk-prediction-system
```

Create and activate a virtual environment:

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application

The application can be launched from anywhere within the repository:

```bash
streamlit run diabetes_mlp_app/app.py
```

Open your browser at `http://localhost:8501` to view and interact with the application.

> **Live Cloud App**: You can also try the deployed app directly without local setup at **[diabetes-risk-prediction-system-mlp.streamlit.app](https://diabetes-risk-prediction-system-mlp.streamlit.app/)**.

---

## 🩺 Web Application Features

- **⚡ Quick-Load Test Profiles**: One-click preset buttons (*🟢 Healthy / Low Risk*, *🟡 Borderline / Pre-diabetic*, *🔴 High Risk Diabetic*) to test clinical edge cases instantly.
- **⚙️ Diagnostic Sensitivity Threshold**: An adjustable decision slider (0.20 to 0.80, default 0.50) enabling clinicians to optimize for higher sensitivity (fewer false negatives) in screening scenarios.
- **📊 Visual Risk Gauge**: Color-coded progress bar and risk tier classification (*Low Risk*, *Moderate / Pre-diabetic Risk*, *High Risk*).
- **📋 Real-time Clinical Biomarker Alerts**: Automatic warning callouts when individual patient attributes cross medical reference ranges (e.g. Fasting Glucose ≥ 126 mg/dL, BMI ≥ 30.0 kg/m²).

---

## ⚠️ Disclaimer

This project is an academic demonstration developed for educational and research purposes. It is not an FDA-approved or certified clinical diagnostic system. Healthcare decisions must always be made in consultation with licensed medical practitioners.
