# 🩺 Diabetes Prediction System

> **AIL 201 Final Project** — Binary Classification using Logistic Regression with an interactive Streamlit dashboard.


---

## Overview

The **Diabetes Prediction System** is a machine-learning-powered web application that predicts whether a patient is at risk of diabetes based on clinical and demographic features. It uses **Logistic Regression** for binary classification and presents results through a clean, interactive **Streamlit** GUI with real-time visualizations.

---

## Features

- ✅ **Auto-trains** the model on startup — no manual steps required
- 🔍 **Patient prediction** with probability scores and risk levels (Low / Medium / High)
- 📊 **Exploratory Data Analysis (EDA)** with charts and statistical summaries
- 🤖 **Model performance** dashboard — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- 📈 **Repeated runs** evaluation (5 seeds) with Mean ± Std Dev stability analysis
- ✅ **Input validation** with clinically meaningful error messages
- 📋 **Clinical reference tables** for Glucose, HbA1c, and BMI
- 💡 **Personalised recommendations** based on prediction outcome

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | Diabetes Prediction Dataset |
| **File** | `real_balanced_3000.csv` |
| **Records** | 3,000 (pre-balanced) |
| **Class split** | 70% Non-Diabetic / 30% Diabetic |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Male / Female / Other |
| `age` | Numeric | Patient age in years |
| `hypertension` | Binary | 1 = Yes, 0 = No |
| `heart_disease` | Binary | 1 = Yes, 0 = No |
| `smoking_history` | Categorical | never / former / current / ever / No Info / not current |
| `bmi` | Numeric | Body Mass Index (kg/m²) |
| `HbA1c_level` | Numeric | Glycated haemoglobin — key clinical diabetes marker (%) |
| `blood_glucose_level` | Numeric | Blood glucose in mg/dL |

**Target:** `diabetes` — `1` = Diabetic, `0` = Non-Diabetic

---


## Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | Logistic Regression |
| **Library** | scikit-learn |
| **Train / Test Split** | 80% / 20% (stratified) |
| **Cross-Validation** | 5-Fold CV |
| **Preprocessing** | StandardScaler + LabelEncoder for categorical features |
| **Max Iterations** | 1,000 |
| **Random State** | 42 |

### Encoded Features Used for Training

```
age, hypertension, heart_disease, bmi,
HbA1c_level, blood_glucose_level,
gender_encoded, smoking_encoded
```

### Risk Level Classification

| Prediction | Probability Threshold | Label |
|-----------|----------------------|-------|
| Non-Diabetic | prob(diabetes) ≤ 0.35 | 🟢 LOW RISK |
| Non-Diabetic | prob(diabetes) > 0.35 | 🟡 MEDIUM RISK |
| Diabetic | Any | 🔴 HIGH RISK |

---

## App Tabs

### 📊 Tab 1 — Dataset & EDA
- Summary metrics (total records, feature count, class counts)
- First 10 rows preview and statistical summary
- Feature description table
- Visualizations: class distribution (pie chart), BMI distribution by outcome, HbA1c distribution by outcome

### 🤖 Tab 2 — Model Performance
- Accuracy, Precision, Recall, F1-Score (displayed as percentages)
- 5-Fold Cross-Validation score
- Confusion Matrix heatmap
- Feature Importance bar chart (Logistic Regression coefficients)
- Full classification report (expandable)

### 🔍 Tab 3 — Predict Patient
- Input form: Gender, Age, Hypertension, Heart Disease, Smoking History, BMI, HbA1c, Blood Glucose
- Real-time input validation with clinically meaningful bounds
- Prediction output: risk label, probability metrics, and personalised recommendation
- Clinical reference table with per-metric status indicators (🟢 Normal / 🟡 Pre-diabetic / 🔴 Diabetic)
- Risk factor summary and probability bar chart

### 📈 Tab 4 — Repeated Runs
- Runs the model 5 times with different random seeds
- Per-run results table (Accuracy, Precision, Recall, F1-Score in %)
- Summary table: Mean ± Std Dev, Min, Max
- Grouped bar chart across all 5 runs

---

## Clinical Reference Ranges

| Metric | Normal (Healthy) | Pre-Diabetic (At Risk) | Diabetic (High Risk) |
|--------|-----------------|----------------------|---------------------|
| Glucose (mg/dL) | 70 – 99 | 100 – 125 | 126+ |
| HbA1c (%) | < 5.7 | 5.7 – 6.4 | ≥ 6.5 |
| BMI (kg/m²) | 18.5 – 24.9 | 25.0 – 29.9 | ≥ 30.0 |

---



## Disclaimer

> ⚠️ **This application is an AI-based prediction tool intended for educational purposes only.**
> It is **not a substitute for professional medical advice, diagnosis, or treatment.**
> Always consult a qualified healthcare provider for any medical decisions.
> app.py is the main file.

---

*AIL 201 Final Project — Diabetes Prediction System*# Diabetes-Prediction-System
