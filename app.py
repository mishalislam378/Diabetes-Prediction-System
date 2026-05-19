#app.py
"""
Diabetes Prediction System – AIL 201 Final Project
Algorithm: Logistic Regression (Binary Classification)
GUI: Streamlit

Main application file
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Import project modules
from constants import COLORS, FEATURE_COLS, DATASET_PATH
from data_loader import load_dataset, get_dataset_info
from model import DiabetesModel, run_multiple_evaluations
from visualization import (
    plot_confusion_matrix, plot_feature_importance, plot_class_distribution,
    plot_bmi_distribution, plot_hba1c_distribution, plot_runs_bargraph,
    plot_prediction_probability
)

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover { background-color: #27ae60; }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "model" not in st.session_state:
    st.session_state.model = DiabetesModel()
    st.session_state.trained = False
    st.session_state.results = None

# ============================================================
# LOAD DATA
# ============================================================
try:
    df = load_dataset(DATASET_PATH)
    dataset_info = get_dataset_info(df)
except FileNotFoundError:
    st.error(f"⚠️ Could not find **{DATASET_PATH}** — place it in the same folder as app.py and restart.")
    st.stop()

# ============================================================
# AUTO-TRAIN MODEL ON STARTUP
# ============================================================
if not st.session_state.trained:
    with st.spinner("⚙️ Loading and training model, please wait..."):
        m = st.session_state.model
        X_train, X_test, y_train, y_test = m.preprocess(df)
        m.train(X_train, y_train)
        st.session_state.results = m.evaluate()
        st.session_state.trained = True

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 Diabetes Prediction")
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown("**Source:** Diabetes Prediction Dataset (Kaggle)")
    st.markdown(f"**Records:** {dataset_info['total_records']:,}")
    st.markdown(f"**Features:** {dataset_info['features']}")
    st.markdown(f"**Diabetic:** {dataset_info['diabetic_percentage']:.1f}%")
    st.markdown("---")
    st.markdown("### ⚙️ Model")
    st.markdown("**Algorithm:** Logistic Regression")
    st.markdown("**Type:** Binary Classification")
    st.markdown("**Split:** 70 / 30 train-test (stratified)")
    st.markdown("**Validation:** 5-Fold CV")
    st.markdown("**Balance:** Pre-balanced dataset — 70% non-diabetic / 30% diabetic")
    st.success("✅ Model ready")

# ============================================================
# MAIN HEADER
# ============================================================
st.title("🩺 Diabetes Prediction System")
st.markdown("**Logistic Regression** · Binary Classification · AIL 201 Final Project")
st.success(f"✅ Dataset loaded — {dataset_info['total_records']:,} records, {dataset_info['features']} features")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset & EDA",
    "🤖 Model Performance",
    "🔍 Predict Patient",
    "📈 Repeated Runs",
])

# ============================================================
# TAB 1 – EDA
# ============================================================
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{dataset_info['total_records']:,}")
    c2.metric("Features", dataset_info['features'])
    c3.metric("Diabetic", dataset_info['diabetic_count'])
    c4.metric("Non-Diabetic", dataset_info['non_diabetic_count'])
    
    st.subheader("Data Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)
    
    st.subheader("Feature Descriptions")
    feat_info = pd.DataFrame({
        'Feature': ['gender', 'age', 'hypertension', 'heart_disease',
                    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'],
        'Type': ['Categorical', 'Numeric', 'Binary', 'Binary',
                 'Categorical', 'Numeric', 'Numeric', 'Numeric'],
        'Description': [
            'Patient gender (Male / Female / Other)',
            'Patient age in years',
            '1 = has hypertension, 0 = does not',
            '1 = has heart disease, 0 = does not',
            'Smoking history (never / former / current / ever / No Info)',
            'Body Mass Index (kg/m²)',
            'Glycated haemoglobin — key clinical diabetes marker',
            'Blood glucose level (mg/dL)',
        ]
    })
    st.dataframe(feat_info, use_container_width=True)
    
    st.subheader("Visualizations")
    v1, v2 = st.columns(2)
    with v1:
        st.pyplot(plot_class_distribution(df))
    with v2:
        st.pyplot(plot_bmi_distribution(df))
    
    st.subheader("HbA1c Level by Outcome")
    st.pyplot(plot_hba1c_distribution(df))

# ============================================================
# TAB 2 – MODEL PERFORMANCE
# ============================================================
with tab2:
    res = st.session_state.results
    st.subheader("Model Performance")
    
    # Display metrics as percentages
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{res['precision']*100:.2f}%")
    m3.metric("Recall", f"{res['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{res['f1']*100:.2f}%")
    
    st.info(f"📊 5-Fold Cross-Validation Mean: **{res['cv_mean']*100:.2f}%** (see Repeated Runs tab for full Mean ± Std Dev analysis)")
    
    st.pyplot(plot_confusion_matrix(res['confusion_matrix']))
    
    st.subheader("Feature Importance")
    st.pyplot(plot_feature_importance(st.session_state.model.feature_importance))
    
    

# ============================================================
# TAB 3 – PREDICTION
# ============================================================
with tab3:
    st.subheader("Enter Patient Information")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])
        age = st.number_input("Age", 1, 120, 45, step=1)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with col2:
        smoking = st.selectbox("Smoking History",
                               ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
        bmi = st.number_input("BMI", 10.0, 70.0, 25.0, step=0.1)
        hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, step=0.1,
                                help="Normal <5.7 | Pre-diabetic 5.7–6.4 | Diabetic ≥6.5")
        glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 500, 120, step=1)
    
    if st.button("🔍 Predict", use_container_width=True):
        # Input validation
        validation_errors = []
        if age < 0 or age > 120:
            validation_errors.append("❌ Age must be between 0 and 120 years")
        if bmi < 10 or bmi > 70:
            validation_errors.append("❌ BMI must be between 10 and 70 (realistic range)")
        if hba1c < 3.0 or hba1c > 15.0:
            validation_errors.append("❌ HbA1c must be between 3.0% and 15.0%")
        if glucose < 50 or glucose > 500:
            validation_errors.append("❌ Blood glucose must be between 50 and 500 mg/dL")
        
        if validation_errors:
            for err in validation_errors:
                st.error(err)
            st.stop()
        
        # Prepare patient features and predict
        m = st.session_state.model
        patient = m.prepare_patient_features(
            gender, smoking, age, hypertension, heart_disease, bmi, hba1c, glucose
        )
        result = m.predict_patient(patient)
        
        prob_d = result['prob'][1]
        is_diabetic = result['label'] == 1
        
        # Display results
        if is_diabetic:
            risk_tag = "🔴 HIGH RISK"
            st.error(f"### ⚠️ HIGH RISK — DIABETIC")
        else:
            if prob_d > 0.35:
                risk_tag = "🟡 MEDIUM RISK"
            else:
                risk_tag = "🟢 LOW RISK"
            st.success(f"### ✅ LOW RISK — NON-DIABETIC")
        
        st.markdown(f"### {risk_tag}")
        
        col_a, col_b = st.columns(2)
        col_a.metric("Probability of Diabetes", f"{prob_d*100:.1f}%")
        col_b.metric("Probability of No Diabetes", f"{result['prob'][0]*100:.1f}%")
        
        # Recommendations
        if is_diabetic:
            st.warning("💊 **Recommendation:** Consult a healthcare provider. Monitor blood glucose and HbA1c regularly.")
        else:
            st.success("🥗 **Recommendation:** Keep up healthy habits — 30 min/day activity and a balanced diet lower long-term risk.")
        
        # Clinical reference ranges
        st.markdown("### 📋 Clinical Reference Ranges")
        clinical_data = {
            'Metric': ['Glucose (mg/dL)', 'HbA1c (%)', 'BMI (kg/m²)'],
            'Your Value': [f"{glucose}", f"{hba1c}", f"{bmi:.1f}"],
            'Normal Range (Healthy)': ['70 – 99', 'Under 5.7', '18.5 – 24.9'],
            'Prediabetic (At Risk)': ['100 – 125', '5.7 – 6.4', '25.0 – 29.9'],
            'Diabetic (High Risk)': ['126+', '6.5 or higher', '30.0+']
        }
        st.dataframe(pd.DataFrame(clinical_data), use_container_width=True)
        
        # Status indicators
        st.markdown("**Your Status:**")
        if glucose < 100:
            st.markdown("- 🟢 **Glucose:** Normal")
        elif glucose <= 125:
            st.markdown("- 🟡 **Glucose:** Prediabetic range")
        else:
            st.markdown("- 🔴 **Glucose:** Diabetic range")
        
        if hba1c < 5.7:
            st.markdown("- 🟢 **HbA1c:** Normal")
        elif hba1c <= 6.4:
            st.markdown("- 🟡 **HbA1c:** Prediabetic range")
        else:
            st.markdown("- 🔴 **HbA1c:** Diabetic range")
        
        # Probability bar chart
        st.pyplot(plot_prediction_probability(result['prob']))
        st.caption("🤖 AI prediction tool — not a substitute for professional medical advice.")

# ============================================================
# TAB 4 – REPEATED RUNS
# ============================================================
with tab4:
    st.subheader("5 Repeated Evaluations — Mean ± Standard Deviation")
    st.markdown("Each run uses a different random seed, varying the train/test split "
                "to show how stable the model is across different data partitions.")
    
    if st.button("▶️ Run 5 Evaluations"):
        with st.spinner("Running 5 evaluations..."):
            metrics_runs = run_multiple_evaluations(df, n_runs=5)
        
        # Per-run breakdown table
        runs_table = pd.DataFrame({
            'Run': [f'Run {i+1}' for i in range(5)],
            'Accuracy': [f"{v*100:.2f}%" for v in metrics_runs['accuracy']],
            'Precision': [f"{v*100:.2f}%" for v in metrics_runs['precision']],
            'Recall': [f"{v*100:.2f}%" for v in metrics_runs['recall']],
            'F1-Score': [f"{v*100:.2f}%" for v in metrics_runs['f1']],
        })
        
        st.markdown("#### 📋 Per-Run Results")
        st.dataframe(runs_table, use_container_width=True)
        
        # Summary statistics
        st.markdown("#### 📊 Summary: Mean ± Std Dev")
        summary = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Mean': [f"{np.mean(metrics_runs[k])*100:.2f}%" for k in ['accuracy', 'precision', 'recall', 'f1']],
            'Std Dev': [f"{np.std(metrics_runs[k])*100:.2f}%" for k in ['accuracy', 'precision', 'recall', 'f1']],
            'Min': [f"{np.min(metrics_runs[k])*100:.2f}%" for k in ['accuracy', 'precision', 'recall', 'f1']],
            'Max': [f"{np.max(metrics_runs[k])*100:.2f}%" for k in ['accuracy', 'precision', 'recall', 'f1']],
        })
        st.dataframe(summary, use_container_width=True)
        
        st.markdown("#### 📈 Bar Chart — All Metrics Per Run")
        st.pyplot(plot_runs_bargraph(metrics_runs, n_runs=5))
        st.success("✅ 5-run evaluation complete!")