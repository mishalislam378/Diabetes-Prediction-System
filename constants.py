#constants.py
"""
Constants and configuration for Diabetes Prediction System
"""

# Column definitions
CATEGORICAL_COLS = ['gender', 'smoking_history']
NUMERIC_COLS = ['age', 'hypertension', 'heart_disease',
                'bmi', 'HbA1c_level', 'blood_glucose_level']
TARGET_COL = 'diabetes'
FEATURE_COLS = NUMERIC_COLS + ['gender_encoded', 'smoking_encoded']

# Color scheme
COLORS = {
    "green":  "#2ecc71",
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "orange": "#f39c12",
    "purple": "#9b59b6",
}

# File paths
DATASET_PATH = "real_balanced_3000.csv"

# Model parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42
LOGISTIC_REGRESSION_MAX_ITER = 1000
CV_FOLDS = 5