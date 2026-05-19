#model.py
"""
Diabetes prediction model using Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from constants import FEATURE_COLS, CV_FOLDS, LOGISTIC_REGRESSION_MAX_ITER, TEST_SIZE, RANDOM_STATE
from data_loader import preprocess_dataframe, split_data


class DiabetesModel:
    """
    Logistic Regression model for diabetes prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.gender_enc = None
        self.smoking_enc = None
        self.X_test = None
        self.y_test = None
        self.is_trained = False
        self.cv_scores = None
        self.feature_importance = None
    
    def preprocess(self, df: pd.DataFrame):
        """
        Preprocess the data and split into train/test sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Preprocess dataframe and get encoders
        df, self.gender_enc, self.smoking_enc = preprocess_dataframe(df)
        
        # Prepare features and target
        X = df[FEATURE_COLS].values
        y = df['diabetes'].values
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train) #learn scaling values and apply scaling
        X_test = self.scaler.transform(X_test)# only applies already learned scaling.model must not learn from testing dataOtherwise data leakage happens.
        
        self.X_test = X_test
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model = LogisticRegression(
            max_iter=LOGISTIC_REGRESSION_MAX_ITER,
            random_state=RANDOM_STATE
        )
        self.model.fit(X_train, y_train) #all training predictions weight error stuff
        
        # Calculate cross-validation scores
        self.cv_scores = cross_val_score(self.model, X_train, y_train, cv=CV_FOLDS) #Model trains/tests 5 times on different splits.
        
        # Calculate feature importance (coefficients)
        self.feature_importance = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Coefficient': self.model.coef_[0] #stores learned weights of features.
        }).sort_values('Coefficient', ascending=False)
        
        self.is_trained = True
    
    def evaluate(self) -> dict:
        """
        Evaluate the model on test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.model.predict(self.X_test)
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),#How many predictions were correct?”
            'precision': precision_score(self.y_test, y_pred),#When model says DIABETIC, how often is it correct?
            'recall': recall_score(self.y_test, y_pred),#Out of all actual diabetic people, how many did model catch?
            'f1': f1_score(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'y_pred': y_pred,
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std(),
        }
    
    def predict_patient(self, raw_input: dict) -> dict:
        """
        Make prediction for a single patient
        
        Args:
            raw_input: Dictionary with patient features
            
        Returns:
            Dictionary with prediction label and probabilities
        """
        row = np.array([[raw_input[f] for f in FEATURE_COLS]])
        row_scaled = self.scaler.transform(row)
        
        return {
            'label': self.model.predict(row_scaled)[0],#[0] is used to extract the first element from the output array because scikit-learn returns predictions in list format even for a single input sample.”
            'prob': self.model.predict_proba(row_scaled)[0]
        }
    
    def prepare_patient_features(self, gender: str, smoking: str, age: int, 
                                  hypertension: int, heart_disease: int, 
                                  bmi: float, hba1c: float, glucose: int) -> dict:
        """
        Prepare patient features for prediction
        
        Args:
            gender: Patient gender
            smoking: Smoking history
            age: Age in years
            hypertension: 0 or 1
            heart_disease: 0 or 1
            bmi: Body Mass Index
            hba1c: HbA1c level
            glucose: Blood glucose level
            
        Returns:
            Dictionary with encoded features
        """
        def safe_encode(encoder, value):
            """Safely encode categorical values"""
            if encoder is None:
                return 0
            return int(encoder.transform([value])[0]) if value in encoder.classes_ else 0
        
        return {
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': glucose,
            'gender_encoded': safe_encode(self.gender_enc, gender),
            'smoking_encoded': safe_encode(self.smoking_enc, smoking),
        }


def run_multiple_evaluations(df, n_runs: int = 5) -> dict:
    """
    Run multiple evaluations with different random seeds
    
    Args:
        df: Input DataFrame
        n_runs: Number of evaluation runs
        
    Returns:
        Dictionary with metrics for each run
    """
    
    
    metrics_runs = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1']}
    
    # Preprocess the dataframe once
    df_enc, gender_enc, smoking_enc = preprocess_dataframe(df)
    X_all = df_enc[FEATURE_COLS].values
    y_all = df_enc['diabetes'].values
    
    for seed in range(n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, random_state=seed, stratify=y_all
        )
        
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        
        lr = LogisticRegression(
            max_iter=LOGISTIC_REGRESSION_MAX_ITER, 
            random_state=seed
        )
        lr.fit(X_tr, y_tr)
        y_pred = lr.predict(X_te)
        
        metrics_runs['accuracy'].append(accuracy_score(y_te, y_pred))
        metrics_runs['precision'].append(precision_score(y_te, y_pred))
        metrics_runs['recall'].append(recall_score(y_te, y_pred))
        metrics_runs['f1'].append(f1_score(y_te, y_pred))
    
    return metrics_runs