#data_loader.py
"""
Data loading and preprocessing utilities
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from constants import TARGET_COL, FEATURE_COLS, TEST_SIZE, RANDOM_STATE


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the diabetes dataset from CSV file
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    df = pd.read_csv(path)
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    return {
        'total_records': len(df),
        'diabetic_count': int(df[TARGET_COL].sum()),
        'non_diabetic_count': int((df[TARGET_COL] == 0).sum()),
        'diabetic_percentage': df[TARGET_COL].mean() * 100,
        'features': len(FEATURE_COLS)
    }


def preprocess_dataframe(df: pd.DataFrame, gender_enc: LabelEncoder = None, 
                         smoking_enc: LabelEncoder = None, fit_encoders: bool = True):
    """
    Preprocess the dataframe by encoding categorical variables
    
    Args:
        df: Input DataFrame
        gender_enc: Existing LabelEncoder for gender (if None, create new)
        smoking_enc: Existing LabelEncoder for smoking_history (if None, create new)
        fit_encoders: Whether to fit the encoders or use existing ones
        
    Returns:
        Tuple of (processed_df, gender_enc, smoking_enc)
    """
    df = df.copy()
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Initialize encoders if not provided
    if gender_enc is None:
        gender_enc = LabelEncoder()
    if smoking_enc is None:
        smoking_enc = LabelEncoder()
    
    # Encode categorical variables
    if fit_encoders:
        df['gender_encoded'] = gender_enc.fit_transform(df['gender'])
        df['smoking_encoded'] = smoking_enc.fit_transform(df['smoking_history'])
    else:
        df['gender_encoded'] = gender_enc.transform(df['gender'])
        df['smoking_encoded'] = smoking_enc.transform(df['smoking_history'])
    
    return df, gender_enc, smoking_enc


def split_data(X, y):
    """
    Split data into training and testing sets
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )