"""
Data Preprocessing Module for Customer Churn Prediction System
Author: ChurnGuard AI Team
Description:
    Handles cleaning, missing value imputation, type casting,
    categorical encoding, feature scaling, and stratified train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Cast TotalCharges to numeric, imputing blank entries with 0.0 (tenure=0)
    - Ensure SeniorCitizen is treated consistently
    - Remove duplicate rows if any
    """
    df = df.copy()
    
    # Clean whitespace in strings
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Missing TotalCharges corresponds to new customers with tenure = 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
        
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def build_preprocessing_pipeline(numerical_cols=None, categorical_cols=None):
    """
    Construct an scikit-learn ColumnTransformer pipeline:
    - Numerical: StandardScaler
    - Categorical: OneHotEncoder (handle_unknown='ignore', sparse_output=False)
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
        
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

def prepare_train_test_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing execution:
    - Loads data
    - Cleans data
    - Extracts X and y (Churn: Yes -> 1, No -> 0)
    - Stratified split
    - Fits preprocessor pipeline on X_train only to prevent data leakage
    - Transforms X_train and X_test
    - Extracts encoded feature names
    """
    df = load_raw_data(data_path)
    df = clean_data(df)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Target encoding
    y = df['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).astype(int)
    X = df.drop(columns=['Churn'])
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Build preprocessor
    preprocessor = build_preprocessing_pipeline()
    
    # Fit preprocessor on X_train only
    preprocessor.fit(X_train)
    
    # Transform
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERICAL_COLS + cat_feature_names
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    
    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_df': X_train_df,
        'X_test_df': X_test_df,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'clean_full_df': df
    }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "..", "data", "customer_churn.csv")
    data = prepare_train_test_data(data_file)
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {data['X_train_df'].shape}")
    print(f"X_test shape: {data['X_test_df'].shape}")
    print(f"Total engineered features: {len(data['feature_names'])}")
