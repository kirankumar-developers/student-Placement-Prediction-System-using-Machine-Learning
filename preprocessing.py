"""
Data Preprocessing Module for Student Placement Prediction System
Handles dataset loading, validation, feature extraction, scaling pipeline creation,
and train-test splitting without data leakage.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DEFAULT_DATASET_PATH = 'student_placement.csv'
TARGET_COLUMN_CANDIDATES = ['Placement', 'placement', 'Placed', 'placed', 'Status', 'status']

def load_and_validate_dataset(filepath=DEFAULT_DATASET_PATH):
    """
    Loads dataset from CSV, validates presence, structure, and integrity.
    Returns: pandas.DataFrame, target_col_name
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found in the project directory.")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset in '{filepath}' is empty.")

    # Identify target column
    target_col = None
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        raise KeyError(
            f"Target column not found in dataset. Expected one of: {TARGET_COLUMN_CANDIDATES}. "
            f"Found columns: {list(df.columns)}"
        )

    # Check and clean duplicates
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[INFO] Removed {initial_len - len(df)} duplicate records.")

    # Handle missing values if any
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[INFO] Found {missing_count} missing values. Handling with forward/median imputations.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for c in categorical_cols:
            df[c] = df[c].fillna(df[c].mode()[0])

    return df, target_col


def encode_target(series):
    """
    Standardizes and maps target variable to binary (1 for Placed, 0 for Not Placed).
    Returns mapped series and mapping dictionary.
    """
    unique_vals = series.astype(str).str.strip().unique()
    
    # Identify positive label
    pos_labels = ['Placed', 'placed', '1', 'Yes', 'yes', 'True', 'true', 'Y', 'y', 'Pass', 'pass']
    neg_labels = ['Not Placed', 'not placed', '0', 'No', 'no', 'False', 'false', 'N', 'n', 'Fail', 'fail']

    placed_val = None
    not_placed_val = None

    for val in unique_vals:
        if val in pos_labels:
            placed_val = val
        elif val in neg_labels:
            not_placed_val = val

    if placed_val is None or not_placed_val is None:
        # Fallback to standard sorting if exact names differ
        if len(unique_vals) == 2:
            sorted_vals = sorted(unique_vals)
            not_placed_val = sorted_vals[0]
            placed_val = sorted_vals[1]
        else:
            raise ValueError(f"Target column must have 2 classes for binary placement classification. Found: {unique_vals}")

    mapping = {placed_val: 1, not_placed_val: 0}
    inverse_mapping = {1: 'Placed', 0: 'Not Placed'}
    
    encoded_series = series.astype(str).str.strip().map(mapping)
    return encoded_series, mapping, inverse_mapping


def extract_feature_metadata(df, target_col):
    """
    Extracts comprehensive metadata for each feature to power dynamic UI forms,
    validation, and descriptions.
    """
    features_df = df.drop(columns=[target_col])
    metadata = {}
    
    for col in features_df.columns:
        dtype = str(features_df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(features_df[col])
        
        # User-friendly label
        friendly_name = col.replace('_', ' ').replace('-', ' ').title()
        if 'Cgpa' in friendly_name:
            friendly_name = 'CGPA'
        elif '10Th' in friendly_name:
            friendly_name = '10th Class Percentage (%)'
        elif '12Th' in friendly_name:
            friendly_name = '12th Class Percentage (%)'
        elif 'Aptitude' in friendly_name:
            friendly_name = 'Aptitude Test Score (0-100)'
        elif 'Communication' in friendly_name:
            friendly_name = 'Communication Score (0-10)'
        elif 'Technical' in friendly_name:
            friendly_name = 'Technical Skill Score (0-10)'
        
        col_info = {
            'name': col,
            'label': friendly_name,
            'is_numeric': bool(is_numeric),
            'dtype': dtype,
            'min': float(features_df[col].min()) if is_numeric else None,
            'max': float(features_df[col].max()) if is_numeric else None,
            'mean': round(float(features_df[col].mean()), 2) if is_numeric else None,
            'std': round(float(features_df[col].std()), 2) if is_numeric else None,
            'median': float(features_df[col].median()) if is_numeric else None,
            'unique_values': [str(x) for x in features_df[col].unique().tolist()] if not is_numeric else None,
            'step': 0.01 if 'float' in dtype or col in ['CGPA', '10th_Percentage', '12th_Percentage', 'Aptitude_Score', 'Communication_Score', 'Technical_Skill_Score'] else 1
        }
        metadata[col] = col_info
        
    return metadata


def build_preprocessor_pipeline(features_df):
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline for scaling numerical features
    and encoding categorical features without data leakage.
    """
    numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor, numeric_features, categorical_features


def prepare_dataset(filepath=DEFAULT_DATASET_PATH, test_size=0.2, random_state=42):
    """
    Loads, cleans, encodes target, prepares train/test split, and fits the ColumnTransformer.
    Returns:
      X_train_raw, X_test_raw, X_train_scaled, X_test_scaled,
      y_train, y_test, preprocessor, feature_names, metadata, df_clean
    """
    df, target_col = load_and_validate_dataset(filepath)
    y, mapping, inverse_mapping = encode_target(df[target_col])
    X = df.drop(columns=[target_col])
    
    metadata = extract_feature_metadata(df, target_col)
    
    # Stratified Train-Test split (80% train, 20% test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor, num_cols, cat_cols = build_preprocessor_pipeline(X)
    
    # Fit preprocessor strictly on training data
    preprocessor.fit(X_train_raw)
    
    X_train_scaled = preprocessor.transform(X_train_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)
    
    feature_names = list(X.columns)
    
    return {
        'df': df,
        'target_col': target_col,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numeric_features': num_cols,
        'categorical_features': cat_cols,
        'metadata': metadata,
        'inverse_mapping': inverse_mapping
    }


if __name__ == '__main__':
    print("[TEST] Running preprocessing verification...")
    data = prepare_dataset()
    print(f"Dataset shape: {data['df'].shape}")
    print(f"Features: {data['feature_names']}")
    print(f"X_train scaled shape: {data['X_train_scaled'].shape}")
    print(f"X_test scaled shape: {data['X_test_scaled'].shape}")
    print(f"y_train distribution:\n{data['y_train'].value_counts(normalize=True)}")
    print(f"y_test distribution:\n{data['y_test'].value_counts(normalize=True)}")
    print("[SUCCESS] Preprocessing pipeline configured successfully.")
