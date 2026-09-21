"""
Data Preprocessing Module
-------------------------
Handles loading, cleaning, and initial preprocessing of customer churn data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform initial cleaning on the churn dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing churn data
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for preprocessing
        
    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist
    ValueError
        If required columns are missing
    """
    df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    df = df.dropna()
    
    df = df[df['TotalCharges'] != ' ']
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    df = df.drop_duplicates()
    
    print(f"After cleaning: {len(df)} records")
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = 'Churn',
    encode_categoricals: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[dict]]:
    """
    Preprocess churn data: encode categoricals, split features/target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str, default='Churn'
        Name of the target column
    encode_categoricals : bool, default=True
        Whether to encode categorical variables
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, dict or None]
        X (features), y (target), and encoding map if encode_categoricals=True
    """
    df = df.copy()
    
    customer_ids = df['customerID']
    df = df.drop(columns=['customerID'])
    
    binary_mappings = {
        'Yes': 1, 'No': 0,
        'Female': 1, 'Male': 0,
        'True': 1, 'False': 0
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [c for c in categorical_columns if c != target_column]
    
    label_encodings = {}
    if encode_categoricals:
        for col in categorical_columns:
            unique_vals = df[col].unique()
            label_encodings[col] = {val: idx for idx, val in enumerate(unique_vals)}
            df[col] = df[col].map(label_encodings[col])
    
    df = df.fillna(df.median(numeric_only=True))
    
    y = df[target_column]
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'No': 0})
    
    X = df.drop(columns=[target_column])
    
    print(f"Processed {X.shape[1]} features from {X.shape[0]} samples")
    print(f"Churn rate: {y.mean():.2%}")
    
    return X, y, label_encodings if encode_categoricals else None


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float, default=0.2
        Proportion of data for test set
    random_state : int, default=42
        Random seed for reproducibility
    stratify : bool, default=True
        Whether to maintain class distribution in splits
        
    Returns
    -------
    Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Scale numerical features.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    scaler_type : str, default='standard'
        Type of scaler ('standard', 'minmax', or 'robust')
        
    Returns
    -------
    Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    scaler = scalers.get(scaler_type, StandardScaler())
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler