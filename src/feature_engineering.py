"""
Feature Engineering Module
------------------------
Advanced feature transformations for improved model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features to capture domain knowledge.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new engineered features
    """
    df = df.copy()
    
    df['Avg_Monthly_Charge'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    df['Charge_Per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    df['Tenure_Group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4+yr']
    )
    
    df['Is_New_Customer'] = (df['tenure'] <= 6).astype(int)
    df['Is_Loyal_Customer'] = (df['tenure'] >= 36).astype(int)
    
    df['Num_AddOns'] = (
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int)
    )
    
    df['Has_Any_AddOn'] = (df['Num_AddOns'] > 0).astype(int)
    df['Has_All_AddOns'] = (df['Num_AddOns'] >= 4).astype(int)
    
    df['Is_Fiber_Optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Has_Phone_And_Internet'] = (
        (df['PhoneService'] == 'Yes') & (df['InternetService'] != 'No')
    ).astype(int)
    
    df['Is_Month_To_Month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Has_Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    df['CLV_Estimate'] = df['TotalCharges'] * (1 + df['tenure'] / 100)
    
    df['Charge_Vs_Avg'] = df['MonthlyCharges'] - df['Avg_Monthly_Charge']
    
    return df


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return groupings of features by category for analysis.
    
    Returns
    -------
    Dict[str, List[str]]
        Feature groups
    """
    return {
        'demographics': ['gender', 'SeniorCitizen', 'Partner', 'Dependents'],
        'service_usage': ['PhoneService', 'MultipleLines', 'InternetService'],
        'addons': [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ],
        'billing': ['Contract', 'PaperlessBilling', 'PaymentMethod'],
        'financial': ['MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Charge'],
        'tenure_based': ['tenure', 'Tenure_Group', 'Is_New_Customer', 'Is_Loyal_Customer'],
        'engineered': ['Num_AddOns', 'Has_Any_AddOn', 'Is_Fiber_Optic', 'CLV_Estimate']
    }


def calculate_churn_risk_score(
    tenure: int,
    monthly_charges: float,
    contract_type: str,
    num_addons: int,
    has_tech_support: bool
) -> float:
    """
    Calculate a simple heuristic churn risk score (baseline for comparison).
    
    Parameters
    ----------
    tenure : int
        Months as customer
    monthly_charges : float
        Monthly bill amount
    contract_type : str
        Contract type
    num_addons : int
        Number of add-on services
    has_tech_support : bool
        Whether customer has tech support
        
    Returns
    -------
    float
        Risk score between 0 and 1
    """
    score = 0.0
    
    if tenure < 6:
        score += 0.3
    elif tenure < 12:
        score += 0.15
    
    if contract_type == 'Month-to-month':
        score += 0.25
    
    if monthly_charges > 70:
        score += 0.15
    
    if num_addons == 0:
        score += 0.2
    elif num_addons < 3:
        score += 0.1
    
    if not has_tech_support:
        score += 0.1
    
    return min(score, 1.0)