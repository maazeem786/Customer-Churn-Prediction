"""
Customer Churn Prediction Package
---------------------------------
A comprehensive ML pipeline for predicting customer churn.
"""

from .data_preprocessing import load_and_clean_data, preprocess_data
from .feature_engineering import engineer_features
from .model_training import train_churn_model, train_with_tuning
from .evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
from .prediction import ChurnPredictor

__all__ = [
    'load_and_clean_data',
    'preprocess_data',
    'engineer_features',
    'train_churn_model',
    'train_with_tuning',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'ChurnPredictor',
]

__version__ = '1.0.0'