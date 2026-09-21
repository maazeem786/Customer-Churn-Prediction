"""
Model Training Module
---------------------
Model selection, training, and hyperparameter tuning for churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib
from typing import Tuple, Optional, Dict, Any


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Tuple[Any, str]:
    """
    Train a churn prediction model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    model_type : str, default='random_forest'
        Type of model to train
    hyperparameters : dict, optional
        Model hyperparameters
        
    Returns
    -------
    Tuple[Any, str]
        Trained model and model name
    """
    models = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'xgboost': XGBClassifier,
        'logistic_regression': LogisticRegression
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = models[model_type]
    
    default_params = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        },
        'logistic_regression': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    }
    
    params = hyperparameters if hyperparameters else default_params.get(model_type, {})
    
    model = model_class(**params)
    
    print(f"Training {model_type}...")
    model.fit(X_train, y_train)
    print(f"Model trained successfully!")
    
    return model, model_type


def train_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    cv_folds: int = 5,
    search_method: str = 'randomized',
    n_iter: int = 20
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a model with hyperparameter tuning.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    model_type : str, default='random_forest'
        Type of model to tune
    cv_folds : int, default=5
        Cross-validation folds
    search_method : str, default='randomized'
        Search method ('randomized' or 'grid')
    n_iter : int, default=20
        Number of iterations for randomized search
        
    Returns
    -------
    Tuple[Any, Dict]
        Best model and best hyperparameters
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.7, 0.8, 0.9]
        }
    }
    
    base_model, _ = train_churn_model(X_train, y_train, model_type)
    
    grid = param_grids.get(model_type, {})
    
    if search_method == 'grid':
        search = GridSearchCV(
            base_model, grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            base_model, grid, n_iter=n_iter, cv=cv_folds,
            scoring='roc_auc', n_jobs=-1, random_state=42
        )
    
    print(f"Tuning {model_type} with {search_method} search...")
    search.fit(X_train, y_train)
    
    print(f"Best ROC-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv_folds : int, default=5
        Cross-validation folds
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
        results.append({
            'Model': name,
            'Mean AUC': cv_scores.mean(),
            'Std AUC': cv_scores.std(),
            'Min AUC': cv_scores.min(),
            'Max AUC': cv_scores.max()
        })
        print(f"  {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return pd.DataFrame(results)


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : Any
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model
        
    Returns
    -------
    Any
        Loaded model
    """
    return joblib.load(filepath)