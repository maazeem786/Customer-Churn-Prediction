"""
Model Evaluation Module
----------------------
Comprehensive model evaluation with multiple metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate model with comprehensive metrics.
    
    Parameters
    ----------
    model : Any
        Trained model with predict or predict_proba method
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    threshold : float, default=0.5
        Classification threshold
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    has_proba = hasattr(model, 'predict_proba')
    if has_proba:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
    else:
        y_proba = None
        auc_score = None
        avg_precision = None
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': auc_score,
        'average_precision': avg_precision,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'])
    }
    
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:           {results['accuracy']:.4f}")
    print(f"Precision:          {results['precision']:.4f}")
    print(f"Recall (Sensitivity): {results['recall']:.4f}")
    print(f"F1-Score:           {results['f1_score']:.4f}")
    if auc_score:
        print(f"ROC-AUC:            {auc_score:.4f}")
        print(f"Average Precision:  {avg_precision:.4f}")
    print("=" * 50)
    
    return results


def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    save_path : str, optional
        Path to save the figure
    normalize : bool, default=True
        Whether to show normalized values
        
    Returns
    -------
    plt.Figure
        Confusion matrix figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='.2%' if normalize else 'd',
        cmap='Blues', cbar=True,
        xticklabels=['Not Churn', 'Churn'],
        yticklabels=['Not Churn', 'Churn'],
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Churn Prediction', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Parameters
    ----------
    model : Any
        Trained model with predict_proba method
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        ROC curve figure
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model does not support predict_proba")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: list,
    top_n: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance for tree-based models.
    
    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, optional
        Show only top N features
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Feature importance figure
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    if top_n:
        indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(indices) * 0.4)))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(indices)))[::-1]
    
    bars = ax.barh(
        range(len(indices)),
        importances[indices],
        color=colors
    )
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Feature Importance - Churn Prediction Model', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, imp in zip(bars, importances[indices]):
        ax.text(imp + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Parameters
    ----------
    model : Any
        Trained model with predict_proba method
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Precision-recall curve figure
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model does not support predict_proba")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='#E63946', lw=2,
            label=f'Precision-Recall Curve (AP = {avg_precision:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='#E63946')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Precision-recall curve saved to {save_path}")
    
    return fig