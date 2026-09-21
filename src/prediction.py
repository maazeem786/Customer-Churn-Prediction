"""
Prediction Module
-----------------
Inference pipeline for making predictions on new customer data.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional
from datetime import datetime


class ChurnPredictor:
    """
    Production-ready churn prediction pipeline.
    """
    
    def __init__(self, model_path: str, encoder_path: Optional[str] = None):
        """
        Initialize the predictor with a trained model.
        
        Parameters
        ----------
        model_path : str
            Path to the saved model file
        encoder_path : str, optional
            Path to the saved label encoder
        """
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path) if encoder_path else None
        self.feature_names = None
        self._set_feature_names()
    
    def _set_feature_names(self):
        """Extract and set feature names from training."""
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_.tolist()
        elif hasattr(self.model, 'n_features_in_'):
            self.feature_names = [f'feature_{i}' 
                                  for i in range(self.model.n_features_in_)]
    
    def _validate_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Validate and convert input dictionary to DataFrame."""
        df = pd.DataFrame([data])
        
        required_features = self.feature_names or []
        for feat in required_features:
            if feat not in df.columns:
                df[feat] = 0
        
        return df
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn for a single customer.
        
        Parameters
        ----------
        customer_data : Dict[str, Any]
            Customer features
            
        Returns
        -------
        Dict[str, Any]
            Prediction results with probability and risk level
        """
        df = self._validate_input(customer_data)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(df)[0]
            churn_probability = float(proba[1])
            prediction = int(self.model.predict(df)[0])
        else:
            prediction = int(self.model.predict(df)[0])
            churn_probability = float(prediction)
        
        if churn_probability >= 0.7:
            risk_level = 'CRITICAL'
            risk_color = 'red'
        elif churn_probability >= 0.5:
            risk_level = 'HIGH'
            risk_color = 'orange'
        elif churn_probability >= 0.3:
            risk_level = 'MEDIUM'
            risk_color = 'yellow'
        else:
            risk_level = 'LOW'
            risk_color = 'green'
        
        return {
            'prediction': 'Churn' if prediction == 1 else 'Retain',
            'churn_probability': churn_probability,
            'churn_percentage': f"{churn_probability:.1%}",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'timestamp': datetime.now().isoformat(),
            'model_version': getattr(self.model, 'version', '1.0')
        }
    
    def predict_batch(self, customer_data_list: list) -> list:
        """
        Predict churn for multiple customers.
        
        Parameters
        ----------
        customer_data_list : list
            List of customer feature dictionaries
            
        Returns
        -------
        list
            List of prediction results
        """
        return [self.predict(customer) for customer in customer_data_list]
    
    def get_risk_factors(self, customer_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Identify key risk factors contributing to churn.
        
        Parameters
        ----------
        customer_data : Dict[str, Any]
            Customer features
            
        Returns
        -------
        Dict[str, float]
            Risk factors with their contribution scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        df = self._validate_input(customer_data)
        importances = self.model.feature_importances_
        
        risk_factors = {}
        for feat, imp in zip(self.feature_names, importances):
            if feat in df.columns:
                value = df[feat].values[0]
                risk_factors[feat] = {
                    'value': float(value),
                    'importance': float(imp),
                    'contribution': float(value * imp)
                }
        
        sorted_factors = sorted(
            risk_factors.items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )[:5]
        
        return dict(sorted_factors)


def create_sample_predictions(model_path: str, n_samples: int = 5) -> None:
    """
    Create sample predictions for demonstration.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model
    n_samples : int, default=5
        Number of sample predictions to generate
    """
    predictor = ChurnPredictor(model_path)
    
    sample_customers = [
        {
            'tenure': 6, 'MonthlyCharges': 85.0, 'TotalCharges': 510.0,
            'Contract': 'Month-to-month', 'PaymentMethod': 'Electronic check',
            'OnlineSecurity': 'No', 'TechSupport': 'No',
            'InternetService': 'Fiber optic', 'PhoneService': 'Yes'
        },
        {
            'tenure': 48, 'MonthlyCharges': 55.0, 'TotalCharges': 2640.0,
            'Contract': 'Two year', 'PaymentMethod': 'Bank transfer',
            'OnlineSecurity': 'Yes', 'TechSupport': 'Yes',
            'InternetService': 'DSL', 'PhoneService': 'Yes'
        },
        {
            'tenure': 3, 'MonthlyCharges': 110.0, 'TotalCharges': 330.0,
            'Contract': 'Month-to-month', 'PaymentMethod': 'Credit card',
            'OnlineSecurity': 'No', 'TechSupport': 'No',
            'InternetService': 'Fiber optic', 'PhoneService': 'Yes'
        }
    ]
    
    print("=" * 60)
    print("SAMPLE CHURN PREDICTIONS")
    print("=" * 60)
    
    for i, customer in enumerate(sample_customers[:n_samples], 1):
        result = predictor.predict(customer)
        print(f"\nCustomer {i}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Churn Probability: {result['churn_percentage']}")
        print(f"  Risk Level: {result['risk_level']}")