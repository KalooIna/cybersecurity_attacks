"""
Model training and evaluation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def initialize_models(self):
        """Initialize multiple models for comparison."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train all models and evaluate them."""
        self.initialize_models()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # Find best model based on F1-score
        self.best_model_name = max(self.results.keys(), 
                                  key=lambda x: self.results[x]['f1_score'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name} "
              f"(F1-Score: {self.results[self.best_model_name]['f1_score']:.4f})")
    
    def get_best_model(self):
        """Return the best performing model."""
        return self.best_model, self.best_model_name
    
    def get_results_summary(self):
        """Get summary of all model results."""
        summary = []
        for name, result in self.results.items():
            summary.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}"
            })
        return pd.DataFrame(summary)
    
    def save_best_model(self, filepath):
        """Save the best model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"Best model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a model from disk."""
        return joblib.load(filepath)

