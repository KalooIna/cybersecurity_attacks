"""
Utility functions for the cybersecurity ML project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        return plt.gcf()
    else:
        print("Model does not support feature importance.")
        return None


def display_data_info(df):
    """Display comprehensive information about the dataset."""
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n\nData Types:")
    print(df.dtypes)
    
    print(f"\n\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print(f"\n\nBasic Statistics:")
    print(df.describe())
    
    if 'Attack Type' in df.columns:
        print(f"\n\nTarget Variable Distribution (Attack Type):")
        print(df['Attack Type'].value_counts())
        print(f"\nUnique Attack Types: {df['Attack Type'].nunique()}")
    
    print("=" * 80)





