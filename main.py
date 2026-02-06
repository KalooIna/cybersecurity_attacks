"""
Main pipeline orchestrator for cybersecurity attack type detection.
This script orchestrates the full ML pipeline from data loading to model evaluation.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import load_dataset, get_column_info
from preprocessing import DataPreprocessor
from model import ModelTrainer
from features import engineer_features
from utils import display_data_info, plot_confusion_matrix, plot_feature_importance

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

def main():
    """Orchestrate the complete ML pipeline."""
    print("="*80)
    print("CYBERSECURITY ATTACK TYPE DETECTION - ML PIPELINE")
    print("="*80)
    
    # Step 1: Load Data
    print("\n[1/7] Loading dataset...")
    try:
        df = load_dataset()
        print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        print("\nPlease place your dataset file in the 'data/' folder.")
        print("Supported formats: CSV, Excel (.xlsx, .xls)")
        sys.exit(1)
    
    # Step 2: Exploratory Data Analysis
    print("\n[2/7] Performing Exploratory Data Analysis...")
    print("\n" + "-"*80)
    display_data_info(df)
    
    # Check columns
    col_info = get_column_info(df)
    print(f"\n[OK] Found {len(col_info['actual'])} columns in dataset")
    if col_info['missing']:
        print(f"[WARNING] Missing expected columns: {col_info['missing']}")
    if col_info['extra']:
        print(f"[INFO] Extra columns: {col_info['extra']}")
    
    # Visualize target distribution
    if 'Attack Type' in df.columns:
        plt.figure(figsize=(12, 6))
        attack_counts = df['Attack Type'].value_counts()
        plt.bar(range(len(attack_counts)), attack_counts.values, color='steelblue')
        plt.xticks(range(len(attack_counts)), attack_counts.index, rotation=45, ha='right')
        plt.xlabel('Attack Type', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Attack Types', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/attack_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Attack type distribution saved to results/attack_type_distribution.png")
        print(f"  Unique attack types: {df['Attack Type'].nunique()}")
        print(f"  Distribution:\n{attack_counts}")
    
    # Missing values visualization
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if len(missing_data) > 0:
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(missing_data)), missing_data.values, color='coral')
        plt.yticks(range(len(missing_data)), missing_data.index)
        plt.xlabel('Number of Missing Values', fontsize=12)
        plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Missing values visualization saved to results/missing_values.png")
    
    # Step 3: Feature Engineering
    print("\n[3/7] Engineering features...")
    try:
        df = engineer_features(df, payload_col='Payload Data')
        print(f"[OK] Feature engineering complete: {df.shape[1]} features")
    except Exception as e:
        print(f"[WARNING] Feature engineering failed: {e}")
        print("Continuing without additional features...")
    
    # Step 4: Data Preprocessing
    print("\n[4/7] Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    if 'Attack Type' not in df.columns:
        print("[ERROR] Error: 'Attack Type' column not found in dataset!")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    X, y = preprocessor.preprocess(df, target_col='Attack Type', fit=True)
    print(f"[OK] Preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[OK] Number of classes: {len(np.unique(y))}")
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    print("[OK] Preprocessor saved to models/preprocessor.pkl")
    
    # Step 5: Train-Test Split
    print("\n[5/7] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Training set: {X_train.shape[0]} samples")
    print(f"[OK] Validation set: {X_val.shape[0]} samples")
    
    # Step 6: Model Training
    print("\n[6/7] Training models...")
    print("This may take a few minutes...\n")
    
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Display results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    results_df = trainer.get_results_summary()
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('results/model_comparison.csv', index=False)
    print(f"\n[OK] Results saved to results/model_comparison.csv")
    
    # Step 7: Model Evaluation
    print("\n[7/7] Evaluating best model...")
    best_model, best_model_name = trainer.get_best_model()
    y_pred = trainer.results[best_model_name]['predictions']
    
    # Classification report
    class_names = preprocessor.target_encoder.classes_
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION REPORT - {best_model_name}")
    print(f"{'='*80}\n")
    report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('results/classification_report.csv')
    print(f"\n[OK] Classification report saved to results/classification_report.csv")
    
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_val, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved to results/confusion_matrix.png")
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        feature_names = preprocessor.feature_columns
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top 20 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Feature importance saved to results/feature_importance.png")
    
    # Save best model
    trainer.save_best_model('models/best_model.pkl')
    print(f"\n[OK] Best model ({best_model_name}) saved to models/best_model.pkl")
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"  Accuracy: {trainer.results[best_model_name]['accuracy']:.4f}")
    print(f"  Precision: {trainer.results[best_model_name]['precision']:.4f}")
    print(f"  Recall: {trainer.results[best_model_name]['recall']:.4f}")
    print(f"  F1-Score: {trainer.results[best_model_name]['f1_score']:.4f}")
    
    print(f"\n[RESULTS] Results saved in 'results/' folder:")
    print(f"  - model_comparison.csv")
    print(f"  - classification_report.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - attack_type_distribution.png")
    if len(missing_data) > 0:
        print(f"  - missing_values.png")
    if hasattr(best_model, 'feature_importances_'):
        print(f"  - feature_importance.png")
    
    print(f"\n[MODELS] Models saved in 'models/' folder:")
    print(f"  - best_model.pkl")
    print(f"  - preprocessor.pkl")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

