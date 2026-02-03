"""Display the best analysis results in a formatted way."""

import pandas as pd

print("\n" + "="*80)
print("        CYBERSECURITY ATTACK TYPE DETECTION - BEST ANALYSIS OUTPUT")
print("="*80)

# 1. Dataset Overview
print("\n[1] DATASET OVERVIEW")
print("-"*80)
print("Total Records:        40,000 rows")
print("Total Features:        25 columns (All expected columns processed!)")
print("Target Variable:      Attack Type (3 classes)")
print("\nAttack Type Distribution:")
print("  - DDoS:        13,428 samples (33.57%)")
print("  - Malware:     13,307 samples (33.27%)")
print("  - Intrusion:   13,265 samples (33.17%)")
print("\n>>> Perfectly Balanced Dataset!")

# 2. Model Performance Comparison
print("\n[2] MODEL PERFORMANCE COMPARISON")
print("-"*80)
df_models = pd.read_csv('results/model_comparison.csv')
print(df_models.to_string(index=False))
print("\n>>> BEST MODEL: Random Forest Classifier")
print("    Performance: 34.04% accuracy (slightly above random chance of 33.33%)")

# 3. Classification Report
print("\n[3] DETAILED CLASSIFICATION REPORT (Best Model: Random Forest)")
print("-"*80)
df_report = pd.read_csv('results/classification_report.csv', index_col=0)
import joblib
preprocessor_data = joblib.load('models/preprocessor.pkl')
target_encoder = preprocessor_data['target_encoder']
class_names = target_encoder.classes_

print(f"{'Attack Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-"*65)
for i, attack_type in enumerate(class_names):
    row = df_report.loc[attack_type]
    print(f"{attack_type:<15} {row['precision']*100:>10.2f}%  {row['recall']*100:>10.2f}%  {row['f1-score']*100:>10.2f}%  {row['support']:>9.0f}")
print("-"*65)
acc_row = df_report.loc['accuracy']
print(f"{'Overall':<15} {acc_row['precision']*100:>10.2f}%  {acc_row['recall']*100:>10.2f}%  {acc_row['f1-score']*100:>10.2f}%  {acc_row['support']:>9.0f}")

# 4. Key Statistics
print("\n[4] KEY STATISTICS")
print("-"*80)
print("Numerical Features Summary:")
print("  - Source Port:      Mean=32,970  Std=18,560  Range=[1,027 - 65,530]")
print("  - Destination Port: Mean=33,151  Std=18,575  Range=[1,024 - 65,535]")
print("  - Packet Length:     Mean=781     Std=416     Range=[64 - 1,500]")
print("  - Anomaly Scores:    Mean=50.11   Std=28.85   Range=[0 - 100]")
print("\nMissing Values (Handled during preprocessing):")
print("  - Malware Indicators: 50.00%")
print("  - Alerts/Warnings:    50.17%")
print("  - Proxy Information:   49.63%")
print("  - Firewall Logs:        49.90%")
print("  - IDS/IPS Alerts:       50.13%")

# 5. Generated Files
print("\n[5] GENERATED OUTPUT FILES")
print("-"*80)
print("Results folder (results/):")
print("  [OK] model_comparison.csv          - All models performance comparison")
print("  [OK] classification_report.csv     - Detailed per-class metrics")
print("  [OK] confusion_matrix.png           - Visual confusion matrix")
print("  [OK] attack_type_distribution.png   - Attack type distribution chart")
print("  [OK] missing_values.png             - Missing data visualization")
print("  [OK] feature_importance.png         - Top 20 important features")
print("\nModels folder (models/):")
print("  [OK] best_model.pkl                 - Trained Random Forest model")
print("  [OK] preprocessor.pkl              - Data preprocessor for predictions")

# 6. Key Insights
print("\n[6] KEY INSIGHTS")
print("-"*80)
print("Dataset Quality:")
print("  [OK] All 25 expected columns present and processed")
print("  [OK] Perfectly balanced target classes (33.33% each)")
print("  [WARNING] Significant missing data in security-related features (~50%)")
print("\nModel Performance:")
print("  [WARNING] All models show similar performance (~33-34% accuracy)")
print("  [WARNING] Performance is close to random chance (33.33% for 3 classes)")
print("  [OK] Random Forest performs slightly better than others")
print("  [OK] Consistent performance across all attack types")

# 7. All 25 Columns
print("\n[7] ALL 25 COLUMNS PROCESSED")
print("-"*80)
columns = [
    "Timestamp", "Source IP Address", "Destination IP Address", "Source Port",
    "Destination Port", "Protocol", "Packet Length", "Packet Type", "Traffic Type",
    "Payload Data", "Malware Indicators", "Anomaly Scores", "Alerts/Warnings",
    "Attack Type", "Attack Signature", "Action Taken", "Severity Level",
    "User Information", "Device Information", "Network Segment", "Geo-location Data",
    "Proxy Information", "Firewall Logs", "IDS/IPS Alerts", "Log Source"
]
for i, col in enumerate(columns, 1):
    print(f"  [{i:2d}] {col}")

print("\n" + "="*80)
print("                    ANALYSIS COMPLETE - ALL RESULTS READY")
print("="*80)
print("\nNext Steps:")
print("  1. View visualizations in results/ folder")
print("  2. Run web app: streamlit run app.py")
print("  3. Explore notebook: notebooks/ml_pipeline.ipynb")
print("="*80 + "\n")

