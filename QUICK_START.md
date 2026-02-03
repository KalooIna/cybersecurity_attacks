# Quick Start Guide

## 🚀 Fastest Way to Get Results

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add Your Dataset

Copy your dataset file to the `data/` folder:
- File name can be: `cybersecurity_attacks.csv`, `cybersecurity_attacks.xlsx`, etc.
- Or run: `python setup_data.py` to automatically copy from Downloads

### Step 3: Run Analysis
```bash
python quick_analysis.py
```

This single command will:
✅ Load and analyze your dataset  
✅ Perform Exploratory Data Analysis (EDA)  
✅ Preprocess all 25 features  
✅ Train 6 different ML models  
✅ Compare model performance  
✅ Save the best model  
✅ Generate all visualizations and reports  

### Step 4: View Results

Check the `results/` folder for:
- `model_comparison.csv` - Performance comparison of all models
- `classification_report.csv` - Detailed classification metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `attack_type_distribution.png` - Distribution of attack types
- `feature_importance.png` - Top important features
- `missing_values.png` - Missing data analysis

Check the `models/` folder for:
- `best_model.pkl` - Trained model ready for predictions
- `preprocessor.pkl` - Data preprocessor for new data

### Step 5: Run Web App (Optional)
```bash
streamlit run app.py
```

Then open your browser to the URL shown (usually http://localhost:8501)

## 📊 Expected Dataset Format

Your dataset should contain these 25 columns (or similar):
1. Timestamp
2. Source IP Address
3. Destination IP Address
4. Source Port
5. Destination Port
6. Protocol
7. Packet Length
8. Packet Type
9. Traffic Type
10. Payload Data
11. Malware Indicators
12. Anomaly Scores
13. Alerts/Warnings
14. Attack Type (target variable)
15. Attack Signature
16. Action Taken
17. Severity Level
18. User Information
19. Device Information
20. Network Segment
21. Geo-location Data
22. Proxy Information
23. Firewall Logs
24. IDS/IPS Alerts
25. Log Source

## ⚠️ Troubleshooting

**Error: Dataset file not found**
- Make sure your CSV/Excel file is in the `data/` folder
- Check the file name matches expected patterns

**Error: Models not found**
- Run `python quick_analysis.py` first to train the models

**Error: Missing columns**
- The script will work with available columns
- Missing expected columns will be noted in the output

## 📝 Next Steps

After running the analysis:
1. Review the results in the `results/` folder
2. Check model performance in `model_comparison.csv`
3. Use the web app for new predictions
4. Modify the notebook for custom analysis

