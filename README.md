# Cyber Security Attack Type Detection

## Project Overview

This project implements an end-to-end machine learning pipeline to detect and classify cybersecurity attack types from network traffic data. The system analyzes 25 different metrics from cybersecurity logs to predict attack types.

## Dataset

The dataset contains 40,000 rows with 25 attributes including:
- Network information (IP addresses, ports, protocols)
- Packet information (length, type, traffic type)
- Security indicators (malware indicators, anomaly scores, alerts)
- Attack information (attack type, signature, severity)
- System information (user, device, network segment, geo-location)
- Security logs (firewall, IDS/IPS, proxy information)

## Project Structure

```
CyberSecurity-ML_Project1/
├── data/
│   └── cybersecurity_attacks.csv (or .xlsx)
├── notebooks/
│   └── ml_pipeline.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   └── utils.py
├── models/
│   └── (trained models will be saved here)
├── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CyberSecurity-ML_Project1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A: Automatic (if dataset is in Downloads folder)**
```bash
python setup_data.py
```

**Option B: Manual**
Place your dataset file (`cybersecurity_attacks.csv` or `.xlsx`) in the `data/` folder.

### 3. Run Quick Analysis (Recommended)

Run the complete ML pipeline and generate all results:
```bash
python quick_analysis.py
```

This will:
- Load and analyze the dataset
- Perform EDA and generate visualizations
- Preprocess the data
- Train multiple models and compare them
- Save the best model and generate evaluation reports
- Create all result files in the `results/` folder

### 4. Run the Web Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The web application allows you to:
- Upload CSV files for batch predictions
- Enter individual data points manually
- View predictions and download results

### 5. Jupyter Notebook (Optional)

For interactive analysis, open the Jupyter notebook:
```bash
jupyter notebook notebooks/ml_pipeline.ipynb
```

## Features

- **Comprehensive EDA**: Visualizations and statistical analysis of all 25 features
- **Feature Engineering**: Handling missing values, encoding categorical variables, feature scaling
- **Multiple Models**: Comparison of various ML algorithms (Random Forest, XGBoost, LightGBM, etc.)
- **Model Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and confusion matrices
- **Web Interface**: User-friendly Streamlit application for real-time predictions

## Models Evaluated

- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- Logistic Regression

## Results

Model performance metrics and comparisons are available in the Jupyter notebook and will be saved in the `models/` directory.

## Contributors

[Your Name/Group Members]

## License

This project is for educational purposes as part of the DSTI Applied MSc program.


