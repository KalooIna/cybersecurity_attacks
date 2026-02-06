"""
Data loading utility for cybersecurity attacks dataset.
Handles both CSV and Excel file formats.
"""

import pandas as pd
import os
from pathlib import Path


def load_dataset(file_path=None):
    """
    Load the cybersecurity attacks dataset from CSV or Excel file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the dataset file. If None, searches in data/ folder.
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if file_path is None:
        # Search in data folder
        data_dir = Path("data")
        
        # Try common file names
        possible_files = [
            "cybersecurity_attacks.csv",
            "cybersecurity_attacks.xlsx",
            "cybersecurity_attacks (1).csv",
            "cybersecurity_attacks (1).xlsx"
        ]
        
        for filename in possible_files:
            file_path = data_dir / filename
            if file_path.exists():
                break
        else:
            # List available files
            available_files = list(data_dir.glob("*"))
            raise FileNotFoundError(
                f"Dataset file not found. Please place your CSV/Excel file in the 'data/' folder.\n"
                f"Available files in data/: {[f.name for f in available_files]}"
            )
    
    # Load based on file extension
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please use CSV or Excel files.")
    
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_column_info(df):
    """
    Get information about the dataset columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    
    Returns:
    --------
    dict
        Column information including expected columns
    """
    expected_columns = [
        'Timestamp', 'Source IP Address', 'Destination IP Address', 'Source Port',
        'Destination Port', 'Protocol', 'Packet Length', 'Packet Type', 'Traffic Type',
        'Payload Data', 'Malware Indicators', 'Anomaly Scores', 'Alerts/Warnings',
        'Attack Type', 'Attack Signature', 'Action Taken', 'Severity Level',
        'User Information', 'Device Information', 'Network Segment', 'Geo-location Data',
        'Proxy Information', 'Firewall Logs', 'IDS/IPS Alerts', 'Log Source'
    ]
    
    actual_columns = list(df.columns)
    
    return {
        'expected': expected_columns,
        'actual': actual_columns,
        'missing': [col for col in expected_columns if col not in actual_columns],
        'extra': [col for col in actual_columns if col not in expected_columns]
    }





