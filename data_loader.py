"""
Data Loading and Basic Validation Module
Handles CSV loading, initial validation, and basic cleaning
"""

import pandas as pd
import numpy as np


def load_dataset(filepath, verbose=True):
    """
    Load the cybersecurity attacks dataset from CSV
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    verbose : bool
        If True, print dataset info
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Shape: {df.shape}")
        print(f"  - Total Records: {df.shape[0]:,}")
        print(f"  - Total Features: {df.shape[1]}")
    
    return df


def get_missing_value_summary(df):
    """
    Generate comprehensive missing value analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values and distinctness
    """
    missing_df = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Distinct_Count': df.nunique(),
        'Distinct_Percentage': (df.nunique() / len(df)) * 100
    }).sort_values('Missing_Count', ascending=False)
    
    return missing_df


def validate_required_columns(df, required_columns):
    """
    Check if required columns exist in dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    required_columns : list
        List of column names that must be present
        
    Returns:
    --------
    tuple
        (bool: all_present, list: missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        print(f"⚠️  Missing required columns: {missing}")
        return False, missing
    
    return True, []


def get_dataset_info(df):
    """
    Print comprehensive dataset information
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    print(f"\n📊 Basic Stats:")
    print(f"  - Total Records: {len(df):,}")
    print(f"  - Total Features: {df.shape[1]}")
    print(f"  - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📋 Data Types:")
    print(df.dtypes.value_counts())
    
    print(f"\n❌ Missing Values:")
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  - Total missing values: {missing:,}")
        print(f"  - Percentage: {(missing / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    else:
        print("  - No missing values found ✓")
