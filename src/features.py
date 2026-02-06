"""
Feature engineering functions for cybersecurity attacks dataset.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from scipy.stats import entropy


def extract_payload_features(payload_text):
    """
    Extract features from payload data.
    
    Parameters:
    -----------
    payload_text : str
        Payload text to analyze
    
    Returns:
    --------
    dict
        Dictionary of extracted features
    """
    features = {}
    
    if pd.isna(payload_text) or len(str(payload_text)) == 0:
        features['payload_entropy'] = 0
        features['payload_length'] = 0
        features['payload_has_special_chars'] = 0
        features['payload_has_numbers'] = 0
        features['payload_has_letters'] = 0
        return features
    
    payload_str = str(payload_text)
    
    # Entropy calculation (high entropy -> likely obfuscation)
    if len(payload_str) > 0:
        freq = Counter(payload_str)
        features['payload_entropy'] = entropy(list(freq.values()), base=2)
    else:
        features['payload_entropy'] = 0
    
    # Length features
    features['payload_length'] = len(payload_str)
    
    # Character type features
    features['payload_has_special_chars'] = 1 if bool(re.search(r'[^a-zA-Z0-9\s]', payload_str)) else 0
    features['payload_has_numbers'] = 1 if bool(re.search(r'\d', payload_str)) else 0
    features['payload_has_letters'] = 1 if bool(re.search(r'[a-zA-Z]', payload_str)) else 0
    
    return features


def is_lorem_ipsum(text):
    """
    Detect if text is Lorem Ipsum or similar generated text.
    
    Parameters:
    -----------
    text : str
        Text to check
    
    Returns:
    --------
    bool
        True if text appears to be Lorem Ipsum
    """
    if pd.isna(text):
        return False
    
    lorem_markers = [
        'lorem', 'ipsum', 'dolor', 'sit amet', 'consectetur', 
        'adipiscing', 'elit', 'sed do', 'eiusmod', 'tempor',
        'incididunt', 'labore', 'dolore', 'magna', 'aliqua'
    ]
    text_lower = str(text).lower()
    matches = sum(1 for marker in lorem_markers if marker in text_lower)
    return matches >= 3


def extract_timestamp_features(df, timestamp_col='Timestamp'):
    """
    Extract time-based features from timestamp column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp column
    timestamp_col : str
        Name of timestamp column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added time features
    """
    df = df.copy()
    
    if timestamp_col not in df.columns:
        return df
    
    # Convert to datetime if not already
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Extract time features
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def create_interaction_features(df):
    """
    Create interaction features between numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added interaction features
    """
    df = df.copy()
    
    # Example: Packet length to port ratio
    if 'Packet Length' in df.columns and 'Destination Port' in df.columns:
        df['packet_length_to_port_ratio'] = (
            df['Packet Length'] / (df['Destination Port'] + 1)  # +1 to avoid division by zero
        )
    
    # Example: Anomaly score to packet length ratio
    if 'Anomaly Scores' in df.columns and 'Packet Length' in df.columns:
        df['anomaly_to_packet_ratio'] = (
            df['Anomaly Scores'] / (df['Packet Length'] + 1)
        )
    
    return df


def engineer_features(df, payload_col='Payload Data'):
    """
    Apply all feature engineering functions to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    payload_col : str
        Name of payload column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Extract payload features
    if payload_col in df.columns:
        payload_features = df[payload_col].apply(extract_payload_features)
        payload_df = pd.DataFrame(payload_features.tolist())
        df = pd.concat([df, payload_df], axis=1)
    
    # Extract timestamp features
    df = extract_timestamp_features(df)
    
    # Create interaction features
    df = create_interaction_features(df)
    
    return df

