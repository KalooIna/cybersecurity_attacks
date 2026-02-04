"""
Feature Engineering Module
Functions for creating derived features from raw data
"""

import pandas as pd
import numpy as np


def create_proxy_features(df):
    """
    Create proxy-related features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Proxy Information' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added proxy features
    """
    if 'Proxy Information' in df.columns:
        # Binary feature: has_proxy
        df['has_proxy'] = df['Proxy Information'].notna().astype(int)
        print(f"✓ Created 'has_proxy' feature")
        print(f"  - Records with proxy: {df['has_proxy'].sum():,} ({df['has_proxy'].sum()/len(df)*100:.2f}%)")
    else:
        print("⚠️  'Proxy Information' column not found, skipping proxy features")
    
    return df


def create_ip_features(df):
    """
    Create IP-related features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with IP address columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added IP features
    """
    if 'Source IP Address' not in df.columns or 'Destination IP Address' not in df.columns:
        print("⚠️  IP address columns not found, skipping IP features")
        return df
    
    # IP Class (first octet)
    df['src_ip_class'] = df['Source IP Address'].str.split('.').str[0].astype(int)
    df['dst_ip_class'] = df['Destination IP Address'].str.split('.').str[0].astype(int)
    
    # Private IP detection
    df['src_is_private'] = df['Source IP Address'].apply(is_private_ip)
    df['dst_is_private'] = df['Destination IP Address'].apply(is_private_ip)
    
    # Bidirectional traffic detection
    source_ips_set = set(df['Source IP Address'])
    dest_ips_set = set(df['Destination IP Address'])
    bidirectional_ips = source_ips_set.intersection(dest_ips_set)
    
    df['is_bidirectional'] = (df['Source IP Address'].isin(bidirectional_ips)) | \
                              (df['Destination IP Address'].isin(bidirectional_ips))
    
    print(f"✓ Created IP features:")
    print(f"  - IP class features (src_ip_class, dst_ip_class)")
    print(f"  - Private IP indicators (src_is_private, dst_is_private)")
    print(f"  - Bidirectional traffic indicator")
    print(f"  - Bidirectional IPs found: {len(bidirectional_ips):,}")
    
    return df


def is_private_ip(ip):
    """
    Check if an IP address is in private range (RFC 1918)
    
    Parameters:
    -----------
    ip : str
        IP address string
        
    Returns:
    --------
    bool
        True if IP is private, False otherwise
    """
    if pd.isna(ip):
        return False
    try:
        parts = str(ip).split('.')
        if len(parts) != 4:
            return False
        first = int(parts[0])
        second = int(parts[1])
        
        # Private IP ranges: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
        if first == 10:
            return True
        if first == 172 and 16 <= second <= 31:
            return True
        if first == 192 and second == 168:
            return True
        return False
    except:
        return False


def create_port_features(df):
    """
    Create port-related features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with port columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added port features
    """
    if 'Source Port' in df.columns:
        df['src_port_category'] = df['Source Port'].apply(categorize_port)
    
    if 'Destination Port' in df.columns:
        df['dst_port_category'] = df['Destination Port'].apply(categorize_port)
    
    print(f"✓ Created port category features")
    
    return df


def categorize_port(port):
    """
    Categorize ports into well-known, registered, or dynamic
    
    Parameters:
    -----------
    port : int
        Port number
        
    Returns:
    --------
    str
        Port category
    """
    if pd.isna(port):
        return 'Unknown'
    try:
        port = int(port)
        if 0 <= port <= 1023:
            return 'Well-known (0-1023)'
        elif 1024 <= port <= 49151:
            return 'Registered (1024-49151)'
        elif 49152 <= port <= 65535:
            return 'Dynamic (49152-65535)'
        else:
            return 'Unknown'
    except:
        return 'Unknown'


def create_packet_features(df):
    """
    Create packet-related features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Packet Length' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added packet features
    """
    if 'Packet Length' not in df.columns:
        print("⚠️  'Packet Length' column not found, skipping packet features")
        return df
    
    # Packet size categories
    df['packet_length_bin'] = pd.cut(
        df['Packet Length'], 
        bins=[0, 100, 500, 1000, 1500, float('inf')],
        labels=['Tiny (0-100)', 'Small (100-500)', 'Medium (500-1000)', 
                'Large (1000-1500)', 'Jumbo (>1500)']
    )
    
    print(f"✓ Created packet length bins")
    
    return df


def create_anomaly_features(df):
    """
    Create anomaly score-related features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'Anomaly Scores' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added anomaly features
    """
    if 'Anomaly Scores' not in df.columns:
        print("⚠️  'Anomaly Scores' column not found, skipping anomaly features")
        return df
    
    anomaly_scores = df['Anomaly Scores'].dropna()
    
    # Create quartile-based categories
    anomaly_bins = [
        anomaly_scores.min(), 
        anomaly_scores.quantile(0.25),
        anomaly_scores.quantile(0.5),
        anomaly_scores.quantile(0.75),
        anomaly_scores.max()
    ]
    anomaly_labels = ['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Critical (75-100%)']
    
    df['anomaly_category'] = pd.cut(
        df['Anomaly Scores'], 
        bins=anomaly_bins, 
        labels=anomaly_labels, 
        include_lowest=True
    )
    
    print(f"✓ Created anomaly score categories")
    
    return df


def create_all_features(df):
    """
    Create all engineered features at once
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all features added
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df = create_proxy_features(df)
    df = create_ip_features(df)
    df = create_port_features(df)
    df = create_packet_features(df)
    df = create_anomaly_features(df)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  - New shape: {df.shape}")
    print(f"  - Total features: {df.shape[1]}")
    
    return df
