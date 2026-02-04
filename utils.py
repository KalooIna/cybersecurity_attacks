"""
Utility Functions Module
Plotting, logging, metrics, and helper functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway


def setup_plotting_style():
    """Set up consistent plotting style across all visualizations"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 10


def plot_attack_distribution(df, column='Attack Type'):
    """
    Plot the distribution of attack types
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name containing attack types
    """
    if column not in df.columns:
        print(f"⚠️  Column '{column}' not found")
        return
    
    plt.figure(figsize=(12, 6))
    df[column].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(f'{column} Distribution', fontsize=14, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{column} Distribution:")
    print(df[column].value_counts())
    print(f"\nPercentages:")
    print((df[column].value_counts() / len(df) * 100).round(2))


def plot_proxy_analysis(df):
    """
    Comprehensive proxy usage visualization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with proxy features
    """
    if 'has_proxy' not in df.columns:
        print("⚠️  'has_proxy' feature not found. Run feature engineering first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall proxy usage pie chart
    proxy_counts = df['has_proxy'].value_counts()
    labels = ['No Proxy', 'With Proxy']
    colors = ['lightcoral', 'lightgreen']
    axes[0, 0].pie(proxy_counts.values, labels=labels, autopct='%1.1f%%', 
                    colors=colors, startangle=90)
    axes[0, 0].set_title('Overall Proxy Usage Distribution', fontsize=14, fontweight='bold')
    
    # 2. Proxy usage by Attack Type
    if 'Attack Type' in df.columns:
        proxy_attack = pd.crosstab(df['Attack Type'], df['has_proxy'], normalize='index') * 100
        proxy_attack.plot(kind='bar', ax=axes[0, 1], stacked=False, 
                         color=['lightcoral', 'lightgreen'])
        axes[0, 1].set_title('Proxy Usage by Attack Type (%)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Attack Type')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].legend(['No Proxy', 'With Proxy'])
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Proxy usage by Severity Level
    if 'Severity Level' in df.columns:
        proxy_severity = pd.crosstab(df['Severity Level'], df['has_proxy'])
        proxy_severity.plot(kind='bar', ax=axes[1, 0], color=['lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Proxy Usage by Severity Level', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Severity Level')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(['No Proxy', 'With Proxy'])
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Proxy by Log Source
    if 'Log Source' in df.columns:
        log_proxy = pd.crosstab(df['Log Source'], df['has_proxy'], normalize='index') * 100
        log_proxy.plot(kind='bar', ax=axes[1, 1], color=['lightcoral', 'lightgreen'])
        axes[1, 1].set_title('Proxy Usage: Firewall vs Server', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Log Source')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].legend(['No Proxy', 'With Proxy'])
        axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()


def plot_ip_analysis(df):
    """
    Visualize IP-related patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with IP features
    """
    if 'src_ip_class' not in df.columns:
        print("⚠️  IP features not found. Run feature engineering first.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Source IP class distribution
    top_20_src_classes = df['src_ip_class'].value_counts().head(20)
    axes[0].bar(range(len(top_20_src_classes)), top_20_src_classes.values, 
                color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(top_20_src_classes)))
    axes[0].set_xticklabels(top_20_src_classes.index, rotation=45)
    axes[0].set_xlabel('IP Class (First Octet)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Top 20 Source IP Classes', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # IP class vs attack type heatmap
    if 'Attack Type' in df.columns:
        ip_attack_matrix = pd.crosstab(
            df['src_ip_class'], 
            df['Attack Type'], 
            normalize='index'
        ) * 100
        
        # Get top 15 IP classes for readability
        top_15_classes = df['src_ip_class'].value_counts().head(15).index
        ip_attack_subset = ip_attack_matrix.loc[top_15_classes]
        
        sns.heatmap(ip_attack_subset, annot=True, fmt='.1f', cmap='YlOrRd', 
                    ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_xlabel('Attack Type', fontsize=12)
        axes[1].set_ylabel('IP Class (First Octet)', fontsize=12)
        axes[1].set_title('Attack Type Distribution by IP Class (%)', 
                         fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_protocol_analysis(df):
    """
    Visualize protocol distribution and patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    if 'Protocol' not in df.columns:
        print("⚠️  'Protocol' column not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    protocol_dist = df['Protocol'].value_counts()
    
    # Protocol pie chart
    axes[0].pie(protocol_dist.values, labels=protocol_dist.index, 
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Protocol Distribution', fontsize=14, fontweight='bold')
    
    # Protocol by attack type
    if 'Attack Type' in df.columns:
        protocol_attack = pd.crosstab(df['Attack Type'], df['Protocol'], normalize='index') * 100
        protocol_attack.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis')
        axes[1].set_xlabel('Attack Type', fontsize=12)
        axes[1].set_ylabel('Percentage', fontsize=12)
        axes[1].set_title('Protocol Distribution by Attack Type (%)', fontsize=14, fontweight='bold')
        axes[1].legend(title='Protocol', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_packet_analysis(df):
    """
    Visualize packet length patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    if 'Packet Length' not in df.columns:
        print("⚠️  'Packet Length' column not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    packet_lengths = df['Packet Length'].dropna()
    
    # Histogram
    axes[0, 0].hist(packet_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Packet Length (bytes)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency (log scale)', fontsize=11)
    axes[0, 0].set_title('Packet Length Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Binned distribution
    if 'packet_length_bin' in df.columns:
        packet_bin_dist = df['packet_length_bin'].value_counts().sort_index()
        packet_bin_dist.plot(kind='bar', ax=axes[0, 1], color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('Packet Length Bins', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('Packet Length Binned Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Box plot by attack type
    if 'Attack Type' in df.columns:
        df.boxplot(column='Packet Length', by='Attack Type', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Attack Type', fontsize=11)
        axes[1, 0].set_ylabel('Packet Length (bytes)', fontsize=11)
        axes[1, 0].set_title('Packet Length by Attack Type', fontsize=13, fontweight='bold')
        axes[1, 0].get_figure().suptitle('')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45, ha='right')
    
    # Bins by attack type
    if 'Attack Type' in df.columns and 'packet_length_bin' in df.columns:
        bin_attack = pd.crosstab(df['Attack Type'], df['packet_length_bin'], normalize='index') * 100
        bin_attack.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='tab10')
        axes[1, 1].set_xlabel('Attack Type', fontsize=11)
        axes[1, 1].set_ylabel('Percentage', fontsize=11)
        axes[1, 1].set_title('Packet Length Bins by Attack Type (%)', fontsize=13, fontweight='bold')
        axes[1, 1].legend(title='Packet Size', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def statistical_test_packet_length(df):
    """
    Perform ANOVA test on packet length across attack types
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    if 'Attack Type' not in df.columns or 'Packet Length' not in df.columns:
        print("⚠️  Required columns not found")
        return
    
    attack_types = df['Attack Type'].unique()
    groups = [df[df['Attack Type'] == attack]['Packet Length'].dropna() for attack in attack_types]
    
    f_stat, p_value = f_oneway(*groups)
    
    print("\n" + "="*80)
    print("PACKET LENGTH ANOVA TEST")
    print("="*80)
    print(f"F-statistic: {f_stat:.2f}")
    print(f"p-value: {p_value:.4e}")
    
    if p_value < 0.001:
        print("✓ Packet Length is HIGHLY discriminative across attack types!")
    elif p_value < 0.05:
        print("✓ Packet Length shows significant differences across attack types")
    else:
        print("⚠️  Packet Length may not be strongly discriminative")


def print_comprehensive_summary(df):
    """
    Print a comprehensive summary of the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE DATASET SUMMARY")
    print("="*80)
    
    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 80)
    print(f"Total Records: {len(df):,}")
    print(f"Total Features: {df.shape[1]}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'Attack Type' in df.columns:
        print(f"\n🎯 ATTACK TYPE DISTRIBUTION")
        print("-" * 80)
        attack_dist = df['Attack Type'].value_counts()
        for attack, count in attack_dist.items():
            print(f"  {attack}: {count:,} ({count/len(df)*100:.2f}%)")
    
    print(f"\n🔍 KEY STATISTICS")
    print("-" * 80)
    
    # Proxy
    if 'has_proxy' in df.columns:
        proxy_pct = (df['has_proxy'].sum() / len(df)) * 100
        print(f"  - Proxy Usage Rate: {proxy_pct:.2f}%")
    
    # IPs
    if 'Source IP Address' in df.columns:
        print(f"  - Unique Source IPs: {df['Source IP Address'].nunique():,}")
        print(f"  - Unique Destination IPs: {df['Destination IP Address'].nunique():,}")
    
    # Packet Length
    if 'Packet Length' in df.columns:
        print(f"  - Average Packet Size: {df['Packet Length'].mean():.2f} bytes")
    
    # Protocol
    if 'Protocol' in df.columns:
        top_protocol = df['Protocol'].value_counts().index[0]
        top_protocol_pct = (df['Protocol'].value_counts().values[0] / len(df)) * 100
        print(f"  - Most Common Protocol: {top_protocol} ({top_protocol_pct:.2f}%)")
    
    # Port
    if 'Destination Port' in df.columns:
        top_port = df['Destination Port'].value_counts().index[0]
        top_port_count = df['Destination Port'].value_counts().values[0]
        print(f"  - Most Targeted Port: {top_port} ({top_port_count:,} times)")
