"""
Data preprocessing utilities for cybersecurity attacks dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class DataPreprocessor:
    """Handles data preprocessing for the cybersecurity dataset."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
    
    def identify_column_types(self, df):
        """Identify categorical and numerical columns."""
        categorical = []
        numerical = []
        
        for col in df.columns:
            if col == 'Attack Type':  # Target variable
                continue
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                categorical.append(col)
            else:
                numerical.append(col)
        
        self.categorical_columns = categorical
        self.numerical_columns = numerical
        return categorical, numerical
    
    def handle_missing_values(self, df, strategy='median'):
        """Handle missing values in the dataset."""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if col == 'Attack Type':
                # For target variable, drop rows with missing values
                df_clean = df_clean.dropna(subset=[col])
                continue
            
            if df_clean[col].isna().sum() > 0:
                if df_clean[col].dtype == 'object':
                    # For categorical, use mode
                    mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_value, inplace=True)
                else:
                    # For numerical, use median or mean
                    if strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        return df_clean
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using Label Encoding."""
        df_encoded = df.copy()
        
        categorical_cols = self.categorical_columns or [col for col in df.columns 
                                                         if df[col].dtype == 'object' and col != 'Attack Type']
        
        for col in categorical_cols:
            if col == 'Attack Type':
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df_encoded[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        # Replace unknown with most frequent
                        df_encoded[col] = df_encoded[col].astype(str).replace(
                            list(unknown_values), 
                            self.label_encoders[col].classes_[0]
                        )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def encode_target(self, y, fit=True):
        """Encode target variable."""
        if fit:
            return self.target_encoder.fit_transform(y)
        else:
            return self.target_encoder.transform(y)
    
    def scale_features(self, X, fit=True):
        """Scale numerical features."""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def preprocess(self, df, target_col='Attack Type', fit=True):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
        fit : bool
            Whether to fit transformers (True for training, False for prediction)
        
        Returns:
        --------
        X : np.ndarray
            Preprocessed features
        y : np.ndarray
            Encoded target (if fit=True and target_col exists)
        """
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Identify column types
        if fit:
            self.identify_column_types(df_processed)
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=fit)
        
        # Separate features and target
        if target_col in df_processed.columns:
            y = df_processed[target_col].values
            X = df_processed.drop(columns=[target_col])
            
            # Encode target
            y_encoded = self.encode_target(y, fit=fit)
        else:
            X = df_processed
            y_encoded = None
        
        # Store feature columns
        if fit:
            self.feature_columns = list(X.columns)
        
        # Scale numerical features
        X_scaled = self.scale_features(X, fit=fit)
        
        if y_encoded is not None:
            return X_scaled, y_encoded
        else:
            return X_scaled
    
    def save(self, filepath):
        """Save preprocessor to disk."""
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }, filepath)
    
    def load(self, filepath):
        """Load preprocessor from disk."""
        data = joblib.load(filepath)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.target_encoder = data['target_encoder']
        self.feature_columns = data['feature_columns']
        self.categorical_columns = data['categorical_columns']
        self.numerical_columns = data['numerical_columns']





