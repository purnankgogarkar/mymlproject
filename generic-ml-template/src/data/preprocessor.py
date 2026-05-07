"""
Data Preprocessing Module

Handles missing values, categorical encoding, feature scaling, and outlier detection.
Supports automatic and manual configuration of preprocessing strategies.

Classes:
    Preprocessor: Main preprocessing class with configurable strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class Preprocessor:
    """
    Comprehensive data preprocessing with automatic strategy detection.
    
    Handles:
    - Missing value imputation (mean, median, mode, drop)
    - Categorical encoding (one-hot, label encoding)
    - Feature scaling (standardization, min-max)
    - Outlier detection and handling (IQR method)
    
    Attributes:
        data (pd.DataFrame): Input dataset
        numeric_cols (list): Numeric column names
        categorical_cols (list): Categorical column names
        scalers (dict): Fitted scalers per column
        encoders (dict): Fitted encoders per column
        imputers (dict): Fitted imputers per column
        outlier_indices (list): Detected outlier row indices
    
    Example:
        >>> preprocessor = Preprocessor(df)
        >>> preprocessor.handle_missing_values(strategy='mean')
        >>> preprocessor.encode_categoricals(method='one-hot')
        >>> preprocessor.scale_features(method='standard')
        >>> df_processed = preprocessor.get_processed_data()
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor with data.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Raises:
            ValueError: If data is empty or not a DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        self.data = data.copy()
        self.original_data = data.copy()
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.outlier_indices = []
        self.missing_value_report = {}
        self.preprocessing_log = []
    
    def handle_missing_values(self, strategy: str = 'auto', 
                             numeric_strategy: str = 'mean',
                             categorical_strategy: str = 'mode') -> 'Preprocessor':
        """
        Handle missing values using specified strategy.
        
        Args:
            strategy (str): 'auto' (auto-detect per column), 'drop', 'mean', 'median', 'mode', 'forward_fill'
            numeric_strategy (str): Strategy for numeric columns ('mean', 'median', 'drop')
            categorical_strategy (str): Strategy for categorical columns ('mode', 'drop', 'unknown')
            
        Returns:
            Preprocessor: Self for method chaining
            
        Raises:
            ValueError: If strategy is not recognized
        """
        valid_strategies = ['auto', 'drop', 'mean', 'median', 'mode', 'forward_fill']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        # Record missing values before imputation
        self.missing_value_report = {
            col: self.data[col].isnull().sum() 
            for col in self.data.columns 
            if self.data[col].isnull().sum() > 0
        }
        
        if strategy == 'drop':
            self.data = self.data.dropna()
            self.preprocessing_log.append(f"Dropped {len(self.original_data) - len(self.data)} rows with missing values")
            
        elif strategy == 'forward_fill':
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            self.preprocessing_log.append("Applied forward fill for missing values")
            
        else:  # auto, mean, median, mode
            for col in self.numeric_cols:
                if self.data[col].isnull().sum() > 0:
                    if strategy == 'auto':
                        strat = numeric_strategy
                    else:
                        strat = strategy
                    
                    imputer = SimpleImputer(strategy=strat)
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                    self.imputers[col] = imputer
            
            for col in self.categorical_cols:
                if self.data[col].isnull().sum() > 0:
                    if strategy == 'auto':
                        strat = categorical_strategy
                    else:
                        strat = strategy
                    
                    if strat == 'drop':
                        self.data = self.data[self.data[col].notna()]
                    elif strat == 'mode':
                        mode_val = self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else 'Unknown'
                        self.data[col] = self.data[col].fillna(mode_val)
                    elif strat == 'unknown':
                        self.data[col] = self.data[col].fillna('Unknown')
        
        self.preprocessing_log.append(f"Missing values handled: {sum(self.missing_value_report.values())} total")
        return self
    
    def encode_categoricals(self, method: str = 'auto', 
                           columns: Optional[List[str]] = None) -> 'Preprocessor':
        """
        Encode categorical variables.
        
        Args:
            method (str): 'one-hot', 'label', 'target', 'auto' (one-hot if <10 unique, label if >=10)
            columns (list): Specific columns to encode (default: all categorical)
            
        Returns:
            Preprocessor: Self for method chaining
            
        Raises:
            ValueError: If method not recognized
        """
        valid_methods = ['one-hot', 'label', 'target', 'auto']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        cols_to_encode = columns if columns else self.categorical_cols
        
        for col in cols_to_encode:
            if col not in self.data.columns:
                continue
            
            unique_count = self.data[col].nunique()
            
            # Decide method if auto
            if method == 'auto':
                encode_method = 'label' if unique_count >= 10 else 'one-hot'
            else:
                encode_method = method
            
            if encode_method == 'one-hot':
                # One-hot encode
                dummies = pd.get_dummies(self.data[col], prefix=col, dtype=int)
                self.data = pd.concat([self.data, dummies], axis=1)
                self.data.drop(col, axis=1, inplace=True)
                self.encoders[col] = {'method': 'one-hot', 'columns': dummies.columns.tolist()}
                
            elif encode_method == 'label':
                # Label encode
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col].astype(str))
                self.encoders[col] = {'method': 'label', 'encoder': encoder}
        
        self.preprocessing_log.append(f"Encoded {len(cols_to_encode)} categorical columns using {method}")
        return self
    
    def scale_features(self, method: str = 'standard', 
                      columns: Optional[List[str]] = None) -> 'Preprocessor':
        """
        Scale numeric features.
        
        Args:
            method (str): 'standard' (z-score), 'minmax' (0-1), 'robust' (IQR-based)
            columns (list): Specific columns to scale (default: all numeric)
            
        Returns:
            Preprocessor: Self for method chaining
            
        Raises:
            ValueError: If method not recognized
        """
        valid_methods = ['standard', 'minmax', 'robust']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        cols_to_scale = columns if columns else self.numeric_cols
        cols_to_scale = [col for col in cols_to_scale if col in self.data.columns]
        
        if method == 'standard':
            scaler = StandardScaler()
            self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
            self.scalers['standard'] = scaler
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
            self.scalers['minmax'] = scaler
            
        elif method == 'robust':
            # Robust scaling: (x - median) / IQR
            for col in cols_to_scale:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    self.data[col] = (self.data[col] - self.data[col].median()) / IQR
        
        self.preprocessing_log.append(f"Scaled {len(cols_to_scale)} numeric columns using {method}")
        return self
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> 'Preprocessor':
        """
        Detect outliers using specified method.
        
        Args:
            method (str): 'iqr' (Interquartile Range), 'zscore' (Z-score >3)
            threshold (float): IQR multiplier (default 1.5)
            
        Returns:
            Preprocessor: Self for method chaining
            
        Raises:
            ValueError: If method not recognized
        """
        valid_methods = ['iqr', 'zscore']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        outlier_mask = pd.Series([False] * len(self.data), index=self.data.index)
        
        for col in self.numeric_cols:
            if col not in self.data.columns:
                continue
            
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                col_outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                col_outliers = z_scores > 3
            
            outlier_mask |= col_outliers
        
        self.outlier_indices = self.data[outlier_mask].index.tolist()
        self.preprocessing_log.append(f"Detected {len(self.outlier_indices)} outliers using {method}")
        return self
    
    def remove_outliers(self) -> 'Preprocessor':
        """
        Remove detected outliers from data.
        
        Returns:
            Preprocessor: Self for method chaining
        """
        if not self.outlier_indices:
            self.preprocessing_log.append("No outliers to remove")
            return self
        
        self.data = self.data.drop(self.outlier_indices)
        self.preprocessing_log.append(f"Removed {len(self.outlier_indices)} outliers")
        return self
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed dataframe.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        return self.data.copy()
    
    def get_report(self) -> Dict:
        """
        Get preprocessing report.
        
        Returns:
            dict: Report with steps taken, missing values, outliers, etc.
        """
        return {
            'original_shape': self.original_data.shape,
            'processed_shape': self.data.shape,
            'missing_values': self.missing_value_report,
            'outliers_detected': len(self.outlier_indices),
            'preprocessing_steps': self.preprocessing_log,
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'encoders': {k: v.get('method', 'unknown') for k, v in self.encoders.items()},
            'scalers': list(self.scalers.keys())
        }
    
    def print_report(self) -> None:
        """Print preprocessing report in formatted output."""
        report = self.get_report()
        print("=" * 60)
        print("PREPROCESSING REPORT")
        print("=" * 60)
        print(f"Original Shape: {report['original_shape']}")
        print(f"Processed Shape: {report['processed_shape']}")
        print(f"\nMissing Values: {report['missing_values']}")
        print(f"Outliers Detected: {report['outliers_detected']}")
        print(f"\nPreprocessing Steps:")
        for step in report['preprocessing_steps']:
            print(f"  - {step}")
        print("=" * 60)
