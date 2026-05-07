"""
Data Explorer Module

Analyzes data and provides insights, recommendations for modeling strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


class DataExplorer:
    """
    Analyze data distributions, correlations, and provide recommendations.
    
    Attributes:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name (if specified)
    """
    
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """
        Initialize DataExplorer.
        
        Args:
            df: Input dataframe
            target_col: Target column name (optional)
        """
        self.df = df
        self.target_col = target_col
        self.insights = {}
    
    def analyze(self) -> Dict:
        """
        Run full data analysis.
        
        Returns:
            Dictionary with all analysis results
        """
        analysis = {
            'dataset_info': self._dataset_info(),
            'missing_analysis': self._missing_analysis(),
            'numeric_analysis': self._numeric_analysis(),
            'categorical_analysis': self._categorical_analysis(),
            'correlations': self._correlation_analysis(),
            'target_analysis': self._target_analysis() if self.target_col else None,
        }
        
        self.insights = analysis
        return analysis
    
    def _dataset_info(self) -> Dict:
        """Get basic dataset information."""
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        return {
            'shape': self.df.shape,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_columns': list(numeric_cols),
            'categorical_columns': list(categorical_cols),
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
        }
    
    def _missing_analysis(self) -> Dict:
        """Analyze missing values."""
        missing_info = {}
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percent': missing_pct,
                }
        
        total_missing_pct = (self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        
        return {
            'columns_with_missing': missing_info,
            'total_missing_percent': total_missing_pct,
            'rows_with_any_missing': self.df.isna().any(axis=1).sum(),
        }
    
    def _numeric_analysis(self) -> Dict:
        """Analyze numeric columns."""
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            return {}
        
        analysis = {}
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                analysis[col] = {
                    'dtype': str(self.df[col].dtype),
                    'count': len(col_data),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'skewness': float(skew(col_data)),
                    'kurtosis': float(kurtosis(col_data)),
                    'zeros_count': (col_data == 0).sum(),
                    'negative_count': (col_data < 0).sum(),
                }
        
        return analysis
    
    def _categorical_analysis(self) -> Dict:
        """Analyze categorical columns."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            return {}
        
        analysis = {}
        for col in categorical_cols:
            col_data = self.df[col].dropna()
            unique_count = col_data.nunique()
            analysis[col] = {
                'dtype': str(self.df[col].dtype),
                'count': len(col_data),
                'unique': unique_count,
                'unique_percent': (unique_count / len(col_data)) * 100 if len(col_data) > 0 else 0,
                'most_common': col_data.value_counts().head(5).to_dict(),
            }
        
        return analysis
    
    def _correlation_analysis(self) -> Dict:
        """Analyze correlations between numeric features."""
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                    })
        
        return {
            'high_correlations': high_corr,
            'correlation_matrix_shape': corr_matrix.shape,
        }
    
    def _target_analysis(self) -> Dict:
        """Analyze target column (if specified)."""
        if self.target_col not in self.df.columns:
            return {'error': f'Target column {self.target_col} not found'}
        
        target = self.df[self.target_col]
        analysis = {
            'column': self.target_col,
            'dtype': str(target.dtype),
            'missing': target.isna().sum(),
        }
        
        if pd.api.types.is_numeric_dtype(target.dtype):
            # Check if it's actually categorical (binary/multiclass with few unique values)
            unique_count = target.nunique()
            # If unique values equal or nearly equal sample count, it's regression
            uniqueness_ratio = unique_count / len(target)
            if unique_count <= 10 and unique_count > 1 and uniqueness_ratio <= 0.5:
                # Classification: few unique values, not all unique
                analysis['problem_type'] = 'Classification'
                analysis['unique_classes'] = unique_count
                analysis['class_distribution'] = target.value_counts().to_dict()
                analysis['class_balance'] = {
                    'most_common_pct': (target.value_counts().max() / len(target)) * 100,
                    'least_common_pct': (target.value_counts().min() / len(target)) * 100,
                    'imbalance_ratio': target.value_counts().max() / target.value_counts().min(),
                }
            else:
                # Regression target
                clean_target = target.dropna()
                analysis['problem_type'] = 'Regression'
                analysis['stats'] = {
                    'mean': float(clean_target.mean()),
                    'std': float(clean_target.std()),
                    'min': float(clean_target.min()),
                    'max': float(clean_target.max()),
                }
        else:
            # String/Object target - always classification
            unique_count = target.nunique()
            analysis['problem_type'] = 'Classification'
            analysis['unique_classes'] = unique_count
            analysis['class_distribution'] = target.value_counts().to_dict()
            analysis['class_balance'] = {
                'most_common_pct': (target.value_counts().max() / len(target)) * 100,
                'least_common_pct': (target.value_counts().min() / len(target)) * 100,
                'imbalance_ratio': target.value_counts().max() / target.value_counts().min(),
            }
        
        return analysis
    
    def recommend_models(self) -> List[Dict]:
        """
        Recommend models based on data characteristics.
        
        Returns:
            List of recommended models with reasoning
        """
        if not self.target_col:
            return []
        
        target = self.df[self.target_col]
        recommendations = []
        
        # Determine problem type
        if pd.api.types.is_numeric_dtype(target.dtype):
            problem_type = 'regression'
        else:
            problem_type = 'classification'
        
        # Get dataset characteristics
        n_samples = len(self.df)
        n_features = len(self.df.select_dtypes(include=np.number).columns)
        
        # Recommend based on characteristics
        if problem_type == 'regression':
            recommendations.append({
                'model': 'LinearRegression',
                'reason': 'Fast baseline model for regression',
                'pros': ['Interpretable', 'Fast training'],
                'cons': ['Assumes linearity'],
            })
            recommendations.append({
                'model': 'RandomForest',
                'reason': 'Handles non-linear patterns, robust to outliers',
                'pros': ['Non-linear', 'Feature importance', 'No scaling needed'],
                'cons': ['Less interpretable', 'Slower inference'],
            })
            recommendations.append({
                'model': 'GradientBoosting',
                'reason': 'Often best performance, sequential boosting',
                'pros': ['High performance', 'Non-linear', 'Feature importance'],
                'cons': ['Slower training', 'Hyperparameter tuning needed'],
            })
            
            if n_samples > 100000:
                recommendations.append({
                    'model': 'XGBoost',
                    'reason': 'Optimized for large datasets',
                    'pros': ['Fast on large data', 'GPU support', 'Regularization'],
                    'cons': ['Complex', 'Memory usage'],
                })
        else:
            # Classification
            recommendations.append({
                'model': 'LogisticRegression',
                'reason': 'Fast interpretable baseline for classification',
                'pros': ['Interpretable', 'Fast', 'Probabilistic'],
                'cons': ['Linear only', 'May underfit'],
            })
            recommendations.append({
                'model': 'RandomForest',
                'reason': 'Robust ensemble method for classification',
                'pros': ['Non-linear', 'Feature importance', 'Handles imbalance'],
                'cons': ['Less interpretable', 'Memory usage'],
            })
            recommendations.append({
                'model': 'GradientBoosting',
                'reason': 'Highest performance via sequential boosting',
                'pros': ['High accuracy', 'Feature importance', 'Non-linear'],
                'cons': ['Hyperparameter tuning needed', 'Slower training'],
            })
            
            # Recommend XGBoost for imbalanced classification
            class_dist = target.value_counts()
            imbalance_ratio = class_dist.max() / class_dist.min()
            if imbalance_ratio > 3:
                recommendations.append({
                    'model': 'XGBoost',
                    'reason': 'Better handling of imbalanced classification',
                    'pros': ['Scale_pos_weight for imbalance', 'Fast', 'GPU support'],
                    'cons': ['Hyperparameter tuning complex'],
                })
        
        return recommendations
    
    def recommend_preprocessing(self) -> Dict:
        """Recommend preprocessing steps based on data."""
        recommendations = {
            'scaling': [],
            'encoding': [],
            'missing_value_handling': [],
        }
        
        # Check for numeric features that need scaling
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            recommendations['scaling'].append('StandardScaler')
            recommendations['scaling'].append('MinMaxScaler')
        
        # Check for categorical features
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            recommendations['encoding'].append('OneHotEncoder')
            recommendations['encoding'].append('LabelEncoder')
        
        # Check for missing values
        missing_info = self._missing_analysis()
        if missing_info['columns_with_missing']:
            recommendations['missing_value_handling'].append('Drop rows/columns')
            recommendations['missing_value_handling'].append('Mean/Median imputation')
            recommendations['missing_value_handling'].append('Forward/Backward fill (for time-series)')
        
        return recommendations
