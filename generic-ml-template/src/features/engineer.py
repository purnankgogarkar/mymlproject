"""
Feature Engineering Module

Automatically generates new features through mathematical transformations,
interactions, and polynomial expansions. Supports custom code execution for
domain-specific feature creation.

Classes:
    FeatureEngineer: Main feature engineering class
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering with automatic and custom strategies.
    
    Generates:
    - Mathematical transformations (log, sqrt, square, cube, exp)
    - Interaction features (between numeric columns)
    - Polynomial features (degree 2-3)
    - Ratio features (feature1 / feature2)
    - Custom features via user-defined functions
    
    Attributes:
        data (pd.DataFrame): Input dataset
        numeric_cols (list): Numeric column names
        categorical_cols (list): Categorical column names
        feature_map (dict): Maps generated features to creation method
        generated_features (list): List of newly generated feature names
        
    Example:
        >>> engineer = FeatureEngineer(df)
        >>> engineer.auto_generate_features()
        >>> engineer.interaction_features()
        >>> df_engineered = engineer.get_engineered_data()
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer with data.
        
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
        self.feature_map = {}
        self.generated_features = []
        self.engineering_log = []
    
    def auto_generate_features(self, transformations: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Automatically generate mathematical transformations.
        
        Args:
            transformations (list): List of transformations to apply
                ['log', 'sqrt', 'square', 'cube', 'exp', 'reciprocal', 'abs']
                Default: ['log', 'sqrt', 'square']
            
        Returns:
            FeatureEngineer: Self for method chaining
            
        Raises:
            ValueError: If invalid transformation specified
        """
        valid_transforms = ['log', 'sqrt', 'square', 'cube', 'exp', 'reciprocal', 'abs']
        transforms = transformations or ['log', 'sqrt', 'square']
        
        for transform in transforms:
            if transform not in valid_transforms:
                raise ValueError(f"Transform must be one of {valid_transforms}")
        
        features_created = 0
        
        for col in self.numeric_cols:
            col_data = self.data[col]
            
            if 'log' in transforms:
                # Log transformation (handle negative/zero values)
                if (col_data > 0).all():
                    new_col = f"{col}_log"
                    self.data[new_col] = np.log(col_data)
                    self.feature_map[new_col] = f"log({col})"
                    self.generated_features.append(new_col)
                    features_created += 1
            
            if 'sqrt' in transforms:
                # Sqrt transformation (only for positive)
                if (col_data >= 0).all():
                    new_col = f"{col}_sqrt"
                    self.data[new_col] = np.sqrt(col_data)
                    self.feature_map[new_col] = f"sqrt({col})"
                    self.generated_features.append(new_col)
                    features_created += 1
            
            if 'square' in transforms:
                new_col = f"{col}_square"
                self.data[new_col] = col_data ** 2
                self.feature_map[new_col] = f"square({col})"
                self.generated_features.append(new_col)
                features_created += 1
            
            if 'cube' in transforms:
                new_col = f"{col}_cube"
                self.data[new_col] = col_data ** 3
                self.feature_map[new_col] = f"cube({col})"
                self.generated_features.append(new_col)
                features_created += 1
            
            if 'exp' in transforms:
                # Exponential (clip to avoid overflow)
                new_col = f"{col}_exp"
                self.data[new_col] = np.exp(np.clip(col_data, -500, 500))
                self.feature_map[new_col] = f"exp({col})"
                self.generated_features.append(new_col)
                features_created += 1
            
            if 'reciprocal' in transforms:
                # Reciprocal (avoid division by zero)
                if (col_data != 0).all():
                    new_col = f"{col}_reciprocal"
                    self.data[new_col] = 1 / col_data
                    self.feature_map[new_col] = f"1/{col}"
                    self.generated_features.append(new_col)
                    features_created += 1
            
            if 'abs' in transforms:
                new_col = f"{col}_abs"
                self.data[new_col] = np.abs(col_data)
                self.feature_map[new_col] = f"abs({col})"
                self.generated_features.append(new_col)
                features_created += 1
        
        self.engineering_log.append(f"Generated {features_created} mathematical transformation features")
        return self
    
    def interaction_features(self, columns: Optional[List[str]] = None,
                            max_features: Optional[int] = None) -> 'FeatureEngineer':
        """
        Generate interaction features between numeric columns.
        
        Args:
            columns (list): Specific column pairs to create interactions for (default: all numeric)
            max_features (int): Maximum interactions to create (default: all pairs)
            
        Returns:
            FeatureEngineer: Self for method chaining
        """
        cols = columns if columns else self.numeric_cols
        cols = [c for c in cols if c in self.data.columns]
        
        interaction_count = 0
        pair_count = 0
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                if max_features and interaction_count >= max_features:
                    break
                
                # Multiplication
                new_col = f"{col1}_x_{col2}"
                self.data[new_col] = self.data[col1] * self.data[col2]
                self.feature_map[new_col] = f"{col1} * {col2}"
                self.generated_features.append(new_col)
                interaction_count += 1
                
                # Addition
                new_col = f"{col1}_plus_{col2}"
                self.data[new_col] = self.data[col1] + self.data[col2]
                self.feature_map[new_col] = f"{col1} + {col2}"
                self.generated_features.append(new_col)
                interaction_count += 1
                
                # Division (avoid division by zero)
                if (self.data[col2] != 0).all():
                    new_col = f"{col1}_div_{col2}"
                    self.data[new_col] = self.data[col1] / self.data[col2]
                    self.feature_map[new_col] = f"{col1} / {col2}"
                    self.generated_features.append(new_col)
                    interaction_count += 1
                
                pair_count += 1
            
            if max_features and interaction_count >= max_features:
                break
        
        self.engineering_log.append(f"Generated {interaction_count} interaction features from {pair_count} pairs")
        return self
    
    def polynomial_features(self, degree: int = 2, 
                          columns: Optional[List[str]] = None,
                          include_bias: bool = False) -> 'FeatureEngineer':
        """
        Generate polynomial features up to specified degree.
        
        Args:
            degree (int): Polynomial degree (2 or 3, default: 2)
            columns (list): Specific columns to polynomialize (default: all numeric)
            include_bias (bool): Whether to include bias term
            
        Returns:
            FeatureEngineer: Self for method chaining
            
        Raises:
            ValueError: If degree not in [2, 3]
        """
        if degree not in [2, 3]:
            raise ValueError("Polynomial degree must be 2 or 3")
        
        cols = columns if columns else self.numeric_cols
        cols = [c for c in cols if c in self.data.columns]
        
        if not cols:
            self.engineering_log.append("No numeric columns for polynomial features")
            return self
        
        # Use sklearn's PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(self.data[cols])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(cols)
        
        # Add new polynomial features (skip first if no bias, or first 2 if with bias)
        start_idx = 2 if include_bias else 1
        
        for i in range(start_idx, len(feature_names)):
            new_col = f"poly_{feature_names[i]}"
            self.data[new_col] = X_poly[:, i]
            self.feature_map[new_col] = feature_names[i]
            self.generated_features.append(new_col)
        
        poly_count = len(feature_names) - start_idx
        self.engineering_log.append(f"Generated {poly_count} degree-{degree} polynomial features")
        return self
    
    def ratio_features(self, numerator_cols: Optional[List[str]] = None,
                      denominator_cols: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Generate ratio features between columns.
        
        Args:
            numerator_cols (list): Columns for numerator (default: all numeric)
            denominator_cols (list): Columns for denominator (default: all numeric)
            
        Returns:
            FeatureEngineer: Self for method chaining
        """
        num_cols = numerator_cols if numerator_cols else self.numeric_cols
        den_cols = denominator_cols if denominator_cols else self.numeric_cols
        
        num_cols = [c for c in num_cols if c in self.data.columns]
        den_cols = [c for c in den_cols if c in self.data.columns]
        
        ratio_count = 0
        
        for num_col in num_cols:
            for den_col in den_cols:
                if num_col == den_col:
                    continue
                if (self.data[den_col] == 0).any():
                    continue
                
                new_col = f"{num_col}_ratio_{den_col}"
                self.data[new_col] = self.data[num_col] / self.data[den_col]
                self.feature_map[new_col] = f"{num_col} / {den_col}"
                self.generated_features.append(new_col)
                ratio_count += 1
        
        self.engineering_log.append(f"Generated {ratio_count} ratio features")
        return self
    
    def custom_features(self, feature_functions: Dict[str, Callable]) -> 'FeatureEngineer':
        """
        Generate custom features using user-defined functions.
        
        Args:
            feature_functions (dict): Mapping of feature_name -> function
                Each function takes the dataframe and returns a Series or value
                
        Returns:
            FeatureEngineer: Self for method chaining
            
        Raises:
            Exception: If custom function fails
            
        Example:
            >>> funcs = {
            ...     'age_squared': lambda df: df['age'] ** 2,
            ...     'income_group': lambda df: pd.cut(df['income'], bins=3)
            ... }
            >>> engineer.custom_features(funcs)
        """
        for feature_name, func in feature_functions.items():
            try:
                result = func(self.data)
                
                if isinstance(result, pd.Series):
                    self.data[feature_name] = result
                else:
                    self.data[feature_name] = result
                
                self.feature_map[feature_name] = "custom"
                self.generated_features.append(feature_name)
            except Exception as e:
                self.engineering_log.append(f"WARNING: Custom feature '{feature_name}' failed: {str(e)}")
                continue
        
        self.engineering_log.append(f"Generated {len(feature_functions)} custom features")
        return self
    
    def get_engineered_data(self) -> pd.DataFrame:
        """
        Get the feature-engineered dataframe.
        
        Returns:
            pd.DataFrame: Data with generated features
        """
        return self.data.copy()
    
    def get_feature_map(self) -> Dict[str, str]:
        """
        Get mapping of generated features to their definitions.
        
        Returns:
            dict: Feature name -> definition mapping
        """
        return self.feature_map.copy()
    
    def get_report(self) -> Dict:
        """
        Get feature engineering report.
        
        Returns:
            dict: Report with features generated, definitions, etc.
        """
        return {
            'original_shape': self.original_data.shape,
            'engineered_shape': self.data.shape,
            'features_generated': len(self.generated_features),
            'generated_feature_names': self.generated_features,
            'feature_map': self.feature_map,
            'engineering_steps': self.engineering_log,
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols
        }
    
    def print_report(self) -> None:
        """Print feature engineering report in formatted output."""
        report = self.get_report()
        print("=" * 60)
        print("FEATURE ENGINEERING REPORT")
        print("=" * 60)
        print(f"Original Shape: {report['original_shape']}")
        print(f"Engineered Shape: {report['engineered_shape']}")
        print(f"Features Generated: {report['features_generated']}")
        print(f"\nGenerated Features:")
        for feat in report['generated_feature_names'][:20]:  # Show first 20
            definition = report['feature_map'].get(feat, 'unknown')
            print(f"  - {feat}: {definition}")
        if len(report['generated_feature_names']) > 20:
            print(f"  ... and {len(report['generated_feature_names']) - 20} more")
        print(f"\nEngineering Steps:")
        for step in report['engineering_steps']:
            print(f"  - {step}")
        print("=" * 60)
