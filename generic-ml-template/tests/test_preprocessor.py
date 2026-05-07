"""
Tests for Data Preprocessing Module

Comprehensive test suite for Preprocessor class covering:
- Missing value handling (mean, median, mode, drop)
- Categorical encoding (one-hot, label)
- Feature scaling (standard, minmax, robust)
- Outlier detection (IQR, zscore)
- Method chaining
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import Preprocessor


class TestPreprocessorInitialization:
    """Test Preprocessor initialization and validation."""
    
    def test_init_with_valid_dataframe(self, sample_dataframe):
        """Test initialization with valid DataFrame."""
        preprocessor = Preprocessor(sample_dataframe)
        assert preprocessor.data is not None
        assert isinstance(preprocessor.data, pd.DataFrame)
    
    def test_init_with_non_dataframe(self):
        """Test initialization with non-DataFrame raises error."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            Preprocessor([1, 2, 3])
    
    def test_init_with_empty_dataframe(self):
        """Test initialization with empty DataFrame raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Preprocessor(pd.DataFrame())
    
    def test_column_detection(self, sample_dataframe):
        """Test numeric and categorical column detection."""
        preprocessor = Preprocessor(sample_dataframe)
        assert len(preprocessor.numeric_cols) > 0
        assert len(preprocessor.categorical_cols) > 0


class TestMissingValueHandling:
    """Test missing value imputation strategies."""
    
    def test_missing_values_mean_strategy(self):
        """Test mean imputation for numeric columns."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, np.nan, 8]
        })
        preprocessor = Preprocessor(df)
        preprocessor.handle_missing_values(strategy='mean')
        
        assert preprocessor.data['A'].isnull().sum() == 0
        assert preprocessor.data['B'].isnull().sum() == 0
        assert 'A' in preprocessor.imputers
    
    def test_missing_values_drop_strategy(self):
        """Test drop strategy for missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, 6, 7, 8]
        })
        original_len = len(df)
        preprocessor = Preprocessor(df)
        preprocessor.handle_missing_values(strategy='drop')
        
        assert len(preprocessor.data) < original_len
        assert preprocessor.data.isnull().sum().sum() == 0
    
    def test_missing_values_forward_fill(self):
        """Test forward fill strategy."""
        df = pd.DataFrame({
            'A': [1, np.nan, np.nan, 4]
        })
        preprocessor = Preprocessor(df)
        preprocessor.handle_missing_values(strategy='forward_fill')
        
        assert preprocessor.data['A'].isnull().sum() == 0
    
    def test_invalid_strategy_raises_error(self, sample_dataframe):
        """Test invalid strategy raises error."""
        preprocessor = Preprocessor(sample_dataframe)
        with pytest.raises(ValueError, match="Strategy must be one of"):
            preprocessor.handle_missing_values(strategy='invalid')
    
    def test_missing_value_report(self):
        """Test missing value report generation."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, np.nan, 6]
        })
        preprocessor = Preprocessor(df)
        preprocessor.handle_missing_values(strategy='drop')
        
        assert isinstance(preprocessor.missing_value_report, dict)


class TestCategoricalEncoding:
    """Test categorical variable encoding."""
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding for categorical variables."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['cat', 'dog', 'cat']
        })
        preprocessor = Preprocessor(df)
        preprocessor.encode_categoricals(method='one-hot')
        
        # One-hot should create new columns
        assert len(preprocessor.data.columns) > len(df.columns)
        assert 'B' not in preprocessor.data.columns  # Original dropped
    
    def test_label_encoding(self):
        """Test label encoding for categorical variables."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['cat', 'dog', 'cat']
        })
        preprocessor = Preprocessor(df)
        preprocessor.encode_categoricals(method='label', columns=['B'])
        
        # Values should be integers
        assert preprocessor.data['B'].dtype in [np.int64, np.int32]
        assert 'B' in preprocessor.encoders
    
    def test_auto_encoding_low_cardinality(self):
        """Test auto encoding uses one-hot for low cardinality."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['cat', 'dog', 'cat']
        })
        preprocessor = Preprocessor(df)
        preprocessor.encode_categoricals(method='auto')
        
        # Low cardinality should use one-hot
        assert len(preprocessor.data.columns) > len(df.columns)
    
    def test_encode_missing_column(self, sample_dataframe):
        """Test encoding non-existent column doesn't raise error."""
        preprocessor = Preprocessor(sample_dataframe)
        # Should not raise error
        preprocessor.encode_categoricals(columns=['nonexistent'])
    
    def test_invalid_encoding_method(self, sample_dataframe):
        """Test invalid encoding method raises error."""
        preprocessor = Preprocessor(sample_dataframe)
        with pytest.raises(ValueError, match="Method must be one of"):
            preprocessor.encode_categoricals(method='invalid')


class TestFeatureScaling:
    """Test feature scaling strategies."""
    
    def test_standard_scaling(self):
        """Test standard (z-score) scaling."""
        df = pd.DataFrame({
            'A': [10.0, 20.0, 30.0, 40.0, 50.0],
            'B': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        preprocessor = Preprocessor(df)
        
        preprocessor.scale_features(method='standard')
        
        # After standard scaling, mean should be ~0, std ~1
        scaled_mean = preprocessor.data['A'].mean()
        assert abs(scaled_mean) < 1e-10
        assert 'standard' in preprocessor.scalers
    
    def test_minmax_scaling(self):
        """Test minmax (0-1) scaling."""
        df = pd.DataFrame({
            'A': [10.0, 20.0, 30.0, 40.0, 50.0],
            'B': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        preprocessor = Preprocessor(df)
        preprocessor.scale_features(method='minmax')
        
        # All values should be between 0 and 1
        for col in preprocessor.numeric_cols:
            assert preprocessor.data[col].min() >= -1e-10
            assert preprocessor.data[col].max() <= 1 + 1e-10
        assert 'minmax' in preprocessor.scalers
    
    def test_robust_scaling(self):
        """Test robust scaling (IQR-based)."""
        df = pd.DataFrame({
            'A': [10.0, 20.0, 30.0, 40.0, 50.0],
            'B': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        preprocessor = Preprocessor(df)
        preprocessor.scale_features(method='robust')
        
        # Data should be scaled but not bounded
        assert preprocessor.data is not None
    
    def test_invalid_scaling_method(self, sample_dataframe):
        """Test invalid scaling method raises error."""
        preprocessor = Preprocessor(sample_dataframe)
        with pytest.raises(ValueError, match="Method must be one of"):
            preprocessor.scale_features(method='invalid')
    
    def test_scale_specific_columns(self):
        """Test scaling specific columns only."""
        df = pd.DataFrame({
            'A': [10.0, 20.0, 30.0, 40.0, 50.0],
            'B': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        preprocessor = Preprocessor(df)
        
        cols_to_scale = ['A']
        preprocessor.scale_features(method='standard', columns=cols_to_scale)
        # Should succeed without error
        assert preprocessor.data is not None


class TestOutlierDetection:
    """Test outlier detection methods."""
    
    def test_iqr_outlier_detection(self):
        """Test IQR-based outlier detection."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 is outlier
            'B': [10, 20, 30, 40, 50, 60]
        })
        preprocessor = Preprocessor(df)
        preprocessor.detect_outliers(method='iqr')
        
        assert len(preprocessor.outlier_indices) > 0
    
    def test_zscore_outlier_detection(self):
        """Test Z-score outlier detection."""
        # Create data with repeated values and extreme outlier
        # Zscore threshold of 3 is conservative; verify method executes without error
        df = pd.DataFrame({
            'A': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50],  # 50 is > 3 sigma from mean of 5s
        })
        preprocessor = Preprocessor(df)
        preprocessor.detect_outliers(method='zscore')
        
        # Verify method runs and detects at least the extreme value
        assert len(preprocessor.outlier_indices) >= 1
    
    def test_invalid_outlier_method(self, sample_dataframe):
        """Test invalid outlier method raises error."""
        preprocessor = Preprocessor(sample_dataframe)
        with pytest.raises(ValueError, match="Method must be one of"):
            preprocessor.detect_outliers(method='invalid')
    
    def test_remove_outliers(self):
        """Test removing detected outliers."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],
        })
        original_len = len(df)
        preprocessor = Preprocessor(df)
        preprocessor.detect_outliers(method='iqr')
        preprocessor.remove_outliers()
        
        assert len(preprocessor.data) < original_len


class TestMethodChaining:
    """Test method chaining functionality."""
    
    def test_chaining_multiple_operations(self):
        """Test chaining multiple preprocessing steps."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, 6, 7, 8],
            'C': ['cat', 'dog', 'cat', 'dog']
        })
        preprocessor = Preprocessor(df)
        result = (preprocessor
                  .handle_missing_values(strategy='mean')
                  .encode_categoricals(method='auto')
                  .scale_features(method='standard'))
        
        # Should return self for chaining
        assert isinstance(result, Preprocessor)
        assert preprocessor.data is not None
    
    def test_chaining_with_outliers(self):
        """Test chaining with outlier detection and removal."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],
        })
        preprocessor = Preprocessor(df)
        result = (preprocessor
                  .detect_outliers(method='iqr')
                  .remove_outliers())
        
        assert isinstance(result, Preprocessor)
        assert len(preprocessor.data) == 5


class TestReportsAndOutput:
    """Test report generation and output."""
    
    def test_get_processed_data(self):
        """Test getting processed data."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        preprocessor = Preprocessor(df)
        processed = preprocessor.get_processed_data()
        
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == len(df)
    
    def test_get_report(self):
        """Test report generation."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan],
            'B': [4, 5, 6]
        })
        preprocessor = Preprocessor(df)
        preprocessor.handle_missing_values(strategy='mean')
        report = preprocessor.get_report()
        
        assert isinstance(report, dict)
        assert 'original_shape' in report
        assert 'processed_shape' in report
        assert 'preprocessing_steps' in report
    
    def test_print_report(self, capsys):
        """Test report printing."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        preprocessor = Preprocessor(df)
        preprocessor.print_report()
        
        captured = capsys.readouterr()
        assert "PREPROCESSING REPORT" in captured.out


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_column_dataframe(self):
        """Test with single column."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        preprocessor = Preprocessor(df)
        assert len(preprocessor.numeric_cols) > 0 or len(preprocessor.categorical_cols) > 0
    
    def test_all_numeric_columns(self):
        """Test with all numeric columns."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        preprocessor = Preprocessor(df)
        assert len(preprocessor.numeric_cols) == 2
        assert len(preprocessor.categorical_cols) == 0
    
    def test_all_categorical_columns(self):
        """Test with all categorical columns."""
        df = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': ['x', 'y', 'z']
        })
        preprocessor = Preprocessor(df)
        assert len(preprocessor.numeric_cols) == 0
        assert len(preprocessor.categorical_cols) == 2
    
    def test_large_dataframe(self):
        """Test with larger dataset."""
        df = pd.DataFrame({
            'A': np.random.rand(1000),
            'B': np.random.rand(1000)
        })
        preprocessor = Preprocessor(df)
        preprocessor.scale_features(method='standard')
        assert len(preprocessor.data) == 1000
