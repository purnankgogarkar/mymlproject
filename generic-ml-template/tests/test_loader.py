"""
Tests for DataLoader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import DataLoader


class TestDataLoaderInitialization:
    """Test DataLoader initialization and validation."""
    
    def test_init_valid_csv(self, sample_iris_csv):
        """Test initialization with valid CSV file."""
        loader = DataLoader(sample_iris_csv)
        assert loader.file_path.exists()
        assert loader.file_path.suffix == '.csv'
    
    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DataLoader('/nonexistent/file.csv')
    
    def test_init_unsupported_format(self, tmp_path):
        """Test initialization with unsupported format."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            DataLoader(str(file_path))


class TestDataLoaderLoad:
    """Test data loading functionality."""
    
    def test_load_csv(self, sample_iris_csv):
        """Test loading CSV file."""
        loader = DataLoader(sample_iris_csv)
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert len(df.columns) == 5
        assert 'species' in df.columns
    
    def test_load_csv_with_sample_size(self, sample_iris_csv):
        """Test loading CSV with sample size limit."""
        loader = DataLoader(sample_iris_csv)
        df = loader.load(sample_size=3)
        
        assert len(df) == 3
    
    def test_auto_type_detection(self, sample_iris_csv):
        """Test automatic type detection."""
        loader = DataLoader(sample_iris_csv)
        df = loader.load()
        
        # Check that numeric columns are detected
        assert pd.api.types.is_numeric_dtype(df['sepal_length'])
        assert pd.api.types.is_numeric_dtype(df['petal_width'])
        # String column should remain object or be converted to category
        assert df['species'].dtype in [object, 'category']


class TestDataLoaderProfile:
    """Test data profiling functionality."""
    
    def test_profile_basic_info(self, sample_iris_csv):
        """Test profile generation."""
        loader = DataLoader(sample_iris_csv)
        loader.load()
        profile = loader.profile()
        
        assert 'shape' in profile
        assert 'rows' in profile
        assert 'columns' in profile
        assert profile['rows'] == 5
        assert profile['columns'] == 5
    
    def test_profile_missing_values(self, sample_titanic_csv):
        """Test profile with missing values."""
        loader = DataLoader(sample_titanic_csv)
        loader.load()
        profile = loader.profile()
        
        # Titanic has missing Age values
        assert 'Age' in profile['missing_values']
        assert profile['missing_values']['Age'] > 0
    
    def test_profile_numeric_statistics(self, sample_iris_csv):
        """Test numeric statistics in profile."""
        loader = DataLoader(sample_iris_csv)
        loader.load()
        profile = loader.profile()
        
        assert 'numeric_stats' in profile
        assert 'sepal_length' in profile['numeric_stats']
    
    def test_profile_without_loading(self):
        """Test profile fails if data not loaded."""
        loader = DataLoader.__new__(DataLoader)
        loader.df = None
        
        with pytest.raises(ValueError):
            loader.profile()


class TestDataLoaderColumnInfo:
    """Test column information retrieval."""
    
    def test_get_column_info_basic(self, sample_iris_csv):
        """Test column info generation."""
        loader = DataLoader(sample_iris_csv)
        loader.load()
        col_info = loader.get_column_info()
        
        assert 'sepal_length' in col_info
        assert 'non_null' in col_info['sepal_length']
        assert 'unique' in col_info['sepal_length']
    
    def test_get_column_info_numeric_stats(self, sample_iris_csv):
        """Test numeric column statistics."""
        loader = DataLoader(sample_iris_csv)
        loader.load()
        col_info = loader.get_column_info()
        
        # Numeric column should have stats
        assert 'min' in col_info['sepal_length']
        assert 'max' in col_info['sepal_length']
        assert 'mean' in col_info['sepal_length']
    
    def test_get_column_info_categorical_stats(self, sample_iris_csv):
        """Test categorical column statistics."""
        loader = DataLoader(sample_iris_csv)
        loader.load()
        col_info = loader.get_column_info()
        
        # Categorical column should have value counts
        assert 'top_values' in col_info['species']
        assert isinstance(col_info['species']['top_values'], dict)


class TestDataLoaderIntegration:
    """Integration tests for DataLoader."""
    
    def test_load_and_profile_iris(self, sample_iris_csv):
        """End-to-end test: load, detect types, profile."""
        loader = DataLoader(sample_iris_csv)
        df = loader.load()
        profile = loader.profile()
        col_info = loader.get_column_info()
        
        # Verify end-to-end flow
        assert df is not None
        assert profile is not None
        assert col_info is not None
        assert len(col_info) == len(df.columns)
    
    def test_load_and_profile_with_missing(self, sample_titanic_csv):
        """Load dataset with missing values and verify handling."""
        loader = DataLoader(sample_titanic_csv)
        df = loader.load()
        profile = loader.profile()
        
        # Verify missing values are detected
        assert profile['missing_percent']['Age'] > 0
    
    def test_load_numeric_regression_data(self, sample_numeric_csv):
        """Load numeric regression dataset."""
        loader = DataLoader(sample_numeric_csv)
        df = loader.load()
        profile = loader.profile()
        
        # All columns should be numeric except potentially index
        assert len(df.select_dtypes(include='number').columns) > 0
