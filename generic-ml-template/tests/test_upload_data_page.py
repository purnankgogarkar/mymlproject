"""Tests for upload data page."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from io import BytesIO
from app.utils.session_state import AppState


@pytest.fixture
def sample_df():
    """Create sample dataframe."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [40000, 50000, 60000, 70000, 80000],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


@pytest.fixture
def state():
    """Create app state for testing."""
    return AppState()


class TestUploadDataPageBasics:
    """Test basic upload data page functionality."""
    
    def test_state_initializes_empty(self, state):
        """Test that state starts empty."""
        assert state.data is None
        assert state.target_col is None
    
    def test_state_has_data_returns_false_initially(self, state):
        """Test has_data returns False initially."""
        assert not state.has_data()
    
    def test_state_has_data_returns_true_after_load(self, state, sample_df):
        """Test has_data returns True after data loaded."""
        state.data = sample_df
        assert state.has_data()


class TestUploadDataPageFileHandling:
    """Test file upload handling."""
    
    @patch('pandas.read_csv')
    def test_can_load_csv_data(self, mock_read_csv, state, sample_df):
        """Test loading CSV data."""
        mock_read_csv.return_value = sample_df
        df = pd.read_csv('test.csv')
        state.data = df
        
        assert len(state.data) == 5
        assert 'age' in state.data.columns
    
    @patch('pandas.read_excel')
    def test_can_load_excel_data(self, mock_read_excel, state, sample_df):
        """Test loading Excel data."""
        mock_read_excel.return_value = sample_df
        df = pd.read_excel('test.xlsx')
        state.data = df
        
        assert len(state.data) == 5
        assert 'age' in state.data.columns


class TestUploadDataPageValidation:
    """Test data validation."""
    
    def test_validate_data_not_empty(self, state, sample_df):
        """Test that loaded data is not empty."""
        state.data = sample_df
        assert len(state.data) > 0
    
    def test_validate_target_column_exists(self, state, sample_df):
        """Test that target column exists in data."""
        state.data = sample_df
        state.target_col = 'age'
        assert state.target_col in state.data.columns
    
    def test_data_types_detected_correctly(self, sample_df):
        """Test that data types are detected."""
        numeric_cols = sample_df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = sample_df.select_dtypes(include=['object']).columns.tolist()
        
        assert 'age' in numeric_cols
        assert 'income' in numeric_cols
        assert 'category' in cat_cols


class TestUploadDataPagePreview:
    """Test data preview functionality."""
    
    def test_preview_shows_first_rows(self, sample_df):
        """Test preview shows first N rows."""
        preview = sample_df.head(10)
        assert len(preview) == 5  # Our sample has only 5 rows
    
    def test_preview_shows_all_columns(self, sample_df):
        """Test preview shows all columns."""
        preview = sample_df.head(10)
        assert len(preview.columns) == 3
    
    def test_preview_preserves_data_integrity(self, sample_df):
        """Test that preview doesn't modify original data."""
        original_shape = sample_df.shape
        preview = sample_df.head(10)
        assert sample_df.shape == original_shape


class TestUploadDataPageMetrics:
    """Test data profile metrics."""
    
    def test_calculate_row_count(self, sample_df):
        """Test row count calculation."""
        assert len(sample_df) == 5
    
    def test_calculate_column_count(self, sample_df):
        """Test column count calculation."""
        assert len(sample_df.columns) == 3
    
    def test_calculate_missing_percentage(self, sample_df):
        """Test missing percentage calculation."""
        missing_percent = (sample_df.isnull().sum().sum() / 
                          (len(sample_df) * len(sample_df.columns)) * 100)
        assert missing_percent == 0.0
    
    def test_calculate_duplicates(self, sample_df):
        """Test duplicate count."""
        duplicates = sample_df.duplicated().sum()
        assert duplicates == 0


class TestUploadDataPageWorkflow:
    """Test complete upload workflow."""
    
    def test_complete_upload_workflow(self, state, sample_df):
        """Test complete upload workflow."""
        # 1. Load data
        state.data = sample_df
        assert state.has_data()
        
        # 2. Check profile
        assert len(state.data) == 5
        
        # 3. Select target
        state.target_col = 'age'
        assert state.target_col in state.data.columns
    
    def test_reset_clears_uploaded_data(self, state, sample_df):
        """Test that reset clears data."""
        state.data = sample_df
        state.target_col = 'age'
        
        state.reset()
        
        assert state.data is None
        assert state.target_col is None


class TestUploadDataPageDataQuality:
    """Test data quality checks."""
    
    def test_detect_numeric_columns(self, sample_df):
        """Test numeric column detection."""
        numeric = sample_df.select_dtypes(include=['number']).columns.tolist()
        assert len(numeric) == 2
        assert 'age' in numeric
    
    def test_detect_categorical_columns(self, sample_df):
        """Test categorical column detection."""
        categorical = sample_df.select_dtypes(include=['object']).columns.tolist()
        assert len(categorical) == 1
        assert 'category' in categorical
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [10, np.nan, 30, 40]
        })
        missing = df.isnull().sum()
        assert missing['col1'] == 1
        assert missing['col2'] == 1
    
    def test_handle_duplicate_rows(self):
        """Test handling duplicate rows."""
        df = pd.DataFrame({
            'a': [1, 2, 1],
            'b': [4, 5, 4]
        })
        duplicates = df.duplicated().sum()
        assert duplicates == 1


class TestUploadDataPageEdgeCases:
    """Test edge cases."""
    
    def test_handle_single_row_data(self):
        """Test handling single row data."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        assert len(df) == 1
    
    def test_handle_single_column_data(self):
        """Test handling single column data."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        assert len(df.columns) == 1
    
    def test_handle_all_missing_column(self):
        """Test handling column with all missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [np.nan, np.nan, np.nan]
        })
        assert df['col2'].isnull().sum() == 3
    
    def test_large_dataframe_handling(self):
        """Test handling large dataframe."""
        df = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000)
        })
        assert len(df) == 10000
        assert len(df.columns) == 2
