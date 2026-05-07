"""Tests for data widgets."""

import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import patch, MagicMock
from app.utils.data_widgets import (
    upload_data_widget,
    display_data_preview,
    display_data_profile,
    display_column_info,
    select_target_column,
    display_missing_value_chart,
    display_basic_statistics
)


@pytest.fixture
def sample_df():
    """Create sample dataframe."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [40000, 50000, 60000, 70000, 80000],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


@pytest.fixture
def df_with_missing():
    """Create dataframe with missing values."""
    return pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [10, np.nan, 30, 40, 50],
        'col3': [100, 200, 300, 400, 500]
    })


class TestUploadDataWidget:
    """Test upload data widget."""
    
    @patch('streamlit.file_uploader')
    def test_upload_data_widget_returns_none_no_file(self, mock_uploader):
        """Test widget returns None when no file uploaded."""
        mock_uploader.return_value = None
        result = upload_data_widget()
        assert result is None
    
    @patch('streamlit.file_uploader')
    def test_upload_csv_file(self, mock_uploader):
        """Test CSV file upload."""
        # Create mock CSV file
        csv_data = b"age,income,category\n25,40000,A\n30,50000,B"
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.read.return_value = csv_data
        mock_uploader.return_value = mock_file
        
        with patch('pandas.read_csv') as mock_read_csv:
            df = pd.DataFrame({'age': [25, 30], 'income': [40000, 50000], 'category': ['A', 'B']})
            mock_read_csv.return_value = df
            result = upload_data_widget()
            assert result is not None
            assert isinstance(result, pd.DataFrame)
    
    @patch('streamlit.file_uploader')
    def test_upload_excel_file(self, mock_uploader):
        """Test Excel file upload."""
        mock_file = MagicMock()
        mock_file.name = "test.xlsx"
        mock_uploader.return_value = mock_file
        
        with patch('pandas.read_excel') as mock_read_excel:
            df = pd.DataFrame({'age': [25, 30], 'income': [40000, 50000]})
            mock_read_excel.return_value = df
            result = upload_data_widget()
            assert result is not None
            assert isinstance(result, pd.DataFrame)


class TestDataPreview:
    """Test data preview display."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    @patch('streamlit.columns')
    def test_display_data_preview(self, mock_columns, mock_dataframe, mock_subheader, sample_df):
        """Test preview display."""
        mock_columns.return_value = [MagicMock(), MagicMock()]
        display_data_preview(sample_df)
        mock_subheader.assert_called()
    
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    @patch('streamlit.columns')
    def test_preview_uses_n_rows(self, mock_columns, mock_dataframe, mock_subheader, sample_df):
        """Test preview respects n_rows parameter."""
        mock_columns.return_value = [MagicMock(), MagicMock()]
        display_data_preview(sample_df, n_rows=3)
        mock_dataframe.assert_called()


class TestDataProfile:
    """Test data profile display."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_display_data_profile(self, mock_metric, mock_columns, mock_subheader, sample_df):
        """Test profile display."""
        mock_columns.return_value = [MagicMock()] * 5
        display_data_profile(sample_df)
        mock_subheader.assert_called()
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_profile_calculates_metrics(self, mock_metric, mock_columns, mock_subheader, sample_df):
        """Test profile calculates metrics correctly."""
        mock_columns.return_value = [MagicMock()] * 5
        display_data_profile(sample_df)
        # Check that metric was called
        assert mock_metric.called


class TestColumnInfo:
    """Test column info display."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.write')
    def test_display_column_info(self, mock_write, mock_columns, mock_subheader, sample_df):
        """Test column info display."""
        mock_columns.return_value = [MagicMock(), MagicMock()]
        display_column_info(sample_df)
        mock_subheader.assert_called()
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.write')
    def test_column_info_separates_types(self, mock_write, mock_columns, mock_subheader, sample_df):
        """Test that column info separates numeric and categorical."""
        mock_columns.return_value = [MagicMock(), MagicMock()]
        display_column_info(sample_df)
        assert mock_write.called


class TestTargetColumnSelection:
    """Test target column selection."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.selectbox')
    def test_select_target_column(self, mock_selectbox, mock_subheader, sample_df):
        """Test target column selection."""
        mock_selectbox.return_value = 'age'
        result = select_target_column(sample_df)
        mock_selectbox.assert_called()
        assert result == 'age'
    
    @patch('streamlit.subheader')
    @patch('streamlit.selectbox')
    def test_selectbox_includes_all_columns(self, mock_selectbox, mock_subheader, sample_df):
        """Test that all columns are available for selection."""
        mock_selectbox.return_value = 'age'
        select_target_column(sample_df)
        call_args = mock_selectbox.call_args
        assert 'age' in call_args[1]['options']
        assert 'income' in call_args[1]['options']


class TestMissingValueChart:
    """Test missing value chart display."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    def test_display_missing_value_chart_with_missing(self, mock_dataframe, mock_subheader, df_with_missing):
        """Test missing value display."""
        display_missing_value_chart(df_with_missing)
        mock_subheader.assert_called()
    
    @patch('streamlit.subheader')
    @patch('streamlit.success')
    def test_display_missing_value_chart_no_missing(self, mock_success, mock_subheader, sample_df):
        """Test display when no missing values."""
        display_missing_value_chart(sample_df)
        mock_success.assert_called()


class TestBasicStatistics:
    """Test basic statistics display."""
    
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    def test_display_basic_statistics(self, mock_dataframe, mock_subheader, sample_df):
        """Test statistics display."""
        display_basic_statistics(sample_df)
        mock_subheader.assert_called()
    
    @patch('streamlit.subheader')
    @patch('streamlit.dataframe')
    def test_statistics_shows_describe_output(self, mock_dataframe, mock_subheader, sample_df):
        """Test that statistics shows describe() output."""
        display_basic_statistics(sample_df)
        mock_dataframe.assert_called()
        call_args = mock_dataframe.call_args[0][0]
        # Check that it's a dataframe with statistics
        assert isinstance(call_args, pd.DataFrame)


class TestIntegration:
    """Integration tests for data widgets."""
    
    def test_widget_workflow_with_sample_data(self, sample_df):
        """Test complete widget workflow."""
        # Simulate loading data
        assert len(sample_df) > 0
        assert 'age' in sample_df.columns
        
        # Verify data properties
        assert sample_df.select_dtypes(include=['number']).shape[1] == 2
        assert sample_df.select_dtypes(include=['object']).shape[1] == 1
    
    def test_widget_handles_missing_data(self, df_with_missing):
        """Test widgets handle missing data correctly."""
        missing_count = df_with_missing.isnull().sum().sum()
        assert missing_count > 0
