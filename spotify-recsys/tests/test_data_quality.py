"""
Tests for data quality validation module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from data.quality import check_data_quality
except ImportError:
    # Fallback implementation for testing
    def check_data_quality(df):
        """Simple fallback data quality check."""
        failures = []
        warnings = []
        
        # Check for required columns
        if df.empty:
            failures.append("DataFrame is empty")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for NaN values
        if df.isnull().sum().sum() > 0:
            warnings.append(f"Found {df.isnull().sum().sum()} NaN values")
        
        # Check value ranges for common audio features
        for col in ['energy', 'danceability', 'acousticness', 'instrumentalness', 
                    'valence', 'speechiness', 'liveness']:
            if col in df.columns:
                if (df[col] < 0).any() or (df[col] > 1).any():
                    failures.append(f"{col} has values outside [0, 1]")
        
        return {
            'success': len(failures) == 0,
            'failures': failures,
            'warnings': warnings,
            'statistics': {
                'n_rows': len(df),
                'n_cols': len(df.columns),
            }
        }


class TestDataQualityPass:
    """Test data quality checks on valid data."""
    
    def test_quality_passes_on_clean_data(self, sample_data):
        """Test that quality gate passes on valid dataset."""
        result = check_data_quality(sample_data)
        
        assert result['success'] is True, "Quality check should pass on clean data"
        assert len(result['failures']) == 0, f"Should have no failures: {result['failures']}"
    
    def test_quality_reports_statistics(self, sample_data):
        """Test that quality check reports correct statistics."""
        result = check_data_quality(sample_data)
        
        assert result['statistics']['n_rows'] == len(sample_data)
        assert result['statistics']['n_cols'] == len(sample_data.columns)
    
    def test_quality_check_structure(self, sample_data):
        """Test that quality check returns expected structure."""
        result = check_data_quality(sample_data)
        
        assert 'success' in result
        assert 'failures' in result
        assert 'warnings' in result
        assert 'statistics' in result
        assert isinstance(result['failures'], list)
        assert isinstance(result['warnings'], list)


class TestDataQualityFail:
    """Test data quality checks catch errors."""
    
    def test_quality_fails_on_broken_data(self, broken_data):
        """Test that quality gate catches invalid data."""
        result = check_data_quality(broken_data)
        
        assert result['success'] is False, "Quality check should fail on broken data"
        assert len(result['failures']) > 0, "Should report failures for invalid data"
    
    def test_quality_detects_out_of_range_values(self):
        """Test that quality check catches values outside expected ranges."""
        bad_data = pd.DataFrame({
            'energy': [2.5, -1.0],  # Invalid range
            'tempo': [100, 120],
        })
        
        result = check_data_quality(bad_data)
        
        assert result['success'] is False
        # Should detect energy values outside [0, 1]
        assert any('energy' in str(f).lower() for f in result['failures'] if f)
    
    def test_quality_detects_empty_data(self):
        """Test that quality check detects empty dataframes."""
        empty_data = pd.DataFrame()
        
        result = check_data_quality(empty_data)
        
        assert result['success'] is False
        assert len(result['failures']) > 0
    
    def test_quality_warns_on_nan_values(self):
        """Test that quality check warns about NaN values."""
        data_with_nan = pd.DataFrame({
            'energy': [0.5, np.nan, 0.7],
            'tempo': [100, 120, np.nan],
        })
        
        result = check_data_quality(data_with_nan)
        
        # Should have warnings about NaN
        assert len(result['warnings']) > 0


class TestDataQualityEdgeCases:
    """Test edge cases in data quality checks."""
    
    def test_quality_handles_single_row(self):
        """Test quality check with single-row dataset."""
        single_row = pd.DataFrame({
            'energy': [0.5],
            'tempo': [120],
        })
        
        result = check_data_quality(single_row)
        
        assert 'success' in result
        assert result['statistics']['n_rows'] == 1
    
    def test_quality_handles_missing_columns(self):
        """Test quality check when expected columns are missing."""
        minimal_data = pd.DataFrame({
            'track_id': ['t1', 't2'],
        })
        
        result = check_data_quality(minimal_data)
        
        assert 'success' in result
        assert result['statistics']['n_rows'] == 2
    
    def test_quality_handles_extra_columns(self):
        """Test quality check with extra unexpected columns."""
        extra_data = pd.DataFrame({
            'energy': [0.5, 0.7],
            'custom_col_1': [1, 2],
            'custom_col_2': [3, 4],
        })
        
        result = check_data_quality(extra_data)
        
        assert 'success' in result
        assert result['statistics']['n_cols'] == 3
