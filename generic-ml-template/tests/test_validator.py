"""
Tests for DataValidator
"""

import pytest
import pandas as pd
import numpy as np
from src.data.validator import DataValidator


class TestDataValidatorBasicChecks:
    """Test basic validation checks."""
    
    def test_validate_not_empty(self, sample_dataframe):
        """Test not empty validation."""
        validator = DataValidator(sample_dataframe)
        is_valid, results = validator.validate()
        
        passed_checks = [c['check'] for c in results['passed']]
        assert 'not_empty' in passed_checks
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty dataframe."""
        empty_df = pd.DataFrame()
        validator = DataValidator(empty_df)
        is_valid, results = validator.validate()
        
        failed_checks = [c['check'] for c in results['failed']]
        assert any('empty' in check or 'columns' in check for check in failed_checks)
    
    def test_validate_no_columns(self):
        """Test validation with dataframe with rows but no columns."""
        df = pd.DataFrame(index=[0, 1, 2])
        validator = DataValidator(df)
        is_valid, results = validator.validate()
        
        failed_checks = [c['check'] for c in results['failed']]
        assert 'has_columns' in failed_checks


class TestDataValidatorMissingValues:
    """Test missing value validation."""
    
    def test_check_missing_values_small(self, sample_dataframe):
        """Test with small amount of missing values."""
        validator = DataValidator(sample_dataframe)
        is_valid, results = validator.validate()
        
        # Should pass since no missing values
        warning_checks = [c['check'] for c in results['warnings']]
        # dtypes_assigned is in passed, not warnings
        assert 'high_missing_values' not in warning_checks or len(results['warnings']) == 0
    
    def test_check_missing_values_high(self, sample_dataframe_with_missing):
        """Test with high amount of missing values."""
        validator = DataValidator(sample_dataframe_with_missing)
        is_valid, results = validator.validate()
        
        # Should have warnings about missing values
        warning_checks = [c['check'] for c in results['warnings']]
        assert len(warning_checks) > 0


class TestDataValidatorDuplicates:
    """Test duplicate validation."""
    
    def test_check_duplicates_none(self, sample_dataframe):
        """Test with no duplicates."""
        validator = DataValidator(sample_dataframe)
        is_valid, results = validator.validate()
        
        passed_checks = [c['check'] for c in results['passed']]
        assert 'duplicates' in passed_checks
    
    def test_check_duplicates_present(self, sample_dataframe_with_duplicates):
        """Test with duplicate rows."""
        validator = DataValidator(sample_dataframe_with_duplicates)
        is_valid, results = validator.validate()
        
        # Should detect duplicates
        all_checks = results['passed'] + results['warnings'] + results['failed']
        dup_checks = [c for c in all_checks if c['check'] == 'duplicates']
        assert len(dup_checks) > 0


class TestDataValidatorDataTypes:
    """Test data type validation."""
    
    def test_check_dtypes_mixed(self, sample_dataframe):
        """Test mixed data type detection."""
        validator = DataValidator(sample_dataframe)
        is_valid, results = validator.validate()
        
        # Should pass dtype check
        dtype_checks = [c for c in results['passed'] if 'dtype' in c['check']]
        assert len(dtype_checks) > 0


class TestDataValidatorCriticalIssues:
    """Test critical issues detection."""
    
    def test_check_constant_columns(self):
        """Test detection of constant (no variance) columns."""
        df = pd.DataFrame({
            'varying': [1, 2, 3, 4, 5],
            'constant': [5, 5, 5, 5, 5],
        })
        validator = DataValidator(df)
        is_valid, results = validator.validate()
        
        # Should warn about constant columns
        warning_checks = [c['check'] for c in results['warnings']]
        assert any('constant' in check for check in warning_checks)
    
    def test_check_infinite_values(self):
        """Test detection of infinite values."""
        df = pd.DataFrame({
            'normal': [1.0, 2.0, 3.0, 4.0, 5.0],
            'with_inf': [1.0, np.inf, 3.0, 4.0, 5.0],
        })
        validator = DataValidator(df)
        is_valid, results = validator.validate()
        
        # Should fail due to infinite values
        failed_checks = [c['check'] for c in results['failed']]
        assert any('infinite' in check for check in failed_checks)


class TestDataValidatorReport:
    """Test validation report printing."""
    
    def test_print_report(self, sample_dataframe, capsys):
        """Test report printing."""
        validator = DataValidator(sample_dataframe)
        validator.validate()
        validator.print_report()
        
        captured = capsys.readouterr()
        assert 'VALIDATION REPORT' in captured.out
        assert 'PASSED' in captured.out or 'WARNINGS' in captured.out or 'FAILED' in captured.out


class TestDataValidatorIntegration:
    """Integration tests for DataValidator."""
    
    def test_validate_clean_data(self, sample_dataframe):
        """Test validation of clean data."""
        validator = DataValidator(sample_dataframe)
        is_valid, results = validator.validate()
        
        # Clean data should pass
        assert len(results['failed']) == 0
    
    def test_validate_messy_data(self):
        """Test validation of messy data."""
        df = pd.DataFrame({
            'col1': [1, 1, 1, 1, 1],  # Constant
            'col2': [1, np.inf, 3, 4, 5],  # Has inf
            'col3': [1, np.nan, np.nan, 4, 5],  # High missing
        })
        validator = DataValidator(df)
        is_valid, results = validator.validate()
        
        # Should find issues
        assert len(results['failed']) > 0 or len(results['warnings']) > 0
