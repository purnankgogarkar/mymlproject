"""
Data Validator Module

Generic validation rules for data quality checks.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate data quality with multiple checks.
    
    Attributes:
        df (pd.DataFrame): Data to validate
        validation_results (Dict): Results from validation
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize validator."""
        self.df = df
        self.validation_results = {}
    
    def validate(self, strict: bool = False) -> Tuple[bool, Dict]:
        """
        Run all validation checks.
        
        Args:
            strict: If True, fail on any warning
            
        Returns:
            Tuple of (is_valid, results_dict)
        """
        results = {
            'passed': [],
            'warnings': [],
            'failed': [],
        }
        
        # Run checks and filter by status
        for check in self._check_not_empty():
            if check['status'] == 'PASSED':
                results['passed'].append(check)
            elif check['status'] == 'WARNING':
                results['warnings'].append(check)
            else:
                results['failed'].append(check)
        
        for check in self._check_missing_values():
            if check['status'] == 'PASSED':
                results['passed'].append(check)
            elif check['status'] == 'WARNING' or check['status'] == 'INFO':
                results['warnings'].append(check)
            else:
                results['failed'].append(check)
        
        for check in self._check_duplicates():
            if check['status'] == 'PASSED':
                results['passed'].append(check)
            elif check['status'] == 'WARNING':
                results['warnings'].append(check)
            else:
                results['failed'].append(check)
        
        for check in self._check_data_types():
            if check['status'] == 'PASSED':
                results['passed'].append(check)
            elif check['status'] == 'WARNING':
                results['warnings'].append(check)
            else:
                results['failed'].append(check)
        
        for check in self._check_critical_issues():
            if check['status'] == 'PASSED':
                results['passed'].append(check)
            elif check['status'] == 'WARNING':
                results['warnings'].append(check)
            else:
                results['failed'].append(check)
        
        is_valid = len(results['failed']) == 0 and (not strict or len(results['warnings']) == 0)
        self.validation_results = results
        
        return is_valid, results
    
    def _check_not_empty(self) -> List[Dict]:
        """Check if dataframe is not empty."""
        checks = []
        
        if len(self.df) == 0:
            checks.append({
                'check': 'not_empty',
                'status': 'FAILED',
                'message': 'DataFrame is empty',
            })
            checks.append({
                'check': 'has_columns',
                'status': 'FAILED',
                'message': 'Cannot have columns in empty DataFrame',
            })
        else:
            checks.append({
                'check': 'not_empty',
                'status': 'PASSED',
                'message': f'DataFrame has {len(self.df)} rows',
            })
            
            if len(self.df.columns) == 0:
                checks.append({
                    'check': 'has_columns',
                    'status': 'FAILED',
                    'message': 'DataFrame has no columns',
                })
            else:
                checks.append({
                    'check': 'has_columns',
                    'status': 'PASSED',
                    'message': f'DataFrame has {len(self.df.columns)} columns',
                })
        
        return checks
    
    def _check_missing_values(self) -> List[Dict]:
        """Check for missing values."""
        checks = []
        missing_pct = (self.df.isna().sum() / len(self.df)) * 100
        
        # Columns with high missing percentage
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            checks.append({
                'check': 'high_missing_values',
                'status': 'WARNING',
                'message': f'{len(high_missing)} columns have >50% missing',
                'columns': high_missing.to_dict(),
            })
        
        # Overall missing percentage
        total_missing_pct = (self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if total_missing_pct > 30:
            checks.append({
                'check': 'overall_missing',
                'status': 'WARNING',
                'message': f'{total_missing_pct:.1f}% of data is missing',
            })
        elif total_missing_pct > 0:
            checks.append({
                'check': 'overall_missing',
                'status': 'INFO',
                'message': f'{total_missing_pct:.2f}% of data is missing',
            })
        
        return checks
    
    def _check_duplicates(self) -> List[Dict]:
        """Check for duplicate rows."""
        checks = []
        dup_count = self.df.duplicated().sum()
        dup_pct = (dup_count / len(self.df)) * 100 if len(self.df) > 0 else 0
        
        if dup_count > 0:
            checks.append({
                'check': 'duplicates',
                'status': 'WARNING' if dup_pct < 10 else 'FAILED',
                'message': f'{dup_count} ({dup_pct:.1f}%) duplicate rows found',
            })
        else:
            checks.append({
                'check': 'duplicates',
                'status': 'PASSED',
                'message': 'No duplicate rows',
            })
        
        return checks
    
    def _check_data_types(self) -> List[Dict]:
        """Check data types."""
        checks = []
        found_issues = False
        
        # Check for mixed types in columns
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                if self.df[col].dtype == 'object':
                    # Try to infer if should be numeric
                    try:
                        pd.to_numeric(col_data)
                        checks.append({
                            'check': 'dtype_inference',
                            'status': 'WARNING',
                            'message': f'Column "{col}" contains numeric data but is type object',
                        })
                        found_issues = True
                    except:
                        pass
        
        checks.append({
            'check': 'dtypes_assigned',
            'status': 'PASSED',
            'message': f'Data types detected: {self.df.dtypes.value_counts().to_dict()}',
        })
        
        return checks
    
    def _check_critical_issues(self) -> List[Dict]:
        """Check for critical issues."""
        checks = []
        
        # Check for infinite values
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            inf_count = np.isinf(self.df[numeric_cols].values).sum()
            if inf_count > 0:
                checks.append({
                    'check': 'infinite_values',
                    'status': 'FAILED',
                    'message': f'{inf_count} infinite values found',
                })
        
        # Check for constant columns (no variance)
        if len(numeric_cols) > 0:
            const_cols = []
            for col in numeric_cols:
                col_clean = self.df[col].dropna()
                # Need at least 2 values to compute std, and std == 0 means no variance
                if len(col_clean) >= 2 and col_clean.std() == 0:
                    const_cols.append(col)
            
            if const_cols:
                checks.append({
                    'check': 'constant_columns',
                    'status': 'WARNING',
                    'message': f'{len(const_cols)} columns have no variance: {const_cols}',
                })
        
        return checks
    
    def print_report(self) -> None:
        """Print validation report."""
        if not self.validation_results:
            print("No validation results. Run validate() first.")
            return
        
        results = self.validation_results
        
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✓ PASSED ({len(results['passed'])}):")
        for check in results['passed']:
            print(f"  ✓ {check['check']}: {check['message']}")
        
        if results['warnings']:
            print(f"\n⚠ WARNINGS ({len(results['warnings'])}):")
            for check in results['warnings']:
                print(f"  ⚠ {check['check']}: {check['message']}")
        
        if results['failed']:
            print(f"\n✗ FAILED ({len(results['failed'])}):")
            for check in results['failed']:
                print(f"  ✗ {check['check']}: {check['message']}")
        
        print("\n" + "="*60 + "\n")
