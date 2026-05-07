"""
Data Loader Module

Handles loading CSV/Excel files with automatic type detection and profiling.
Supports both tabular and time-series data formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Generic data loader for CSV/Excel files with auto-type detection.
    
    Attributes:
        file_path (Path): Path to data file
        df (pd.DataFrame): Loaded dataframe
        metadata (Dict): Data profiling metadata
    """
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.tsv', '.parquet'}
    
    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize DataLoader with file path validation.
        
        Args:
            file_path: Path to CSV/Excel file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        self.file_path = Path(file_path)
        self.df = None
        self.metadata = {}
        
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Validate file exists and has supported format."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
    
    def load(self, sample_size: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            sample_size: If specified, load only first N rows (for large datasets)
            **kwargs: Additional arguments passed to pandas read function
            
        Returns:
            Loaded DataFrame
        """
        suffix = self.file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                self.df = pd.read_csv(self.file_path, nrows=sample_size, **kwargs)
            elif suffix == '.tsv':
                self.df = pd.read_csv(self.file_path, sep='\t', nrows=sample_size, **kwargs)
            elif suffix in {'.xlsx', '.xls'}:
                self.df = pd.read_excel(self.file_path, nrows=sample_size, **kwargs)
            elif suffix == '.parquet':
                self.df = pd.read_parquet(self.file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {suffix}")
            
            logger.info(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            self._auto_detect_types()
            return self.df
        
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
    
    def _auto_detect_types(self) -> None:
        """
        Auto-detect and convert data types intelligently.
        Attempts to convert columns to numeric, datetime, or category types.
        """
        for col in self.df.columns:
            # Try numeric conversion
            if self.df[col].dtype == 'object':
                try:
                    # Try datetime first
                    if self._is_datetime(col):
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        logger.info(f"  {col}: datetime")
                        continue
                    
                    # Try numeric
                    converted = pd.to_numeric(self.df[col], errors='coerce')
                    if converted.isna().sum() < len(converted) * 0.5:  # <50% NaN
                        self.df[col] = converted
                        logger.info(f"  {col}: numeric")
                        continue
                    
                    # Try category if few unique values
                    if self.df[col].nunique() < len(self.df) * 0.05:  # <5% unique
                        self.df[col] = self.df[col].astype('category')
                        logger.info(f"  {col}: category")
                        continue
                    
                    logger.info(f"  {col}: object")
                
                except Exception as e:
                    logger.debug(f"Could not convert {col}: {e}")
    
    def _is_datetime(self, col: str) -> bool:
        """Check if column might be datetime."""
        sample = self.df[col].dropna().head(10)
        date_patterns = ['/', '-', ':']
        return any(pattern in str(sample.iloc[0]) for pattern in date_patterns if len(sample) > 0)
    
    def profile(self) -> Dict:
        """
        Generate data profile with statistics.
        
        Returns:
            Dictionary with profiling metadata
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        profile = {
            'shape': self.df.shape,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': {col: self.df[col].isna().sum() for col in self.df.columns},
            'missing_percent': {col: (self.df[col].isna().sum() / len(self.df)) * 100 
                              for col in self.df.columns},
            'duplicates': self.df.duplicated().sum(),
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns},
        }
        
        # Add numeric statistics
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            profile['numeric_stats'] = self.df[numeric_cols].describe().to_dict()
        
        self.metadata = profile
        return profile
    
    def get_column_info(self) -> Dict:
        """Get detailed information about each column."""
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        info = {}
        for col in self.df.columns:
            dtype = self.df[col].dtype
            info[col] = {
                'dtype': str(dtype),
                'non_null': len(self.df[col].dropna()),
                'null_count': self.df[col].isna().sum(),
                'null_percent': (self.df[col].isna().sum() / len(self.df)) * 100,
                'unique': self.df[col].nunique(),
                'duplicate_rows': self.df[col].duplicated().sum(),
            }
            
            # Add numeric/categorical specific stats
            if pd.api.types.is_numeric_dtype(dtype):
                info[col].update({
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'median': self.df[col].median(),
                })
            elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
                info[col]['top_values'] = self.df[col].value_counts().head(5).to_dict()
        
        return info
