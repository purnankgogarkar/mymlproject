"""Data quality checks."""
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    'track_id', 'track_name', 'artist', 'album',
    'valence', 'energy', 'danceability', 'acousticness',
    'loudness', 'tempo', 'speechiness', 'instrumentalness',
]

NUMERIC_COLUMNS = [
    'valence', 'energy', 'danceability', 'acousticness',
    'loudness', 'tempo', 'speechiness', 'instrumentalness',
]


def check_data_quality(df):
    """Run 5 quality checks on dataframe."""
    result = {
        'success': True,
        'failures': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check 1: Schema validation
    result = _check_schema(df, result)
    
    # Check 2: Row count
    result = _check_row_count(df, result)
    
    # Check 3: Null rates
    result = _check_null_rates(df, result)
    
    # Check 4: Value ranges
    result = _check_value_ranges(df, result)
    
    # Check 5: Target distribution (if classification target exists)
    result = _check_target_distribution(df, result)
    
    # Set success flag
    result['success'] = len(result['failures']) == 0
    
    return result


def _check_schema(df, result):
    """Check 1: Schema validation - required columns + dtypes."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        result['failures'].append(f"Missing required columns: {missing_cols}")
    
    # Check numeric columns are numeric
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                result['failures'].append(f"Column '{col}' should be numeric but is {df[col].dtype}")
    
    return result


def _check_row_count(df, result):
    """Check 2: Row count - at least 100, warn if < 1000."""
    row_count = len(df)
    result['statistics']['total_rows'] = row_count
    
    if row_count < 100:
        result['failures'].append(f"Too few rows: {row_count} (need >= 100)")
    elif row_count < 1000:
        result['warnings'].append(f"Low row count: {row_count} (recommend >= 1000)")
    
    return result


def _check_null_rates(df, result):
    """Check 3: Null rates - no col > 50%, warn if > 20%."""
    null_counts = df.isnull().sum()
    null_rates = (null_counts / len(df)) * 100
    
    result['statistics']['total_nulls_by_column'] = null_counts.to_dict()
    result['statistics']['null_rates_by_column'] = null_rates.to_dict()
    
    for col in df.columns:
        null_rate = null_rates[col]
        
        if null_rate > 50:
            result['failures'].append(
                f"Column '{col}' has {null_rate:.1f}% nulls (> 50%)"
            )
        elif null_rate > 20:
            result['warnings'].append(
                f"Column '{col}' has {null_rate:.1f}% nulls (> 20%)"
            )
    
    return result


def _check_value_ranges(df, result):
    """Check 4: Value ranges - numeric columns within bounds."""
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        
        # Skip if all nulls
        if df[col].isnull().all():
            continue
        
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Check for negative counts (if col is count-like)
        if 'count' in col.lower() or 'plays' in col.lower():
            if col_min < 0:
                result['failures'].append(
                    f"Column '{col}' has negative values: min={col_min}"
                )
        
        # Check for extreme percentages
        if col in ['valence', 'energy', 'danceability', 'acousticness', 
                   'speechiness', 'instrumentalness']:
            if col_max > 1.5 or col_min < -0.1:
                result['warnings'].append(
                    f"Column '{col}' out of expected range [0, 1]: min={col_min:.2f}, max={col_max:.2f}"
                )
    
    return result


def _check_target_distribution(df, result):
    """Check 5: Target distribution (genre/mood classification)."""
    # Check if there's a classification column (genre, mood, etc.)
    target_cols = [col for col in df.columns if col.lower() in ['genre', 'mood', 'target']]
    
    if not target_cols:
        # Skip if no target column
        return result
    
    target_col = target_cols[0]
    
    # Count unique values
    unique_count = df[target_col].nunique()
    result['statistics']['target_classes'] = unique_count
    
    if unique_count < 2:
        result['failures'].append(
            f"Target column '{target_col}' has < 2 classes: {unique_count}"
        )
    
    # Check class balance
    value_counts = df[target_col].value_counts()
    class_pcts = (value_counts / len(df)) * 100
    
    for class_name, pct in class_pcts.items():
        if pct < 5:
            result['warnings'].append(
                f"Target class '{class_name}' is imbalanced: {pct:.1f}% of data"
            )
    
    result['statistics']['target_distribution'] = value_counts.to_dict()
    
    return result


def print_quality_report(result):
    """Print formatted quality report."""
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    # Overall status
    status = "✓ PASSED" if result['success'] else "✗ FAILED"
    print(f"\nStatus: {status}")
    
    # Failures
    if result['failures']:
        print(f"\n[FAILURES] ({len(result['failures'])})")
        for failure in result['failures']:
            print(f"  ✗ {failure}")
    
    # Warnings
    if result['warnings']:
        print(f"\n[WARNINGS] ({len(result['warnings'])})")
        for warning in result['warnings']:
            print(f"  ⚠ {warning}")
    
    # Statistics
    print(f"\n[STATISTICS]")
    for key, value in result['statistics'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Load dataset
    import os
    
    data_path = "data/raw/spotify_tracks.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        result = check_data_quality(df)
        print_quality_report(result)
    else:
        print(f"Error: {data_path} not found!")
        print("Download dataset from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
