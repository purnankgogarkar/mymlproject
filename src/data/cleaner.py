"""Data cleaning pipeline."""
import pandas as pd
import numpy as np
import os
import sys

try:
    from .quality import check_data_quality, print_quality_report
except ImportError:
    # Handle direct execution
    from quality import check_data_quality, print_quality_report


NUMERIC_COLUMNS = [
    'valence', 'energy', 'danceability', 'acousticness',
    'loudness', 'tempo', 'speechiness', 'instrumentalness',
]

CATEGORICAL_COLUMNS = [
    'track_id', 'track_name', 'artist', 'album',
]


def clean_data(df):
    """Clean data through multiple steps."""
    print("\n" + "="*60)
    print("DATA CLEANING PIPELINE")
    print("="*60)
    
    initial_rows = len(df)
    print(f"\nInitial rows: {initial_rows:,}")
    
    df_clean = df.copy()
    
    # Step 1: Drop columns with > 50% nulls
    df_clean = _drop_high_null_columns(df_clean)
    
    # Step 2: Handle nulls - drop rows with target/critical nulls
    df_clean = _handle_nulls(df_clean)
    
    # Step 3: Remove duplicates
    df_clean = _remove_duplicates(df_clean)
    
    # Step 4: Convert dtypes
    df_clean = _convert_dtypes(df_clean)
    
    # Step 5: Save cleaned data
    output_path = "data/processed/spotify_cleaned.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to {output_path}")
    
    final_rows = len(df_clean)
    rows_removed = initial_rows - final_rows
    print(f"\nFinal rows: {final_rows:,}")
    print(f"Rows removed: {rows_removed:,} ({(rows_removed/initial_rows)*100:.1f}%)")
    
    # Step 6: Re-run quality gate
    print("\n--- Quality Gate on Cleaned Data ---")
    quality_result = check_data_quality(df_clean)
    print_quality_report(quality_result)
    
    return df_clean, quality_result


def _drop_high_null_columns(df):
    """Drop columns with > 50% nulls."""
    null_rates = (df.isnull().sum() / len(df)) * 100
    high_null_cols = null_rates[null_rates > 50].index.tolist()
    
    if high_null_cols:
        print(f"\nDropping columns with > 50% nulls: {high_null_cols}")
        df = df.drop(columns=high_null_cols)
    
    return df


def _handle_nulls(df):
    """Handle null values."""
    initial_rows = len(df)
    
    # Drop rows with critical column nulls
    critical_cols = [col for col in ['track_id', 'track_name'] if col in df.columns]
    df = df.dropna(subset=critical_cols)
    
    # For other columns: drop remaining rows with nulls
    # (Not time series, so no forward-fill)
    df = df.dropna()
    
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"\nDropped rows with nulls: {rows_dropped}")
    
    return df


def _remove_duplicates(df):
    """Remove exact row duplicates, keep first."""
    initial_rows = len(df)
    
    # Drop duplicates based on all columns
    df = df.drop_duplicates(keep='first')
    
    # Also drop duplicates based on track_id (if exists)
    if 'track_id' in df.columns:
        dups_track = initial_rows - len(df.drop_duplicates(subset=['track_id'], keep='first'))
        if dups_track > 0:
            df = df.drop_duplicates(subset=['track_id'], keep='first')
            print(f"\nRemoved duplicate track_ids: {dups_track}")
    
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Removed exact duplicates: {rows_removed}")
    
    return df


def _convert_dtypes(df):
    """Convert columns to correct dtypes."""
    for col in df.columns:
        # Convert numeric columns
        if col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert categorical columns to string
        elif col in CATEGORICAL_COLUMNS:
            df[col] = df[col].astype(str)
    
    print("\n✓ Data types converted")
    
    return df


if __name__ == "__main__":
    try:
        from .loader import load_data
    except ImportError:
        from loader import load_data
    
    # Load raw data
    data_path = "data/raw/spotify_tracks.csv"
    
    if os.path.exists(data_path):
        print("Loading raw data...")
        df_raw = load_data(data_path)
        
        # Clean data
        df_clean, quality_result = clean_data(df_raw)
        
        print(f"\n✓ Cleaning complete!")
        print(f"  Raw: {len(df_raw):,} rows")
        print(f"  Clean: {len(df_clean):,} rows")
        print(f"  Quality: {'PASSED' if quality_result['success'] else 'FAILED'}")
    else:
        print(f"Error: {data_path} not found!")
        print("Download dataset from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
