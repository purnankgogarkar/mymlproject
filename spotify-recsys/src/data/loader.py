"""Data loading and analysis."""
import pandas as pd
import os


def load_data(filepath):
    """Load CSV file."""
    df = pd.read_csv(filepath)
    return df


def print_shape(df):
    """Print dataset shape."""
    rows, cols = df.shape
    print(f"\n=== Dataset Shape ===")
    print(f"Rows: {rows:,}")
    print(f"Columns: {cols}")


def print_columns(df):
    """Print column names and data types."""
    print(f"\n=== Columns & Data Types ===")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"  {col}: {dtype}")


def print_summary_stats(df):
    """Print summary statistics for numeric columns."""
    print(f"\n=== Summary Statistics (Numeric) ===")
    print(df.describe())


def print_missing_values(df):
    """Print missing value counts and percentages."""
    print(f"\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing.values,
        'Percentage': missing_pct.values
    })
    
    # Show only columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")


def analyze_data(filepath):
    """Run all analysis steps."""
    df = load_data(filepath)
    
    print_shape(df)
    print_columns(df)
    print_summary_stats(df)
    print_missing_values(df)
    
    return df


if __name__ == "__main__":
    # Load from data/raw/spotify_tracks.csv
    data_path = "data/raw/spotify_tracks.csv"
    
    if os.path.exists(data_path):
        df = analyze_data(data_path)
    else:
        print(f"Error: {data_path} not found!")
        print("Download dataset from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
