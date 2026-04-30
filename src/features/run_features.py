"""
Feature Engineering Pipeline Orchestrator.

Loads cleaned data → creates engineered features → selects optimal features → saves.
Includes timing and before/after statistics.
"""

import time
import pandas as pd
import os
import sys
import numpy as np

try:
    from .engineering import create_features, select_features, save_engineered_data
except ImportError:
    # Support direct execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from engineering import create_features, select_features, save_engineered_data


def run_feature_pipeline(
    input_path="data/processed/spotify_cleaned.csv",
    output_engineered="data/processed/spotify_engineered.csv",
    output_selected="data/processed/spotify_features.csv",
):
    """
    Run full feature engineering pipeline with timing and statistics.
    
    Args:
        input_path: Path to cleaned data
        output_engineered: Path to save engineered features
        output_selected: Path to save selected features
    
    Returns:
        tuple: (df_selected, selected_features, elapsed_time)
    """
    t0 = time.time()
    
    print("🚀 FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # ===== LOAD DATA =====
    print(f"\n📥 Loading cleaned data from {input_path}...")
    try:
        df_cleaned = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {input_path}")
        print("   Run cleaner.py first: python src/data/cleaner.py")
        return None, None, None
    
    print(f"✓ Loaded {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns")
    print(f"  Columns: {', '.join(df_cleaned.columns[:5])}...")
    
    # ===== CREATE ENGINEERED FEATURES =====
    print(f"\n⚙️ Creating engineered features...")
    try:
        df_engineered = create_features(df_cleaned)
    except Exception as e:
        print(f"❌ Feature creation failed: {e}")
        return None, None, None
    
    print(f"\n✓ Engineered features created")
    print(f"  Shape: {df_engineered.shape[0]:,} rows × {df_engineered.shape[1]} columns")
    
    # ===== SELECT FEATURES (SKIP - USE ALL) =====
    print(f"\n🔍 Using all engineered features (no selection to preserve signal)...")
    
    # Get all numeric columns as selected features
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = numeric_cols
    df_selected = df_engineered.copy()
    
    print(f"✓ Using {len(selected_features)} features (all engineered)")
    print(f"  Shape: {df_selected.shape[0]:,} rows × {df_selected.shape[1]} columns")
    
    # ===== SAVE ENGINEERED DATA =====
    print(f"\n💾 Saving engineered features...")
    try:
        os.makedirs(os.path.dirname(output_engineered), exist_ok=True)
        df_engineered.to_csv(output_engineered, index=False)
        print(f"✓ Saved to {output_engineered}")
    except Exception as e:
        print(f"⚠ Warning: Could not save engineered features: {e}")
    
    # ===== SAVE SELECTED FEATURES =====
    print(f"\n💾 Saving selected features...")
    try:
        os.makedirs(os.path.dirname(output_selected), exist_ok=True)
        df_selected.to_csv(output_selected, index=False)
        print(f"✓ Saved to {output_selected}")
    except Exception as e:
        print(f"❌ Error saving selected features: {e}")
        return None, None, None
    
    # ===== SUMMARY STATISTICS =====
    elapsed = time.time() - t0
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    print(f"\n📊 Shape Progression:")
    print(f"  1. Cleaned data:     {df_cleaned.shape[0]:6,} rows × {df_cleaned.shape[1]:3d} cols")
    print(f"  2. Engineered:       {df_engineered.shape[0]:6,} rows × {df_engineered.shape[1]:3d} cols (+{df_engineered.shape[1] - df_cleaned.shape[1]} features)")
    print(f"  3. Selected:         {df_selected.shape[0]:6,} rows × {df_selected.shape[1]:3d} cols ({(df_selected.shape[1]/df_engineered.shape[1]*100):.1f}% retained)")
    
    print(f"\n📋 Selected Features ({len(selected_features)} total):")
    # Get numeric columns only
    numeric_selected = [col for col in selected_features if col in df_engineered.columns]
    non_numeric_selected = [col for col in df_selected.columns if col not in numeric_selected]
    
    if non_numeric_selected:
        print(f"\n  Non-numeric (metadata):")
        for col in non_numeric_selected:
            print(f"    • {col}")
    
    print(f"\n  Numeric features ({len(numeric_selected)}):")
    for i, col in enumerate(numeric_selected, 1):
        var = df_engineered[col].var()
        print(f"    {i:2d}. {col:40s} (var: {var:.8f})")
    
    print(f"\n⏱️ Timing:")
    print(f"  Total elapsed: {elapsed:.2f} seconds")
    print(f"  Per feature: {(elapsed/len(numeric_selected)*1000):.2f} ms")
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Ready for model training")
    
    return df_selected, selected_features, elapsed


if __name__ == "__main__":
    df_selected, selected_features, elapsed = run_feature_pipeline(
        input_path="data/processed/spotify_cleaned.csv",
        output_engineered="data/processed/spotify_engineered.csv",
        output_selected="data/processed/spotify_features.csv",
    )
    
    if df_selected is None:
        print("\n❌ Pipeline failed. Exiting.")
        exit(1)
    
    print(f"\n📁 Output files:")
    print(f"   • data/processed/spotify_engineered.csv — all engineered features")
    print(f"   • data/processed/spotify_features.csv — selected features only")
