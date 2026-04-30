"""
Feature Engineering Module for Spotify Recommendation Engine.

Creates domain-specific, statistical, and interaction features from raw audio characteristics.
Each feature is documented with business logic explaining WHY it matters for recommendations.

Categories:
1. Domain Features (music theory & audio semantics)
2. Statistical Features (distributional properties)
3. Interaction Features (combined effects of multiple dimensions)
"""

import pandas as pd
import numpy as np
import os


def create_features(df):
    """
    Engineer 10+ new features across 3+ categories from audio features.
    
    Args:
        df: DataFrame with audio features (valence, energy, danceability, etc.)
    
    Returns:
        pd.DataFrame: Original + engineered features
    """
    df_engineered = df.copy()
    
    print("🔧 Creating Engineered Features...\n")
    
    # ===== DOMAIN FEATURES (Music Theory & Audio Semantics) =====
    
    # Feature 1: Valence-Energy "Vibe" Score
    # WHY: Happy/positive (high valence) + Intense (high energy) = uplifting, energizing track
    # Use case: Recommend for workout playlists, party moods, motivation
    df_engineered['vibe_uplifting'] = df['valence'] * df['energy']
    print("✓ Feature 1: vibe_uplifting = valence × energy")
    print("  → Captures 'happy & energetic' vibe for mood-based recommendations")
    
    # Feature 2: Danceability-Tempo Efficiency
    # WHY: High danceability + appropriate tempo = rhythm-friendly dance track
    # Normalized by dividing by mean tempo to capture relative efficiency
    # Use case: Dance playlist recommendations, genre classification
    mean_tempo = df['tempo'].mean()
    df_engineered['dance_rhythm_match'] = (df['danceability'] * df['tempo']) / mean_tempo
    print("\n✓ Feature 2: dance_rhythm_match = (danceability × tempo) / mean_tempo")
    print("  → Measures how well danceability aligns with tempo rhythm")
    
    # Feature 3: Acoustic-Electric Balance Index
    # WHY: Inverse relationship — low acousticness + high energy = electric, synth-heavy
    # Use case: Genre identification (acoustic vs electronic), mood classification
    df_engineered['electric_index'] = (1 - df['acousticness']) * df['energy']
    print("\n✓ Feature 3: electric_index = (1 - acousticness) × energy")
    print("  → Identifies electronic/synth-heavy vs acoustic tracks")
    
    # Feature 4: Complexity Score (Instrumental Density)
    # WHY: High instrumentalness + high energy = complex instrumentation (progressive, jazz, etc.)
    # Use case: Genre/style classification, appeal to musicians/audiophiles
    df_engineered['instrumental_complexity'] = df['instrumentalness'] * df['energy']
    print("\n✓ Feature 4: instrumental_complexity = instrumentalness × energy")
    print("  → Captures complex instrumental arrangements (progressive, jazz)")
    
    # Feature 5: Lyrical Presence Index
    # WHY: High speechiness + low instrumentalness = vocal-heavy, lyrics-focused
    # Use case: Identify rap, spoken word, storytelling tracks
    df_engineered['vocal_intensity'] = df['speechiness'] * (1 - df['instrumentalness'])
    print("\n✓ Feature 5: vocal_intensity = speechiness × (1 - instrumentalness)")
    print("  → Identifies vocal-heavy tracks (rap, spoken word, storytelling)")
    
    # Feature 6: Loudness-Energy Consistency
    # WHY: Loudness should correlate with energy; deviation = production quirk
    # Use case: Identify mastered/compressed tracks, quality assessment
    loudness_norm = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())
    df_engineered['loudness_energy_consistency'] = 1 - np.abs(loudness_norm - df['energy'])
    print("\n✓ Feature 6: loudness_energy_consistency = 1 - |loudness_norm - energy|")
    print("  → Detects tracks with consistent production (correlated loudness/energy)")
    
    # ===== STATISTICAL FEATURES (Distributional Properties) =====
    
    # Feature 7: Audio Feature Variance (Complexity/Richness)
    # WHY: High variance = diverse instrumentation, evolving dynamics; low variance = repetitive
    # Use case: Distinguish background music (low variance) from engaging tracks (high variance)
    audio_features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
    df_engineered['feature_variance'] = df[audio_features].var(axis=1)
    print("\n✓ Feature 7: feature_variance = variance([valence, energy, danceability, acousticness, tempo])")
    print("  → High variance = rich, evolving track; low variance = repetitive/hypnotic")
    
    # Feature 8: Loudness Z-Score (Relative Loudness)
    # WHY: Track loudness relative to distribution = compression/mastering quality indicator
    # Use case: Quality assessment, mastering style classification
    loudness_mean = df['loudness'].mean()
    loudness_std = df['loudness'].std()
    df_engineered['loudness_zscore'] = (df['loudness'] - loudness_mean) / (loudness_std + 1e-8)
    print("\n✓ Feature 8: loudness_zscore = (loudness - mean) / std")
    print("  → Identifies exceptionally loud (commercial) or quiet (intimate) tracks")
    
    # Feature 9: Tempo Percentile Rank
    # WHY: Track tempo relative to distribution = genre indicator (slow ballad vs fast dance)
    # Use case: Genre classification, pace classification
    df_engineered['tempo_percentile'] = df['tempo'].rank(pct=True)
    print("\n✓ Feature 9: tempo_percentile = percentile rank of tempo")
    print("  → Identifies genre pace: slow ballads (0.0-0.3) vs fast dance (0.7-1.0)")
    
    # ===== INTERACTION FEATURES (Combined Effects) =====
    
    # Feature 10: Chill-Out Index
    # WHY: Low energy + high acousticness = relaxing, intimate track
    # Use case: Chill/relax playlist recommendations
    df_engineered['chill_index'] = (1 - df['energy']) * df['acousticness']
    print("\n✓ Feature 10: chill_index = (1 - energy) × acousticness")
    print("  → Identifies relaxing, intimate acoustic tracks (chill playlists)")
    
    # Feature 11: Party Potential Score
    # WHY: High danceability + high energy + high valence = ultimate party track
    # Use case: Party/club playlist recommendations
    df_engineered['party_potential'] = (df['danceability'] * df['energy'] * df['valence']) ** (1/3)
    print("\n✓ Feature 11: party_potential = cbrt(danceability × energy × valence)")
    print("  → Geometric mean of party drivers (dance, energy, happiness)")
    
    # Feature 12: Silence-Depth Index
    # WHY: Low energy + low loudness + low tempo = quiet, introspective track
    # Use case: Meditation, focus, ambient playlist recommendations
    energy_norm = df['energy'] / df['energy'].max()
    loudness_norm = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())
    tempo_norm = df['tempo'] / df['tempo'].max()
    df_engineered['silence_depth'] = (1 - energy_norm) * (1 - loudness_norm) * (1 - tempo_norm)
    print("\n✓ Feature 12: silence_depth = (1 - energy_norm) × (1 - loudness_norm) × (1 - tempo_norm)")
    print("  → Identifies deeply quiet, meditative tracks")
    
    # Summary
    print("\n" + "="*70)
    print(f"✅ FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_engineered.columns) - len(df.columns)}")
    print(f"Total features: {len(df_engineered.columns)}")
    print(f"\nNew feature columns:")
    new_cols = [col for col in df_engineered.columns if col not in df.columns]
    for i, col in enumerate(new_cols, 1):
        print(f"  {i:2d}. {col}")
    
    return df_engineered


def save_engineered_data(df_engineered, output_path="data/processed/spotify_engineered.csv"):
    """
    Save engineered features to CSV.
    
    Args:
        df_engineered: DataFrame with engineered features
        output_path: Path to save CSV
    
    Returns:
        str: Path where data was saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_engineered.to_csv(output_path, index=False)
    print(f"\n💾 Engineered data saved to {output_path}")
    return output_path


def select_features(df, correlation_threshold=0.95, variance_threshold=0.01):
    """
    Select features by removing:
    1. Features with correlation > threshold (keep first, drop redundant)
    2. Features with variance < threshold * overall_variance
    
    Args:
        df: DataFrame with numeric features
        correlation_threshold: Max correlation to keep both features (default 0.95)
        variance_threshold: Min variance as fraction of overall variance (default 0.01)
    
    Returns:
        tuple: (list of selected feature names, reduced dataframe)
    """
    print("\n🔍 FEATURE SELECTION\n")
    print("="*70)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()
    
    print(f"Starting with {len(numeric_cols)} numeric features")
    
    # ===== STEP 1: Remove highly correlated features =====
    print(f"\n📊 Step 1: Removing highly correlated features (threshold > {correlation_threshold})")
    
    corr_matrix = df_numeric.corr().abs()
    dropped_corr = set()
    
    # Upper triangle of correlation matrix
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                # Keep first (lower index), drop second
                dropped_corr.add(col_j)
                print(f"  ⚠ Dropping '{col_j}' (corr={corr_matrix.iloc[i, j]:.3f} with '{col_i}')")
    
    if not dropped_corr:
        print("  ✓ No highly correlated features found")
    
    # ===== STEP 2: Remove low-variance features =====
    print(f"\n📉 Step 2: Removing low-variance features (threshold < {variance_threshold} × overall_variance)")
    
    overall_variance = df_numeric.var().mean()
    variance_threshold_value = variance_threshold * overall_variance
    
    print(f"  • Overall mean variance: {overall_variance:.6f}")
    print(f"  • Variance threshold: {variance_threshold_value:.6f}")
    
    dropped_variance = []
    for col in df_numeric.columns:
        if col not in dropped_corr:
            col_variance = df_numeric[col].var()
            if col_variance < variance_threshold_value:
                dropped_variance.append((col, col_variance))
                print(f"  ⚠ Dropping '{col}' (variance={col_variance:.6f} < threshold)")
    
    if not dropped_variance:
        print("  ✓ No low-variance features found")
    
    # ===== Build selected features list =====
    dropped_var_names = [col for col, _ in dropped_variance]
    selected_features = [col for col in numeric_cols 
                        if col not in dropped_corr and col not in dropped_var_names]
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("FEATURE SELECTION SUMMARY")
    print("="*70)
    print(f"Original features: {len(numeric_cols)}")
    print(f"Dropped (correlation): {len(dropped_corr)}")
    print(f"Dropped (low variance): {len(dropped_variance)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"\n✅ Selected {len(selected_features)} features:")
    for i, col in enumerate(selected_features, 1):
        var = df_numeric[col].var()
        print(f"  {i:2d}. {col:35s} (variance: {var:.6f})")
    
    # Return reduced dataframe (keep non-numeric + selected numeric)
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    df_selected = df[non_numeric_cols + selected_features].copy()
    
    return selected_features, df_selected


if __name__ == "__main__":
    print("🚀 Spotify Recommendation Engine - Feature Engineering\n")
    
    # Load cleaned data
    try:
        df = pd.read_csv('data/processed/spotify_cleaned.csv')
        print(f"✓ Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns\n")
    except FileNotFoundError:
        print("❌ Error: Cleaned data not found at 'data/processed/spotify_cleaned.csv'")
        print("   Run cleaner.py first: python src/data/cleaner.py")
        exit(1)
    
    # Create engineered features
    try:
        df_engineered = create_features(df)
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        exit(1)
    
    # Save engineered data
    try:
        save_engineered_data(df_engineered)
    except Exception as e:
        print(f"❌ Failed to save engineered data: {e}")
        exit(1)
    
    # Display sample
    print("\n📋 Sample of engineered features:")
    print(df_engineered[['track_id', 'track_name', 'vibe_uplifting', 'dance_rhythm_match', 
                         'electric_index', 'chill_index', 'party_potential', 'silence_depth']].head(10))
    
    print("\n✅ Feature engineering complete! Ready for model training.")
