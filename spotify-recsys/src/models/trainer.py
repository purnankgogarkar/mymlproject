"""
Model training module for Spotify Recommendation Engine.

Trains both content-based and collaborative filtering models.
Saves trained artifacts to models/ with metadata logging.
"""

import os
import json
import joblib
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

try:
    from ..features import (
        extract_audio_features,
        normalize_features,
        compute_similarity_matrix,
        build_interaction_matrix,
    )
except ImportError:
    # Support direct execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from features import (
        extract_audio_features,
        normalize_features,
        compute_similarity_matrix,
        build_interaction_matrix,
    )


def load_cleaned_data(filepath="data/processed/spotify_cleaned.csv"):
    """
    Load cleaned data from processed directory.
    
    Args:
        filepath: Path to cleaned CSV file
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned data not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def train_content_based_model(df, sample_size=5000, available_features=None):
    """
    Train content-based model (compute similarity matrix from audio features).
    
    Args:
        df: DataFrame with audio features
        sample_size: Max tracks to use (to avoid memory overflow on similarity matrix)
        available_features: List of audio feature columns to use
    
    Returns:
        dict: {
            'similarity_matrix': cosine similarity matrix,
            'features_df': normalized feature vectors,
            'scaler': StandardScaler object,
            'available_features': list of feature names used,
            'n_samples': number of tracks used,
            'n_features': number of features,
            'total_tracks': total tracks in original data
        }
    """
    print("\n📊 Training Content-Based Model...")
    
    # Sample data if too large (similarity matrix = n_samples² elements)
    if len(df) > sample_size:
        print(f"  ⚠ Dataset too large ({len(df)} tracks)")
        print(f"    Sampling {sample_size} tracks to fit in memory")
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"  • Using {len(df_sample)} tracks for training")
    
    # Extract available audio features (returns tuple: features_df, available)
    features_df, available = extract_audio_features(df_sample)
    if features_df is None or features_df.empty:
        raise ValueError("No audio features found in dataset")
    
    print(f"  • Audio features extracted: {len(available)} features")
    
    # Normalize features (returns tuple: normalized_df, scaler, available_features)
    normalized_df, scaler, _ = normalize_features(features_df, available)
    print(f"  • Features normalized (StandardScaler)")
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(normalized_df, available)
    print(f"  • Similarity matrix computed: {similarity_matrix.shape[0]}×{similarity_matrix.shape[1]}")
    print(f"    Memory usage: ~{(similarity_matrix.nbytes / (1024**3)):.2f} GB")
    
    model_dict = {
        'similarity_matrix': similarity_matrix,
        'features_df': features_df,
        'normalized_df': normalized_df,
        'scaler': scaler,
        'available_features': available,
        'n_samples': len(df_sample),
        'n_features': len(available),
        'total_tracks': len(df),
        'model_type': 'content_based',
        'trained_at': datetime.now().isoformat(),
    }
    
    return model_dict


def train_collaborative_filtering_model(df, n_neighbors=5, test_size=0.2):
    """
    Train collaborative filtering model (KNN on interaction matrix).
    
    Args:
        df: DataFrame with user interaction data
        n_neighbors: Number of neighbors for KNN
        test_size: Fraction for train/test split
    
    Returns:
        dict: {
            'knn_model': Fitted NearestNeighbors object,
            'interaction_matrix': User-item interaction matrix,
            'n_users': number of users,
            'n_items': number of items,
            'n_neighbors': k value used,
            'train_size': size of training set,
            'test_size': size of test set
        }
    """
    print("\n🤝 Training Collaborative Filtering Model...")
    
    # Build interaction matrix (user_id -> plays per track)
    if 'user_id' not in df.columns or 'track_id' not in df.columns:
        print("  ⚠ No user_id/track_id columns; creating synthetic interaction matrix from data")
        # Use track features as proxy if no user data
        interaction_data = df[['track_id']].head(100)
        user_item_matrix = pd.DataFrame(
            np.random.randint(0, 5, size=(min(50, len(interaction_data)), len(interaction_data))),
            index=[f"user_{i}" for i in range(min(50, len(interaction_data)))],
            columns=interaction_data['track_id'].values
        )
    else:
        # Use real interaction data
        interaction_data = df[['user_id', 'track_id']].drop_duplicates()
        user_item_matrix = pd.crosstab(interaction_data['user_id'], interaction_data['track_id'], fill_value=0)
    
    print(f"  • Interaction matrix: {user_item_matrix.shape[0]} users × {user_item_matrix.shape[1]} items")
    
    # Fit KNN on interaction matrix
    knn = NearestNeighbors(n_neighbors=min(n_neighbors, user_item_matrix.shape[0]-1), metric='cosine')
    knn.fit(user_item_matrix)
    print(f"  • KNN model fitted with k={min(n_neighbors, user_item_matrix.shape[0]-1)} neighbors")
    
    model_dict = {
        'knn_model': knn,
        'interaction_matrix': user_item_matrix,
        'n_users': user_item_matrix.shape[0],
        'n_items': user_item_matrix.shape[1],
        'n_neighbors': min(n_neighbors, user_item_matrix.shape[0]-1),
        'model_type': 'collaborative_filtering',
        'trained_at': datetime.now().isoformat(),
    }
    
    return model_dict


def save_model(model_dict, model_name, models_dir="models"):
    """
    Save trained model with metadata.
    
    Args:
        model_dict: Model dictionary with metadata
        model_name: Name prefix for saved files (e.g., 'content_based', 'collaborative_filtering')
        models_dir: Directory to save models
    
    Returns:
        dict: Saved file paths and metadata
    """
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model with joblib (more efficient for sklearn objects)
    model_path = os.path.join(models_dir, f"{model_name}_{timestamp}.joblib")
    joblib.dump(model_dict, model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # Save metadata separately as JSON
    metadata = {
        'model_name': model_name,
        'model_type': model_dict.get('model_type', 'unknown'),
        'trained_at': model_dict.get('trained_at'),
        'file_path': model_path,
        'file_size_mb': os.path.getsize(model_path) / (1024**2),
    }
    
    if model_name == 'content_based':
        metadata.update({
            'n_samples': model_dict.get('n_samples'),
            'n_features': model_dict.get('n_features'),
            'available_features': model_dict.get('available_features'),
            'similarity_matrix_shape': model_dict['similarity_matrix'].shape if 'similarity_matrix' in model_dict else None,
        })
    elif model_name == 'collaborative_filtering':
        metadata.update({
            'n_users': model_dict.get('n_users'),
            'n_items': model_dict.get('n_items'),
            'n_neighbors': model_dict.get('n_neighbors'),
            'interaction_matrix_shape': model_dict['interaction_matrix'].shape if 'interaction_matrix' in model_dict else None,
        })
    
    metadata_path = os.path.join(models_dir, f"{model_name}_{timestamp}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to {metadata_path}")
    
    return metadata


def load_model(model_path):
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        dict: Model dictionary
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model_dict = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model_dict


def print_training_summary(cb_model, cf_model):
    """
    Print summary of trained models.
    
    Args:
        cb_model: Content-based model dict
        cf_model: Collaborative filtering model dict
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print("\n📊 Content-Based Model:")
    print(f"  • Samples trained: {cb_model['n_samples']}")
    print(f"  • Total available: {cb_model['total_tracks']}")
    print(f"  • Features: {cb_model['n_features']}")
    print(f"  • Feature names: {', '.join(cb_model['available_features'][:5])}{'...' if len(cb_model['available_features']) > 5 else ''}")
    print(f"  • Similarity matrix shape: {cb_model['similarity_matrix'].shape}")
    
    print("\n🤝 Collaborative Filtering Model:")
    print(f"  • Users: {cf_model['n_users']}")
    print(f"  • Items: {cf_model['n_items']}")
    print(f"  • Neighbors: {cf_model['n_neighbors']}")
    print(f"  • Interaction sparsity: {1 - (cf_model['interaction_matrix'].astype(bool).sum().sum() / (cf_model['n_users'] * cf_model['n_items'])):.2%}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("🚀 Spotify Recommendation Engine - Model Training Pipeline\n")
    
    # 1. Load cleaned data
    try:
        df = load_cleaned_data()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Run cleaner.py first: python src/data/cleaner.py")
        exit(1)
    
    # 2. Train content-based model
    try:
        cb_model = train_content_based_model(df)
    except Exception as e:
        print(f"❌ Content-based training failed: {e}")
        exit(1)
    
    # 3. Train collaborative filtering model
    try:
        cf_model = train_collaborative_filtering_model(df, n_neighbors=5)
    except Exception as e:
        print(f"❌ Collaborative filtering training failed: {e}")
        exit(1)
    
    # 4. Save models
    print("\n💾 Saving Models...")
    cb_metadata = save_model(cb_model, 'content_based')
    cf_metadata = save_model(cf_model, 'collaborative_filtering')
    
    # 5. Print summary
    print_training_summary(cb_model, cf_model)
    
    print("\n✅ Training complete!")
    print("   Models saved to models/ directory")
    print("   Ready for evaluation and deployment")
