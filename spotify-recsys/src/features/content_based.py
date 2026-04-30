"""Content-based filtering features."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Audio features for content-based recommendations
AUDIO_FEATURES = [
    'valence',           # Musical positiveness
    'energy',            # Intensity/activity
    'danceability',      # How suitable for dancing
    'acousticness',      # Acoustic vs electric
    'loudness',          # Volume (dB)
    'tempo',             # Beats per minute
    'speechiness',       # Spoken words
    'instrumentalness',  # Lack of vocals
]


def extract_audio_features(df):
    """Extract audio features from dataframe."""
    # Filter to available columns
    available = [f for f in AUDIO_FEATURES if f in df.columns]
    features_df = df[['track_id', 'track_name'] + available].copy()
    return features_df, available


def normalize_features(features_df, available_features):
    """Normalize audio features to 0-1 scale."""
    scaler = StandardScaler()
    
    # Scale features
    features_scaled = scaler.fit_transform(features_df[available_features])
    
    # Create normalized dataframe
    normalized_df = features_df.copy()
    normalized_df[available_features] = features_scaled
    
    return normalized_df, scaler, available_features


def compute_similarity_matrix(normalized_df, available_features):
    """Compute cosine similarity between all tracks."""
    feature_vectors = normalized_df[available_features].values
    similarity = cosine_similarity(feature_vectors)
    return similarity


def get_recommendations_cb(track_id, similarity_matrix, df, k=10):
    """Get top-K similar tracks based on content."""
    # Find track index
    track_idx = df[df['track_id'] == track_id].index
    
    if len(track_idx) == 0:
        return None
    
    track_idx = track_idx[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[track_idx]))
    
    # Sort by similarity (descending), exclude original track
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
    
    # Get track indices
    track_indices = [i[0] for i in sim_scores]
    scores = [s[1] for s in sim_scores]
    
    # Return recommendations
    recommendations = df.iloc[track_indices].copy()
    recommendations['similarity_score'] = scores
    
    return recommendations[['track_id', 'track_name', 'artist', 'similarity_score']]


if __name__ == "__main__":
    # Test
    print("Content-Based Filtering Module")
    print(f"Audio features: {AUDIO_FEATURES}")
