"""Collaborative filtering features."""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_interaction_matrix(user_item_data, n_neighbors=5):
    """Build user-item interaction matrix and fit KNN model."""
    # Expects: user_item_data with columns [user_id, track_id, rating/plays]
    # Pivot to create sparse matrix
    interaction_matrix = user_item_data.pivot_table(
        index='user_id',
        columns='track_id',
        values='plays',
        fill_value=0
    )
    
    # Fit KNN on interaction matrix
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
    knn.fit(interaction_matrix)
    
    return interaction_matrix, knn


def get_recommendations_cf(user_id, knn, interaction_matrix, df, k=10):
    """Get recommendations using collaborative filtering."""
    if user_id not in interaction_matrix.index:
        return None
    
    user_idx = interaction_matrix.index.get_loc(user_id)
    user_vector = interaction_matrix.iloc[user_idx].values.reshape(1, -1)
    
    # Find similar users
    distances, indices = knn.kneighbors(user_vector)
    
    # Get tracks from similar users (not yet rated by target user)
    recommendations = {}
    
    for idx in indices[0][1:]:  # Exclude self (idx 0)
        similar_user_id = interaction_matrix.index[idx]
        similar_user_tracks = interaction_matrix.iloc[idx]
        
        # Get tracks rated by similar user but not by target user
        target_user_tracks = interaction_matrix.iloc[user_idx]
        new_tracks = similar_user_tracks[target_user_tracks == 0]
        
        for track_id, plays in new_tracks.items():
            if track_id not in recommendations:
                recommendations[track_id] = 0
            recommendations[track_id] += plays
    
    # Sort by score (descending)
    top_tracks = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]
    
    if len(top_tracks) == 0:
        return None
    
    track_ids = [t[0] for t in top_tracks]
    scores = [t[1] for t in top_tracks]
    
    # Normalize scores
    max_score = max(scores) if max(scores) > 0 else 1
    scores = [s / max_score for s in scores]
    
    recommendations_df = pd.DataFrame({
        'track_id': track_ids,
        'cf_score': scores
    })
    
    return recommendations_df


if __name__ == "__main__":
    print("Collaborative Filtering Module")
    print("Requires user-track interaction data")
