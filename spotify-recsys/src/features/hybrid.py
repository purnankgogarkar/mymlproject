"""Hybrid recommendation blending."""
import pandas as pd
import numpy as np


def blend_recommendations(cb_recs, cf_recs, cb_weight=0.5, cf_weight=0.5):
    """Blend content-based and collaborative filtering recommendations."""
    
    # Normalize weights
    total_weight = cb_weight + cf_weight
    cb_weight = cb_weight / total_weight
    cf_weight = cf_weight / total_weight
    
    blended = {}
    
    # Add CB recommendations
    if cb_recs is not None:
        for idx, row in cb_recs.iterrows():
            track_id = row['track_id']
            score = row.get('similarity_score', 0)
            blended[track_id] = {'cb_score': score, 'cf_score': 0}
    
    # Add CF recommendations
    if cf_recs is not None:
        for idx, row in cf_recs.iterrows():
            track_id = row['track_id']
            score = row.get('cf_score', 0)
            if track_id not in blended:
                blended[track_id] = {'cb_score': 0, 'cf_score': 0}
            blended[track_id]['cf_score'] = score
    
    # Compute blend scores
    blended_scores = []
    for track_id, scores in blended.items():
        blend_score = (scores['cb_score'] * cb_weight + 
                       scores['cf_score'] * cf_weight)
        blended_scores.append({
            'track_id': track_id,
            'cb_score': scores['cb_score'],
            'cf_score': scores['cf_score'],
            'blend_score': blend_score
        })
    
    # Sort by blend score
    blended_df = pd.DataFrame(blended_scores)
    blended_df = blended_df.sort_values('blend_score', ascending=False)
    
    return blended_df


def rank_hybrid(recommendations_df, k=10):
    """Rank and return top-K hybrid recommendations."""
    if recommendations_df is None or len(recommendations_df) == 0:
        return None
    
    return recommendations_df.head(k)


if __name__ == "__main__":
    print("Hybrid Recommendation Blender")
    print("Combines CB (content-based) + CF (collaborative filtering)")
