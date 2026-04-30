"""Feature engineering - CB, CF, Hybrid."""
from .content_based import (
    extract_audio_features,
    normalize_features,
    compute_similarity_matrix,
    get_recommendations_cb,
)
from .collaborative_filtering import (
    build_interaction_matrix,
    get_recommendations_cf,
)
from .hybrid import (
    blend_recommendations,
    rank_hybrid,
)
from .engineering import (
    create_features,
    save_engineered_data,
    select_features,
)

__all__ = [
    'extract_audio_features',
    'normalize_features',
    'compute_similarity_matrix',
    'get_recommendations_cb',
    'build_interaction_matrix',
    'get_recommendations_cf',
    'blend_recommendations',
    'rank_hybrid',
    'create_features',
    'save_engineered_data',
    'select_features',
]
