"""
Pytest fixtures and test utilities for Spotify RecSys project.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample Spotify dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'track_id': [f'track_{i}' for i in range(n_samples)],
        'track_name': [f'Song {i}' for i in range(n_samples)],
        'energy': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(50, 200, n_samples),
        'danceability': np.random.uniform(0, 1, n_samples),
        'loudness': np.random.uniform(-20, 5, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'instrumentalness': np.random.uniform(0, 1, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
        'speechiness': np.random.uniform(0, 1, n_samples),
        'liveness': np.random.uniform(0, 1, n_samples),
        'duration_ms': np.random.uniform(120000, 600000, n_samples),
        'popularity': np.random.randint(0, 100, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def broken_data():
    """Create a broken/invalid dataset for testing error handling."""
    return pd.DataFrame({
        'track_id': ['t1', 't2'],
        'energy': [2.5, -1.0],  # Out of range [0, 1]
        'tempo': [np.nan, np.nan],  # All NaN
    })


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Get data directory path."""
    return project_root / 'data' / 'processed'


@pytest.fixture
def models_dir(project_root):
    """Get models directory path."""
    return project_root / 'models'
