"""
Tests for model predictions and inference.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def dummy_model():
    """Create a simple dummy model for testing."""
    # Train a simple model
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model = GradientBoostingClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)
    
    return model


@pytest.fixture
def test_features():
    """Create test feature vectors."""
    np.random.seed(42)
    return np.random.rand(10, 10)


@pytest.fixture
def scaler_fixture():
    """Create a fitted scaler."""
    X = np.random.rand(100, 10)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


class TestModelPredictions:
    """Test model prediction functionality."""
    
    def test_model_makes_predictions(self, dummy_model, test_features):
        """Test that model can make predictions."""
        predictions = dummy_model.predict(test_features)
        
        assert predictions is not None
        assert len(predictions) == len(test_features)
    
    def test_predictions_are_binary_classification(self, dummy_model, test_features):
        """Test that predictions are binary (0 or 1)."""
        predictions = dummy_model.predict(test_features)
        
        unique_predictions = np.unique(predictions)
        assert len(unique_predictions) <= 2, "Should be binary classification"
        assert all(p in [0, 1] for p in unique_predictions)
    
    def test_predictions_have_correct_shape(self, dummy_model, test_features):
        """Test that predictions have correct shape."""
        predictions = dummy_model.predict(test_features)
        
        assert predictions.shape[0] == test_features.shape[0]
    
    def test_model_produces_probabilities(self, dummy_model, test_features):
        """Test that model can produce probability predictions."""
        proba = dummy_model.predict_proba(test_features)
        
        assert proba is not None
        assert proba.shape[0] == len(test_features)
        assert proba.shape[1] == 2  # Binary classification


class TestPredictionRanges:
    """Test that predictions are in expected ranges."""
    
    def test_probabilities_in_valid_range(self, dummy_model, test_features):
        """Test that probabilities are between 0 and 1."""
        proba = dummy_model.predict_proba(test_features)
        
        assert np.all(proba >= 0), "Probabilities should be >= 0"
        assert np.all(proba <= 1), "Probabilities should be <= 1"
    
    def test_probabilities_sum_to_one(self, dummy_model, test_features):
        """Test that class probabilities sum to 1."""
        proba = dummy_model.predict_proba(test_features)
        
        sums = np.sum(proba, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(len(test_features)))
    
    def test_decision_scores_are_numeric(self, dummy_model, test_features):
        """Test that decision function produces numeric scores."""
        if hasattr(dummy_model, 'decision_function'):
            scores = dummy_model.decision_function(test_features)
            assert scores is not None
            assert np.all(np.isfinite(scores))


class TestModelLoading:
    """Test model persistence (save/load)."""
    
    def test_model_can_be_saved(self, dummy_model, tmp_path):
        """Test that model can be serialized with joblib."""
        model_path = tmp_path / "test_model.pkl"
        
        joblib.dump(dummy_model, str(model_path))
        
        assert model_path.exists()
    
    def test_model_can_be_loaded(self, dummy_model, tmp_path, test_features):
        """Test that saved model can be loaded and used."""
        model_path = tmp_path / "test_model.pkl"
        
        # Save model
        joblib.dump(dummy_model, str(model_path))
        
        # Load model
        loaded_model = joblib.load(str(model_path))
        
        # Verify loaded model works
        original_predictions = dummy_model.predict(test_features)
        loaded_predictions = loaded_model.predict(test_features)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_loaded_model_matches_original(self, dummy_model, tmp_path, test_features):
        """Test that predictions from loaded model match original."""
        model_path = tmp_path / "test_model.pkl"
        
        joblib.dump(dummy_model, str(model_path))
        loaded_model = joblib.load(str(model_path))
        
        original_proba = dummy_model.predict_proba(test_features)
        loaded_proba = loaded_model.predict_proba(test_features)
        
        np.testing.assert_array_almost_equal(original_proba, loaded_proba)


class TestModelConsistency:
    """Test model consistency and determinism."""
    
    def test_model_deterministic_predictions(self, dummy_model, test_features):
        """Test that same input produces same prediction."""
        pred1 = dummy_model.predict(test_features)
        pred2 = dummy_model.predict(test_features)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_model_consistent_probabilities(self, dummy_model, test_features):
        """Test that same input produces same probabilities."""
        proba1 = dummy_model.predict_proba(test_features)
        proba2 = dummy_model.predict_proba(test_features)
        
        np.testing.assert_array_almost_equal(proba1, proba2)
    
    def test_predictions_independent_of_order(self, dummy_model, test_features):
        """Test that prediction order doesn't affect predictions."""
        predictions_forward = dummy_model.predict(test_features)
        predictions_reversed = dummy_model.predict(test_features[::-1])
        
        # Predictions should match when we reverse back
        np.testing.assert_array_equal(
            predictions_forward,
            predictions_reversed[::-1]
        )


class TestModelEdgeCases:
    """Test edge cases in model predictions."""
    
    def test_model_handles_single_sample(self, dummy_model):
        """Test model prediction on single sample."""
        sample = np.random.rand(1, 10)
        
        prediction = dummy_model.predict(sample)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    def test_model_handles_zero_features(self, dummy_model):
        """Test model with zero-valued features."""
        zero_sample = np.zeros((1, 10))
        
        prediction = dummy_model.predict(zero_sample)
        
        assert prediction is not None
        assert prediction[0] in [0, 1]
    
    def test_model_handles_max_features(self, dummy_model):
        """Test model with maximum-valued features."""
        max_sample = np.ones((1, 10))
        
        prediction = dummy_model.predict(max_sample)
        
        assert prediction is not None
        assert prediction[0] in [0, 1]
    
    def test_model_handles_large_batch(self, dummy_model):
        """Test model on large batch of samples."""
        large_batch = np.random.rand(1000, 10)
        
        predictions = dummy_model.predict(large_batch)
        
        assert len(predictions) == 1000
        assert all(p in [0, 1] for p in predictions)


class TestModelAttributes:
    """Test model attributes and properties."""
    
    def test_model_has_feature_importances(self, dummy_model):
        """Test that model exposes feature importance."""
        assert hasattr(dummy_model, 'feature_importances_')
        assert len(dummy_model.feature_importances_) == 10
    
    def test_feature_importances_sum_reasonably(self, dummy_model):
        """Test that feature importances are reasonable."""
        importances = dummy_model.feature_importances_
        
        assert np.all(importances >= 0), "Importances should be non-negative"
        assert np.sum(importances) > 0, "At least one feature should be important"
    
    def test_model_has_classes(self, dummy_model):
        """Test that model has classes attribute."""
        assert hasattr(dummy_model, 'classes_')
        assert len(dummy_model.classes_) == 2
        assert 0 in dummy_model.classes_
        assert 1 in dummy_model.classes_
