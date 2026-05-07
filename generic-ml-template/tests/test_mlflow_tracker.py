"""
Tests for MLflow Tracker

Test experiment tracking, metrics logging, and model artifacts.
"""

import pytest
import os
import tempfile
import pickle
from unittest.mock import MagicMock, patch

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.config.mlflow_tracker import MLflowTracker


# Module-level model class for pickling tests
class SimpleModel:
    """Simple picklable model for testing."""
    def __init__(self):
        self.params = {'n_estimators': 100}


@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Clean up any active MLflow runs before and after each test."""
    # Clean up before test
    if MLFLOW_AVAILABLE:
        try:
            mlflow.end_run()
        except:
            pass
    
    yield
    
    # Clean up after test
    if MLFLOW_AVAILABLE:
        try:
            mlflow.end_run()
        except:
            pass


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowTrackerInitialization:
    """Test MLflow tracker initialization."""
    
    def test_init_raises_without_mlflow(self):
        """Test initialization raises without mlflow installed."""
        with patch('src.config.mlflow_tracker.MLFLOW_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="MLflow not installed"):
                MLflowTracker("test_experiment")
    
    def test_init_with_experiment_name(self):
        """Test initialization with experiment name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            
            assert tracker.experiment_name == "test_experiment"
            # tracking_uri is normalized to file:// format for Windows compatibility
            assert tracker.tracking_uri.startswith("file:///")
    
    def test_init_with_tags(self):
        """Test initialization with tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tags = {'author': 'test', 'version': '1.0'}
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir, tags=tags)
            
            assert tracker.tags == tags


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowRunManagement:
    """Test MLflow run management."""
    
    def test_start_run(self):
        """Test starting a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            result = tracker.start_run(run_name="test_run")
            
            assert isinstance(result, MLflowTracker)
            assert result is tracker
            assert tracker.run_id is not None
            
            tracker.end_run()
    
    def test_start_run_returns_self(self):
        """Test start_run returns self for chaining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            result = tracker.start_run()
            
            assert isinstance(result, MLflowTracker)
            tracker.end_run()


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowLogging:
    """Test logging functionality."""
    
    def test_log_params(self):
        """Test logging parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            
            result = tracker.log_params({
                'n_estimators': 100,
                'max_depth': 10,
            })
            
            assert isinstance(result, MLflowTracker)
            tracker.end_run()
    
    def test_log_metrics(self):
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            
            result = tracker.log_metrics({
                'accuracy': 0.95,
                'f1_score': 0.92,
            })
            
            assert isinstance(result, MLflowTracker)
            tracker.end_run()
    
    def test_log_metrics_with_step(self):
        """Test logging metrics with step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            
            tracker.log_metrics({'loss': 0.5}, step=1)
            tracker.log_metrics({'loss': 0.3}, step=2)
            
            tracker.end_run()
    
    def test_log_params_returns_self(self):
        """Test log_params returns self for chaining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            
            result = tracker.log_params({'param': 1})
            
            assert isinstance(result, MLflowTracker)
            tracker.end_run()


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowArtifacts:
    """Test artifact logging."""
    
    def test_log_config(self):
        """Test logging config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            
            config = {'model': 'RandomForest', 'n_estimators': 100}
            result = tracker.log_config(config)
            
            assert isinstance(result, MLflowTracker)
            tracker.end_run()
    
    def test_log_model(self):
        """Test logging model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            
            # Use module-level SimpleModel for pickling
            model = SimpleModel()
            result = tracker.log_model(model)
            
            assert isinstance(result, MLflowTracker)
            tracker.end_run()


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowChaining:
    """Test method chaining."""
    
    def test_full_chain(self):
        """Test full method chaining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            
            result = (tracker
                      .start_run()
                      .log_params({'n_estimators': 100})
                      .log_metrics({'accuracy': 0.95})
                      .end_run())
            
            assert isinstance(result, MLflowTracker)


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowUtilities:
    """Test utility methods."""
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            'model': {
                'params': {
                    'n_estimators': 100
                }
            }
        }
        
        flattened = MLflowTracker._flatten_dict(nested)
        
        assert 'model.params.n_estimators' in flattened
        assert flattened['model.params.n_estimators'] == 100
    
    def test_flatten_dict_with_custom_separator(self):
        """Test dictionary flattening with custom separator."""
        nested = {'a': {'b': 1}}
        flattened = MLflowTracker._flatten_dict(nested, sep='_')
        
        assert 'a_b' in flattened


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowRunEnding:
    """Test run ending."""
    
    def test_end_run(self):
        """Test ending a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            run_id = tracker.run_id
            
            result = tracker.end_run()
            
            assert isinstance(result, MLflowTracker)
            assert tracker.run_id is None


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
class TestMLflowEdgeCases:
    """Test edge cases."""
    
    def test_log_non_numeric_metrics_ignored(self):
        """Test non-numeric metrics are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker("test_experiment", tracking_uri=tmpdir)
            tracker.start_run()
            
            # Should not raise
            tracker.log_metrics({
                'accuracy': 0.95,
                'model_name': 'RandomForest',  # This should be ignored
            })
            
            tracker.end_run()
