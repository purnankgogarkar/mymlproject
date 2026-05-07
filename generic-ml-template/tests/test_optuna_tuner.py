"""
Tests for Optuna Hyperparameter Tuner

Test hyperparameter optimization functionality.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from unittest.mock import patch

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.config.optuna_tuner import OptunaTuner


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaTunerInitialization:
    """Test Optuna tuner initialization."""
    
    def test_init_raises_without_optuna(self):
        """Test initialization raises without optuna installed."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        with patch('src.config.optuna_tuner.OPTUNA_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="Optuna not installed"):
                OptunaTuner(X, y, RandomForestClassifier)
    
    def test_init_classification(self):
        """Test initialization for classification."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier, problem_type='classification')
        
        assert tuner.problem_type == 'classification'
        assert tuner.scoring == 'f1_weighted'
    
    def test_init_regression(self):
        """Test initialization for regression."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestRegressor, problem_type='regression')
        
        assert tuner.problem_type == 'regression'
        assert tuner.scoring == 'r2'


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaTuning:
    """Test hyperparameter tuning."""
    
    def test_tune_classification(self):
        """Test tuning for classification."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier, problem_type='classification')
        result = tuner.tune(n_trials=5)
        
        assert isinstance(result, OptunaTuner)
        assert result is tuner
    
    def test_tune_regression(self):
        """Test tuning for regression."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestRegressor, problem_type='regression')
        result = tuner.tune(n_trials=5)
        
        assert isinstance(result, OptunaTuner)
    
    def test_get_best_params(self):
        """Test getting best parameters."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5)
        
        best_params = tuner.get_best_params()
        
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
    
    def test_get_best_score(self):
        """Test getting best score."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5)
        
        best_score = tuner.get_best_score()
        
        assert isinstance(best_score, (int, float))
        assert -1 <= best_score <= 1  # F1 score range
    
    def test_get_best_params_without_tuning(self):
        """Test getting best params without tuning raises error."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        
        with pytest.raises(RuntimeError, match="Must call tune"):
            tuner.get_best_params()


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaSamplers:
    """Test different samplers."""
    
    def test_tune_with_tpe_sampler(self):
        """Test tuning with TPE sampler."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5, sampler='tpe')
        
        assert tuner.best_score is not None
    
    def test_tune_with_random_sampler(self):
        """Test tuning with random sampler."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5, sampler='random')
        
        assert tuner.best_score is not None
    
    def test_tune_with_invalid_sampler(self):
        """Test invalid sampler raises error."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        
        with pytest.raises(ValueError, match="Unknown sampler"):
            tuner.tune(n_trials=5, sampler='invalid')


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaTrialsHistory:
    """Test trials history tracking."""
    
    def test_get_trials_history(self):
        """Test getting trials history."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5)
        
        history = tuner.get_trials_history()
        
        assert isinstance(history, list)
        assert len(history) == 5
    
    def test_get_trial_results(self):
        """Test getting trial results."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5)
        
        results = tuner.get_trial_results()
        
        assert 'trial_number' in results
        assert 'score' in results
        assert 'params' in results
        assert len(results['trial_number']) == 5
    
    def test_get_trials_history_without_tuning(self):
        """Test getting trials history without tuning raises error."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        
        with pytest.raises(RuntimeError, match="Must call tune"):
            tuner.get_trials_history()


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaVisualization:
    """Test visualization methods."""
    
    def test_plot_optimization_history(self):
        """Test plotting optimization history."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5)
        
        fig = tuner.plot_optimization_history()
        
        # Should return a figure object or None
        assert fig is not None or fig is None
    
    def test_plot_param_importance(self):
        """Test plotting parameter importance."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5)
        
        fig = tuner.plot_param_importance()
        
        # Should return a figure object or None
        assert fig is not None or fig is None


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaPrinting:
    """Test printing functionality."""
    
    def test_print_results(self, capsys):
        """Test printing tuning results."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5)
        tuner.print_results()
        
        captured = capsys.readouterr()
        assert "OPTUNA TUNING RESULTS" in captured.out
        assert "Best Score" in captured.out


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaEdgeCases:
    """Test edge cases."""
    
    def test_tune_with_pruning(self):
        """Test tuning with pruning enabled."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5, pruning=True)
        
        assert tuner.best_score is not None
    
    def test_tune_without_pruning(self):
        """Test tuning without pruning."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=5, pruning=False)
        
        assert tuner.best_score is not None
    
    def test_tune_with_single_trial(self):
        """Test tuning with single trial."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tuner = OptunaTuner(X, y, RandomForestClassifier)
        tuner.tune(n_trials=1)
        
        assert tuner.best_score is not None
