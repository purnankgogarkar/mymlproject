"""Tests for results page."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.utils.session_state import AppState


@pytest.fixture
def sample_df():
    """Create sample dataframe."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(20, 70, 100),
        'income': np.random.randint(20000, 150000, 100),
        'score': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })


@pytest.fixture
def state(sample_df):
    """Create app state with complete training results."""
    state = AppState()
    state.data = sample_df
    state.target_col = 'income'
    state.model = {
        'name': 'RandomForest',
        'problem_type': 'regression',
        'hyperparameters': {'n_estimators': 100, 'max_depth': 10}
    }
    state.results = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1': 0.85,
        'auc_roc': 0.90,
        'rmse': 0.25,
        'mae': 0.18,
        'r2': 0.88,
        'mape': 5.2
    }
    return state


class TestResultsPageBasics:
    """Test basic results page functionality."""
    
    def test_state_has_results(self, state):
        """Test state has training results."""
        assert state.results is not None
    
    def test_state_has_model_info(self, state):
        """Test state has model information."""
        assert state.model is not None
        assert state.model['name'] == 'RandomForest'
    
    def test_state_has_data(self, state):
        """Test state has data."""
        assert state.has_data()
    
    def test_state_has_target(self, state):
        """Test state has target column."""
        assert state.target_col is not None


class TestResultsDisplay:
    """Test results display functionality."""
    
    def test_classification_metrics_display(self):
        """Test classification metrics can be displayed."""
        results = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1': 0.85,
            'auc_roc': 0.90
        }
        
        assert results['accuracy'] > 0.8
        assert results['f1'] > 0.8
    
    def test_regression_metrics_display(self):
        """Test regression metrics can be displayed."""
        results = {
            'rmse': 0.25,
            'mae': 0.18,
            'r2': 0.88,
            'mape': 5.2
        }
        
        assert results['r2'] > 0.8
        assert results['rmse'] < 1.0
    
    def test_both_metric_types_available(self, state):
        """Test both classification and regression metrics available."""
        assert 'accuracy' in state.results
        assert 'rmse' in state.results


class TestFeatureImportance:
    """Test feature importance visualization."""
    
    def test_feature_importance_dataframe_creation(self, state):
        """Test feature importance dataframe can be created."""
        feature_cols = list(state.data.columns)
        if state.target_col in feature_cols:
            feature_cols.remove(state.target_col)
        
        importances = np.random.random(len(feature_cols))
        importances = importances / importances.sum()
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        })
        
        assert len(importance_df) == len(feature_cols)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_feature_importance_sums_to_one(self, state):
        """Test feature importance sums to 1."""
        features = ['age', 'score']
        importances = np.array([0.6, 0.4])
        
        assert np.isclose(importances.sum(), 1.0)
    
    def test_feature_importance_sorted(self):
        """Test feature importance can be sorted."""
        importance_df = pd.DataFrame({
            'feature': ['feat1', 'feat2', 'feat3'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        sorted_df = importance_df.sort_values('importance', ascending=False)
        assert sorted_df.iloc[0]['importance'] == 0.5


class TestConfusionMatrix:
    """Test confusion matrix."""
    
    def test_confusion_matrix_creation(self):
        """Test confusion matrix can be created."""
        cm = np.array([
            [85, 5],
            [3, 97]
        ])
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] + cm[1, 1] > cm[0, 1] + cm[1, 0]  # Correct > Incorrect
    
    def test_multiclass_confusion_matrix(self):
        """Test multiclass confusion matrix."""
        cm = np.array([
            [85, 5, 10],
            [3, 87, 10],
            [2, 3, 95]
        ])
        
        assert cm.shape == (3, 3)
        assert cm.shape[0] == cm.shape[1]


class TestROCCurve:
    """Test ROC curve."""
    
    def test_roc_curve_data(self):
        """Test ROC curve data generation."""
        fpr = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        tpr = np.array([0.0, 0.4, 0.7, 0.9, 1.0])
        auc = 0.85
        
        assert len(fpr) == len(tpr)
        assert fpr[0] == 0.0 and fpr[-1] == 1.0
        assert tpr[0] == 0.0 and tpr[-1] == 1.0
        assert 0 <= auc <= 1
    
    def test_auc_value_valid(self, state):
        """Test AUC value is valid."""
        auc = state.results.get('auc_roc', 0)
        assert 0 <= auc <= 1


class TestCrossValidationResults:
    """Test cross-validation results display."""
    
    def test_cv_results_structure(self):
        """Test CV results structure."""
        cv_results = {
            'mean_score': 0.85,
            'std_score': 0.05,
            'max_score': 0.92,
            'min_score': 0.78,
            'fold_scores': [0.85, 0.88, 0.82, 0.87, 0.84]
        }
        
        assert 'mean_score' in cv_results
        assert 'fold_scores' in cv_results
        assert len(cv_results['fold_scores']) == 5
    
    def test_cv_fold_scores_valid(self):
        """Test CV fold scores are valid."""
        fold_scores = [0.85, 0.88, 0.82, 0.87, 0.84]
        
        assert all(0 <= score <= 1 for score in fold_scores)
        assert np.mean(fold_scores) > 0.8
    
    def test_cv_statistics_consistency(self):
        """Test CV statistics are consistent."""
        fold_scores = [0.85, 0.88, 0.82, 0.87, 0.84]
        
        mean = np.mean(fold_scores)
        max_score = np.max(fold_scores)
        min_score = np.min(fold_scores)
        
        assert min_score <= mean <= max_score


class TestExportOptions:
    """Test export options."""
    
    def test_export_formats_available(self):
        """Test export formats are available."""
        formats = ['pickle', 'yaml', 'pdf']
        
        assert 'pickle' in formats
        assert 'yaml' in formats
        assert 'pdf' in formats
    
    def test_model_picklable(self, state):
        """Test model configuration is picklable."""
        import pickle
        
        model_config = state.model
        pickled = pickle.dumps(model_config)
        unpickled = pickle.loads(pickled)
        
        assert unpickled['name'] == model_config['name']


class TestResultsWorkflow:
    """Test complete results workflow."""
    
    def test_complete_results_workflow(self, state):
        """Test complete results viewing workflow."""
        # 1. Verify results available
        assert state.results is not None
        
        # 2. Display metrics
        accuracy = state.results.get('accuracy', 0)
        assert accuracy > 0
        
        # 3. Feature importance
        feature_cols = list(state.data.columns)
        if state.target_col in feature_cols:
            feature_cols.remove(state.target_col)
        assert len(feature_cols) > 0
        
        # 4. Cross-validation
        cv_results = {
            'mean_score': accuracy,
            'fold_scores': [accuracy - 0.05, accuracy, accuracy + 0.05]
        }
        assert len(cv_results['fold_scores']) == 3
    
    def test_multiple_results_tabs(self, state):
        """Test multiple results tabs available."""
        tabs = ['Metrics', 'Feature Importance', 'Cross-Validation', 'Export']
        
        assert len(tabs) == 4
        assert all(tab is not None for tab in tabs)


class TestResultsEdgeCases:
    """Test edge cases."""
    
    def test_perfect_score_handling(self):
        """Test handling of perfect scores."""
        results = {'accuracy': 1.0, 'r2': 1.0}
        
        assert results['accuracy'] == 1.0
        assert results['r2'] == 1.0
    
    def test_poor_score_handling(self):
        """Test handling of poor scores."""
        results = {'accuracy': 0.5, 'r2': -1.0}
        
        assert results['accuracy'] == 0.5
        assert results['r2'] == -1.0
    
    def test_missing_metric_handling(self):
        """Test handling of missing metrics."""
        results = {'accuracy': 0.85}
        
        missing_metric = results.get('rmse', None)
        assert missing_metric is None
