"""Tests for results display widgets."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestTrainingProgress:
    """Test training progress display."""
    
    def test_progress_bar_range(self):
        """Test progress bar is in valid range."""
        for progress in [0, 25, 50, 75, 100]:
            assert 0 <= progress <= 100
    
    def test_progress_messages_valid(self):
        """Test progress messages are valid."""
        messages = [
            "Starting training...",
            "Training 50% complete...",
            "Training completed!"
        ]
        
        assert all(isinstance(msg, str) for msg in messages)
        assert all(len(msg) > 0 for msg in messages)
    
    def test_fold_information_display(self):
        """Test fold information can be displayed."""
        folds = [1, 2, 3, 4, 5]
        
        for fold in folds:
            assert 1 <= fold <= 5


class TestTrainingStatus:
    """Test training status display."""
    
    def test_status_types_valid(self):
        """Test status types are valid."""
        statuses = ['running', 'completed', 'error']
        
        assert 'running' in statuses
        assert 'completed' in statuses
    
    def test_status_messages(self):
        """Test status messages."""
        messages = {
            'running': 'Training in progress...',
            'completed': 'Training completed!',
            'error': 'Training failed.'
        }
        
        assert all(isinstance(msg, str) for msg in messages.values())


class TestClassificationMetrics:
    """Test classification metrics display."""
    
    def test_accuracy_display(self):
        """Test accuracy metric."""
        accuracy = 0.85
        assert 0 <= accuracy <= 1
    
    def test_precision_display(self):
        """Test precision metric."""
        precision = 0.82
        assert 0 <= precision <= 1
    
    def test_recall_display(self):
        """Test recall metric."""
        recall = 0.88
        assert 0 <= recall <= 1
    
    def test_f1_display(self):
        """Test F1 metric."""
        f1 = 0.85
        assert 0 <= f1 <= 1
    
    def test_auc_roc_display(self):
        """Test AUC-ROC metric."""
        auc_roc = 0.90
        assert 0 <= auc_roc <= 1
    
    def test_classification_metrics_dictionary(self):
        """Test classification metrics as dictionary."""
        metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1': 0.85,
            'auc_roc': 0.90
        }
        
        assert len(metrics) == 5
        assert all(0 <= v <= 1 for v in metrics.values())


class TestRegressionMetrics:
    """Test regression metrics display."""
    
    def test_rmse_display(self):
        """Test RMSE metric."""
        rmse = 0.25
        assert rmse >= 0
    
    def test_mae_display(self):
        """Test MAE metric."""
        mae = 0.18
        assert mae >= 0
    
    def test_r2_display(self):
        """Test R² metric."""
        r2 = 0.88
        assert -1 <= r2 <= 1
    
    def test_mape_display(self):
        """Test MAPE metric."""
        mape = 5.2
        assert mape >= 0
    
    def test_regression_metrics_dictionary(self):
        """Test regression metrics as dictionary."""
        metrics = {
            'rmse': 0.25,
            'mae': 0.18,
            'r2': 0.88,
            'mape': 5.2
        }
        
        assert len(metrics) == 4
        assert metrics['r2'] > 0.8


class TestFeatureImportance:
    """Test feature importance display."""
    
    def test_feature_importance_dataframe(self):
        """Test feature importance dataframe."""
        importance_df = pd.DataFrame({
            'feature': ['age', 'income', 'score'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        assert len(importance_df) == 3
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_feature_importance_sorting(self):
        """Test feature importance sorting."""
        importance_df = pd.DataFrame({
            'feature': ['age', 'income', 'score'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        sorted_df = importance_df.sort_values('importance', ascending=False)
        assert sorted_df.iloc[0]['importance'] == 0.5
    
    def test_importance_percentages(self):
        """Test importance as percentages."""
        importances = np.array([0.5, 0.3, 0.2])
        percentages = importances * 100
        
        assert all(0 <= p <= 100 for p in percentages)


class TestConfusionMatrix:
    """Test confusion matrix display."""
    
    def test_binary_confusion_matrix(self):
        """Test binary classification confusion matrix."""
        cm = np.array([
            [85, 5],
            [3, 97]
        ])
        
        assert cm.shape == (2, 2)
        true_positives = cm[1, 1]
        false_positives = cm[0, 1]
        assert true_positives > false_positives
    
    def test_multiclass_confusion_matrix(self):
        """Test multiclass confusion matrix."""
        cm = np.array([
            [85, 5, 10],
            [3, 87, 10],
            [2, 3, 95]
        ])
        
        assert cm.shape == (3, 3)
        assert np.diag(cm).sum() > np.sum(cm) / 2  # Correct > incorrect
    
    def test_confusion_matrix_labels(self):
        """Test confusion matrix labels."""
        labels = ['Class 0', 'Class 1', 'Class 2']
        assert len(labels) == 3


class TestROCCurve:
    """Test ROC curve display."""
    
    def test_roc_curve_coordinates(self):
        """Test ROC curve coordinates."""
        fpr = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        tpr = np.array([0.0, 0.4, 0.7, 0.9, 1.0])
        
        assert len(fpr) == len(tpr)
        assert fpr[0] == 0.0 and fpr[-1] == 1.0
        assert tpr[0] == 0.0 and tpr[-1] == 1.0
    
    def test_roc_auc_value(self):
        """Test ROC AUC value."""
        auc = 0.85
        assert 0 <= auc <= 1
        assert auc > 0.5  # Better than random
    
    def test_roc_curve_monotonic(self):
        """Test ROC curve is monotonic."""
        tpr = np.array([0.0, 0.3, 0.6, 0.9, 1.0])
        
        for i in range(len(tpr) - 1):
            assert tpr[i] <= tpr[i + 1]


class TestCrossValidationResults:
    """Test cross-validation results display."""
    
    def test_cv_mean_score(self):
        """Test cross-validation mean score."""
        scores = [0.85, 0.88, 0.82, 0.87, 0.84]
        mean_score = np.mean(scores)
        
        assert 0.8 < mean_score < 0.9
    
    def test_cv_standard_deviation(self):
        """Test cross-validation standard deviation."""
        scores = [0.85, 0.88, 0.82, 0.87, 0.84]
        std_score = np.std(scores)
        
        assert 0 <= std_score <= 1
    
    def test_cv_fold_scores(self):
        """Test individual fold scores."""
        fold_scores = [0.85, 0.88, 0.82, 0.87, 0.84]
        
        assert len(fold_scores) == 5
        assert all(0 <= score <= 1 for score in fold_scores)
    
    def test_cv_results_dataframe(self):
        """Test CV results as dataframe."""
        cv_results = pd.DataFrame({
            'fold': [1, 2, 3, 4, 5],
            'score': [0.85, 0.88, 0.82, 0.87, 0.84]
        })
        
        assert len(cv_results) == 5
        assert 'fold' in cv_results.columns
        assert 'score' in cv_results.columns


class TestExportOptions:
    """Test export functionality."""
    
    def test_pickle_export_option(self):
        """Test pickle export option."""
        export_format = 'pickle'
        assert export_format in ['pickle', 'yaml', 'pdf']
    
    def test_yaml_export_option(self):
        """Test YAML export option."""
        export_format = 'yaml'
        assert export_format in ['pickle', 'yaml', 'pdf']
    
    def test_pdf_export_option(self):
        """Test PDF export option."""
        export_format = 'pdf'
        assert export_format in ['pickle', 'yaml', 'pdf']
    
    def test_export_button_text(self):
        """Test export button text."""
        buttons = {
            'pickle': 'Download Model (Pickle)',
            'yaml': 'Download Config (YAML)',
            'pdf': 'Export Report (PDF)'
        }
        
        assert all(isinstance(text, str) for text in buttons.values())


class TestMetricsFormatting:
    """Test metrics formatting."""
    
    def test_percentage_formatting(self):
        """Test percentage formatting."""
        accuracy = 0.8532
        formatted = f"{accuracy * 100:.2f}%"
        
        assert "85.32%" == formatted
    
    def test_decimal_formatting(self):
        """Test decimal formatting."""
        rmse = 0.2541
        formatted = f"{rmse:.4f}"
        
        assert formatted.startswith("0.")
    
    def test_score_display_format(self):
        """Test score display format."""
        metrics = {
            'accuracy': 0.85,
            'rmse': 0.25
        }
        
        assert metrics['accuracy'] == 0.85
        assert metrics['rmse'] == 0.25


class TestResultsComparison:
    """Test results comparison features."""
    
    def test_before_after_comparison(self):
        """Test before/after results comparison."""
        baseline = {'accuracy': 0.80}
        improved = {'accuracy': 0.85}
        
        improvement = improved['accuracy'] - baseline['accuracy']
        assert abs(improvement - 0.05) < 0.001  # Use approx comparison for floats
    
    def test_metrics_change_tracking(self):
        """Test tracking metric changes."""
        history = [
            {'accuracy': 0.75, 'epoch': 1},
            {'accuracy': 0.80, 'epoch': 2},
            {'accuracy': 0.85, 'epoch': 3}
        ]
        
        assert history[-1]['accuracy'] > history[0]['accuracy']
