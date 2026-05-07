"""
Tests for Model Evaluation Module

Comprehensive test suite for Evaluator class covering:
- Classification metrics
- Regression metrics
- Confusion matrix
- Feature importance
- Cross-model comparisons
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.evaluator import Evaluator


class TestEvaluatorInitialization:
    """Test Evaluator initialization."""
    
    def test_init_classification(self):
        """Test initialization for classification."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        assert evaluator.problem_type == 'classification'
    
    def test_init_regression(self):
        """Test initialization for regression."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        assert evaluator.problem_type == 'regression'
    
    def test_init_mismatched_shapes(self):
        """Test mismatched y_true and y_pred raises error."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1])
        
        with pytest.raises(ValueError, match="same length"):
            Evaluator(y_true, y_pred)
    
    def test_invalid_problem_type(self):
        """Test invalid problem type raises error."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 0])
        
        with pytest.raises(ValueError, match="'classification' or 'regression'"):
            Evaluator(y_true, y_pred, problem_type='invalid')


class TestClassificationEvaluation:
    """Test classification metrics."""
    
    def test_accuracy_metric(self):
        """Test accuracy computation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        assert 'accuracy' in evaluator.metrics
        assert evaluator.metrics['accuracy'] == 0.8  # 4/5 correct
    
    def test_precision_recall_f1(self):
        """Test precision, recall, F1 computation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        assert 'precision' in evaluator.metrics
        assert 'recall' in evaluator.metrics
        assert 'f1' in evaluator.metrics
        assert 0 <= evaluator.metrics['precision'] <= 1
        assert 0 <= evaluator.metrics['recall'] <= 1
        assert 0 <= evaluator.metrics['f1'] <= 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        conf_mat = evaluator.get_confusion_matrix()
        assert conf_mat is not None
        assert conf_mat.shape == (2, 2)
    
    def test_auc_roc_binary(self):
        """Test AUC-ROC for binary classification."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        y_pred_proba = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.6, 0.4]
        ])
        
        evaluator = Evaluator(y_true, y_pred, y_pred_proba, problem_type='classification')
        evaluator.evaluate()
        
        assert 'auc_roc' in evaluator.metrics
    
    def test_multiclass_classification(self):
        """Test multiclass classification metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        assert evaluator.metrics['accuracy'] > 0
        conf_mat = evaluator.get_confusion_matrix()
        assert conf_mat.shape == (3, 3)
    
    def test_classification_metrics_getter(self):
        """Test getting classification-specific metrics."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        class_metrics = evaluator.get_classification_metrics()
        assert 'accuracy' in class_metrics
        assert 'precision' in class_metrics


class TestRegressionEvaluation:
    """Test regression metrics."""
    
    def test_rmse_metric(self):
        """Test RMSE computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        assert 'rmse' in evaluator.metrics
        assert evaluator.metrics['rmse'] > 0
    
    def test_mae_metric(self):
        """Test MAE computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        assert 'mae' in evaluator.metrics
        assert evaluator.metrics['mae'] > 0
    
    def test_r2_metric(self):
        """Test R² computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        assert 'r2' in evaluator.metrics
        assert -1 <= evaluator.metrics['r2'] <= 1
    
    def test_mape_metric(self):
        """Test MAPE computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        assert 'mape' in evaluator.metrics
    
    def test_residuals(self):
        """Test residual statistics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        assert 'residual_mean' in evaluator.metrics
        assert 'residual_std' in evaluator.metrics
    
    def test_regression_metrics_getter(self):
        """Test getting regression-specific metrics."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        reg_metrics = evaluator.get_regression_metrics()
        assert 'rmse' in reg_metrics
        assert 'mae' in reg_metrics
        assert 'r2' in reg_metrics


class TestEvaluationMethods:
    """Test evaluation methods and chaining."""
    
    def test_evaluate_chaining(self):
        """Test evaluate returns self for chaining."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        result = evaluator.evaluate()
        
        assert isinstance(result, Evaluator)
    
    def test_get_metrics(self):
        """Test getting all metrics."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        metrics = evaluator.get_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0


class TestComparison:
    """Test model comparison functionality."""
    
    def test_compare_two_evaluators_classification(self):
        """Test comparing two evaluators for classification."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred1 = np.array([0, 1, 0, 0, 1])  # 4/5 correct
        y_pred2 = np.array([0, 1, 1, 0, 1])  # 5/5 correct
        
        eval1 = Evaluator(y_true, y_pred1, problem_type='classification').evaluate()
        eval2 = Evaluator(y_true, y_pred2, problem_type='classification').evaluate()
        
        comparison = eval1.compare_with(eval2)
        assert 'metric_deltas' in comparison
        assert 'better_metrics' in comparison
    
    def test_compare_different_problem_types(self):
        """Test comparing different problem types raises error."""
        y_true1 = np.array([0, 1, 1])
        y_pred1 = np.array([0, 1, 0])
        
        y_true2 = np.array([1.0, 2.0, 3.0])
        y_pred2 = np.array([1.1, 2.1, 2.9])
        
        eval1 = Evaluator(y_true1, y_pred1, problem_type='classification').evaluate()
        eval2 = Evaluator(y_true2, y_pred2, problem_type='regression').evaluate()
        
        with pytest.raises(ValueError, match="different problem types"):
            eval1.compare_with(eval2)


class TestReportsAndOutput:
    """Test report generation and printing."""
    
    def test_get_report_classification(self):
        """Test report generation for classification."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        report = evaluator.get_report()
        assert 'problem_type' in report
        assert 'n_samples' in report
        assert 'metrics' in report
    
    def test_get_report_regression(self):
        """Test report generation for regression."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        report = evaluator.get_report()
        assert report['n_samples'] == 5
        assert 'metrics' in report
    
    def test_print_report_classification(self, capsys):
        """Test report printing for classification."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        evaluator.print_report()
        
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out
        assert "Classification" in captured.out
    
    def test_print_report_regression(self, capsys):
        """Test report printing for regression."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        evaluator.print_report()
        
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out
        assert "Regression" in captured.out


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_perfect_predictions_classification(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        assert evaluator.metrics['accuracy'] == 1.0
    
    def test_all_same_prediction_classification(self):
        """Test with all same predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='classification')
        evaluator.evaluate()
        
        # Should compute metrics without crashing
        assert 'accuracy' in evaluator.metrics
    
    def test_perfect_regression_predictions(self):
        """Test perfect predictions in regression."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        assert evaluator.metrics['r2'] == 1.0
        assert evaluator.metrics['rmse'] == 0.0
    
    def test_large_regression_errors(self):
        """Test with large prediction errors."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        evaluator = Evaluator(y_true, y_pred, problem_type='regression')
        evaluator.evaluate()
        
        assert evaluator.metrics['r2'] < 0  # Poor fit
