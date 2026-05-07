"""
Tests for Generic Trainer Module

Comprehensive test suite for GenericTrainer class covering:
- Initialization and problem type detection
- Training with different model types
- Cross-validation
- Predictions (both continuous and probabilistic)
- Feature importance extraction
- Method chaining
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.trainer import GenericTrainer


class TestTrainerInitialization:
    """Test GenericTrainer initialization and setup."""
    
    def test_init_with_valid_data(self):
        """Test initialization with valid data."""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        assert trainer.X is not None
        assert trainer.y is not None
        assert trainer.problem_type == 'classification'
    
    def test_init_mismatched_shapes(self):
        """Test initialization with mismatched X and y shapes."""
        X = pd.DataFrame(np.random.rand(100, 10))
        y = pd.Series(np.random.randint(0, 2, 50))
        
        with pytest.raises(ValueError, match="same length"):
            GenericTrainer(X, y)
    
    def test_invalid_problem_type(self):
        """Test invalid problem type raises error."""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        with pytest.raises(ValueError, match="'classification' or 'regression'"):
            GenericTrainer(X, y, problem_type='invalid')
    
    def test_auto_detect_classification(self):
        """Test auto-detection of classification problem."""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y)  # No problem_type specified
        assert trainer.problem_type == 'classification'
    
    def test_auto_detect_regression(self):
        """Test auto-detection of regression problem."""
        X, y = make_regression(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y)  # No problem_type specified
        assert trainer.problem_type == 'regression'


class TestTrainerModels:
    """Test training with different model types."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        return X, pd.Series(y)
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        return X, pd.Series(y)
    
    def test_train_logistic_regression(self, classification_data):
        """Test training LogisticRegression."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('LogisticRegression')
        
        assert trainer.model is not None
        assert trainer.model_name == 'LogisticRegression'
    
    def test_train_random_forest_classification(self, classification_data):
        """Test training RandomForest classifier."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('RandomForest')
        
        assert trainer.model is not None
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_gradient_boosting_classification(self, classification_data):
        """Test training GradientBoosting classifier."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('GradientBoosting')
        
        assert trainer.model is not None
        assert trainer.cv_scores['mean_score'] > 0
    
    def test_train_xgboost_classification(self, classification_data):
        """Test training XGBoost classifier."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('XGBoost')
        
        assert trainer.model is not None
    
    def test_train_linear_regression(self, regression_data):
        """Test training LinearRegression."""
        X, y = regression_data
        trainer = GenericTrainer(X, y, problem_type='regression')
        trainer.train('LinearRegression')
        
        assert trainer.model is not None
    
    def test_train_random_forest_regression(self, regression_data):
        """Test training RandomForest regressor."""
        X, y = regression_data
        trainer = GenericTrainer(X, y, problem_type='regression')
        trainer.train('RandomForest')
        
        assert trainer.model is not None
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_gradient_boosting_regression(self, regression_data):
        """Test training GradientBoosting regressor."""
        X, y = regression_data
        trainer = GenericTrainer(X, y, problem_type='regression')
        trainer.train('GradientBoosting')
        
        assert trainer.model is not None
    
    def test_train_invalid_model(self, classification_data):
        """Test training invalid model raises error."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        
        with pytest.raises(ValueError, match="not available"):
            trainer.train('InvalidModel')
    
    def test_train_regression_model_on_classification(self, classification_data):
        """Test training regression model on classification problem."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        
        # LinearRegression should not be in classification models
        with pytest.raises(ValueError, match="not available"):
            trainer.train('LinearRegression')


class TestTrainerCrossValidation:
    """Test cross-validation functionality."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X)
        return X, pd.Series(y)
    
    def test_cv_scores_computed(self, classification_data):
        """Test CV scores are computed."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification', cv_folds=5)
        trainer.train('RandomForest')
        
        cv_scores = trainer.get_cv_scores()
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        assert cv_scores['mean_score'] >= 0
    
    def test_different_cv_folds(self, classification_data):
        """Test training with different CV folds."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification', cv_folds=3)
        trainer.train('RandomForest')
        
        assert trainer.cv_folds == 3
    
    def test_cv_override(self, classification_data):
        """Test overriding CV folds during training."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification', cv_folds=5)
        trainer.train('RandomForest', cv_folds=3)
        
        assert trainer.cv_folds == 3


class TestTrainerPredictions:
    """Test prediction functionality."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X)
        return X, pd.Series(y)
    
    def test_predict_without_training(self, classification_data):
        """Test predict without training raises error."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        
        with pytest.raises(ValueError, match="not trained"):
            trainer.predict(X)
    
    def test_predict_after_training(self, classification_data):
        """Test predictions after training."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('RandomForest')
        
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba_classification(self, classification_data):
        """Test probability predictions for classification."""
        X, y = classification_data
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('LogisticRegression')
        
        proba = trainer.predict_proba(X)
        assert proba.shape == (len(y), 2)  # Binary classification
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_predict_proba_regression_error(self):
        """Test predict_proba raises error for regression."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='regression')
        trainer.train('LinearRegression')
        
        with pytest.raises(ValueError, match="only available for classification"):
            trainer.predict_proba(X)


class TestFeatureImportance:
    """Test feature importance extraction."""
    
    def test_feature_importance_tree_model(self):
        """Test feature importance extraction from tree model."""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('RandomForest')
        
        importance = trainer.get_feature_importance()
        assert importance is not None
        assert len(importance) == 10
    
    def test_no_feature_importance_linear_model(self):
        """Test linear model returns None for feature importance."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('LogisticRegression')
        
        importance = trainer.get_feature_importance()
        assert importance is None


class TestMethodChaining:
    """Test method chaining capability."""
    
    def test_chaining_train_predict(self):
        """Test chaining train and predict methods."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        result = trainer.train('RandomForest')
        
        assert isinstance(result, GenericTrainer)
        predictions = trainer.predict(X)
        assert len(predictions) == len(y)


class TestReportsAndOutput:
    """Test report generation."""
    
    def test_get_report(self):
        """Test report generation."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('RandomForest')
        
        report = trainer.get_report()
        assert 'model_name' in report
        assert 'problem_type' in report
        assert 'cv_scores' in report
    
    def test_print_report(self, capsys):
        """Test report printing."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('RandomForest')
        trainer.print_report()
        
        captured = capsys.readouterr()
        assert "TRAINING REPORT" in captured.out


class TestAvailableModels:
    """Test available models listing."""
    
    def test_available_classification_models(self):
        """Test listing available classification models."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        models = trainer.get_available_models()
        
        assert 'classification' in models
        assert 'RandomForest' in models['classification']
        assert 'LogisticRegression' in models['classification']
    
    def test_available_regression_models(self):
        """Test listing available regression models."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='regression')
        models = trainer.get_available_models()
        
        assert 'regression' in models
        assert 'RandomForest' in models['regression']
        assert 'LinearRegression' in models['regression']


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_feature(self):
        """Test with single feature."""
        X = pd.DataFrame(np.random.rand(100, 1), columns=['feat_0'])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('LogisticRegression')
        
        assert trainer.model is not None
    
    def test_many_features(self):
        """Test with many features."""
        X = pd.DataFrame(np.random.rand(50, 100))
        y = pd.Series(np.random.randint(0, 2, 50))
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('LogisticRegression')
        
        assert trainer.model is not None
    
    def test_large_dataset(self):
        """Test with larger dataset."""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        trainer = GenericTrainer(X, y, problem_type='classification')
        trainer.train('RandomForest', cv_folds=3)
        
        assert trainer.cv_scores['mean_score'] > 0
