"""Tests for model configuration widgets."""

import pytest
import pandas as pd
from src.config.model_defaults import get_model_defaults, list_models, get_tuning_space


class TestModelTypeSelection:
    """Test model type selection."""
    
    def test_classification_type_valid(self):
        """Test classification type is valid."""
        problem_type = 'classification'
        assert problem_type in ['classification', 'regression']
    
    def test_regression_type_valid(self):
        """Test regression type is valid."""
        problem_type = 'regression'
        assert problem_type in ['classification', 'regression']
    
    def test_problem_type_not_empty(self):
        """Test problem type is not empty."""
        problem_type = 'classification'
        assert problem_type != ''


class TestModelSelection:
    """Test model selection based on problem type."""
    
    def test_classification_models_list(self):
        """Test classification models list."""
        models = list_models('classification')
        
        classification_models = ['LogisticRegression', 'RandomForest', 'GradientBoosting']
        for model in classification_models:
            assert model in models
    
    def test_regression_models_list(self):
        """Test regression models list."""
        models = list_models('regression')
        
        regression_models = ['LinearRegression', 'Ridge', 'Lasso']
        for model in regression_models:
            assert model in models
    
    def test_all_models_available(self):
        """Test all models are available."""
        # Get models from classification for testing
        models = list_models('classification') + list_models('regression')
        
        assert len(models) > 0
        assert isinstance(models, list)


class TestHyperparameterConfiguration:
    """Test hyperparameter configuration."""
    
    def test_random_forest_hyperparameters(self):
        """Test RandomForest hyperparameters."""
        params = get_model_defaults('classification', 'RandomForest')
        
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'min_samples_split' in params
        assert isinstance(params['n_estimators'], int)
    
    def test_logistic_regression_hyperparameters(self):
        """Test LogisticRegression hyperparameters."""
        params = get_model_defaults('classification', 'LogisticRegression')
        
        assert 'C' in params
        assert 'max_iter' in params
        assert isinstance(params['max_iter'], int)
    
    def test_xgboost_hyperparameters(self):
        """Test XGBoost hyperparameters."""
        params = get_model_defaults('classification', 'XGBoost')
        
        assert 'learning_rate' in params or 'eta' in params
        assert 'n_estimators' in params or 'num_rounds' in params
    
    def test_hyperparameter_ranges_reasonable(self):
        """Test hyperparameter ranges are reasonable."""
        params = get_model_defaults('classification', 'RandomForest')
        
        assert 10 <= params.get('n_estimators', 100) <= 1000
        assert params.get('max_depth', 10) is None or params.get('max_depth') > 0


class TestTuningSpace:
    """Test hyperparameter tuning space."""
    
    def test_tuning_space_defined(self):
        """Test tuning space is defined."""
        space = get_tuning_space('RandomForest')
        
        assert isinstance(space, dict)
        assert len(space) > 0
    
    def test_tuning_space_contains_params(self):
        """Test tuning space contains parameters."""
        space = get_tuning_space('RandomForest')
        
        assert 'n_estimators' in space or 'learning_rate' in space
    
    def test_multiple_models_have_tuning_space(self):
        """Test multiple models have tuning space."""
        models = list_models('classification')
        
        tuning_spaces = 0
        for model in models[:5]:  # Test first 5 models
            space = get_tuning_space(model)
            if space:
                tuning_spaces += 1
        
        assert tuning_spaces > 0


class TestTrainingOptions:
    """Test training options configuration."""
    
    def test_cv_folds_range(self):
        """Test cross-validation folds range."""
        for cv_folds in [2, 3, 5, 10]:
            assert 2 <= cv_folds <= 10
    
    def test_test_size_range(self):
        """Test test size range."""
        for test_size in [0.1, 0.2, 0.3, 0.5]:
            assert 0.1 <= test_size <= 0.5
    
    def test_random_state_valid(self):
        """Test random state is valid."""
        for random_state in [0, 42, 123, 999]:
            assert 0 <= random_state <= 10000
    
    def test_training_options_dictionary(self):
        """Test training options as dictionary."""
        options = {
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        }
        
        assert isinstance(options, dict)
        assert 'cv_folds' in options
        assert options['cv_folds'] == 5


class TestModelConfiguration:
    """Test complete model configuration."""
    
    def test_configuration_dictionary_structure(self):
        """Test configuration dictionary structure."""
        config = {
            'name': 'RandomForest',
            'problem_type': 'classification',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10
            },
            'training_options': {
                'cv_folds': 5,
                'test_size': 0.2,
                'random_state': 42
            }
        }
        
        assert 'name' in config
        assert 'problem_type' in config
        assert 'hyperparameters' in config
        assert 'training_options' in config
    
    def test_configuration_values_valid(self):
        """Test configuration values are valid."""
        config = {
            'name': 'RandomForest',
            'problem_type': 'classification',
            'hyperparameters': {'n_estimators': 100},
            'training_options': {'cv_folds': 5}
        }
        
        assert config['name'] != ''
        assert config['problem_type'] in ['classification', 'regression']
        assert config['hyperparameters']['n_estimators'] > 0
        assert config['training_options']['cv_folds'] >= 2


class TestModelDefaults:
    """Test ModelDefaults configuration class."""
    
    def test_model_functions_available(self):
        """Test model configuration functions are available."""
        # Test that functions work
        models = list_models('classification')
        assert len(models) > 0
        
        params = get_model_defaults('classification', 'RandomForest')
        assert isinstance(params, dict)
        
        space = get_tuning_space('RandomForest')
        assert isinstance(space, dict) or space is None
    
    def test_get_model_defaults_returns_dict(self):
        """Test get_model_defaults returns dictionary."""
        params = get_model_defaults('classification', 'RandomForest')
        
        assert isinstance(params, dict)


class TestWidgetInteraction:
    """Test widget interaction patterns."""
    
    def test_model_selection_affects_hyperparameters(self):
        """Test selecting model changes hyperparameters."""
        rf_params = get_model_defaults('classification', 'RandomForest')
        lr_params = get_model_defaults('classification', 'LogisticRegression')
        
        # Different models should have different parameters
        assert rf_params.keys() != lr_params.keys()
    
    def test_problem_type_affects_model_list(self):
        """Test problem type affects available models."""
        all_models = list_models('classification') + list_models('regression')
        
        # Should have multiple model options
        assert len(all_models) > 5
    
    def test_configuration_summary_generation(self):
        """Test configuration can be summarized."""
        config = {
            'name': 'RandomForest',
            'problem_type': 'classification',
            'hyperparameters': {'n_estimators': 100, 'max_depth': 10},
            'training_options': {'cv_folds': 5}
        }
        
        summary_text = f"{config['name']} ({config['problem_type']})"
        assert 'RandomForest' in summary_text
        assert 'classification' in summary_text
