"""
Tests for Model Defaults Registry

Test hyperparameter defaults and tuning spaces.
"""

import pytest
from src.config.model_defaults import (
    get_model_defaults, get_tuning_space, list_models,
    update_defaults, MODEL_DEFAULTS
)


class TestModelDefaults:
    """Test model defaults retrieval."""
    
    def test_get_classification_defaults(self):
        """Test getting defaults for classification model."""
        defaults = get_model_defaults('classification', 'RandomForest')
        
        assert isinstance(defaults, dict)
        assert 'n_estimators' in defaults
        assert defaults['n_estimators'] == 100
    
    def test_get_regression_defaults(self):
        """Test getting defaults for regression model."""
        defaults = get_model_defaults('regression', 'LinearRegression')
        
        assert isinstance(defaults, dict)
    
    def test_get_all_classification_models(self):
        """Test defaults for all classification models."""
        models = [
            'LogisticRegression', 'RandomForest', 'GradientBoosting',
            'XGBoost', 'LightGBM', 'SVM', 'KNeighbors', 'DecisionTree', 'NeuralNetwork'
        ]
        
        for model_name in models:
            defaults = get_model_defaults('classification', model_name)
            assert isinstance(defaults, dict)
            assert len(defaults) > 0 or model_name == 'LinearRegression'
    
    def test_get_all_regression_models(self):
        """Test defaults for all regression models."""
        models = [
            'LinearRegression', 'Ridge', 'Lasso', 'RandomForest',
            'GradientBoosting', 'XGBoost', 'LightGBM', 'SVM',
            'KNeighbors', 'DecisionTree', 'NeuralNetwork'
        ]
        
        for model_name in models:
            defaults = get_model_defaults('regression', model_name)
            assert isinstance(defaults, dict)
    
    def test_get_invalid_problem_type(self):
        """Test invalid problem type raises error."""
        with pytest.raises(ValueError, match="Unknown problem type"):
            get_model_defaults('invalid_type', 'RandomForest')
    
    def test_get_invalid_model_name(self):
        """Test invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_defaults('classification', 'InvalidModel')
    
    def test_defaults_are_copies(self):
        """Test that returned defaults are copies, not references."""
        defaults1 = get_model_defaults('classification', 'RandomForest')
        defaults1['n_estimators'] = 999
        
        defaults2 = get_model_defaults('classification', 'RandomForest')
        
        # Original defaults should not be modified
        assert defaults2['n_estimators'] == 100


class TestTuningSpaces:
    """Test hyperparameter tuning spaces."""
    
    def test_get_tuning_space(self):
        """Test getting tuning space for a model."""
        space = get_tuning_space('RandomForest')
        
        assert isinstance(space, dict)
        assert 'n_estimators' in space
        assert 'max_depth' in space
    
    def test_tuning_space_format(self):
        """Test tuning space has correct format."""
        space = get_tuning_space('RandomForest')
        
        # Each parameter should have tuple format: (type, min, max) or (type, choices)
        for key, value in space.items():
            assert isinstance(value, tuple)
            assert len(value) >= 2
            assert value[0] in ['int', 'uniform', 'loguniform', 'categorical']
    
    def test_get_tuning_space_invalid_model(self):
        """Test invalid model raises error."""
        with pytest.raises(ValueError, match="No tuning space defined"):
            get_tuning_space('InvalidModel')
    
    def test_tuning_space_not_modified_after_get(self):
        """Test that returned tuning space is a copy."""
        space1 = get_tuning_space('RandomForest')
        space1['new_param'] = ('int', 1, 10)
        
        space2 = get_tuning_space('RandomForest')
        
        assert 'new_param' not in space2


class TestListModels:
    """Test listing available models."""
    
    def test_list_classification_models(self):
        """Test listing classification models."""
        models = list_models('classification')
        
        assert isinstance(models, list)
        assert len(models) == 9
        assert 'RandomForest' in models
        assert 'XGBoost' in models
    
    def test_list_regression_models(self):
        """Test listing regression models."""
        models = list_models('regression')
        
        assert isinstance(models, list)
        assert len(models) == 11
        assert 'RandomForest' in models
        assert 'LinearRegression' in models
    
    def test_list_invalid_problem_type(self):
        """Test invalid problem type raises error."""
        with pytest.raises(ValueError):
            list_models('invalid_type')


class TestUpdateDefaults:
    """Test updating default hyperparameters."""
    
    def test_update_single_param(self):
        """Test updating single hyperparameter."""
        original = get_model_defaults('classification', 'RandomForest')['n_estimators']
        
        update_defaults('classification', 'RandomForest', {'n_estimators': 200})
        
        updated = get_model_defaults('classification', 'RandomForest')['n_estimators']
        assert updated == 200
        
        # Restore original
        update_defaults('classification', 'RandomForest', {'n_estimators': original})
    
    def test_update_multiple_params(self):
        """Test updating multiple hyperparameters."""
        original_depth = get_model_defaults('classification', 'RandomForest')['max_depth']
        original_est = get_model_defaults('classification', 'RandomForest')['n_estimators']
        
        update_defaults('classification', 'RandomForest', {
            'n_estimators': 150,
            'max_depth': 15
        })
        
        updated = get_model_defaults('classification', 'RandomForest')
        assert updated['n_estimators'] == 150
        assert updated['max_depth'] == 15
        
        # Restore
        update_defaults('classification', 'RandomForest', {
            'n_estimators': original_est,
            'max_depth': original_depth
        })
    
    def test_update_invalid_problem_type(self):
        """Test updating with invalid problem type raises error."""
        with pytest.raises(ValueError):
            update_defaults('invalid_type', 'RandomForest', {'n_estimators': 200})
    
    def test_update_invalid_model_name(self):
        """Test updating with invalid model name raises error."""
        with pytest.raises(ValueError):
            update_defaults('classification', 'InvalidModel', {'n_estimators': 200})


class TestModelDefaultsStructure:
    """Test the structure of MODEL_DEFAULTS."""
    
    def test_has_classification_key(self):
        """Test MODEL_DEFAULTS has classification key."""
        assert 'classification' in MODEL_DEFAULTS
    
    def test_has_regression_key(self):
        """Test MODEL_DEFAULTS has regression key."""
        assert 'regression' in MODEL_DEFAULTS
    
    def test_classification_has_9_models(self):
        """Test classification has 9 models."""
        assert len(MODEL_DEFAULTS['classification']) == 9
    
    def test_regression_has_11_models(self):
        """Test regression has 11 models."""
        assert len(MODEL_DEFAULTS['regression']) == 11
    
    def test_all_defaults_are_dicts(self):
        """Test all defaults are dictionaries."""
        for problem_type, models in MODEL_DEFAULTS.items():
            for model_name, params in models.items():
                assert isinstance(params, dict)
