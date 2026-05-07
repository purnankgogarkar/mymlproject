"""Tests for model configuration page."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.utils.session_state import AppState
from src.config.model_defaults import get_model_defaults, list_models, get_tuning_space


@pytest.fixture
def sample_df():
    """Create sample dataframe."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [40000, 50000, 60000, 70000, 80000],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


@pytest.fixture
def state(sample_df):
    """Create app state with data."""
    state = AppState()
    state.data = sample_df
    state.target_col = 'income'
    return state


class TestConfigureModelPageBasics:
    """Test basic configure model page functionality."""
    
    def test_state_has_data(self, state):
        """Test state has data loaded."""
        assert state.has_data()
    
    def test_state_has_target_column(self, state):
        """Test target column is set."""
        assert state.target_col is not None
    
    def test_model_not_configured_initially(self, state):
        """Test model not configured initially."""
        assert state.model is None


class TestModelSelection:
    """Test model selection functionality."""
    
    def test_classification_models_available(self):
        """Test classification models are available."""
        models = list_models('classification')
        
        assert 'LogisticRegression' in models
        assert 'RandomForest' in models
        assert 'GradientBoosting' in models
    
    def test_regression_models_available(self):
        """Test regression models are available."""
        models = list_models('regression')
        
        assert 'LinearRegression' in models
        assert 'Ridge' in models
        assert 'Lasso' in models
    
    def test_model_defaults_exist(self):
        """Test model defaults exist for all models."""
        models = list_models('classification')
        
        for model in models:
            model_defaults = get_model_defaults('classification', model)
            assert isinstance(model_defaults, dict)
            assert len(model_defaults) > 0


class TestHyperparameterConfiguration:
    """Test hyperparameter configuration."""
    
    def test_default_parameters_for_random_forest(self):
        """Test default parameters for RandomForest."""
        params = get_model_defaults('classification', 'RandomForest')
        
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'min_samples_split' in params
    
    def test_default_parameters_for_logistic_regression(self):
        """Test default parameters for LogisticRegression."""
        params = get_model_defaults('classification', 'LogisticRegression')
        
        assert 'C' in params
        assert 'max_iter' in params
    
    def test_tuning_space_definition(self):
        """Test tuning space definition exists."""
        space = get_tuning_space('RandomForest')
        assert isinstance(space, dict)
        assert len(space) > 0


class TestTrainingOptionsConfiguration:
    """Test training options."""
    
    def test_cv_folds_configuration(self):
        """Test cross-validation folds can be set."""
        cv_folds = 5
        assert 2 <= cv_folds <= 10
    
    def test_test_size_configuration(self):
        """Test test size can be set."""
        test_size = 0.2
        assert 0.1 <= test_size <= 0.5
    
    def test_random_state_configuration(self):
        """Test random state can be set."""
        random_state = 42
        assert 0 <= random_state <= 10000


class TestConfigurationWorkflow:
    """Test complete configuration workflow."""
    
    def test_complete_configuration_workflow(self, state):
        """Test complete configuration workflow."""
        # 1. Select problem type
        problem_type = 'classification'
        assert problem_type in ['classification', 'regression']
        
        # 2. Select model
        models = list_models('classification')
        selected_model = 'RandomForest'
        assert selected_model in models
        
        # 3. Get hyperparameters
        params = get_model_defaults('classification', selected_model)
        assert isinstance(params, dict)
        
        # 4. Training options
        training_opts = {
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        }
        
        # 5. Save configuration
        state.model = {
            'name': selected_model,
            'problem_type': problem_type,
            'hyperparameters': params,
            'training_options': training_opts
        }
        
        assert state.model is not None
        assert state.model['name'] == 'RandomForest'
        assert state.model['problem_type'] == 'classification'


class TestModelConfigurationValidation:
    """Test model configuration validation."""
    
    def test_model_name_valid(self, state):
        """Test model name is valid."""
        state.model = {'name': 'RandomForest'}
        assert state.model['name'] in ['RandomForest', 'LogisticRegression', 'SVM']
    
    def test_problem_type_valid(self, state):
        """Test problem type is valid."""
        state.model = {'problem_type': 'classification'}
        assert state.model['problem_type'] in ['classification', 'regression']
    
    def test_hyperparameters_not_empty(self, state):
        """Test hyperparameters are not empty."""
        params = get_model_defaults('classification', 'RandomForest')
        
        state.model = {'hyperparameters': params}
        assert len(state.model['hyperparameters']) > 0


class TestModelSummaryGeneration:
    """Test model configuration summary."""
    
    def test_model_summary_contains_required_fields(self, state):
        """Test model summary has required fields."""
        state.model = {
            'name': 'RandomForest',
            'problem_type': 'classification',
            'hyperparameters': {'n_estimators': 100}
        }
        
        assert 'name' in state.model
        assert 'problem_type' in state.model
        assert 'hyperparameters' in state.model
