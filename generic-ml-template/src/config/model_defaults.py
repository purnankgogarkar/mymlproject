"""
Model Defaults and Hyperparameter Tuning Spaces

Pre-tuned hyperparameters for all ML models and search spaces for Optuna.
"""

from typing import Dict, Any, List, Tuple

# Pre-tuned hyperparameters for all models
MODEL_DEFAULTS = {
    'classification': {
        'LogisticRegression': {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42,
        },
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'GradientBoosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'XGBoost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        },
        'LightGBM': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        },
        'SVM': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42,
        },
        'KNeighbors': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
        },
        'DecisionTree': {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'NeuralNetwork': {
            'hidden_layer_sizes': (100, 50),
            'learning_rate': 0.001,
            'max_iter': 500,
            'random_state': 42,
        },
    },
    'regression': {
        'LinearRegression': {},
        'Ridge': {
            'alpha': 1.0,
        },
        'Lasso': {
            'alpha': 1.0,
            'max_iter': 1000,
        },
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'GradientBoosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'XGBoost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        },
        'LightGBM': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        },
        'SVM': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
        },
        'KNeighbors': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
        },
        'DecisionTree': {
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
        },
        'NeuralNetwork': {
            'hidden_layer_sizes': (100, 50),
            'learning_rate': 0.001,
            'max_iter': 500,
            'random_state': 42,
        },
    }
}

# Hyperparameter search spaces for Optuna
TUNING_SPACES = {
    'LogisticRegression': {
        'C': ('loguniform', 1e-4, 1e2),
        'solver': ('categorical', ['lbfgs', 'liblinear']),
    },
    'RandomForest': {
        'n_estimators': ('int', 50, 300),
        'max_depth': ('int', 5, 50),
        'min_samples_split': ('int', 2, 10),
        'min_samples_leaf': ('int', 1, 5),
    },
    'GradientBoosting': {
        'n_estimators': ('int', 50, 300),
        'learning_rate': ('loguniform', 0.001, 0.1),
        'max_depth': ('int', 3, 10),
        'min_samples_split': ('int', 2, 10),
    },
    'XGBoost': {
        'n_estimators': ('int', 50, 300),
        'learning_rate': ('loguniform', 0.001, 0.1),
        'max_depth': ('int', 3, 10),
        'subsample': ('uniform', 0.5, 1.0),
        'colsample_bytree': ('uniform', 0.5, 1.0),
    },
    'LightGBM': {
        'n_estimators': ('int', 50, 300),
        'learning_rate': ('loguniform', 0.001, 0.1),
        'num_leaves': ('int', 15, 100),
        'min_data_in_leaf': ('int', 10, 50),
    },
    'SVM': {
        'C': ('loguniform', 1e-2, 1e3),
        'kernel': ('categorical', ['linear', 'rbf', 'poly']),
        'gamma': ('categorical', ['scale', 'auto']),
    },
    'KNeighbors': {
        'n_neighbors': ('int', 3, 15),
        'weights': ('categorical', ['uniform', 'distance']),
    },
    'DecisionTree': {
        'max_depth': ('int', 5, 30),
        'min_samples_split': ('int', 2, 10),
        'min_samples_leaf': ('int', 1, 5),
    },
}


def get_model_defaults(problem_type: str, model_name: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for a model.
    
    Args:
        problem_type: 'classification' or 'regression'
        model_name: Name of model (e.g., 'RandomForest')
        
    Returns:
        Dictionary of default hyperparameters
        
    Raises:
        ValueError: If problem type or model name invalid
    """
    if problem_type not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    if model_name not in MODEL_DEFAULTS[problem_type]:
        raise ValueError(f"Unknown model for {problem_type}: {model_name}")
    
    return MODEL_DEFAULTS[problem_type][model_name].copy()


def get_tuning_space(model_name: str) -> Dict[str, Tuple]:
    """
    Get hyperparameter search space for Optuna.
    
    Args:
        model_name: Name of model
        
    Returns:
        Dictionary of parameter search ranges
        
    Raises:
        ValueError: If model not found in tuning spaces
    """
    if model_name not in TUNING_SPACES:
        raise ValueError(f"No tuning space defined for: {model_name}")
    
    return TUNING_SPACES[model_name].copy()


def list_models(problem_type: str) -> List[str]:
    """
    List all available models for a problem type.
    
    Args:
        problem_type: 'classification' or 'regression'
        
    Returns:
        List of model names
    """
    if problem_type not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    return list(MODEL_DEFAULTS[problem_type].keys())


def update_defaults(problem_type: str, model_name: str, hyperparams: Dict[str, Any]) -> None:
    """
    Update default hyperparameters for a model.
    
    Args:
        problem_type: 'classification' or 'regression'
        model_name: Name of model
        hyperparams: New hyperparameters to set
    """
    if problem_type not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    if model_name not in MODEL_DEFAULTS[problem_type]:
        raise ValueError(f"Unknown model for {problem_type}: {model_name}")
    
    # Update the defaults
    MODEL_DEFAULTS[problem_type][model_name].update(hyperparams)
