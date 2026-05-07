"""
YAML Configuration Loader

Loads, validates, and manages ML pipeline configuration files with:
- Environment variable substitution
- Auto-detection and defaults
- Type validation
- Method chaining support
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import re


class ConfigLoader:
    """Load and validate YAML configuration files for ML pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigLoader.
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config_path = config_path
        self.config = {}
        self.defaults = self._get_defaults()
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str) -> 'ConfigLoader':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            self for method chaining
            
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        if raw_config is None:
            raw_config = {}
        
        # Substitute environment variables
        self.config = self._substitute_env_vars(raw_config)
        self.config_path = config_path
        
        # Merge with defaults
        self._apply_defaults()
        
        return self
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate configuration structure and values.
        
        Returns:
            Tuple of (is_valid, error_list)
        """
        errors = []
        
        # Validate data config
        if 'data' in self.config:
            data_cfg = self.config['data']
            if 'path' in data_cfg:
                if not os.path.exists(data_cfg['path']):
                    errors.append(f"Data path not found: {data_cfg['path']}")
            
            if 'target' not in data_cfg:
                errors.append("Data config missing 'target' field")
            
            if 'test_size' in data_cfg:
                test_size = data_cfg['test_size']
                if not (0 < test_size < 1):
                    errors.append(f"test_size must be between 0 and 1, got {test_size}")
        
        # Validate model config
        if 'model' in self.config:
            model_cfg = self.config['model']
            if 'name' not in model_cfg:
                errors.append("Model config missing 'name' field")
            else:
                valid_models = {
                    'classification': ['LogisticRegression', 'RandomForest', 'GradientBoosting', 
                                     'XGBoost', 'LightGBM', 'SVM', 'KNeighbors', 
                                     'DecisionTree', 'NeuralNetwork'],
                    'regression': ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest',
                                 'GradientBoosting', 'XGBoost', 'LightGBM', 'SVM',
                                 'KNeighbors', 'DecisionTree', 'NeuralNetwork']
                }
                model_name = model_cfg['name']
                problem_type = model_cfg.get('type', 'classification')
                
                if problem_type in valid_models:
                    if model_name not in valid_models[problem_type]:
                        errors.append(f"Unknown model: {model_name} for {problem_type}")
        
        # Validate evaluation config
        if 'evaluation' in self.config:
            eval_cfg = self.config['evaluation']
            if 'cv_folds' in eval_cfg:
                cv_folds = eval_cfg['cv_folds']
                if not isinstance(cv_folds, int) or cv_folds < 2:
                    errors.append(f"cv_folds must be integer >= 2, got {cv_folds}")
        
        # Validate preprocessing config
        valid_strategies = {
            'missing_value_strategy': ['drop', 'mean', 'median', 'mode', 'forward_fill', 'auto'],
            'encoding_method': ['one-hot', 'label', 'auto'],
            'scaling_method': ['standard', 'minmax', 'robust'],
            'outlier_method': ['iqr', 'zscore']
        }
        
        if 'preprocessing' in self.config:
            preproc_cfg = self.config['preprocessing']
            for key, valid_values in valid_strategies.items():
                if key in preproc_cfg:
                    value = preproc_cfg[key]
                    if value not in valid_values:
                        errors.append(f"{key} must be one of {valid_values}, got {value}")
        
        return len(errors) == 0, errors
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data loading configuration."""
        return self.config.get('data', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.config.get('preprocessing', {})
    
    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.config.get('feature_engineering', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.config.get('mlflow', {})
    
    def get_optuna_config(self) -> Dict[str, Any]:
        """Get Optuna configuration."""
        return self.config.get('optuna', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot-notation key.
        
        Args:
            key: Key path (e.g., 'model.name', 'preprocessing.scaling_method')
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary."""
        return self.config.copy()
    
    def to_yaml(self, output_path: str) -> 'ConfigLoader':
        """
        Save config to YAML file.
        
        Args:
            output_path: Path to save config
            
        Returns:
            self for method chaining
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        return self
    
    def print_config(self) -> None:
        """Pretty-print configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False))
        print("="*60 + "\n")
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively substitute environment variables in config.
        
        Supports ${VAR_NAME} and $VAR_NAME syntax.
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR_NAME}
            def replace_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            config = re.sub(r'\$\{([^}]+)\}', replace_var, config)
            
            # Replace $VAR_NAME (word boundary)
            def replace_var2(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            config = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', replace_var2, config)
            return config
        else:
            return config
    
    def _apply_defaults(self) -> None:
        """Merge config with defaults."""
        for key, default_value in self.defaults.items():
            if key not in self.config:
                self.config[key] = default_value
            elif isinstance(default_value, dict):
                # Merge nested dicts
                if isinstance(self.config[key], dict):
                    for sub_key, sub_default in default_value.items():
                        if sub_key not in self.config[key]:
                            self.config[key][sub_key] = sub_default
    
    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'data': {
                'test_size': 0.2,
                'random_state': 42,
            },
            'preprocessing': {
                'missing_value_strategy': 'auto',
                'encoding_method': 'auto',
                'scaling_method': 'standard',
                'outlier_method': 'iqr',
            },
            'feature_engineering': {
                'auto_generate': False,
                'transformations': [],
                'interactions': False,
                'polynomials': {
                    'enabled': False,
                    'degree': 2,
                }
            },
            'model': {
                'type': 'classification',
            },
            'evaluation': {
                'cv_folds': 5,
                'metrics': ['accuracy'],
            },
            'mlflow': {
                'enabled': False,
            },
            'optuna': {
                'enabled': False,
            }
        }
