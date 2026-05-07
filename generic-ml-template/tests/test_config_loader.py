"""
Tests for Configuration Loader

Test YAML loading, validation, environment variable substitution, and defaults.
"""

import pytest
import yaml
import os
import tempfile
from pathlib import Path
from src.config.config_loader import ConfigLoader


class TestConfigLoaderInitialization:
    """Test ConfigLoader initialization."""
    
    def test_init_without_path(self):
        """Test initialization without config path."""
        loader = ConfigLoader()
        assert loader.config == {}
        assert loader.config_path is None
    
    def test_init_with_valid_path(self, tmp_path):
        """Test initialization with valid config file."""
        config_data = {
            'data': {'path': 'test.csv', 'target': 'y'},
            'model': {'name': 'RandomForest'}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        assert loader.config_path == str(config_file)
        assert loader.config['model']['name'] == 'RandomForest'


class TestConfigLoaderLoading:
    """Test YAML loading functionality."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading valid YAML config."""
        config_data = {
            'data': {'path': 'data.csv', 'target': 'target'},
            'model': {'name': 'RandomForest', 'type': 'classification'}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader()
        loader.load(str(config_file))
        
        assert loader.config['model']['name'] == 'RandomForest'
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        loader = ConfigLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load('/nonexistent/path/config.yaml')
    
    def test_load_empty_yaml(self, tmp_path):
        """Test loading empty YAML file."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            f.write("")
        
        loader = ConfigLoader()
        loader.load(str(config_file))
        
        # Should have defaults applied
        assert 'data' in loader.config
        assert 'model' in loader.config
    
    def test_load_returns_self(self, tmp_path):
        """Test load returns self for method chaining."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({}, f)
        
        loader = ConfigLoader()
        result = loader.load(str(config_file))
        
        assert isinstance(result, ConfigLoader)
        assert result is loader


class TestConfigLoaderValidation:
    """Test configuration validation."""
    
    def test_validate_valid_config(self, tmp_path):
        """Test validation of valid config."""
        config_data = {
            'data': {'path': str(tmp_path / "data.csv"), 'target': 'y'},
            'model': {'name': 'RandomForest', 'type': 'classification'},
            'evaluation': {'cv_folds': 5}
        }
        
        # Create dummy data file
        (tmp_path / "data.csv").touch()
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        is_valid, errors = loader.validate()
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_missing_data_path(self):
        """Test validation fails for missing data path."""
        loader = ConfigLoader()
        loader.config = {
            'data': {'path': '/nonexistent/file.csv', 'target': 'y'}
        }
        
        is_valid, errors = loader.validate()
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_invalid_test_size(self):
        """Test validation fails for invalid test_size."""
        loader = ConfigLoader()
        loader.config = {
            'data': {'test_size': 1.5}
        }
        
        is_valid, errors = loader.validate()
        
        assert not is_valid
        assert any('test_size' in e for e in errors)
    
    def test_validate_invalid_cv_folds(self):
        """Test validation fails for invalid cv_folds."""
        loader = ConfigLoader()
        loader.config = {
            'evaluation': {'cv_folds': 1}
        }
        
        is_valid, errors = loader.validate()
        
        assert not is_valid
        assert any('cv_folds' in e for e in errors)
    
    def test_validate_invalid_preprocessing_strategy(self):
        """Test validation fails for invalid preprocessing strategy."""
        loader = ConfigLoader()
        loader.config = {
            'preprocessing': {'missing_value_strategy': 'invalid_strategy'}
        }
        
        is_valid, errors = loader.validate()
        
        assert not is_valid
        assert any('missing_value_strategy' in e for e in errors)


class TestConfigLoaderGetters:
    """Test getter methods."""
    
    def test_get_data_config(self):
        """Test getting data config."""
        loader = ConfigLoader()
        loader.config = {
            'data': {'path': 'test.csv', 'target': 'y'}
        }
        
        data_cfg = loader.get_data_config()
        assert data_cfg['path'] == 'test.csv'
    
    def test_get_model_config(self):
        """Test getting model config."""
        loader = ConfigLoader()
        loader.config = {
            'model': {'name': 'RandomForest', 'type': 'classification'}
        }
        
        model_cfg = loader.get_model_config()
        assert model_cfg['name'] == 'RandomForest'
    
    def test_get_evaluation_config(self):
        """Test getting evaluation config."""
        loader = ConfigLoader()
        loader.config = {
            'evaluation': {'cv_folds': 5}
        }
        
        eval_cfg = loader.get_evaluation_config()
        assert eval_cfg['cv_folds'] == 5
    
    def test_get_with_dot_notation(self):
        """Test getting value with dot notation."""
        loader = ConfigLoader()
        loader.config = {
            'model': {'hyperparams': {'n_estimators': 100}}
        }
        
        value = loader.get('model.hyperparams.n_estimators')
        assert value == 100
    
    def test_get_nonexistent_key(self):
        """Test getting nonexistent key returns None."""
        loader = ConfigLoader()
        loader.config = {}
        
        value = loader.get('nonexistent.key')
        assert value is None
    
    def test_get_with_default(self):
        """Test getting with default value."""
        loader = ConfigLoader()
        loader.config = {}
        
        value = loader.get('missing.key', default='default_value')
        assert value == 'default_value'


class TestConfigLoaderEnvironmentVariables:
    """Test environment variable substitution."""
    
    def test_substitute_env_var_dollar_brace(self):
        """Test substituting ${VAR_NAME} style env vars."""
        os.environ['TEST_PATH'] = '/test/path'
        
        loader = ConfigLoader()
        config = {'data': {'path': '${TEST_PATH}/data.csv'}}
        result = loader._substitute_env_vars(config)
        
        assert result['data']['path'] == '/test/path/data.csv'
    
    def test_substitute_env_var_dollar(self):
        """Test substituting $VAR_NAME style env vars."""
        os.environ['TEST_VAR'] = 'test_value'
        
        loader = ConfigLoader()
        config = {'model': {'param': '$TEST_VAR'}}
        result = loader._substitute_env_vars(config)
        
        assert result['model']['param'] == 'test_value'
    
    def test_substitute_undefined_env_var(self):
        """Test undefined env var remains unchanged."""
        loader = ConfigLoader()
        config = {'data': {'path': '${UNDEFINED_VAR}/data.csv'}}
        result = loader._substitute_env_vars(config)
        
        # Should remain unchanged
        assert result['data']['path'] == '${UNDEFINED_VAR}/data.csv'


class TestConfigLoaderDefaults:
    """Test default value merging."""
    
    def test_defaults_applied_on_load(self, tmp_path):
        """Test defaults are applied when loading config."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'model': {'name': 'RandomForest'}}, f)
        
        loader = ConfigLoader(str(config_file))
        
        # Check defaults were applied
        assert 'data' in loader.config
        assert loader.config['evaluation']['cv_folds'] == 5
    
    def test_config_overrides_defaults(self, tmp_path):
        """Test config values override defaults."""
        config_data = {'evaluation': {'cv_folds': 10}}
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ConfigLoader(str(config_file))
        
        assert loader.config['evaluation']['cv_folds'] == 10


class TestConfigLoaderSerialization:
    """Test serialization and output methods."""
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        loader = ConfigLoader()
        loader.config = {'model': {'name': 'RandomForest'}}
        
        config_dict = loader.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['model']['name'] == 'RandomForest'
    
    def test_to_yaml(self, tmp_path):
        """Test saving config to YAML file."""
        loader = ConfigLoader()
        loader.config = {
            'model': {'name': 'RandomForest'},
            'evaluation': {'cv_folds': 5}
        }
        
        output_file = tmp_path / "output.yaml"
        loader.to_yaml(str(output_file))
        
        assert output_file.exists()
        
        # Verify saved content
        with open(output_file) as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['model']['name'] == 'RandomForest'
    
    def test_to_yaml_returns_self(self, tmp_path):
        """Test to_yaml returns self for method chaining."""
        loader = ConfigLoader()
        output_file = tmp_path / "output.yaml"
        
        result = loader.to_yaml(str(output_file))
        
        assert isinstance(result, ConfigLoader)
        assert result is loader
