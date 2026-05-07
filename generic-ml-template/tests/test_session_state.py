"""Tests for session state management."""

import pytest
from app.utils.session_state import AppState, get_state


class TestAppStateInitialization:
    """Test AppState initialization."""
    
    def test_app_state_creation(self):
        """Test AppState can be created."""
        state = AppState()
        assert state is not None
    
    def test_app_state_defaults(self):
        """Test AppState has correct default values."""
        state = AppState()
        assert state.data is None
        assert state.config is None
        assert state.model is None
        assert state.results is None
        assert state.target_col is None
    
    def test_app_state_has_data(self):
        """Test has_data method."""
        state = AppState()
        assert state.has_data() is False
        
        state.data = {"test": "data"}
        assert state.has_data() is True
    
    def test_app_state_has_config(self):
        """Test has_config method."""
        state = AppState()
        assert state.has_config() is False
        
        state.config = {"model": "RandomForest"}
        assert state.has_config() is True
    
    def test_app_state_has_model(self):
        """Test has_model method."""
        state = AppState()
        assert state.has_model() is False
        
        state.model = {"trained": True}
        assert state.has_model() is True
    
    def test_app_state_has_results(self):
        """Test has_results method."""
        state = AppState()
        assert state.has_results() is False
        
        state.results = {"accuracy": 0.95}
        assert state.has_results() is True


class TestAppStateReset:
    """Test AppState reset functionality."""
    
    def test_reset_all_state(self):
        """Test reset() clears all state."""
        state = AppState()
        state.data = {"test": "data"}
        state.config = {"model": "RF"}
        state.model = {"trained": True}
        state.results = {"accuracy": 0.95}
        
        state.reset()
        
        assert state.data is None
        assert state.config is None
        assert state.model is None
        assert state.results is None
    
    def test_reset_results_only(self):
        """Test reset_results() clears only results."""
        state = AppState()
        state.data = {"test": "data"}
        state.config = {"model": "RF"}
        state.results = {"accuracy": 0.95}
        
        state.reset_results()
        
        assert state.data is not None
        assert state.config is not None
        assert state.results is None


class TestAppStateWorkflow:
    """Test typical workflow scenarios."""
    
    def test_upload_data_workflow(self):
        """Test workflow: upload data."""
        state = AppState()
        assert not state.has_data()
        
        state.data = {"rows": 100, "cols": 5}
        state.target_col = "target"
        
        assert state.has_data()
        assert state.target_col == "target"
    
    def test_full_workflow(self):
        """Test full workflow: data → config → train → results."""
        state = AppState()
        
        # Step 1: Upload data
        assert not state.has_data()
        state.data = {"rows": 100}
        state.target_col = "target"
        assert state.has_data()
        
        # Step 2: Configure model
        assert not state.has_config()
        state.config = {"model": "RandomForest"}
        assert state.has_config()
        
        # Step 3: Train
        assert not state.has_model()
        state.model = {"trained": True}
        assert state.has_model()
        
        # Step 4: Results
        assert not state.has_results()
        state.results = {"accuracy": 0.95}
        assert state.has_results()
    
    def test_retrain_workflow(self):
        """Test workflow: clear results and retrain."""
        state = AppState()
        state.data = {"rows": 100}
        state.config = {"model": "RF"}
        state.model = {"trained": True}
        state.results = {"accuracy": 0.90}
        
        # Change config
        state.config = {"model": "GB"}
        
        # Clear old results
        state.reset_results()
        
        assert state.has_data()
        assert state.has_config()
        assert state.has_model()  # Old model still there
        assert not state.has_results()  # Results cleared


class TestAppStateDataclassFeatures:
    """Test dataclass features."""
    
    def test_state_can_be_serialized(self):
        """Test state can be converted to dict."""
        state = AppState()
        state.data = {"test": "data"}
        
        # Should be able to access as dict-like
        assert hasattr(state, '__dict__')
        assert 'data' in state.__dict__
