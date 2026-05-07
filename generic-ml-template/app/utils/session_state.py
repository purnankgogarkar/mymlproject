"""Session state management for Streamlit app.

Manages application state across page reruns using Streamlit's session_state.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class AppState:
    """Central application state management.
    
    Stores all app data that needs to persist across Streamlit reruns:
    - Uploaded data and metadata
    - Configuration settings
    - Trained models and results
    - UI state (current page, expanded sections, etc.)
    """
    
    # Data state
    data: Optional[Any] = None
    target_col: Optional[str] = None
    data_profile: Optional[Dict[str, Any]] = None
    
    # Configuration state
    config: Optional[Dict[str, Any]] = None
    
    # Model state
    model: Optional[Any] = None
    trainer: Optional[Any] = None
    preprocessor: Optional[Any] = None
    
    # Results state
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    evaluation_report: Optional[Dict[str, Any]] = None
    
    # UI state
    current_page: str = "home"
    show_advanced: bool = False
    
    # Experiment tracking
    mlflow_enabled: bool = False
    optuna_enabled: bool = False
    
    def initialize(self):
        """Initialize all state variables in Streamlit session_state.
        
        Called once to set up empty state if not already present.
        """
        for key, value in self.__dict__.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def reset(self):
        """Reset all state to defaults.
        
        Used when user clicks 'New Project' or similar.
        """
        for key in self.__dict__.keys():
            setattr(self, key, None)
    
    def reset_results(self):
        """Reset only results and metrics.
        
        Used when user changes configuration and needs to retrain.
        """
        self.results = None
        self.metrics = None
        self.evaluation_report = None
    
    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.data is not None
    
    def has_config(self) -> bool:
        """Check if model is configured."""
        return self.config is not None
    
    def has_model(self) -> bool:
        """Check if model is trained."""
        return self.model is not None
    
    def has_results(self) -> bool:
        """Check if results are available."""
        return self.results is not None


def get_state() -> AppState:
    """Get or create the application state.
    
    This is the main entry point for accessing app state. It ensures
    the state exists and is initialized.
    
    Returns:
        AppState: The current application state
    """
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
        st.session_state.app_state.initialize()
    return st.session_state.app_state


def init_session_state():
    """Initialize session state for the app.
    
    Call this once at the beginning of streamlit_app.py to set up
    all state variables before any pages are rendered.
    """
    state = get_state()
    return state
