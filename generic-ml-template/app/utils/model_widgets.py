"""Model configuration utilities for Streamlit."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.config.model_defaults import get_model_defaults, list_models, get_tuning_space


def select_model_type() -> str:
    """Select problem type (classification vs regression).
    
    Returns:
        Problem type: 'classification' or 'regression'
    """
    st.subheader("🎯 Problem Type")
    
    problem_type = st.radio(
        "What type of problem are you solving?",
        options=['classification', 'regression'],
        format_func=lambda x: "📊 Classification" if x == 'classification' else "📈 Regression",
        horizontal=True
    )
    
    return problem_type


def select_model(problem_type: str) -> str:
    """Select ML model based on problem type.
    
    Args:
        problem_type: 'classification' or 'regression'
        
    Returns:
        Model name
    """
    st.subheader("🤖 Select Model")
    
    models = list_models(problem_type)
    
    if problem_type == 'classification':
        available_models = [m for m in models if m in [
            'LogisticRegression', 'RandomForest', 'GradientBoosting',
            'XGBoost', 'LightGBM', 'SVM', 'KNeighbors', 'DecisionTree', 'NeuralNetwork'
        ]]
    else:
        available_models = [m for m in models if m in [
            'LinearRegression', 'Ridge', 'Lasso', 'RandomForest',
            'GradientBoosting', 'XGBoost', 'LightGBM', 'SVM',
            'KNeighbors', 'DecisionTree', 'NeuralNetwork'
        ]]
    
    selected_model = st.selectbox(
        "Choose a model:",
        options=available_models,
        help=f"Available {problem_type} models"
    )
    
    return selected_model


def configure_model_hyperparameters(model_name: str, problem_type: str) -> Dict[str, Any]:
    """Configure hyperparameters for selected model.
    
    Args:
        model_name: Name of the model
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of hyperparameters
    """
    st.subheader("⚙️ Hyperparameters")
    
    default_params = get_model_defaults(problem_type, model_name)
    
    params = {}
    
    # Display default parameters
    st.write(f"**Default parameters for {model_name}:**")
    
    with st.expander("View defaults", expanded=True):
        for key, value in default_params.items():
            st.write(f"  • `{key}`: {value}")
    
    # Option to customize
    use_defaults = st.checkbox("Use default parameters", value=True)
    
    if use_defaults:
        params = default_params.copy()
    else:
        st.write("**Customize parameters:**")
        
        # Model-specific parameter configuration
        if model_name in ['RandomForest', 'GradientBoosting', 'DecisionTree']:
            params['max_depth'] = st.slider(
                "Max Depth",
                min_value=1,
                max_value=20,
                value=default_params.get('max_depth', 10)
            )
            params['min_samples_split'] = st.slider(
                "Min Samples Split",
                min_value=2,
                max_value=20,
                value=default_params.get('min_samples_split', 2)
            )
        
        if model_name in ['RandomForest', 'GradientBoosting']:
            params['n_estimators'] = st.slider(
                "Number of Trees",
                min_value=10,
                max_value=500,
                value=default_params.get('n_estimators', 100),
                step=10
            )
        
        if model_name in ['XGBoost', 'LightGBM']:
            params['learning_rate'] = st.slider(
                "Learning Rate",
                min_value=0.001,
                max_value=0.5,
                value=float(default_params.get('learning_rate', 0.1)),
                step=0.01
            )
        
        if model_name in ['KNeighbors']:
            params['n_neighbors'] = st.slider(
                "Number of Neighbors",
                min_value=1,
                max_value=20,
                value=default_params.get('n_neighbors', 5)
            )
        
        if model_name in ['SVM', 'LogisticRegression']:
            params['C'] = st.slider(
                "Regularization (C)",
                min_value=0.001,
                max_value=100.0,
                value=float(default_params.get('C', 1.0)),
                step=0.1
            )
    
    return params


def configure_training_options() -> Dict[str, Any]:
    """Configure training options (CV, test size, etc).
    
    Returns:
        Training configuration dictionary
    """
    st.subheader("📋 Training Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of folds for K-fold cross-validation"
        )
    
    with col2:
        test_size = st.slider(
            "Test Size Ratio",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
    
    with col3:
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=10000,
            value=42,
            help="Seed for reproducibility"
        )
    
    return {
        'cv_folds': cv_folds,
        'test_size': test_size,
        'random_state': random_state
    }


def show_model_summary(model_name: str, problem_type: str, params: Dict[str, Any]):
    """Display model configuration summary.
    
    Args:
        model_name: Name of the model
        problem_type: 'classification' or 'regression'
        params: Hyperparameters dictionary
    """
    st.subheader("📌 Configuration Summary")
    
    with st.expander("View Summary", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Configuration**")
            st.write(f"  • **Type:** {problem_type.capitalize()}")
            st.write(f"  • **Model:** {model_name}")
        
        with col2:
            st.write("**Hyperparameters**")
            for key, value in params.items():
                st.write(f"  • **{key}:** {value}")
