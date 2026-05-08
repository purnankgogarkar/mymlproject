"""Training and results utilities for Streamlit."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
import pickle
import json
import io
from datetime import datetime


def show_training_progress(progress: int, message: str = ""):
    """Display training progress.
    
    Args:
        progress: Progress percentage (0-100)
        message: Status message
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(progress / 100)
    
    with col2:
        st.metric("Progress", f"{progress}%")
    
    if message:
        st.info(message)


def display_training_status(status: Dict[str, Any]):
    """Display training status metrics.
    
    Args:
        status: Status dictionary with metrics
    """
    st.subheader("⚡ Training Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Epoch", status.get('epoch', 0))
    
    with col2:
        st.metric("Loss", f"{status.get('loss', 0):.4f}")
    
    with col3:
        st.metric("Accuracy", f"{status.get('accuracy', 0):.4f}")
    
    with col4:
        st.metric("Elapsed Time", status.get('elapsed_time', '0s'))


def display_model_results(results: Dict[str, Any], problem_type: str):
    """Display model training results.
    
    Args:
        results: Results dictionary with metrics
        problem_type: 'classification' or 'regression'
    """
    st.subheader("📊 Model Results")
    
    if problem_type == 'classification':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
        
        with col2:
            st.metric("Precision", f"{results.get('precision', 0):.4f}")
        
        with col3:
            st.metric("Recall", f"{results.get('recall', 0):.4f}")
        
        with col4:
            st.metric("F1 Score", f"{results.get('f1', 0):.4f}")
        
        # Additional classification metrics
        col5, col6 = st.columns(2)
        
        with col5:
            st.metric("AUC-ROC", f"{results.get('auc_roc', 0):.4f}")
        
        with col6:
            st.metric("Log Loss", f"{results.get('log_loss', 0):.4f}")
    
    else:  # regression
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{results.get('rmse', 0):.4f}")
        
        with col2:
            st.metric("MAE", f"{results.get('mae', 0):.4f}")
        
        with col3:
            st.metric("R² Score", f"{results.get('r2', 0):.4f}")
        
        with col4:
            st.metric("MAPE", f"{results.get('mape', 0):.2f}%")


def display_regression_equation(equation: str, coefficients: Optional[Dict[str, float]] = None):
    """Display regression equation.
    
    Args:
        equation: The regression equation string
        coefficients: Dictionary of feature names to coefficient values
    """
    if not equation:
        return
    
    st.subheader("📐 Regression Equation")
    
    # Display equation in a highlighted box
    st.code(equation, language="python")
    
    # Display coefficients if available
    if coefficients:
        with st.expander("📊 View Coefficients"):
            coef_df = pd.DataFrame(
                list(coefficients.items()),
                columns=["Feature", "Coefficient"]
            ).sort_values("Coefficient", key=abs, ascending=False)
            
            st.dataframe(coef_df, use_container_width=True)
            
            # Visualization
            fig_data = {
                'Feature': coef_df['Feature'],
                'Coefficient': coef_df['Coefficient']
            }
            st.bar_chart(coef_df.set_index('Feature')['Coefficient'])


def generate_importance_equation(importance_df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """Generate an equation-style representation from feature importances.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        
    Returns:
        Tuple of (equation_string, importance_dict)
    """
    if importance_df.empty:
        return "No features available", {}
    
    # Sort by importance descending
    importance_sorted = importance_df.sort_values('importance', ascending=False)
    
    # Normalize importances to percentages (0-100)
    importance_values = importance_sorted['importance'].values
    max_importance = importance_values.max()
    
    if max_importance > 0:
        normalized_values = (importance_values / max_importance) * 100
    else:
        normalized_values = importance_values
    
    # Build equation string similar to regression equation
    equation_parts = ["ŷ = base_prediction"]
    importance_dict = {}
    
    for feature, importance in zip(importance_sorted['feature'], normalized_values):
        # Format the importance value
        if importance > 0:
            equation_parts.append(f"+ {importance:.2f}·{feature}")
        else:
            equation_parts.append(f"- {abs(importance):.2f}·{feature}")
        
        # Store in dict for table display
        importance_dict[feature] = float(importance)
    
    # Combine parts into equation
    equation = " ".join(equation_parts)
    
    return equation, importance_dict


def display_feature_importance(importance_df: pd.DataFrame):
    """Display feature importance with equation-style representation.
    
    Args:
        importance_df: DataFrame with feature importance data
    """
    # Sort for display
    importance_sorted = importance_df.sort_values('importance', ascending=True)
    
    # Generate importance-based equation
    equation, importance_dict = generate_importance_equation(importance_df)
    
    # Display equation section
    st.subheader("📊 Feature Contribution Equation")
    st.info("🎯 Features are ranked by their importance in making predictions. Higher values indicate stronger influence on the model output.")
    
    # Display equation
    st.code(equation, language="python")
    
    # Display coefficients table (showing normalized importance)
    if importance_dict:
        with st.expander("📋 View Feature Importance Scores"):
            importance_display_df = pd.DataFrame(
                list(importance_dict.items()),
                columns=["Feature", "Importance Score (%)"]
            ).sort_values("Importance Score (%)", ascending=False)
            
            st.dataframe(importance_display_df, use_container_width=True)
            
            # Visualization of importances
            st.bar_chart(importance_display_df.set_index('Feature')['Importance Score (%)'])


def display_confusion_matrix(cm: np.ndarray, labels: Optional[list] = None):
    """Display confusion matrix.
    
    Args:
        cm: Confusion matrix numpy array
        labels: Class labels
    """
    st.subheader("🔲 Confusion Matrix")
    
    from app.utils.visualizations import create_confusion_matrix_plot
    
    fig = create_confusion_matrix_plot(cm, labels=labels)
    st.plotly_chart(fig, use_container_width=True)


def display_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float):
    """Display ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: AUC score
    """
    st.subheader("📈 ROC Curve")
    
    from app.utils.visualizations import create_roc_curve
    
    fig = create_roc_curve(fpr, tpr, auc)
    st.plotly_chart(fig, use_container_width=True)


def export_model_options(model_name: str, problem_type: str):
    """Show export options for trained model.
    
    Args:
        model_name: Name of the model
        problem_type: 'classification' or 'regression'
    """
    st.subheader("💾 Export Model")
    
    # Generate mock model data for download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2, col3 = st.columns(3)
    
    # 1. Download Model (Pickle)
    with col1:
        # Create mock model object
        mock_model = {
            'name': model_name,
            'type': problem_type,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Serialize to pickle bytes
        model_bytes = io.BytesIO()
        pickle.dump(mock_model, model_bytes)
        model_bytes.seek(0)
        
        st.download_button(
            label="📥 Download Model (Pickle)",
            data=model_bytes.getvalue(),
            file_name=f"{model_name}_{timestamp}.pkl",
            mime="application/octet-stream",
            key="export_pickle"
        )
    
    # 2. Download Config (YAML)
    with col2:
        # Create YAML config
        config_data = f"""name: {model_name}
problem_type: {problem_type}
timestamp: {timestamp}
version: 1.0

hyperparameters:
  max_depth: 10
  n_estimators: 100
  random_state: 42

training:
  test_size: 0.2
  cv_folds: 5
  random_state: 42

feature_scaling: standard
missing_value_strategy: mean
"""
        
        st.download_button(
            label="📋 Download Config (YAML)",
            data=config_data,
            file_name=f"{model_name}_{timestamp}_config.yaml",
            mime="text/yaml",
            key="export_yaml"
        )
    
    # 3. Download Report (JSON)
    with col3:
        # Create JSON report with equation for regression
        report_data = {
            'model_name': model_name,
            'problem_type': problem_type,
            'timestamp': timestamp,
            'version': '1.0'
        }
        
        # Add equation for regression models
        if problem_type == 'regression':
            report_data['regression_equation'] = {
                'equation': 'ŷ = 50000 + 200.50·square_feet - 5000.25·age + 10000.75·bedrooms',
                'coefficients': {
                    'square_feet': 200.50,
                    'age': -5000.25,
                    'bedrooms': 10000.75,
                    'intercept': 50000.00
                },
                'description': 'Predictive linear regression equation'
            }
            report_data['metrics'] = {
                'rmse': 2500.50,
                'mae': 1850.25,
                'r2_score': 0.92,
                'mape': 2.3
            }
        else:
            report_data['metrics'] = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'auc_roc': 0.90
            }
        
        report_data.update({
            'features_count': 12,
            'training_samples': 1000,
            'test_samples': 250,
            'cross_validation_folds': 5
        })
        
        report_json = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="📊 Download Report (JSON)",
            data=report_json,
            file_name=f"{model_name}_{timestamp}_report.json",
            mime="application/json",
            key="export_json"
        )
    
    st.success("✅ Click any button above to download!")


def display_cross_validation_results(cv_results: Dict[str, Any]):
    """Display cross-validation results.
    
    Args:
        cv_results: Cross-validation results dictionary
    """
    st.subheader("🔄 Cross-Validation Results")
    
    # Create DataFrame from results
    cv_df = pd.DataFrame(cv_results)
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Score", f"{cv_results.get('mean_score', 0):.4f}")
    
    with col2:
        st.metric("Std Dev", f"{cv_results.get('std_score', 0):.4f}")
    
    with col3:
        st.metric("Max Score", f"{cv_results.get('max_score', 0):.4f}")
    
    with col4:
        st.metric("Min Score", f"{cv_results.get('min_score', 0):.4f}")
    
    # Display fold scores
    with st.expander("View Fold Scores"):
        if 'fold_scores' in cv_results:
            fold_df = pd.DataFrame({
                'Fold': range(1, len(cv_results['fold_scores']) + 1),
                'Score': cv_results['fold_scores']
            })
            st.dataframe(fold_df, use_container_width=True)
