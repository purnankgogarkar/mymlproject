"""Training and results utilities for Streamlit."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time


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


def display_feature_importance(importance_df: pd.DataFrame):
    """Display feature importance.
    
    Args:
        importance_df: DataFrame with feature importance data
    """
    st.subheader("🎯 Feature Importance")
    
    # Sort and display
    importance_sorted = importance_df.sort_values('importance', ascending=True)
    
    # Bar chart
    st.bar_chart(importance_sorted.set_index('feature')['importance'])
    
    # Table
    with st.expander("View Details"):
        st.dataframe(importance_sorted, use_container_width=True)


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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Model (Pickle)", key="export_pickle"):
            st.info("Model export to pickle format ready")
    
    with col2:
        if st.button("📋 Download Config (YAML)", key="export_yaml"):
            st.info("Configuration export to YAML ready")
    
    with col3:
        if st.button("📊 Download Report (PDF)", key="export_pdf"):
            st.info("Report generation ready")


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
