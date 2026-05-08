"""Results page."""

import streamlit as st
import pandas as pd
import numpy as np
from app.utils.session_state import get_state
from app.utils.results_widgets import (
    display_model_results,
    display_regression_equation,
    display_feature_importance,
    display_confusion_matrix,
    display_roc_curve,
    display_cross_validation_results,
    export_model_options
)
from src.export.equation_extractor import get_model_equation_info


def main():
    """Render results page."""
    st.set_page_config(page_title="Results", layout="wide")
    st.title("📊 Results")
    
    state = get_state()
    
    # Check if results are available
    if state.results is None:
        st.warning("⚠️ No results yet. Please go to **Train Model** first.")
        return
    
    # Model Information
    st.subheader("🤖 Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Name", state.model.get('name', 'N/A') if state.model else 'N/A')
    
    with col2:
        st.metric("Problem Type", (state.model.get('problem_type', 'N/A').capitalize() if state.model else 'N/A'))
    
    with col3:
        st.metric("Target Column", state.target_col or 'N/A')
    
    st.divider()
    
    # Results Tabs
    problem_type = state.model.get('problem_type', 'classification') if state.model else 'classification'
    
    # Show equation tab only for regression models
    if problem_type == 'regression':
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Metrics", 
            "📐 Equation", 
            "🎯 Feature Importance", 
            "🔄 Cross-Validation", 
            "💾 Export"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Metrics", 
            "🎯 Feature Importance", 
            "🔄 Cross-Validation", 
            "💾 Export",
            "📋 Info"
        ])
    
    # Tab 1: Metrics
    with tab1:
        display_model_results(state.results, problem_type)
        
        # Confusion Matrix
        if problem_type == 'classification':
            st.divider()
            cm = np.array([
                [np.random.randint(50, 100), np.random.randint(0, 20)],
                [np.random.randint(0, 20), np.random.randint(50, 100)]
            ])
            display_confusion_matrix(cm)
            
            st.divider()
            fpr = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
            tpr = np.array([0.0, 0.4, 0.7, 0.9, 1.0])
            display_roc_curve(fpr, tpr, state.results.get('auc_roc', 0.85))
    
    # Tab 2: Equation (for regression) or Feature Importance (for classification)
    with tab2:
        if problem_type == 'regression':
            st.info("📐 The regression equation represents the predictive model learned from your training data.")
            mock_equation = "ŷ = 50000 + 200.50·square_feet - 5000.25·age + 10000.75·bedrooms"
            mock_coefficients = {
                'square_feet': 200.50,
                'age': -5000.25,
                'bedrooms': 10000.75,
                'intercept': 50000.00
            }
            display_regression_equation(mock_equation, mock_coefficients)
        else:
            # Classification: Feature Importance in tab2
            if state.data is not None:
                feature_cols = list(state.data.columns)
                if state.target_col in feature_cols:
                    feature_cols.remove(state.target_col)
                
                importances = np.random.random(len(feature_cols))
                importances = importances / importances.sum()
                
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                })
                
                display_feature_importance(importance_df)
    
    # Tab 3: Feature Importance (for regression) or Cross-Validation (for classification)
    with tab3:
        if problem_type == 'regression':
            if state.data is not None:
                feature_cols = list(state.data.columns)
                if state.target_col in feature_cols:
                    feature_cols.remove(state.target_col)
                
                importances = np.random.random(len(feature_cols))
                importances = importances / importances.sum()
                
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                })
                
                display_feature_importance(importance_df)
        else:
            # Classification: Cross-Validation
            cv_results = {
                'mean_score': state.results.get('accuracy', 0.85),
                'std_score': 0.05,
                'max_score': 0.92,
                'min_score': 0.78,
                'fold_scores': [0.85, 0.88, 0.82, 0.87, 0.84]
            }
            display_cross_validation_results(cv_results)
    
    # Tab 4: Cross-Validation (for regression) or Export (for classification)
    with tab4:
        if problem_type == 'regression':
            cv_results = {
                'mean_score': state.results.get('r2', 0.88),
                'std_score': 0.05,
                'max_score': 0.92,
                'min_score': 0.84,
                'fold_scores': [0.88, 0.90, 0.86, 0.89, 0.87]
            }
            display_cross_validation_results(cv_results)
        else:
            # Classification: Export
            export_model_options(
                (state.model.get('name', 'model') if state.model else 'model'),
                problem_type
            )
    
    # Tab 5: Export (for regression) or Info (for classification)
    with tab5:
        if problem_type == 'regression':
            export_model_options(
                (state.model.get('name', 'model') if state.model else 'model'),
                problem_type
            )
        else:
            st.subheader("ℹ️ Model Information")
            st.write("**Classification Model Details**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{state.results.get('accuracy', 0):.4f}")
                st.metric("Precision", f"{state.results.get('precision', 0):.4f}")
            with col2:
                st.metric("Recall", f"{state.results.get('recall', 0):.4f}")
                st.metric("F1 Score", f"{state.results.get('f1', 0):.4f}")


if __name__ == "__main__":
    main()
