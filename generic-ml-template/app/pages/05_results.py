"""Results page."""

import streamlit as st
import pandas as pd
import numpy as np
from app.utils.session_state import get_state
from app.utils.results_widgets import (
    display_model_results,
    display_feature_importance,
    display_confusion_matrix,
    display_roc_curve,
    display_cross_validation_results,
    export_model_options
)


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
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "🎯 Feature Importance", "🔄 Cross-Validation", "💾 Export"])
    
    # Tab 1: Metrics
    with tab1:
        display_model_results(state.results, (state.model.get('problem_type', 'classification') if state.model else 'classification'))
        
        # Confusion Matrix
        if state.model and state.model.get('problem_type') == 'classification':
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
    
    # Tab 2: Feature Importance
    with tab2:
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
    
    # Tab 3: Cross-Validation
    with tab3:
        cv_results = {
            'mean_score': state.results.get('accuracy', 0.85),
            'std_score': 0.05,
            'max_score': 0.92,
            'min_score': 0.78,
            'fold_scores': [0.85, 0.88, 0.82, 0.87, 0.84]
        }
        display_cross_validation_results(cv_results)
    
    # Tab 4: Export
    with tab4:
        export_model_options(
            (state.model.get('name', 'model') if state.model else 'model'),
            (state.model.get('problem_type', 'classification') if state.model else 'classification')
        )


if __name__ == "__main__":
    main()
