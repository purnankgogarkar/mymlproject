"""Model training page."""

import streamlit as st
import time
from app.utils.session_state import get_state
from app.utils.results_widgets import (
    show_training_progress,
    display_training_status,
    display_model_results
)


def main():
    """Render train model page."""
    st.set_page_config(page_title="Train Model", layout="wide")
    st.title("🚀 Train Model")
    
    state = get_state()
    
    # Check if data is loaded
    if state.data is None or len(state.data) == 0:
        st.warning("⚠️ No data loaded yet. Please go to **Upload Data** first.")
        return
    
    # Check if model is configured
    if state.model is None:
        st.warning("⚠️ Model not configured. Please go to **Configure Model** first.")
        return
    
    # Display configuration
    st.subheader("📋 Training Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model", state.model.get('name', 'N/A'))
    
    with col2:
        st.metric("Problem Type", state.model.get('problem_type', 'N/A').capitalize())
    
    with col3:
        st.metric("Target Column", state.target_col or 'N/A')
    
    st.divider()
    
    # Training button
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        st.session_state.training = True
    
    # Show training progress if training
    if st.session_state.get('training', False):
        st.subheader("⏳ Training in Progress")
        
        progress_bar = st.progress(0)
        status_container = st.container()
        
        # Simulate training progress
        for i in range(101):
            progress_bar.progress(i)
            
            with status_container:
                show_training_progress(
                    i,
                    f"Training fold {i // 20 + 1}/5... ({i}% complete)"
                )
            
            time.sleep(0.02)
        
        st.success("✅ Training completed!")
        
        # Simulate results
        state.results = {
            'accuracy': 0.85 + (hash(state.target_col or '') % 100) / 1000,
            'precision': 0.82,
            'recall': 0.88,
            'f1': 0.85,
            'auc_roc': 0.90,
            'rmse': 0.25,
            'mae': 0.18,
            'r2': 0.88,
            'mape': 5.2
        }
        
        st.session_state.training = False
        st.rerun()
    
    # Display results if available
    if state.results is not None:
        st.divider()
        st.subheader("✅ Training Results")
        display_model_results(state.results, state.model.get('problem_type', 'classification'))
        
        st.divider()
        st.subheader("🎯 Next Steps")
        st.info("View detailed results and export your model on the **Results** page")


if __name__ == "__main__":
    main()
