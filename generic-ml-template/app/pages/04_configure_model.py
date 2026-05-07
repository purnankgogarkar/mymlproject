"""Model configuration page."""

import streamlit as st
from app.utils.session_state import get_state
from app.utils.model_widgets import (
    select_model_type,
    select_model,
    configure_model_hyperparameters,
    configure_training_options,
    show_model_summary
)


def main():
    """Render configure model page."""
    st.set_page_config(page_title="Configure Model", layout="wide")
    st.title("⚙️ Configure Model")
    
    state = get_state()
    
    # Check if data is loaded
    if state.data is None or len(state.data) == 0:
        st.warning("⚠️ No data loaded yet. Please go to **Upload Data** first.")
        return
    
    st.write(f"Configuring model for **{state.target_col}** target column")
    st.divider()
    
    # Step 1: Select problem type
    problem_type = select_model_type()
    
    # Step 2: Select model
    model_name = select_model(problem_type)
    
    # Step 3: Configure hyperparameters
    st.divider()
    params = configure_model_hyperparameters(model_name, problem_type)
    
    # Step 4: Training options
    st.divider()
    training_opts = configure_training_options()
    
    # Step 5: Summary
    st.divider()
    show_model_summary(model_name, problem_type, params)
    
    # Save configuration
    if st.button("✅ Save Configuration", use_container_width=True, type="primary"):
        state.model = {
            'name': model_name,
            'problem_type': problem_type,
            'hyperparameters': params,
            'training_options': training_opts
        }
        st.success("✅ Model configuration saved!")
        st.info("You can now proceed to **Train Model** page")


if __name__ == "__main__":
    main()
