"""Data upload page."""

import streamlit as st
import pandas as pd
from app.utils.session_state import get_state
from app.utils.data_widgets import (
    upload_data_widget,
    display_data_preview,
    display_data_profile,
    display_column_info,
    select_target_column,
    display_missing_value_chart,
    display_basic_statistics
)


def main():
    """Render upload data page."""
    st.set_page_config(page_title="Upload Data", layout="wide")
    st.title("📤 Upload Data")
    
    state = get_state()
    
    # Step 1: Upload file
    st.header("Step 1: Upload Data")
    st.write("Upload a CSV or Excel file with your data")
    
    uploaded_df = upload_data_widget()
    
    if uploaded_df is not None:
        # Show success message
        st.success(f"✅ File loaded successfully! ({len(uploaded_df)} rows, {len(uploaded_df.columns)} columns)")
        
        # Step 2: Preview data
        st.header("Step 2: Preview Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_data_preview(uploaded_df)
        
        with col2:
            display_data_profile(uploaded_df)
        
        # Step 3: Column info
        st.header("Step 3: Column Types")
        display_column_info(uploaded_df)
        
        # Step 4: Missing values
        display_missing_value_chart(uploaded_df)
        
        # Step 5: Statistics
        st.header("Step 4: Basic Statistics")
        display_basic_statistics(uploaded_df)
        
        # Step 6: Select target
        st.header("Step 5: Select Target Column")
        target_col = select_target_column(uploaded_df)
        
        # Save to state
        if st.button("✅ Load and Continue", use_container_width=True, type="primary"):
            state.data = uploaded_df
            state.target_col = target_col
            st.success("✅ Data loaded! You can now explore or configure your model.")
            st.balloons()
    else:
        st.info("👆 Upload a file to get started")


if __name__ == "__main__":
    main()
