"""Configure Model page - Visual model configuration builder."""

import streamlit as st
from app.utils.session_state import init_session_state

# Initialize state
state = init_session_state()

st.title("⚙️ Configure Model")
st.markdown("Set up your ML pipeline configuration with visual options.")

st.info("📋 Coming soon! Page structure ready.")
st.write("This page will support:")
st.write("- Data configuration (train/test split, CV folds)")
st.write("- Preprocessing options (scaling, encoding, imputation)")
st.write("- Feature engineering settings")
st.write("- Model selection and hyperparameter tuning")
st.write("- Evaluation metrics selection")
