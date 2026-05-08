"""Generic ML Dashboard - Main Streamlit App.

Interactive web UI for the entire ML pipeline with data upload,
exploration, model configuration, training, and results visualization.
"""

import streamlit as st
from app.utils.session_state import init_session_state

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Generic ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/purnankgogarkar/mymlproject",
        "Report a bug": "https://github.com/purnankgogarkar/mymlproject/issues",
        "About": "Generic ML Template Framework v1.0"
    }
)

# ============================================================================
# Initialize Session State
# ============================================================================

state = init_session_state()

# ============================================================================
# Custom CSS & Styling
# ============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 3px solid #0078D4;
    }
    
    /* Status indicators */
    .status-ready { color: #28a745; font-weight: bold; }
    .status-pending { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    
    /* Feature cards */
    .feature-card {
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #0078D4;
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar - Navigation & Status
# ============================================================================

with st.sidebar:
    st.markdown("---")
    st.title("🤖 ML Dashboard")
    st.markdown("---")
    
    # Status indicators
    st.subheader("📊 Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "✅" if state.has_data() else "⏳"
        st.write(f"{status} Data")
    
    with col2:
        status = "✅" if state.has_config() else "⏳"
        st.write(f"{status} Config")
    
    with col3:
        status = "✅" if state.has_model() else "⏳"
        st.write(f"{status} Model")
    
    with col4:
        status = "✅" if state.has_results() else "⏳"
        st.write(f"{status} Results")
    
    st.markdown("---")
    
    # Navigation
    st.subheader("🧭 Navigation")
    st.page_link("pages/01_upload_data.py", label="Upload Data", icon="📤")
    st.page_link("pages/02_explore_data.py", label="Explore Data", icon="📊")
    st.page_link("pages/03_clean_data.py", label="Clean Data", icon="🧹")
    st.page_link("pages/04_configure_model.py", label="Configure Model", icon="⚙️")
    st.page_link("pages/05_train_model.py", label="Train Model", icon="🚀")
    st.page_link("pages/06_results.py", label="Results", icon="📈")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    if st.button("🔄 New Project"):
        state.reset()
        st.success("Project reset!")
        st.rerun()
    
    if state.has_model():
        if st.button("🗑️ Clear Results"):
            state.reset_results()
            st.info("Results cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Information
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Generic ML Template Framework**
        
        v1.0 - May 2026
        
        A production-ready ML pipeline framework for tabular data.
        
        [📖 Documentation](https://github.com/purnankgogarkar/mymlproject)
        [🐛 Report Issues](https://github.com/purnankgogarkar/mymlproject/issues)
        """)

# ============================================================================
# Main Content - Home Page
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>🤖 Generic ML Dashboard</h1>
    <p>Build, train, and compare machine learning models with no code!</p>
</div>
""", unsafe_allow_html=True)

# Quick status
if state.has_data() and state.has_config() and state.has_model() and state.has_results():
    st.success("✅ Complete workflow executed! View results in the Results page.")
elif state.has_data() and state.has_config() and state.has_model():
    st.info("✅ Model trained! View detailed results in the Results page.")
elif state.has_data() and state.has_config():
    st.info("⏳ Configuration ready. Click 'Train Model' to start training.")
elif state.has_data():
    st.info("⏳ Data loaded. Configure your model in the 'Configure Model' page.")
else:
    st.info("👋 Welcome! Start by uploading your data in the 'Upload Data' page.")

# ============================================================================
# Feature Cards
# ============================================================================

st.markdown("## ✨ What You Can Do")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📤 Upload Data
    Load your CSV or Excel files with automatic type detection and validation.
    - Supports various file formats
    - Automatic data profiling
    - Type detection (numeric, categorical, datetime)
    """)

with col2:
    st.markdown("""
    ### 📊 Explore Data
    Interactive data analysis with visualizations.
    - Distribution plots
    - Correlation analysis
    - Missing data detection
    - Statistical summaries
    """)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ⚙️ Configure Model
    Visual configuration builder (no coding needed).
    - Data preprocessing options
    - Feature engineering settings
    - 20+ model algorithms
    - Hyperparameter tuning
    """)

with col2:
    st.markdown("""
    ### 🚀 Train Model
    Real-time training with progress monitoring.
    - Cross-validation support
    - MLflow experiment tracking
    - Optuna hyperparameter optimization
    - Live metrics display
    """)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📈 View Results
    Comprehensive results visualization.
    - Performance metrics
    - Feature importance
    - Prediction analysis
    - Model comparison
    """)

with col2:
    st.markdown("""
    ### 💾 Export Everything
    Download your results and configurations.
    - Export metrics as CSV
    - Save configurations as YAML
    - Export predictions
    - Share experiments
    """)

# ============================================================================
# Getting Started
# ============================================================================

st.markdown("---")
st.markdown("## 🚀 Quick Start")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown("**Step 1**\n📤 Upload")
with col2:
    st.markdown("→")
with col3:
    st.markdown("**Step 2**\n📊 Explore")
with col4:
    st.markdown("→")
with col5:
    st.markdown("**Step 3**\n⚙️ Config")
with col6:
    st.markdown("→")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown("")
with col2:
    st.markdown("")
with col3:
    st.markdown("")
with col4:
    st.markdown("→")
with col5:
    st.markdown("**Step 4**\n🚀 Train")
with col6:
    st.markdown("")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown("")
with col2:
    st.markdown("")
with col3:
    st.markdown("")
with col4:
    st.markdown("↓")
with col5:
    st.markdown("**Step 5**\n📈 Results")
with col6:
    st.markdown("")

# ============================================================================
# Example Workflows
# ============================================================================

st.markdown("---")
st.markdown("## 📚 Example Workflows")

tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "Compare Models"])

with tab1:
    st.markdown("""
    ### Iris Classification
    Predict flower species from measurements.
    
    **Steps:**
    1. Download [iris.csv](example_data/iris.csv)
    2. Upload in 'Upload Data' page
    3. Set target column to 'species'
    4. Use 'RandomForest' model
    5. View feature importance in Results
    """)

with tab2:
    st.markdown("""
    ### Housing Regression
    Predict house prices from features.
    
    **Steps:**
    1. Download [housing.csv](example_data/housing.csv)
    2. Upload in 'Upload Data' page
    3. Set target column to 'price'
    4. Use 'GradientBoosting' model
    5. Analyze predictions
    """)

with tab3:
    st.markdown("""
    ### Model Comparison
    Compare multiple models on same data.
    
    **Steps:**
    1. Train with 'RandomForest'
    2. Export results
    3. Retrain with 'GradientBoosting'
    4. Compare metrics side-by-side
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Version:** 1.0")
with col2:
    st.markdown("[📖 Docs](https://github.com/purnankgogarkar/mymlproject)")
with col3:
    st.markdown("[🐛 Issues](https://github.com/purnankgogarkar/mymlproject/issues)")
