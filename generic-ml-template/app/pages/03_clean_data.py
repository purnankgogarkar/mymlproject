"""Data cleaning and preprocessing page."""

import streamlit as st
import pandas as pd
import numpy as np
from app.utils.session_state import get_state
from src.data.preprocessor import Preprocessor
from app.utils.visualizations import create_missing_data_plot


def main():
    """Render data cleaning page."""
    st.set_page_config(page_title="Clean Data", layout="wide")
    st.title("🧹 Clean & Prepare Data")
    
    state = get_state()
    
    # Check if data is loaded
    if state.data is None or len(state.data) == 0:
        st.warning("⚠️ No data loaded yet. Please go to **Upload Data** first.")
        return
    
    st.write("Configure data cleaning options: handle missing values, encode categories, scale features, and remove outliers.")
    st.divider()
    
    df = state.data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Initialize cleaning configuration
    if 'clean_config' not in st.session_state:
        st.session_state.clean_config = {
            'missing_strat': 'auto',
            'num_missing': 'mean',
            'cat_missing': 'mode',
            'encode_method': 'auto',
            'scale_method': None,
            'outlier_method': None,
            'outlier_threshold': 1.5
        }
    
    cfg = st.session_state.clean_config
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "❌ Missing Values",
        "🏷️ Categorical Encoding",
        "📊 Feature Scaling",
        "🎯 Outlier Detection",
        "👁️ Preview"
    ])
    
    # TAB 1: Missing Values
    with tab1:
        st.subheader("Handle Missing Values")
        
        missing_stats = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values('Missing', ascending=False)
        
        missing_cols = missing_stats[missing_stats['Missing'] > 0]
        
        if len(missing_cols) > 0:
            st.warning(f"**{len(missing_cols)}** columns with missing values")
            st.dataframe(missing_cols, use_container_width=True)
            
            try:
                missing_plot = create_missing_data_plot(df)
                st.plotly_chart(missing_plot, use_container_width=True)
            except:
                pass
            
            col_left, col_right = st.columns(2)
            with col_left:
                cfg['missing_strat'] = st.selectbox(
                    "Strategy",
                    ['auto', 'mean', 'median', 'mode', 'drop'],
                    help="How to handle missing values"
                )
            
            with col_right:
                if cfg['missing_strat'] == 'auto':
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        cfg['num_missing'] = st.selectbox("Numeric", ['mean', 'median'], key='nm')
                    with col_r2:
                        cfg['cat_missing'] = st.selectbox("Categorical", ['mode', 'unknown'], key='cm')
        else:
            st.success("✅ No missing values!")
    
    # TAB 2: Categorical Encoding
    with tab2:
        st.subheader("Encode Categorical Features")
        
        if len(categorical_cols) > 0:
            st.info(f"Found **{len(categorical_cols)}** categorical columns")
            
            cat_info = []
            for col in categorical_cols:
                cat_info.append({
                    'Column': col,
                    'Unique': df[col].nunique(),
                    'Most Common': df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A'
                })
            
            st.dataframe(pd.DataFrame(cat_info), use_container_width=True)
            
            cfg['encode_method'] = st.selectbox(
                "Encoding Method",
                ['auto', 'one-hot', 'label'],
                help="auto: smart selection | one-hot: binary vectors | label: integer mapping"
            )
        else:
            st.info("No categorical columns")
    
    # TAB 3: Feature Scaling
    with tab3:
        st.subheader("Scale Numeric Features")
        
        if len(numeric_cols) > 0:
            st.info(f"Found **{len(numeric_cols)}** numeric columns")
            
            enable_scale = st.checkbox("Enable Scaling", value=False)
            if enable_scale:
                cfg['scale_method'] = st.selectbox(
                    "Method",
                    ['standard', 'minmax', 'robust'],
                    help="standard: Z-score | minmax: [0,1] range | robust: IQR-based"
                )
            else:
                cfg['scale_method'] = None
        else:
            st.info("No numeric columns")
    
    # TAB 4: Outlier Detection
    with tab4:
        st.subheader("Detect & Remove Outliers")
        
        if len(numeric_cols) > 0:
            enable_outliers = st.checkbox("Detect Outliers", value=False)
            if enable_outliers:
                cfg['outlier_method'] = st.selectbox(
                    "Method",
                    ['iqr', 'zscore'],
                    help="iqr: Interquartile Range | zscore: Statistical (>3σ)"
                )
                if cfg['outlier_method'] == 'iqr':
                    cfg['outlier_threshold'] = st.slider("IQR Threshold", 1.0, 3.0, 1.5)
            else:
                cfg['outlier_method'] = None
        else:
            st.info("No numeric columns")
    
    # TAB 5: Preview
    with tab5:
        st.subheader("Preview Cleaned Data")
        
        try:
            preview = df.copy()
            
            # Show stats
            col_shape, col_missing = st.columns(2)
            with col_shape:
                st.metric("Rows", preview.shape[0])
            with col_missing:
                st.metric("Missing Values", preview.isnull().sum().sum())
            
            st.write("**Data Preview**")
            st.dataframe(preview.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Apply Button
    st.divider()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("✅ Apply Cleaning & Continue", use_container_width=True, type="primary"):
            try:
                cleaned = df.copy()
                
                # Apply missing value handling
                if cfg['missing_strat'] != 'auto':
                    prep = Preprocessor(cleaned)
                    prep.handle_missing_values(
                        strategy=cfg['missing_strat'],
                        numeric_strategy=cfg['num_missing'],
                        categorical_strategy=cfg['cat_missing']
                    )
                    cleaned = prep.get_processed_data()
                
                # Apply outlier removal
                if cfg.get('outlier_method'):
                    prep = Preprocessor(cleaned)
                    prep.detect_outliers(
                        method=cfg['outlier_method'],
                        threshold=cfg['outlier_threshold']
                    )
                    prep.remove_outliers()
                    cleaned = prep.get_processed_data()
                
                # Apply encoding
                if cfg['encode_method']:
                    prep = Preprocessor(cleaned)
                    prep.encode_categoricals(method=cfg['encode_method'])
                    cleaned = prep.get_processed_data()
                
                # Apply scaling
                if cfg['scale_method']:
                    prep = Preprocessor(cleaned)
                    numeric_scaled = cleaned.select_dtypes(include=[np.number]).columns.tolist()
                    prep.scale_features(method=cfg['scale_method'], columns=numeric_scaled)
                    cleaned = prep.get_processed_data()
                
                state.data = cleaned
                st.success(f"✅ Cleaned! New shape: {cleaned.shape}")
                st.balloons()
            
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("⏭️ Skip", use_container_width=True):
            st.info("Using original data")


if __name__ == "__main__":
    main()
