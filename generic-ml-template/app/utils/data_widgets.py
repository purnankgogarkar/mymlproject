"""Streamlit widgets for data handling."""

import streamlit as st
import pandas as pd
from typing import Tuple, Optional
from src.data.loader import DataLoader
from src.data.explorer import DataExplorer


def upload_data_widget() -> Optional[pd.DataFrame]:
    """File upload widget for CSV/Excel files.
    
    Returns:
        DataFrame if file uploaded, None otherwise
    """
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:
            # Load file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    return None


def display_data_preview(df: pd.DataFrame, n_rows: int = 10):
    """Display data preview.
    
    Args:
        df: Input dataframe
        n_rows: Number of rows to show
    """
    st.subheader("📋 Data Preview")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(
            df.head(n_rows),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))


def display_data_profile(df: pd.DataFrame):
    """Display data profile metrics.
    
    Args:
        df: Input dataframe
    """
    st.subheader("📊 Data Profile")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    duplicates = df.duplicated().sum()
    numeric_cols = df.select_dtypes(include=['number']).shape[1]
    categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    
    with col1:
        st.metric("Rows", len(df))
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        st.metric("Missing %", f"{missing_percent:.1f}%")
    
    with col4:
        st.metric("Duplicates", duplicates)
    
    with col5:
        st.metric("Numeric", numeric_cols)


def display_column_info(df: pd.DataFrame):
    """Display column information (types and stats).
    
    Args:
        df: Input dataframe
    """
    st.subheader("🔍 Column Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numeric Columns**")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            for col in numeric_cols:
                st.write(f"  • {col}")
        else:
            st.write("  None")
    
    with col2:
        st.write("**Categorical Columns**")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            for col in cat_cols:
                st.write(f"  • {col}")
        else:
            st.write("  None")


def select_target_column(df: pd.DataFrame) -> str:
    """Widget to select target column.
    
    Args:
        df: Input dataframe
        
    Returns:
        Selected column name
    """
    st.subheader("🎯 Select Target Column")
    
    target_col = st.selectbox(
        "Which column do you want to predict?",
        options=df.columns.tolist(),
        help="This will be your model's target variable"
    )
    
    return target_col


def display_missing_value_chart(df: pd.DataFrame):
    """Display missing values per column.
    
    Args:
        df: Input dataframe
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing'] > 0].sort_values('Missing %', ascending=False)
    
    if len(missing) > 0:
        st.subheader("❌ Missing Values")
        st.dataframe(missing, use_container_width=True)
    else:
        st.subheader("✅ Missing Values")
        st.success("No missing values detected!")


def display_basic_statistics(df: pd.DataFrame):
    """Display basic statistics.
    
    Args:
        df: Input dataframe
    """
    st.subheader("⚡ Basic Statistics")
    
    stats = df.describe().T
    st.dataframe(stats, use_container_width=True)
