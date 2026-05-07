"""Data exploration page."""

import streamlit as st
import pandas as pd
from app.utils.session_state import get_state
from app.utils.visualizations import (
    create_distribution_plot,
    create_categorical_plot,
    create_correlation_heatmap,
    create_missing_data_plot,
    create_box_plot,
    create_scatter_plot
)
from src.data.explorer import DataExplorer


def main():
    """Render explore data page."""
    st.set_page_config(page_title="Explore Data", layout="wide")
    st.title("📊 Explore Data")
    
    state = get_state()
    
    # Check if data is loaded
    if state.data is None or len(state.data) == 0:
        st.warning("⚠️ No data loaded yet. Please go to **Upload Data** first.")
        return
    
    df = state.data
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distributions",
        "🔗 Correlations",
        "❌ Missing Data",
        "📊 Statistics",
        "💡 Recommendations"
    ])
    
    # Tab 1: Distributions
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if numeric_cols:
                selected_numeric = st.multiselect(
                    "Select numeric columns to visualize",
                    numeric_cols,
                    default=numeric_cols[:min(2, len(numeric_cols))]
                )
            else:
                st.info("No numeric columns found")
                selected_numeric = []
        
        with col2:
            if categorical_cols:
                selected_categorical = st.multiselect(
                    "Select categorical columns to visualize",
                    categorical_cols,
                    default=[]
                )
            else:
                selected_categorical = []
        
        # Plot numeric distributions
        for col in selected_numeric:
            fig = create_distribution_plot(df, col)
            st.plotly_chart(fig, use_container_width=True)
        
        # Plot categorical distributions
        for col in selected_categorical:
            fig = create_categorical_plot(df, col)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Correlations
    with tab2:
        st.subheader("Feature Correlations")
        
        if len(numeric_cols) > 1:
            fig = create_correlation_heatmap(df[numeric_cols])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show highest correlations
            st.subheader("Highest Correlations")
            corr_matrix = df[numeric_cols].corr()
            
            # Get upper triangle to avoid duplicates
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df.head(10), use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    # Tab 3: Missing Data
    with tab3:
        st.subheader("Missing Values Analysis")
        
        fig = create_missing_data_plot(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing value statistics
        missing_stats = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values('Missing Count', ascending=False)
        
        missing_stats = missing_stats[missing_stats['Missing Count'] > 0]
        
        if len(missing_stats) > 0:
            st.dataframe(missing_stats, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
    
    # Tab 4: Statistics
    with tab4:
        st.subheader("Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Overall Statistics**")
            st.dataframe(df.describe().T, use_container_width=True)
        
        with col2:
            st.write("**Data Info**")
            info_data = {
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null': df.count(),
                'Null': df.isnull().sum()
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True)
    
    # Tab 5: Recommendations
    with tab5:
        st.subheader("ML Model Recommendations")
        
        try:
            explorer = DataExplorer(df, target_col=state.target_col)
            recommendations = explorer.recommend_models()
            
            st.write("Based on your data characteristics, consider these models:")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['model']}"):
                    st.write(f"**Reason:** {rec['reason']}")
                    st.write(f"**Details:** {rec['details']}")
        
        except Exception as e:
            st.warning(f"Could not generate recommendations: {str(e)}")


if __name__ == "__main__":
    main()
