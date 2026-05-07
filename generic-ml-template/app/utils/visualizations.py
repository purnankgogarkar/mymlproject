"""Plotly visualization utilities for data exploration."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List


def create_distribution_plot(df: pd.DataFrame, column: str, nbins: int = 30):
    """Create histogram with KDE overlay.
    
    Args:
        df: Input dataframe
        column: Column name to plot
        nbins: Number of bins for histogram
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=f"Distribution of {column}",
        labels={column: column},
        marginal="box"
    )
    fig.update_layout(
        showlegend=False,
        hovermode='x unified',
        height=400
    )
    return fig


def create_categorical_plot(df: pd.DataFrame, column: str):
    """Create bar chart for categorical column.
    
    Args:
        df: Input dataframe
        column: Column name to plot
        
    Returns:
        Plotly figure
    """
    value_counts = df[column].value_counts()
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=f"Distribution of {column}",
        labels={'x': column, 'y': 'Count'}
    )
    fig.update_layout(
        showlegend=False,
        hovermode='x unified',
        height=400
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlation Matrix"):
    """Create correlation heatmap.
    
    Args:
        df: Input dataframe (numeric columns only)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.values,
        texttemplate='%.2f',
        textfont={"size": 9},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=700
    )
    
    return fig


def create_missing_data_plot(df: pd.DataFrame):
    """Create missing data visualization.
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure
    """
    missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_percent = missing_percent[missing_percent > 0]
    
    if len(missing_percent) == 0:
        # No missing data
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No missing values detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="green")
        )
        fig.update_layout(title="Missing Data Analysis", height=300)
        return fig
    
    fig = px.bar(
        x=missing_percent.values,
        y=missing_percent.index,
        orientation='h',
        title="Missing Data Percentage",
        labels={'x': '% Missing', 'y': 'Column'},
        color=missing_percent.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        showlegend=False,
        hovermode='y unified',
        height=max(300, len(missing_percent) * 20)
    )
    
    return fig


def create_feature_importance_plot(
    importance_series: pd.Series,
    title: str = "Feature Importance",
    n_features: int = 20
):
    """Create feature importance bar chart.
    
    Args:
        importance_series: Series with feature names and importance values
        title: Plot title
        n_features: Number of top features to show
        
    Returns:
        Plotly figure
    """
    # Sort and take top N
    top_importance = importance_series.sort_values(ascending=True).tail(n_features)
    
    fig = px.bar(
        y=top_importance.index,
        x=top_importance.values,
        orientation='h',
        title=title,
        labels={'x': 'Importance', 'y': 'Feature'},
        color=top_importance.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        showlegend=False,
        hovermode='y unified',
        height=max(400, len(top_importance) * 25)
    )
    
    return fig


def create_confusion_matrix_plot(cm: np.ndarray, labels: Optional[List[str]] = None):
    """Create confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (numpy array)
        labels: Class labels (optional)
        
    Returns:
        Plotly figure
    """
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500
    )
    
    return fig


def create_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve"
):
    """Create ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC={auc_score:.3f})',
        line=dict(color='#0078D4', width=3)
    ))
    
    # Diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        hovermode='closest',
        height=500,
        width=600
    )
    
    return fig


def create_box_plot(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None):
    """Create box plot for outlier detection.
    
    Args:
        df: Input dataframe
        y_col: Column to plot on y-axis
        x_col: Optional categorical column for grouping
        
    Returns:
        Plotly figure
    """
    fig = px.box(
        df,
        y=y_col,
        x=x_col,
        title=f"Box Plot: {y_col}",
        points="outliers"
    )
    
    fig.update_layout(
        hovermode='closest',
        height=400
    )
    
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None
):
    """Create scatter plot.
    
    Args:
        df: Input dataframe
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Optional column for color encoding
        size_col: Optional column for size encoding
        
    Returns:
        Plotly figure
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=f"{x_col} vs {y_col}",
        hover_data=df.columns.tolist()
    )
    
    fig.update_layout(
        hovermode='closest',
        height=500
    )
    
    return fig
