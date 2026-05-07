"""Tests for visualization utilities."""

import pytest
import pandas as pd
import numpy as np
from app.utils.visualizations import (
    create_distribution_plot,
    create_categorical_plot,
    create_correlation_heatmap,
    create_missing_data_plot,
    create_feature_importance_plot,
    create_confusion_matrix_plot,
    create_roc_curve,
    create_box_plot,
    create_scatter_plot
)


@pytest.fixture
def sample_df():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100),
        'numeric3': np.random.exponential(2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })


@pytest.fixture
def df_with_missing():
    """Create dataframe with missing values."""
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5] * 10,
        'col2': [1.0, np.nan, 3.0, 4.0, 5.0] * 10,
        'col3': [1, 2, 3, 4, 5] * 10
    })
    return df


class TestDistributionPlot:
    """Test distribution plot creation."""
    
    def test_create_distribution_plot_returns_figure(self, sample_df):
        """Test that distribution plot returns valid figure."""
        fig = create_distribution_plot(sample_df, 'numeric1')
        assert fig is not None
        assert hasattr(fig, 'update_layout')
    
    def test_distribution_plot_has_title(self, sample_df):
        """Test that distribution plot has title."""
        fig = create_distribution_plot(sample_df, 'numeric1')
        assert 'numeric1' in fig.layout.title.text
    
    def test_distribution_plot_custom_bins(self, sample_df):
        """Test custom bin count."""
        fig = create_distribution_plot(sample_df, 'numeric1', nbins=50)
        assert fig is not None


class TestCategoricalPlot:
    """Test categorical plot creation."""
    
    def test_create_categorical_plot_returns_figure(self, sample_df):
        """Test that categorical plot returns valid figure."""
        fig = create_categorical_plot(sample_df, 'category')
        assert fig is not None
        assert hasattr(fig, 'update_layout')
    
    def test_categorical_plot_has_title(self, sample_df):
        """Test that categorical plot has title."""
        fig = create_categorical_plot(sample_df, 'category')
        assert 'category' in fig.layout.title.text


class TestCorrelationHeatmap:
    """Test correlation heatmap creation."""
    
    def test_create_correlation_heatmap_returns_figure(self, sample_df):
        """Test that correlation heatmap returns valid figure."""
        numeric_df = sample_df.select_dtypes(include=['number'])
        fig = create_correlation_heatmap(numeric_df)
        assert fig is not None
        assert hasattr(fig, 'update_layout')
    
    def test_correlation_heatmap_has_title(self, sample_df):
        """Test that heatmap has default title."""
        numeric_df = sample_df.select_dtypes(include=['number'])
        fig = create_correlation_heatmap(numeric_df)
        assert fig.layout.title.text is not None
    
    def test_correlation_heatmap_custom_title(self, sample_df):
        """Test custom title."""
        numeric_df = sample_df.select_dtypes(include=['number'])
        fig = create_correlation_heatmap(numeric_df, title="Custom Title")
        assert "Custom Title" in fig.layout.title.text


class TestMissingDataPlot:
    """Test missing data plot creation."""
    
    def test_missing_data_plot_with_missing_values(self, df_with_missing):
        """Test plot with missing data."""
        fig = create_missing_data_plot(df_with_missing)
        assert fig is not None
    
    def test_missing_data_plot_no_missing_values(self, sample_df):
        """Test plot with no missing data."""
        fig = create_missing_data_plot(sample_df)
        assert fig is not None
        # Should have annotation about no missing values
        assert len(fig.data) > 0 or len(fig.layout.annotations) > 0


class TestFeatureImportancePlot:
    """Test feature importance plot creation."""
    
    def test_feature_importance_plot_returns_figure(self):
        """Test that feature importance plot returns valid figure."""
        importance = pd.Series({'feat1': 0.5, 'feat2': 0.3, 'feat3': 0.2})
        fig = create_feature_importance_plot(importance)
        assert fig is not None
    
    def test_feature_importance_plot_with_custom_title(self):
        """Test custom title."""
        importance = pd.Series({'feat1': 0.5, 'feat2': 0.3})
        fig = create_feature_importance_plot(importance, title="Custom Importance")
        assert "Custom Importance" in fig.layout.title.text
    
    def test_feature_importance_plot_n_features(self):
        """Test limiting number of features."""
        importance = pd.Series({f'feat{i}': 0.5 - i*0.05 for i in range(30)})
        fig = create_feature_importance_plot(importance, n_features=10)
        assert fig is not None


class TestConfusionMatrixPlot:
    """Test confusion matrix plot creation."""
    
    def test_confusion_matrix_plot_binary(self):
        """Test binary confusion matrix."""
        cm = np.array([[95, 5], [3, 97]])
        fig = create_confusion_matrix_plot(cm)
        assert fig is not None
        assert 'Confusion Matrix' in fig.layout.title.text
    
    def test_confusion_matrix_plot_multiclass(self):
        """Test multiclass confusion matrix."""
        cm = np.array([[50, 2, 3], [1, 48, 4], [2, 1, 47]])
        fig = create_confusion_matrix_plot(cm)
        assert fig is not None
    
    def test_confusion_matrix_plot_with_labels(self):
        """Test with custom labels."""
        cm = np.array([[95, 5], [3, 97]])
        labels = ['Negative', 'Positive']
        fig = create_confusion_matrix_plot(cm, labels=labels)
        assert fig is not None


class TestROCCurve:
    """Test ROC curve plot creation."""
    
    def test_roc_curve_returns_figure(self):
        """Test that ROC curve returns valid figure."""
        fpr = np.array([0.0, 0.2, 0.5, 1.0])
        tpr = np.array([0.0, 0.6, 0.8, 1.0])
        fig = create_roc_curve(fpr, tpr, auc_score=0.85)
        assert fig is not None
    
    def test_roc_curve_has_title(self):
        """Test that ROC curve has title."""
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.8, 1.0])
        fig = create_roc_curve(fpr, tpr, auc_score=0.9)
        assert fig.layout.title.text is not None
    
    def test_roc_curve_custom_title(self):
        """Test custom title."""
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.8, 1.0])
        fig = create_roc_curve(fpr, tpr, auc_score=0.9, title="Custom ROC")
        assert "Custom ROC" in fig.layout.title.text


class TestBoxPlot:
    """Test box plot creation."""
    
    def test_box_plot_single_variable(self, sample_df):
        """Test box plot for single variable."""
        fig = create_box_plot(sample_df, 'numeric1')
        assert fig is not None
    
    def test_box_plot_with_grouping(self, sample_df):
        """Test box plot with grouping variable."""
        fig = create_box_plot(sample_df, 'numeric1', x_col='category')
        assert fig is not None


class TestScatterPlot:
    """Test scatter plot creation."""
    
    def test_scatter_plot_basic(self, sample_df):
        """Test basic scatter plot."""
        fig = create_scatter_plot(sample_df, 'numeric1', 'numeric2')
        assert fig is not None
    
    def test_scatter_plot_with_color(self, sample_df):
        """Test scatter plot with color encoding."""
        fig = create_scatter_plot(sample_df, 'numeric1', 'numeric2', color_col='category')
        assert fig is not None
    
    def test_scatter_plot_with_size(self, sample_df):
        """Test scatter plot with size encoding."""
        fig = create_scatter_plot(sample_df, 'numeric1', 'numeric2', size_col='numeric3')
        assert fig is not None
