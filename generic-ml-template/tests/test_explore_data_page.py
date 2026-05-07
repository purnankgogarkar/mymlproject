"""Tests for explore data page."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.utils.session_state import AppState


@pytest.fixture
def sample_df():
    """Create sample dataframe for exploration."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(20, 70, 100),
        'income': np.random.randint(20000, 150000, 100),
        'score': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })


@pytest.fixture
def state(sample_df):
    """Create app state with data."""
    state = AppState()
    state.data = sample_df
    state.target_col = 'income'
    return state


class TestExploreDataPageBasics:
    """Test basic explore page functionality."""
    
    def test_state_has_data(self, state):
        """Test state has data."""
        assert state.has_data()
    
    def test_state_has_target_column(self, state):
        """Test state has target column set."""
        assert state.target_col is not None
        assert state.target_col in state.data.columns
    
    def test_detect_numeric_columns(self, state):
        """Test numeric column detection."""
        numeric_cols = state.data.select_dtypes(include=['number']).columns.tolist()
        assert 'age' in numeric_cols
        assert 'income' in numeric_cols
    
    def test_detect_categorical_columns(self, state):
        """Test categorical column detection."""
        cat_cols = state.data.select_dtypes(include=['object']).columns.tolist()
        assert 'category' in cat_cols
        assert 'region' in cat_cols


class TestExploreDataDistributions:
    """Test distribution visualization."""
    
    def test_numeric_distribution_possible(self, state):
        """Test that numeric distributions can be created."""
        numeric_cols = state.data.select_dtypes(include=['number']).columns.tolist()
        assert len(numeric_cols) > 0
    
    def test_categorical_distribution_possible(self, state):
        """Test that categorical distributions can be created."""
        cat_cols = state.data.select_dtypes(include=['object']).columns.tolist()
        assert len(cat_cols) > 0
    
    def test_distribution_column_selection(self, state):
        """Test selecting columns for distribution."""
        numeric_cols = state.data.select_dtypes(include=['number']).columns.tolist()
        selected = numeric_cols[:2]
        assert len(selected) > 0


class TestExploreDataCorrelations:
    """Test correlation analysis."""
    
    def test_correlation_matrix_calculation(self, state):
        """Test correlation matrix can be calculated."""
        numeric_df = state.data.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        assert corr.shape[0] == corr.shape[1]
    
    def test_correlation_values_in_range(self, state):
        """Test correlation values are in [-1, 1] range."""
        numeric_df = state.data.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        assert (corr >= -1).all().all()
        assert (corr <= 1).all().all()
    
    def test_find_highest_correlations(self, state):
        """Test finding highest correlations."""
        numeric_df = state.data.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        
        # Extract pairs
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append(abs(corr.iloc[i, j]))
        
        assert len(corr_pairs) > 0


class TestExploreDataMissing:
    """Test missing data analysis."""
    
    def test_detect_missing_values(self, state):
        """Test missing value detection."""
        missing = state.data.isnull().sum()
        assert isinstance(missing, pd.Series)
    
    def test_calculate_missing_percentage(self, state):
        """Test missing percentage calculation."""
        missing_percent = (state.data.isnull().sum() / len(state.data) * 100)
        assert all(0 <= pct <= 100 for pct in missing_percent)
    
    def test_handle_no_missing_values(self, state):
        """Test handling data with no missing values."""
        missing = state.data.isnull().sum()
        total_missing = missing.sum()
        assert total_missing == 0


class TestExploreDataStatistics:
    """Test statistics display."""
    
    def test_descriptive_statistics(self, state):
        """Test descriptive statistics calculation."""
        stats = state.data.describe()
        assert 'count' in stats.index
        assert 'mean' in stats.index
    
    def test_statistics_for_numeric_only(self, state):
        """Test statistics calculated for numeric columns only."""
        numeric_df = state.data.select_dtypes(include=['number'])
        stats = numeric_df.describe()
        assert len(stats.columns) > 0
    
    def test_data_type_info(self, state):
        """Test data type information."""
        dtypes = state.data.dtypes
        assert len(dtypes) == len(state.data.columns)


class TestExploreDataRecommendations:
    """Test model recommendations."""
    
    def test_can_get_recommendations(self, state):
        """Test that recommendations can be generated."""
        from src.data.explorer import DataExplorer
        explorer = DataExplorer(state.data, target_col=state.target_col)
        recommendations = explorer.recommend_models()
        assert len(recommendations) > 0
    
    def test_recommendation_structure(self, state):
        """Test recommendation structure."""
        from src.data.explorer import DataExplorer
        explorer = DataExplorer(state.data, target_col=state.target_col)
        recommendations = explorer.recommend_models()
        
        for rec in recommendations:
            assert 'model' in rec
            assert 'reason' in rec
            # Recommendations have either 'details' or 'pros'/'cons'
            assert 'details' in rec or ('pros' in rec and 'cons' in rec)


class TestExploreDataWorkflow:
    """Test complete exploration workflow."""
    
    def test_complete_exploration_workflow(self, state):
        """Test complete exploration workflow."""
        # 1. Check data loaded
        assert state.has_data()
        
        # 2. Analyze distributions
        numeric_cols = state.data.select_dtypes(include=['number']).columns.tolist()
        assert len(numeric_cols) > 0
        
        # 3. Analyze correlations
        numeric_df = state.data.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        assert corr.shape[0] > 0
        
        # 4. Check missing data
        missing = state.data.isnull().sum()
        assert len(missing) == len(state.data.columns)
        
        # 5. Get statistics
        stats = state.data.describe()
        assert len(stats.columns) > 0


class TestExploreDataEdgeCases:
    """Test edge cases."""
    
    def test_single_numeric_column(self):
        """Test exploration with single numeric column."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        assert len(numeric) == 1
    
    def test_single_categorical_column(self):
        """Test exploration with single categorical column."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'subcategory': ['X', 'Y', 'X', 'Y', 'X']
        })
        cat = df.select_dtypes(include=['object']).columns.tolist()
        assert len(cat) == 2
    
    def test_high_cardinality_categorical(self):
        """Test handling high cardinality categorical columns."""
        df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000)
        })
        cat = df.select_dtypes(include=['object']).columns.tolist()
        # ID as int, not treated as categorical
        assert len(cat) == 0
    
    def test_all_same_value_column(self):
        """Test column with all same values."""
        df = pd.DataFrame({
            'const': [1, 1, 1, 1, 1],
            'var': [1, 2, 3, 4, 5]
        })
        var_col_std = df['const'].std()
        assert var_col_std == 0


class TestExploreDataPerformance:
    """Test performance with larger datasets."""
    
    def test_explore_large_dataset(self):
        """Test exploration with larger dataset."""
        large_df = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000),
            'col3': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        assert len(large_df) == 10000
        
        # Test correlation calculation
        numeric = large_df.select_dtypes(include=['number'])
        corr = numeric.corr()
        assert corr.shape == (2, 2)
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        df = pd.DataFrame({
            'col1': np.random.randn(100000),
            'col2': np.random.randn(100000)
        })
        
        # Describe should work efficiently
        stats = df.describe()
        assert stats.shape[0] > 0
