"""
Tests for DataExplorer
"""

import pytest
import pandas as pd
import numpy as np
from src.data.explorer import DataExplorer


class TestDataExplorerInitialization:
    """Test DataExplorer initialization."""
    
    def test_init_with_dataframe(self, sample_dataframe):
        """Test initialization with dataframe."""
        explorer = DataExplorer(sample_dataframe)
        assert explorer.df is not None
        assert len(explorer.df) == 5
    
    def test_init_with_target_col(self, sample_dataframe):
        """Test initialization with target column."""
        explorer = DataExplorer(sample_dataframe, target_col='numeric_col')
        assert explorer.target_col == 'numeric_col'


class TestDataExplorerAnalyze:
    """Test data analysis functionality."""
    
    def test_analyze_basic(self, sample_dataframe):
        """Test basic analysis."""
        explorer = DataExplorer(sample_dataframe)
        analysis = explorer.analyze()
        
        assert 'dataset_info' in analysis
        assert 'missing_analysis' in analysis
        assert 'numeric_analysis' in analysis
        assert 'categorical_analysis' in analysis
    
    def test_dataset_info(self, sample_dataframe):
        """Test dataset information extraction."""
        explorer = DataExplorer(sample_dataframe)
        info = explorer._dataset_info()
        
        assert info['rows'] == 5
        assert info['columns'] == 4
        assert 'numeric_columns' in info
        assert 'categorical_columns' in info
    
    def test_numeric_analysis(self, sample_dataframe):
        """Test numeric column analysis."""
        explorer = DataExplorer(sample_dataframe)
        analysis = explorer._numeric_analysis()
        
        assert 'numeric_col' in analysis
        assert 'mean' in analysis['numeric_col']
        assert 'std' in analysis['numeric_col']
        assert 'min' in analysis['numeric_col']
        assert 'max' in analysis['numeric_col']
    
    def test_categorical_analysis(self, sample_dataframe):
        """Test categorical column analysis."""
        explorer = DataExplorer(sample_dataframe)
        analysis = explorer._categorical_analysis()
        
        assert 'string_col' in analysis
        assert 'unique' in analysis['string_col']
        assert analysis['string_col']['unique'] == 5


class TestDataExplorerTargetAnalysis:
    """Test target variable analysis."""
    
    def test_target_analysis_regression(self, sample_numeric_csv):
        """Test target analysis for regression."""
        # First load the data
        df = pd.read_csv(sample_numeric_csv)
        explorer = DataExplorer(df, target_col='price')
        analysis = explorer._target_analysis()
        
        assert analysis['problem_type'] == 'Regression'
        assert 'stats' in analysis
        assert 'mean' in analysis['stats']
    
    def test_target_analysis_classification(self, sample_titanic_csv):
        """Test target analysis for classification."""
        # First load the data
        df = pd.read_csv(sample_titanic_csv)
        explorer = DataExplorer(df, target_col='Survived')
        analysis = explorer._target_analysis()
        
        assert analysis['problem_type'] == 'Classification'
        assert 'class_distribution' in analysis
    
    def test_target_not_found(self, sample_dataframe):
        """Test error when target column not found."""
        explorer = DataExplorer(sample_dataframe, target_col='nonexistent')
        analysis = explorer._target_analysis()
        
        assert 'error' in analysis


class TestDataExplorerRecommendations:
    """Test model and preprocessing recommendations."""
    
    def test_recommend_models_regression(self, sample_numeric_csv):
        """Test model recommendations for regression."""
        df = pd.read_csv(sample_numeric_csv)
        explorer = DataExplorer(df, target_col='price')
        recommendations = explorer.recommend_models()
        
        assert len(recommendations) > 0
        model_names = [r['model'] for r in recommendations]
        assert 'RandomForest' in model_names or 'GradientBoosting' in model_names
    
    def test_recommend_models_classification(self, sample_titanic_csv):
        """Test model recommendations for classification."""
        df = pd.read_csv(sample_titanic_csv)
        explorer = DataExplorer(df, target_col='Survived')
        recommendations = explorer.recommend_models()
        
        assert len(recommendations) > 0
        model_names = [r['model'] for r in recommendations]
        assert 'LogisticRegression' in model_names or 'RandomForest' in model_names
    
    def test_recommend_preprocessing(self, sample_dataframe_with_missing):
        """Test preprocessing recommendations."""
        explorer = DataExplorer(sample_dataframe_with_missing)
        recommendations = explorer.recommend_preprocessing()
        
        assert 'missing_value_handling' in recommendations
        assert len(recommendations['missing_value_handling']) > 0


class TestDataExplorerIntegration:
    """Integration tests for DataExplorer."""
    
    def test_full_analysis_workflow(self, sample_titanic_csv):
        """Test full analysis workflow."""
        df = pd.read_csv(sample_titanic_csv)
        explorer = DataExplorer(df, target_col='Survived')
        
        # Run full analysis
        analysis = explorer.analyze()
        recommendations = explorer.recommend_models()
        preprocessing = explorer.recommend_preprocessing()
        
        # Verify results
        assert analysis is not None
        assert recommendations is not None
        assert preprocessing is not None
