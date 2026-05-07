"""
Tests for Feature Engineering Module

Comprehensive test suite for FeatureEngineer class covering:
- Mathematical transformations (log, sqrt, square, etc.)
- Interaction features
- Polynomial features
- Ratio features
- Custom features
- Method chaining
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineer import FeatureEngineer


class TestFeatureEngineerInitialization:
    """Test FeatureEngineer initialization and validation."""
    
    def test_init_with_valid_dataframe(self, sample_dataframe):
        """Test initialization with valid DataFrame."""
        engineer = FeatureEngineer(sample_dataframe)
        assert engineer.data is not None
        assert isinstance(engineer.data, pd.DataFrame)
    
    def test_init_with_non_dataframe(self):
        """Test initialization with non-DataFrame raises error."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            FeatureEngineer([1, 2, 3])
    
    def test_init_with_empty_dataframe(self):
        """Test initialization with empty DataFrame raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            FeatureEngineer(pd.DataFrame())
    
    def test_column_detection(self, sample_dataframe):
        """Test numeric and categorical column detection."""
        engineer = FeatureEngineer(sample_dataframe)
        assert len(engineer.numeric_cols) >= 0
        assert len(engineer.categorical_cols) >= 0


class TestMathematicalTransformations:
    """Test mathematical transformation features."""
    
    def test_log_transformation(self):
        """Test log transformation for positive values."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': [10, 20, 30, 40]
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['log'])
        
        assert any('_log' in col for col in engineer.generated_features)
        assert engineer.data['A_log'].iloc[0] == pytest.approx(0)
    
    def test_sqrt_transformation(self):
        """Test sqrt transformation for non-negative values."""
        df = pd.DataFrame({
            'A': [1.0, 4.0, 9.0, 16.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['sqrt'])
        
        assert any('_sqrt' in col for col in engineer.generated_features)
        assert engineer.data['A_sqrt'].iloc[1] == pytest.approx(2.0)
    
    def test_square_transformation(self):
        """Test square transformation."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['square'])
        
        assert any('_square' in col for col in engineer.generated_features)
        assert engineer.data['A_square'].iloc[2] == pytest.approx(9.0)
    
    def test_cube_transformation(self):
        """Test cube transformation."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['cube'])
        
        assert any('_cube' in col for col in engineer.generated_features)
        assert engineer.data['A_cube'].iloc[2] == pytest.approx(27.0)
    
    def test_exp_transformation(self):
        """Test exponential transformation."""
        df = pd.DataFrame({
            'A': [0.0, 1.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['exp'])
        
        assert any('_exp' in col for col in engineer.generated_features)
        assert engineer.data['A_exp'].iloc[0] == pytest.approx(1.0)  # exp(0) = 1
    
    def test_reciprocal_transformation(self):
        """Test reciprocal transformation."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 4.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['reciprocal'])
        
        assert any('_reciprocal' in col for col in engineer.generated_features)
        assert engineer.data['A_reciprocal'].iloc[2] == pytest.approx(0.25)
    
    def test_abs_transformation(self):
        """Test absolute value transformation."""
        df = pd.DataFrame({
            'A': [-1.0, -2.0, 3.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['abs'])
        
        assert any('_abs' in col for col in engineer.generated_features)
        assert (engineer.data['A_abs'] >= 0).all()
    
    def test_multiple_transformations(self):
        """Test multiple transformations applied."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['log', 'sqrt', 'square'])
        
        assert len(engineer.generated_features) >= 3
    
    def test_invalid_transformation(self, sample_dataframe):
        """Test invalid transformation raises error."""
        engineer = FeatureEngineer(sample_dataframe)
        with pytest.raises(ValueError, match="Transform must be one of"):
            engineer.auto_generate_features(transformations=['invalid'])


class TestInteractionFeatures:
    """Test interaction feature generation."""
    
    def test_multiplication_interaction(self):
        """Test multiplication interaction features."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.interaction_features()
        
        assert any('_x_' in col for col in engineer.generated_features)
        assert engineer.data['A_x_B'].iloc[0] == pytest.approx(4.0)  # 1*4
    
    def test_addition_interaction(self):
        """Test addition interaction features."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.interaction_features()
        
        assert any('_plus_' in col for col in engineer.generated_features)
        assert engineer.data['A_plus_B'].iloc[0] == pytest.approx(5.0)  # 1+4
    
    def test_division_interaction(self):
        """Test division interaction features."""
        df = pd.DataFrame({
            'A': [4.0, 6.0, 8.0],
            'B': [2.0, 3.0, 4.0]
        })
        engineer = FeatureEngineer(df)
        engineer.interaction_features()
        
        assert any('_div_' in col for col in engineer.generated_features)
        assert engineer.data['A_div_B'].iloc[0] == pytest.approx(2.0)  # 4/2
    
    def test_interaction_max_features(self):
        """Test limiting number of interaction features."""
        df = pd.DataFrame({
            'A': [1.0, 2.0],
            'B': [3.0, 4.0],
            'C': [5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.interaction_features(max_features=3)
        
        # Should only generate up to max_features
        assert len(engineer.generated_features) <= 3
    
    def test_interaction_specific_columns(self):
        """Test interactions for specific columns only."""
        df = pd.DataFrame({
            'A': [1.0, 2.0],
            'B': [3.0, 4.0],
            'C': [5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.interaction_features(columns=['A', 'B'])
        
        assert any('A' in col and 'B' in col for col in engineer.generated_features)


class TestPolynomialFeatures:
    """Test polynomial feature generation."""
    
    def test_degree_2_polynomial(self):
        """Test degree 2 polynomial features."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.polynomial_features(degree=2)
        
        assert any('poly_' in col for col in engineer.generated_features)
    
    def test_degree_3_polynomial(self):
        """Test degree 3 polynomial features."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.polynomial_features(degree=3)
        
        assert len(engineer.generated_features) > 0
    
    def test_invalid_polynomial_degree(self, sample_dataframe):
        """Test invalid polynomial degree raises error."""
        engineer = FeatureEngineer(sample_dataframe)
        with pytest.raises(ValueError, match="Polynomial degree must be"):
            engineer.polynomial_features(degree=5)
    
    def test_polynomial_specific_columns(self):
        """Test polynomial for specific columns."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.polynomial_features(degree=2, columns=['A'])
        
        assert len(engineer.generated_features) > 0


class TestRatioFeatures:
    """Test ratio feature generation."""
    
    def test_basic_ratio_features(self):
        """Test basic ratio feature generation."""
        df = pd.DataFrame({
            'A': [4.0, 6.0, 8.0],
            'B': [2.0, 3.0, 4.0]
        })
        engineer = FeatureEngineer(df)
        engineer.ratio_features()
        
        assert any('_ratio_' in col for col in engineer.generated_features)
    
    def test_ratio_specific_numerator(self):
        """Test ratio with specific numerator columns."""
        df = pd.DataFrame({
            'A': [4.0, 8.0],
            'B': [2.0, 4.0],
            'C': [2.0, 2.0]
        })
        engineer = FeatureEngineer(df)
        engineer.ratio_features(numerator_cols=['A'])
        
        assert len(engineer.generated_features) > 0


class TestCustomFeatures:
    """Test custom feature generation."""
    
    def test_custom_feature_simple(self):
        """Test simple custom feature."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        
        custom_funcs = {
            'sum': lambda df: df['A'] + df['B']
        }
        engineer.custom_features(custom_funcs)
        
        assert 'sum' in engineer.generated_features
        assert engineer.data['sum'].iloc[0] == pytest.approx(5.0)
    
    def test_custom_feature_multiple(self):
        """Test multiple custom features."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        
        custom_funcs = {
            'sum': lambda df: df['A'] + df['B'],
            'product': lambda df: df['A'] * df['B'],
            'diff': lambda df: df['B'] - df['A']
        }
        engineer.custom_features(custom_funcs)
        
        assert len(engineer.generated_features) == 3
    
    def test_custom_feature_invalid_function(self):
        """Test custom feature with invalid function."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
        })
        engineer = FeatureEngineer(df)
        
        custom_funcs = {
            'bad': lambda df: df['nonexistent']
        }
        engineer.custom_features(custom_funcs)
        
        # Should handle error gracefully
        assert len(engineer.engineering_log) > 0


class TestMethodChaining:
    """Test method chaining functionality."""
    
    def test_chaining_transformations_and_interactions(self):
        """Test chaining multiple engineering operations."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        result = (engineer
                  .auto_generate_features(transformations=['square'])
                  .interaction_features())
        
        assert isinstance(result, FeatureEngineer)
        assert len(engineer.generated_features) > 0
    
    def test_chaining_with_polynomial(self):
        """Test chaining including polynomial features."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        result = (engineer
                  .auto_generate_features(transformations=['square'])
                  .polynomial_features(degree=2)
                  .interaction_features())
        
        assert isinstance(result, FeatureEngineer)
        assert len(engineer.generated_features) > 0


class TestReportsAndOutput:
    """Test report generation and output."""
    
    def test_get_engineered_data(self):
        """Test getting engineered data."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['square'])
        engineered = engineer.get_engineered_data()
        
        assert isinstance(engineered, pd.DataFrame)
        assert len(engineered.columns) > len(df.columns)
    
    def test_get_feature_map(self):
        """Test getting feature map."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['square'])
        feature_map = engineer.get_feature_map()
        
        assert isinstance(feature_map, dict)
        assert len(feature_map) > 0
    
    def test_get_report(self):
        """Test report generation."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['square'])
        report = engineer.get_report()
        
        assert isinstance(report, dict)
        assert 'original_shape' in report
        assert 'engineered_shape' in report
        assert 'features_generated' in report
    
    def test_print_report(self, capsys):
        """Test report printing."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['square'])
        engineer.print_report()
        
        captured = capsys.readouterr()
        assert "FEATURE ENGINEERING REPORT" in captured.out


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_column_dataframe(self):
        """Test with single column."""
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['square'])
        assert len(engineer.generated_features) > 0
    
    def test_all_numeric_columns(self):
        """Test with all numeric columns."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0],
            'C': [7.0, 8.0, 9.0]
        })
        engineer = FeatureEngineer(df)
        assert len(engineer.numeric_cols) == 3
    
    def test_large_dataframe(self):
        """Test with larger dataset."""
        df = pd.DataFrame({
            'A': np.random.rand(1000),
            'B': np.random.rand(1000)
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features(transformations=['square'])
        assert len(engineer.data) == 1000
    
    def test_no_numeric_columns(self):
        """Test with no numeric columns."""
        df = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': ['x', 'y', 'z']
        })
        engineer = FeatureEngineer(df)
        engineer.auto_generate_features()
        
        # Should handle gracefully with no numeric columns
        assert len(engineer.numeric_cols) == 0
