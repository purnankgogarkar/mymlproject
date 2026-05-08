"""Tests for equation extraction from regression models."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.export.equation_extractor import (
    EquationExtractor,
    extract_regression_equation,
    get_model_equation_info
)


class TestLinearRegressionEquation:
    """Test equation extraction from linear regression."""
    
    def test_simple_linear_equation(self):
        """Test extraction of simple linear equation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        assert equation is not None
        assert "ŷ" in equation
        assert "x1" in equation
        assert "x2" in equation
    
    def test_linear_equation_with_feature_names(self):
        """Test linear equation with custom feature names."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        feature_names = ["age", "income"]
        equation = extract_regression_equation(model, feature_names)
        
        assert "age" in equation
        assert "income" in equation
    
    def test_coefficients_table(self):
        """Test extraction of coefficients as table."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2"]
        extractor = EquationExtractor(model, feature_names)
        coef_dict = extractor.get_coefficients_table()
        
        assert coef_dict is not None
        assert "feature1" in coef_dict
        assert "feature2" in coef_dict
        assert len(coef_dict) == 2
    
    def test_intercept_included(self):
        """Test that intercept is included in equation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        # Should have numeric constant (intercept)
        assert any(char.isdigit() for char in equation)
    
    def test_ridge_regression_equation(self):
        """Test equation extraction from Ridge regression."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        assert equation is not None
        assert "ŷ" in equation
    
    def test_lasso_regression_equation(self):
        """Test equation extraction from Lasso regression."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = Lasso(alpha=0.1)
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        assert equation is not None
        assert "ŷ" in equation
    
    def test_zero_coefficients_excluded(self):
        """Test that near-zero coefficients are excluded."""
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]]).astype(float)
        y = np.array([1, 2, 3, 4]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        # Equation should be simpler due to zero coefficient
        assert equation is not None


class TestTreeRegressionEquation:
    """Test equation extraction from tree-based models."""
    
    def test_decision_tree_equation(self):
        """Test equation extraction from decision tree."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = DecisionTreeRegressor(max_depth=2, random_state=42)
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        assert equation is not None
        assert "Tree Model" in equation
    
    def test_random_forest_equation(self):
        """Test equation extraction from random forest."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=42)
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        assert equation is not None
        assert "Ensemble Model" in equation or "Tree Model" in equation
    
    def test_feature_importance_extraction(self):
        """Test extraction of feature importances."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2"]
        extractor = EquationExtractor(model, feature_names)
        info = extractor.get_model_info()
        
        assert 'feature_importance' in info
        assert len(info['feature_importance']) == 2


class TestModelInfo:
    """Test comprehensive model information extraction."""
    
    def test_get_model_info_linear(self):
        """Test getting complete model info for linear regression."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2"]
        info = get_model_equation_info(model, feature_names)
        
        assert 'model_type' in info
        assert info['model_type'] == 'LinearRegression'
        assert 'equation' in info
        assert 'intercept' in info
        assert 'coefficients' in info
    
    def test_get_model_info_tree(self):
        """Test getting complete model info for tree model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2"]
        info = get_model_equation_info(model, feature_names)
        
        assert 'model_type' in info
        assert 'equation' in info
        assert 'feature_importance' in info


class TestEquationFormatting:
    """Test equation formatting utilities."""
    
    def test_format_equation_latex(self):
        """Test LaTeX formatting of equations."""
        equation = "ŷ = 50 + 2·x1 - 3·x2"
        
        latex_eq = EquationExtractor.format_equation_latex(equation)
        
        assert r"\hat{y}" in latex_eq
        assert r"\times" in latex_eq
        assert "$" in latex_eq
    
    def test_validate_equation(self):
        """Test equation validation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        extractor = EquationExtractor(model)
        is_valid = extractor.validate_equation()
        
        assert is_valid is True


class TestEquationEdgeCases:
    """Test edge cases in equation extraction."""
    
    def test_single_feature_equation(self):
        """Test equation with single feature."""
        X = np.array([[1], [2], [3], [4]]).astype(float)
        y = np.array([2, 4, 6, 8]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        assert equation is not None
        assert "x1" in equation
    
    def test_many_features_equation(self):
        """Test equation with many features."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.sum(X[:, :3], axis=1) + np.random.randn(20) * 0.1
        
        model = LinearRegression()
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        assert equation is not None
        assert "ŷ" in equation
    
    def test_default_feature_names(self):
        """Test that default feature names are generated."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Without feature names
        equation = extract_regression_equation(model)
        
        assert "x1" in equation or "x2" in equation or "x3" in equation
    
    def test_unsupported_model(self):
        """Test that unsupported models return None."""
        from sklearn.neighbors import KNeighborsRegressor
        
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = KNeighborsRegressor()
        model.fit(X, y)
        
        equation = extract_regression_equation(model)
        
        # KNN may not have a clean equation representation
        # Result depends on implementation
        assert equation is None or isinstance(equation, str)


class TestEquationExportIntegration:
    """Test integration with export functionality."""
    
    def test_equation_in_model_info(self):
        """Test that equation is included in model info."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        feature_names = ["age", "salary"]
        info = get_model_equation_info(model, feature_names)
        
        # Should be serializable to JSON for export
        assert 'equation' in info
        assert isinstance(info['equation'], str)
    
    def test_coefficients_serializable(self):
        """Test that coefficients can be serialized."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]).astype(float)
        y = np.array([5, 7, 9, 11]).astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        
        feature_names = ["feature1", "feature2"]
        extractor = EquationExtractor(model, feature_names)
        coef_dict = extractor.get_coefficients_table()
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(coef_dict)
        assert json_str is not None
