"""Extract human-readable equations from trained regression models."""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class EquationExtractor:
    """Extract and format regression equations from trained models."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize equation extractor.
        
        Args:
            model: Trained regression model
            feature_names: List of feature names (if None, uses generic names)
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = type(model).__name__
    
    def extract_equation(self) -> Optional[str]:
        """Extract regression equation from model.
        
        Returns:
            Human-readable equation string, or None if not supported
        """
        if isinstance(self.model, (LinearRegression, Ridge, Lasso, ElasticNet)):
            return self._extract_linear_equation()
        
        elif isinstance(self.model, DecisionTreeRegressor):
            return self._extract_tree_equation()
        
        elif isinstance(self.model, RandomForestRegressor):
            return self._extract_tree_ensemble_equation()
        
        elif isinstance(self.model, SVR):
            return self._extract_svr_equation()
        
        else:
            return None
    
    def _extract_linear_equation(self) -> str:
        """Extract equation for linear-based models."""
        intercept = self.model.intercept_
        coefficients = self.model.coef_
        
        # Get feature names
        if self.feature_names is None:
            feature_names = [f"x{i+1}" for i in range(len(coefficients))]
        else:
            feature_names = self.feature_names
        
        # Build equation
        terms = []
        
        # Add intercept
        if abs(intercept) > 1e-10:
            terms.append(f"{intercept:.4f}")
        
        # Add coefficient terms
        for coef, name in zip(coefficients, feature_names):
            if abs(coef) > 1e-10:  # Skip near-zero coefficients
                sign = "+" if coef > 0 else "-"
                abs_coef = abs(coef)
                
                if terms and coef > 0:
                    terms.append(f"+ {abs_coef:.4f}·{name}")
                elif terms:
                    terms.append(f"- {abs_coef:.4f}·{name}")
                else:
                    terms.append(f"{coef:.4f}·{name}")
        
        # Combine terms
        equation = " ".join(terms) if terms else "0"
        return f"ŷ = {equation}"
    
    def _extract_tree_equation(self) -> str:
        """Extract simplified equation for decision tree.
        
        For trees, we extract the most important features and their ranges.
        Full tree extraction is complex, so we provide a summary.
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        feature_names = self.feature_names or [f"x{i+1}" for i in range(len(importances))]
        
        # Get top 3 features
        top_indices = np.argsort(importances)[-3:][::-1]
        
        equation_parts = []
        for idx in top_indices:
            if importances[idx] > 0.01:  # Only show features with >1% importance
                pct = importances[idx] * 100
                equation_parts.append(f"{feature_names[idx]} ({pct:.1f}%)")
        
        if equation_parts:
            return f"Tree Model: f({', '.join(equation_parts)})"
        else:
            return "Tree Model: No significant features"
    
    def _extract_tree_ensemble_equation(self) -> str:
        """Extract simplified equation for random forest."""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        feature_names = self.feature_names or [f"x{i+1}" for i in range(len(importances))]
        
        # Get top 3 features
        top_indices = np.argsort(importances)[-3:][::-1]
        
        equation_parts = []
        for idx in top_indices:
            if importances[idx] > 0.01:
                pct = importances[idx] * 100
                equation_parts.append(f"{feature_names[idx]} ({pct:.1f}%)")
        
        if equation_parts:
            return f"Ensemble Model: ŷ = f({', '.join(equation_parts)})"
        else:
            return "Ensemble Model: No significant features"
    
    def _extract_svr_equation(self) -> str:
        """Extract simplified equation for SVM regressor."""
        kernel = getattr(self.model, 'kernel', 'rbf')
        return f"SVM Regression: ŷ = f(X) [kernel={kernel}]"
    
    def get_coefficients_table(self) -> Optional[Dict[str, float]]:
        """Get coefficients as dictionary for display.
        
        Returns:
            Dict mapping feature names to coefficients, or None if not applicable
        """
        if not hasattr(self.model, 'coef_'):
            return None
        
        coefficients = self.model.coef_
        feature_names = self.feature_names or [f"x{i+1}" for i in range(len(coefficients))]
        
        coef_dict = {
            name: coef for name, coef in zip(feature_names, coefficients)
        }
        
        # Sort by absolute value
        return dict(sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Returns:
            Dictionary with model details
        """
        info = {
            'model_type': self.model_type,
            'equation': self.extract_equation(),
        }
        
        # Add intercept if available
        if hasattr(self.model, 'intercept_'):
            info['intercept'] = float(self.model.intercept_)
        
        # Add coefficients if available
        if hasattr(self.model, 'coef_'):
            info['coefficients'] = self.get_coefficients_table()
        
        # Add feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_names or [f"x{i+1}" for i in range(len(importances))]
            info['feature_importance'] = {
                name: float(imp) for name, imp in zip(feature_names, importances)
            }
        
        return info
    
    @staticmethod
    def format_equation_latex(equation: str) -> str:
        """Convert equation to LaTeX format for better display.
        
        Args:
            equation: Plain text equation
        
        Returns:
            LaTeX formatted equation
        """
        # Replace symbols
        latex = equation.replace("ŷ", r"\hat{y}")
        latex = latex.replace("·", r" \times ")
        latex = latex.replace("+", r" + ")
        latex = latex.replace("-", r" - ")
        
        return f"${latex}$"
    
    def validate_equation(self) -> bool:
        """Validate that equation was successfully extracted.
        
        Returns:
            True if equation was extracted, False otherwise
        """
        equation = self.extract_equation()
        return equation is not None


def extract_regression_equation(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> Optional[str]:
    """Convenience function to extract equation.
    
    Args:
        model: Trained regression model
        feature_names: List of feature names
    
    Returns:
        Human-readable equation string
    """
    extractor = EquationExtractor(model, feature_names)
    return extractor.extract_equation()


def get_model_equation_info(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get complete equation and model information.
    
    Args:
        model: Trained regression model
        feature_names: List of feature names
    
    Returns:
        Dictionary with equation, coefficients, and other info
    """
    extractor = EquationExtractor(model, feature_names)
    return extractor.get_model_info()
