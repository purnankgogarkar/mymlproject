"""
Pytest fixtures for testing
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_iris_csv(tmp_path):
    """Create sample iris dataset CSV."""
    iris_data = {
        'sepal_length': [5.1, 4.9, 4.7, 7.0, 6.4],
        'sepal_width': [3.5, 3.0, 3.2, 3.2, 3.2],
        'petal_length': [1.4, 1.4, 1.3, 4.7, 4.5],
        'petal_width': [0.2, 0.2, 0.2, 1.4, 1.5],
        'species': ['setosa', 'setosa', 'setosa', 'virginica', 'virginica']
    }
    df = pd.DataFrame(iris_data)
    file_path = tmp_path / "iris.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_titanic_csv(tmp_path):
    """Create sample titanic dataset CSV."""
    titanic_data = {
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, np.nan, 35.0, 35.0],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05]
    }
    df = pd.DataFrame(titanic_data)
    file_path = tmp_path / "titanic.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_numeric_csv(tmp_path):
    """Create sample numeric regression dataset."""
    housing_data = {
        'price': [280000, 335000, 155000, 275000, 310000],
        'area_sqft': [2100, 2500, 1200, 2300, 2600],
        'bedrooms': [3, 4, 2, 3, 4],
        'age_years': [15, 5, 40, 20, 8],
        'garage_spaces': [2, 2, 1, 2, 3]
    }
    df = pd.DataFrame(housing_data)
    file_path = tmp_path / "housing.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe with mixed types."""
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'string_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True],
    })


@pytest.fixture
def sample_dataframe_with_missing():
    """Create sample dataframe with missing values."""
    return pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': ['a', np.nan, 'c', 'd', 'e'],
        'col3': [1.1, 2.2, 3.3, np.nan, 5.5],
    })


@pytest.fixture
def sample_dataframe_with_duplicates():
    """Create sample dataframe with duplicate rows."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 1],
        'value': [10, 20, 30, 40, 50, 10],
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
    })
