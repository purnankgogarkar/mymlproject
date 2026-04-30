# Testing Guide

## Overview

The test suite for the Spotify Recommendation Engine includes unit tests for:
- **Data Quality**: Validation pipeline and error handling
- **Feature Engineering**: Feature creation and ranges
- **Model Inference**: Predictions and serialization

## Test Files

### `conftest.py`
Pytest fixtures and test utilities:
- `sample_data` — Valid 100-row Spotify dataset
- `broken_data` — Invalid dataset for error testing
- `project_root`, `data_dir`, `models_dir` — Path fixtures

### `test_data_quality.py`
Tests for data quality validation:
- **Pass tests**: Valid datasets should pass validation
- **Fail tests**: Invalid data (out-of-range, NaN, empty) should be caught
- **Edge cases**: Single rows, missing columns, extra columns

### `test_features.py`
Tests for feature engineering:
- **Dimensionality**: New features added correctly
- **No NaN**: All engineered features are valid
- **Ranges**: Features within expected bounds (0-1, etc.)
- **Preservation**: Original columns preserved
- **Edge cases**: Constant values, identical values, row ordering

### `test_model.py`
Tests for model predictions:
- **Predictions**: Model makes valid predictions
- **Ranges**: Predictions in [0, 1] for probabilities
- **Persistence**: Models can be saved/loaded
- **Consistency**: Same input → same prediction
- **Attributes**: Feature importance, classes

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_features.py -v
```

### Run specific test class
```bash
pytest tests/test_model.py::TestModelPredictions -v
```

### Run specific test function
```bash
pytest tests/test_data_quality.py::TestDataQualityPass::test_quality_passes_on_clean_data -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run with output
```bash
pytest tests/ -v -s
```

### Run faster (stop on first failure)
```bash
pytest tests/ -x
```

### Run in parallel (requires pytest-xdist)
```bash
pip install pytest-xdist
pytest tests/ -n auto
```

## Test Statistics

- **Total test cases**: 50+
- **Data Quality**: 15 tests
- **Feature Engineering**: 20 tests
- **Model Inference**: 15+ tests

## Expected Output

All tests should **PASS** when run:

```
tests/test_data_quality.py::TestDataQualityPass::test_quality_passes_on_clean_data PASSED
tests/test_data_quality.py::TestDataQualityPass::test_quality_reports_statistics PASSED
tests/test_data_quality.py::TestDataQualityPass::test_quality_check_structure PASSED
...
tests/test_features.py::TestFeatureEngineering::test_features_increase_dimensionality PASSED
tests/test_features.py::TestFeatureEngineering::test_features_output_correct_count PASSED
...
tests/test_model.py::TestModelPredictions::test_model_makes_predictions PASSED
tests/test_model.py::TestModelPredictions::test_predictions_are_binary_classification PASSED
...

==================== 50+ passed in 5.23s ====================
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/ -v --tb=short
```

## Troubleshooting

### Tests fail with import errors
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest tests/ -v
```

### Fixtures not found
```bash
# Ensure conftest.py is in tests/ directory
# Run pytest from project root
cd spotify-recsys
pytest tests/ -v
```

### Memory issues with large datasets
- Reduce test dataset size in conftest.py
- Run tests without coverage: `pytest tests/ -v`

### Slow tests
```bash
# Run without output capture (faster)
pytest tests/ -v -s

# Run in parallel
pytest tests/ -n auto
```

## Adding New Tests

### Template
```python
def test_something(sample_data):
    """Test description."""
    # Arrange
    result = some_function(sample_data)
    
    # Act & Assert
    assert result is not None
```

### Best Practices
1. One assertion per test when possible
2. Use fixtures for reusable data
3. Test both success and failure cases
4. Use descriptive test names
5. Group related tests in classes

## Coverage Goals

- Data pipeline: >90%
- Feature engineering: >85%
- Model inference: >80%
- Overall target: >80%

Run coverage reports:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```
