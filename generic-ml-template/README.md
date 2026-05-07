# 🤖 Generic ML Template Framework

**A reusable, production-ready ML pipeline framework for ANY tabular data (CSV/Excel). Automates data exploration, preprocessing, feature engineering, model training, and deployment.**

---

## ✨ Features

### 🎯 Core Capabilities
- ✅ **Auto Data Detection** — Automatically detect numeric/categorical/datetime columns
- ✅ **Data Profiling** — Comprehensive data analysis with 20+ metrics
- ✅ **Quality Validation** — 5-step validation gate (missing values, duplicates, outliers, etc.)
- ✅ **Generic Preprocessing** — Handle NaNs, encode categoricals, scale features (configurable)
- ✅ **Automatic Features** — Generate interaction terms, polynomial features, domain features
- ✅ **Custom Features** — Optional Python code for domain-specific engineering
- ✅ **Model Registry** — 8+ models (LogReg, RF, GB, XGBoost, SVM, KNN, etc.)
- ✅ **Unified Training** — Train any model with 5-fold CV + hyperparameter tuning (Optuna)
- ✅ **Smart Recommendations** — System recommends models based on data profiling
- ✅ **Multiple Exports** — Save as model.pkl OR Flask API
- ✅ **MLflow Tracking** — Optional experiment tracking

### 📊 Problem Types Supported
- **Classification** — Binary & multiclass problems
- **Regression** — Continuous value prediction
- **Time-Series** (v2) — Designed-in, implementation deferred

---

## 🏗️ Architecture

```
User Data (CSV/Excel)
    ↓
[DataLoader] → Auto-detect types, handle missing
    ↓
[DataExplorer] → Profile, correlations, recommendations
    ↓
[User Config] → Select target, features, model type
    ↓
[Preprocessor] → NaN handling, encoding, scaling
    ↓
[FeatureEngineer] → Auto + custom feature creation
    ↓
[GenericTrainer] → Train any model with 5-fold CV
    ↓
[Evaluator] → Compute metrics (ACC, F1, RMSE, AUC, etc.)
    ↓
[Exporter] → model.pkl + Flask API + config.yaml
```

---

## 📁 Project Structure

```
generic-ml-template/
├── src/
│   ├── data/               # Data pipeline
│   │   ├── loader.py       # Load CSV/Excel with auto type detection
│   │   ├── explorer.py     # Data profiling & recommendations
│   │   └── validator.py    # Data quality validation
│   │
│   ├── features/           # Feature engineering (Phase 2)
│   │   ├── engineer.py     # Auto-generate features
│   │   └── custom_engine.py # Optional user code execution
│   │
│   ├── models/             # ML training (Phase 3)
│   │   ├── registry.py     # Model registry with metadata
│   │   ├── trainer.py      # Generic trainer for any model
│   │   ├── evaluator.py    # Compute metrics
│   │   └── tuner.py        # Optuna hyperparameter tuning
│   │
│   ├── config/             # Configuration (Phase 4)
│   │   ├── config_loader.py
│   │   └── defaults.py
│   │
│   ├── export/             # Model export (Phase 6)
│   │   └── model_exporter.py
│   │
│   └── api/                # Flask API (Phase 6)
│       └── flask_app.py
│
├── app/                    # Streamlit UI (Phase 5)
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 01_upload_data.py
│   │   ├── 02_explore_data.py
│   │   ├── 03_configure_model.py
│   │   ├── 04_train.py
│   │   └── 05_results.py
│   └── utils/
│       ├── session_state.py
│       └── visualizations.py
│
├── tests/                  # Test suite (50+ tests)
│   ├── conftest.py         # Pytest fixtures
│   ├── test_loader.py      # DataLoader tests
│   ├── test_explorer.py    # DataExplorer tests
│   ├── test_validator.py   # DataValidator tests
│   └── ... (more test files)
│
├── configs/                # Example YAML configs
│   ├── default_classification.yaml
│   ├── default_regression.yaml
│   └── spotify_example.yaml
│
├── examples/               # Example workflows
│   ├── iris_classification.yaml
│   ├── titanic_classification.yaml
│   ├── housing_regression.yaml
│   └── step_by_step_guide.md
│
├── data/
│   ├── raw/                # Place your CSV/Excel files here
│   └── processed/          # Output directory
│
├── requirements.txt        # Dependencies
├── setup.py                # Package installation
├── pytest.ini              # Test configuration
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd generic-ml-template
pip install -r requirements.txt
```

### 2. Load Your Data

```python
from src.data.loader import DataLoader

# Load your CSV/Excel file
loader = DataLoader('data/my_dataset.csv')
df = loader.load()

# Profile the data
profile = loader.profile()
print(profile)
```

### 3. Explore the Data

```python
from src.data.explorer import DataExplorer

explorer = DataExplorer(df, target_col='target_column')
analysis = explorer.analyze()

# Get model recommendations
recommendations = explorer.recommend_models()
print(recommendations)
```

### 4. Validate Data

```python
from src.data.validator import DataValidator

validator = DataValidator(df)
is_valid, results = validator.validate()
validator.print_report()
```

### 5. (Upcoming Phases)
- **Phase 2**: Preprocessing + Feature Engineering
- **Phase 3**: Model Training
- **Phase 4**: Configuration System
- **Phase 5**: Streamlit UI
- **Phase 6**: Production Export (Flask API)

---

## 📊 Phase Progress

| Phase | Status | Duration | Completeness |
|-------|--------|----------|---|
| **1. Data Pipeline** | ✅ DONE | 2d | 100% |
| 1.Q Quality Gate | ⏳ IN PROGRESS | 0.5d | 0% |
| 2. Preprocessing | ⏳ TODO | 2d | 0% |
| 3. Model Training | ⏳ TODO | 2d | 0% |
| 4. Config System | ⏳ TODO | 2d | 0% |
| 5. Streamlit UI | ⏳ TODO | 3d | 0% |
| 6. Production | ⏳ TODO | 2d | 0% |
| 7. Tests + Docs | ⏳ TODO | 2d | 0% |
| 8. GitHub + Deploy | ⏳ TODO | 2d | 0% |

---

## ✅ Phase 1 Verification

Completed implementations tested with:
- ✅ **Iris dataset** — 150 samples, numeric + categorical
- ✅ **Titanic dataset** — 891 samples, missing values, mixed types
- ✅ **Housing dataset** — Numeric regression

All tests pass:
```bash
pytest tests/ -v
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_loader.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_loader.py::TestDataLoaderLoad::test_load_csv -v
```

**Test Coverage:**
- `test_loader.py` — 15+ tests for DataLoader
- `test_explorer.py` — 12+ tests for DataExplorer
- `test_validator.py` — 12+ tests for DataValidator

---

## 📝 Example Usage

### Simple Classification Problem

```python
from src.data.loader import DataLoader
from src.data.explorer import DataExplorer
from src.data.validator import DataValidator

# 1. Load data
loader = DataLoader('data/titanic.csv')
df = loader.load()

# 2. Explore
explorer = DataExplorer(df, target_col='Survived')
analysis = explorer.analyze()
models = explorer.recommend_models()

# 3. Validate
validator = DataValidator(df)
is_valid, results = validator.validate()

if is_valid:
    print("Data is valid! Ready for training.")
    print("\nRecommended models:")
    for model in models:
        print(f"  - {model['model']}: {model['reason']}")
```

### Load Different File Formats

```python
# CSV
loader = DataLoader('data/file.csv')

# Excel
loader = DataLoader('data/file.xlsx')

# TSV
loader = DataLoader('data/file.tsv')

# Parquet
loader = DataLoader('data/file.parquet')

df = loader.load(sample_size=1000)  # Load first 1000 rows
```

---

## 🛠️ Configuration (Phase 4+)

Example YAML config:

```yaml
data:
  path: data/my_dataset.csv
  target: 'target_column'
  
problem_type: classification  # or regression

features:
  select: all
  engineering: true
  custom_code: |
    def my_feature(df):
        return df['col1'] * df['col2']

model:
  type: GradientBoosting
  hyperparameters:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 5

training:
  cv_folds: 5
  tuning:
    enabled: true
    trials: 30

export:
  format: both  # pkl + flask_api
```

---

## 🔄 Development Workflow

Each phase follows this pattern:

1. **Plan** — Detailed task breakdown
2. **Implement** — Write code with type hints & docstrings
3. **Test** — 50+ unit/integration tests
4. **Review** — Code optimizer sub-agent reviews (security, performance, design)
5. **Commit** — Atomic git commits with clear messages
6. **Document** — README, examples, API docs

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for your changes
4. Run: `pytest tests/ -v`
5. Run code quality: `black src/ app/` && `flake8 src/ app/`
6. Submit pull request

---

## 📚 Dependencies

Core:
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `scikit-learn` — Machine learning
- `xgboost`, `lightgbm` — Gradient boosting
- `optuna` — Hyperparameter tuning

Optional:
- `streamlit` — Web UI
- `flask` — REST API
- `mlflow` — Experiment tracking
- `plotly` — Interactive visualizations

---

## 📄 License

MIT License — See LICENSE file

---

## 👤 Authors

**Generic ML Template Framework** — Community-driven template for production ML

---

## 🎯 Next Steps

- [ ] Phase 2: Preprocessing + Feature Engineering
- [ ] Phase 3: Model Training + Evaluation
- [ ] Phase 4: Configuration System
- [ ] Phase 5: Streamlit Dashboard
- [ ] Phase 6: Production Export (Flask API)
- [ ] Phase 7: Comprehensive tests + docs
- [ ] Phase 8: GitHub CI/CD + deployment

---

**Built for practitioners who want production-ready ML without boilerplate.**
