# Phase 4: Config System + MLflow Integration

**Duration:** 2 days (Days 7-8)  
**Goal:** Enable YAML-based configuration management with optional MLflow experiment tracking and hyperparameter tuning  
**Estimated Tests:** 25+ tests  

---

## 📋 Phase Overview

### Problem Statement
- Currently, models are trained via direct Python API calls
- No centralized configuration management → hard to reproduce experiments
- No experiment tracking → can't compare model runs
- No hyperparameter tuning framework → manual optimization only

### Solution
1. **ConfigLoader** — Load & validate YAML configs with auto-detection
2. **Model Defaults** — Pre-tuned hyperparameters per model type
3. **MLflow Integration** — Track experiments, metrics, artifacts (optional)
4. **Optuna Tuning** — Automated hyperparameter search (optional)

---

## 🏗️ Components to Build

### 1. ConfigLoader (`src/config/config_loader.py`)

**Purpose:** Load and validate YAML configuration files

**Structure:**
```yaml
# config.yaml
data:
  path: data/my_dataset.csv
  target: 'target_column'
  test_size: 0.2
  random_state: 42

preprocessing:
  missing_value_strategy: mean
  encoding_method: auto
  scaling_method: standard
  outlier_method: iqr

feature_engineering:
  auto_generate: true
  transformations: [log, sqrt, square]
  interactions: true
  polynomials:
    enabled: true
    degree: 2

model:
  type: classification  # auto-detected if omitted
  name: RandomForest
  hyperparams:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5

evaluation:
  cv_folds: 5
  metrics: [accuracy, f1, auc_roc]

mlflow:
  enabled: false
  experiment_name: my_experiment
  tracking_uri: http://localhost:5000

optuna:
  enabled: false
  n_trials: 100
  search_space:
    n_estimators: [50, 200]
    max_depth: [5, 20]
```

**Key Methods:**
- `load(config_path)` → Load YAML
- `validate()` → Validate config structure
- `get_data_config()` → Data pipeline config
- `get_preprocessing_config()` → Preprocessing settings
- `get_feature_engineering_config()` → Feature engineering
- `get_model_config()` → Model name + hyperparams
- `get_evaluation_config()` → Evaluation settings
- `to_dict()` → Return as dictionary
- `print_config()` → Pretty-print

**Features:**
- Auto-load defaults if keys missing
- Type validation (path exists, numeric ranges, enum values)
- Environment variable substitution (`${VAR_NAME}`)
- Schema validation using pydantic (optional)
- Method chaining support

---

### 2. Model Defaults Registry (`src/config/model_defaults.py`)

**Purpose:** Store pre-tuned hyperparameters per model

**Structure:**
```python
MODEL_DEFAULTS = {
    'classification': {
        'LogisticRegression': {'C': 1.0, 'max_iter': 1000, ...},
        'RandomForest': {'n_estimators': 100, 'max_depth': 10, ...},
        'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, ...},
        'XGBoost': {'n_estimators': 100, 'max_depth': 6, ...},
        'LightGBM': {'n_estimators': 100, 'num_leaves': 31, ...},
        'SVM': {'C': 1.0, 'kernel': 'rbf', ...},
        'KNeighbors': {'n_neighbors': 5, ...},
        'DecisionTree': {'max_depth': 10, ...},
        'NeuralNetwork': {'hidden_layer_sizes': (100, 50), ...},
    },
    'regression': {
        'LinearRegression': {},
        'Ridge': {'alpha': 1.0, ...},
        'Lasso': {'alpha': 1.0, ...},
        'RandomForest': {'n_estimators': 100, 'max_depth': None, ...},
        # ... more regression models
    }
}
```

**Key Methods:**
- `get_defaults(problem_type, model_name)` → Return hyperparams
- `get_tuning_space(model_name)` → Return ranges for Optuna
- `list_models(problem_type)` → List all available models
- `update_defaults(model_name, hyperparams)` → Custom overrides

---

### 3. MLflow Integration (`src/config/mlflow_tracker.py`)

**Purpose:** Track experiments, metrics, and artifacts (optional)

**Key Methods:**
- `__init__(experiment_name, tracking_uri)` → Initialize experiment
- `log_config(config_dict)` → Log config as artifact
- `log_metrics(metrics_dict)` → Log all metrics
- `log_params(params_dict)` → Log hyperparameters
- `log_model(model, model_name)` → Log trained model
- `end_run()` → End experiment run
- `get_best_run(metric)` → Retrieve best historical run

**Features:**
- Auto-create experiment if not exists
- URI support (local, remote, S3, GCS)
- Artifact logging (config, model, data)
- Metric tracking (train, validation, test)
- Hyperparameter logging

---

### 4. Optuna Hyperparameter Tuner (`src/config/optuna_tuner.py`)

**Purpose:** Automated hyperparameter search (optional)

**Key Methods:**
- `__init__(X, y, model_name, n_trials, cv_folds)` → Setup tuner
- `define_search_space(param_ranges)` → Define search ranges
- `tune()` → Run optimization
- `get_best_params()` → Best hyperparameters found
- `get_best_score()` → Best CV score
- `get_trials_history()` → All trial results
- `plot_optimization()` → Visualize optimization

**Features:**
- Support pruning (stop bad trials early)
- Parallel trials (configurable)
- Different samplers (TPE, Random, Bayesian)
- Save/load optimization state

---

## 📝 Tasks

### Task 1: ConfigLoader Implementation
- [ ] Create `src/config/config_loader.py`
- [ ] Implement YAML loading with auto-detection
- [ ] Add config validation (schema, path existence)
- [ ] Add environment variable substitution
- [ ] Add method chaining support
- [ ] Write 8+ tests

### Task 2: Model Defaults Registry
- [ ] Create `src/config/model_defaults.py`
- [ ] Define defaults for all 20 models (9 class + 11 reg)
- [ ] Implement getter methods
- [ ] Add tuning space definitions for Optuna
- [ ] Write 4+ tests

### Task 3: MLflow Integration (Optional)
- [ ] Create `src/config/mlflow_tracker.py`
- [ ] Implement experiment creation/tracking
- [ ] Add metric/param logging
- [ ] Add model logging
- [ ] Write 6+ tests

### Task 4: Optuna Tuner (Optional)
- [ ] Create `src/config/optuna_tuner.py`
- [ ] Implement hyperparameter search
- [ ] Add search space definitions
- [ ] Add visualization support
- [ ] Write 4+ tests

### Task 5: Integration Tests
- [ ] Test full workflow: config → train → track → evaluate
- [ ] Test with different datasets
- [ ] Test config validation edge cases
- [ ] Write 5+ integration tests

### Task 6: Documentation & Examples
- [ ] Create example configs (classification, regression, tuning)
- [ ] Document YAML schema
- [ ] Add Quick Start guide
- [ ] Update README.md

---

## 🎯 Success Criteria

✅ **Core (Must Have)**
- ConfigLoader loads and validates YAML configs
- Model defaults registry complete for all 20 models
- 25+ tests passing (100% pass rate)
- Full integration with Phase 3 GenericTrainer
- README updated with Phase 4 progress

✅ **Optional (Nice to Have)**
- MLflow experiment tracking working
- Optuna hyperparameter tuning functional
- Visualization of tuning process
- Example configs for common use cases

---

## 🗂️ File Structure

```
src/config/
├── __init__.py
├── config_loader.py       # YAML config loading + validation
├── model_defaults.py      # Pre-tuned hyperparameters
├── mlflow_tracker.py      # MLflow experiment tracking (optional)
└── optuna_tuner.py        # Optuna hyperparameter search (optional)

tests/
├── test_config_loader.py  # 8+ tests
├── test_model_defaults.py # 4+ tests
├── test_mlflow_tracker.py # 6+ tests (if implemented)
├── test_optuna_tuner.py   # 4+ tests (if implemented)
└── test_phase4_integration.py # 5+ integration tests

configs/
├── default_classification.yaml
├── default_regression.yaml
├── iris_example.yaml
├── titanic_example.yaml
└── optuna_tuning.yaml

docs/
└── config_schema.md       # YAML schema documentation
```

---

## 📊 Estimated Timeline

| Task | Duration | Tests |
|------|----------|-------|
| ConfigLoader | 4 hours | 8 |
| Model Defaults | 2 hours | 4 |
| MLflow Integration | 3 hours | 6 |
| Optuna Tuner | 3 hours | 4 |
| Integration Tests | 2 hours | 5 |
| Documentation | 1 hour | - |
| **TOTAL** | **15 hours (2 days)** | **27+ tests** |

---

## 🚀 Implementation Order

**Day 7:**
1. ConfigLoader (most foundational)
2. Model Defaults Registry
3. Integration tests for both

**Day 8:**
4. MLflow Integration (if enabled in config)
5. Optuna Tuner (if enabled in config)
6. Final tests & documentation update

---

## ✅ Phase Completion Checklist

- [ ] ConfigLoader module created + tested
- [ ] Model defaults registry complete
- [ ] 25+ tests passing (100%)
- [ ] Full end-to-end workflow: config → train → evaluate
- [ ] Example configs created
- [ ] Phase 4 code committed to git
- [ ] README.md updated with Phase 4 progress
- [ ] Skills file updated
