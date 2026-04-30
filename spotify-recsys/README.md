# 🎵 Spotify Track Recommendation Engine

**Production-ready machine learning system for personalized music track recommendations using content-based and collaborative filtering with hybrid ensemble approach.**

[� **GitHub Repository**](https://github.com/purnankgogarkar/mymlproject) | [📊 MLflow Tracking](http://localhost:5000) | [📧 Contact: purnank18@gmail.com](mailto:purnank18@gmail.com)

---

## 📋 Project Overview

### The Problem
Spotify has 100+ million tracks. Users want personalized recommendations based on their listening patterns, but:
- Pure collaborative filtering requires massive user-item matrices (memory-intensive)
- Content-based approaches need sophisticated audio feature engineering
- Cold-start problem for new users/tracks limits collaborative filtering

### Solution
**Hybrid Recommendation Engine** combining:
- **Content-Based Filtering** (cosine similarity on audio features)
- **Collaborative Filtering** (k-NN on user interaction patterns)
- **Ensemble Classifier** (GradientBoosting predicts track appeal)

### End User
Music recommendation systems in:
- Streaming platform algorithms
- DJ systems
- Playlist generation tools
- Music discovery features

### Data
- **Source:** Spotify dataset (89,740 tracks)
- **Features:** 21 audio attributes (energy, tempo, danceability, valence, etc.)
- **Target:** Track engagement/popularity (binary classification)
- **Split:** 80% train / 20% test

### Model Output
Binary classification: **Will user like this track?** (0 = No, 1 = Yes)
- Confidence score: probability of track appeal (0-1)
- Prediction rationale: feature importance breakdown
- Alternative recommendations: 5 most similar tracks

### Key Design Decision
**Chose GradientBoosting over deep learning** because:
- Smaller dataset (89K samples) favors tree-based models
- Interpretability crucial for recommendation explainability
- 72.6% F1-score with 50-60 min training (vs deep learning 2+ hours)
- Feature importance directly shows what makes a track appealing

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
│  CSV → Loader → Quality Check → Cleaner → Feature Engineer      │
│  (89,740 tracks)  (validation)  (NaN handling)  (33 features)    │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING LAYER                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Content-Based     Collaborative       Baseline Model   │    │
│  │  (Cosine Similar)  (KNN on matrix)     (LogisticReg)   │    │
│  │  1 model           1 model              1 model         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                       ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   ENSEMBLE: 5 Models Comparison (5-fold CV)            │    │
│  │  ┌─────────────────┬──────────────┬────────────────┐    │    │
│  │  │LogisticReg 63%  │RandomForest  │GradientBoost   │    │    │
│  │  │(Baseline)       │72%           │72.6% 🏆       │    │    │
│  │  └─────────────────┴──────────────┴────────────────┘    │    │
│  │  + XGBoost (71%) + SVM (57%)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                       ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Hyperparameter Tuning (Optuna 30 trials)              │    │
│  │  Best: max_depth=12, lr=0.15, n_estimators=100        │    │
│  │  Result: 72.63% F1, 0.8008 AUC (test set)             │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT & SERVING                            │
│  ┌──────────────────┬──────────────┬──────────────────────┐     │
│  │ MLflow Tracking  │ Streamlit    │ Docker Compose       │     │
│  │ (metrics/logs)   │ (4-page UI)  │ (containerized)      │     │
│  │ :5000            │ :8501        │ (volume-mounted)     │     │
│  └──────────────────┴──────────────┴──────────────────────┘     │
│                                                                   │
│  CI/CD Pipeline (GitHub Actions):                               │
│  Test → Lint → Security → Build → Deploy                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results Summary

### Model Comparison (5-Fold Cross-Validation)

| Model | Accuracy | F1-Score | Precision | Recall | AUC-ROC | Train Time |
|-------|----------|----------|-----------|--------|---------|-----------|
| **Logistic Regression** (Baseline) | 63% | 63.2% | 65% | 61% | 0.712 | 2 sec |
| Random Forest Classifier | 71% | 71.8% | 72% | 71% | 0.784 | 15 sec |
| **Gradient Boosting** 🏆 | **72%** | **72.34%** | **73%** | **71%** | **0.801** | 45 min |
| XGBoost Classifier | 70% | 71.0% | 71% | 70% | 0.778 | 30 min |
| Support Vector Machine | 56% | 57.0% | 58% | 56% | 0.701 | 90 min |

### Test Set Performance (Best Model)
- **F1-Score:** 72.63% (↑ 15% vs baseline)
- **AUC-ROC:** 0.8008 (↑ 12% vs baseline)
- **Precision:** 73.5% (fewer false positives)
- **Recall:** 71.8% (catches most relevant tracks)

### Improvement Over Baseline
| Metric | Baseline | Winner | Improvement |
|--------|----------|--------|------------|
| F1-Score | 63.2% | 72.63% | **+15.3%** |
| AUC-ROC | 0.712 | 0.8008 | **+12.5%** |
| Precision | 65% | 73.5% | **+13.1%** |

---

## 🛠️ Tech Stack

| Component | Tool | Version | Purpose |
|-----------|------|---------|---------|
| **Language** | Python | 3.9+ | Data science & ML |
| **Data Processing** | pandas | 1.3.0+ | DataFrames & manipulation |
| **Numerical Computing** | NumPy | 1.21.0+ | Array operations |
| **ML Library** | scikit-learn | 0.24.0+ | Classification & metrics |
| **Hyperparameter Tuning** | Optuna | 2.10.0+ | 30-trial Bayesian optimization |
| **Experiment Tracking** | MLflow | 1.20.0+ | Metrics, params, model logging |
| **Web Dashboard** | Streamlit | 1.0.0+ | Interactive 4-page portfolio UI |
| **Visualization** | Plotly | 5.0.0+ | Interactive charts |
| **Visualization** | Matplotlib | 3.4.0+ | Static plots |
| **Visualization** | Seaborn | 0.11.0+ | Statistical plots |
| **Testing** | pytest | 6.2.0+ | 50+ unit tests |
| **Code Quality** | flake8 | 4.0.0+ | PEP 8 linting |
| **Code Quality** | black | 22.0.0+ | Code formatting |
| **Code Quality** | isort | 5.10.0+ | Import sorting |
| **Security** | bandit | 1.7.0+ | Vulnerability scanning |
| **Containerization** | Docker | latest | Image building |
| **Orchestration** | docker-compose | latest | Multi-container deployment |
| **CI/CD** | GitHub Actions | - | Automated testing & builds |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip or conda
- Git
- (Optional) Docker & docker-compose
- (Optional) MLflow

### 1. Clone Repository
```bash
git clone https://github.com/purnankgogarkar/mymlproject.git
cd mymlproject/spotify-recsys
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n spotify-recsys python=3.9
conda activate spotify-recsys
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data
```bash
# Place your CSV file in data/raw/
# Expected: spotify_data.csv with 89,740 tracks
```

### 5. Verify Installation
```bash
python -m pytest tests/ -v  # Should see 50+ tests pass
```

---

## ▶️ How to Run

### Option 1: Full Data Pipeline
```bash
# Step 1: Load & validate data
python -m src.data.run_data_pipeline

# Step 2: Engineer features
python -m src.features.run_features

# Step 3: Compare models (5 models, 5-fold CV)
python -m src.models.compare_models

# Step 4: Hyperparameter tuning (Optuna 30 trials, ~50 min)
python -m src.models.tuning

# Step 5: Train & save production model (with MLflow logging)
python -m src.models.run_training
```

### Option 2: Launch Streamlit Dashboard
```bash
# Interactive portfolio UI with 4 pages
python -m streamlit run app/streamlit_app.py

# Access: http://localhost:8501
```

### Option 3: Run MLflow Tracking Server
```bash
# View all experiment runs, metrics, and models
python -m mlflow ui --host 127.0.0.1 --port 5000

# Access: http://localhost:5000
```

### Option 4: Docker (Recommended for Production)
```bash
# Build image
docker build -t spotify-recsys:latest .

# Run with docker-compose (starts Streamlit + MLflow)
docker-compose up -d

# Access:
#   Streamlit: http://localhost:8501
#   MLflow: http://localhost:5000
#   Logs: docker-compose logs -f
```

### Option 5: Run Tests
```bash
# All tests (50+ cases)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov=app

# Specific test file
python -m pytest tests/test_model.py -v
```

### Option 6: Linting & Code Quality
```bash
# Check code style
flake8 src/ app/

# Auto-format code
black src/ app/

# Sort imports
isort src/ app/

# Security scan
bandit -r src/ app/
```

---

## 🎨 Feature Engineering

### Engineered Features (12 total)

#### Domain Features (Based on Audio Domain Knowledge)
| Feature | Formula | Range | Rationale |
|---------|---------|-------|-----------|
| `vibe_uplifting` | (energy × valence × 0.5) / 100 | [0, 1] | Combines energy & happiness for uplifting vibe |
| `dance_rhythm_match` | (danceability × tempo) / 200 | [0, 1] | Danceability × tempo predicts dance potential |
| `electric_index` | (1 - acousticness) × energy | [0, 1] | Electronic/synthesized sound indicator |
| `instrumental_complexity` | instrumentalness × (1 - speechiness) | [0, 1] | Pure instrumental sophistication |
| `vocal_intensity` | (1 - instrumentalness) × energy | [0, 1] | Vocal-forward presence strength |
| `loudness_energy_consistency` | 1 - abs(loudness_z - energy_z) | [0, 1] | Are loudness & energy aligned? |

#### Statistical Features (Computed from Audio Properties)
| Feature | Computation | Range | Rationale |
|---------|-------------|-------|-----------|
| `feature_variance` | StdDev of [energy, tempo, …] | [0, 1] | Musical diversity within track |
| `loudness_zscore` | (loudness - mean) / std | [-3, 3] | Loudness relative to dataset norm |
| `tempo_percentile` | rank(tempo) / count | [0, 1] | Track tempo rarity (slow vs fast) |

#### Interaction Features (Cross-Feature Relationships)
| Feature | Formula | Range | Rationale |
|---------|---------|-------|-----------|
| `chill_index` | (acousticness × valence) × (1 - energy) | [0, 1] | Relaxing/chill vibe predictor |
| `party_potential` | (energy × danceability × tempo) / 10000 | [0, 1] | Probability track fits party playlist |
| `silence_depth` | (1 - speechiness) × (1 - liveness) | [0, 1] | Studio recording cleanness/isolation |

### Feature Selection
- **Original features:** 21 (from Spotify API)
- **Engineered features:** +12 = 33 total
- **Feature selection:** Disabled (aggressive thresholds dropped too much signal)
- **Result:** All 33 features retained to preserve model signal

---

## 💡 Key Decisions & Lessons

### 1. **GradientBoosting over Deep Learning** ✓
- **Decision:** Use scikit-learn's GradientBoostingClassifier instead of neural network
- **Rationale:** 
  - Dataset size (89K) too small for deep learning efficiency
  - Tree-based models interpretable (critical for recommendations)
  - 72.6% F1 achieved with 50 min training vs 2+ hours for deep nets
- **Lesson:** Not every problem needs deep learning; domain-appropriate models win

### 2. **Disabled Aggressive Feature Selection** ✗ → ✓
- **Mistake:** Initial thresholds (correlation_threshold=0.95, variance_threshold=0.01) dropped 95% of features
- **Result:** Only 1 feature remained; model collapsed to 51% F1
- **Fix:** Disabled feature selection entirely
- **Lesson:** Validate selection logic before applying; aggressive filtering is rarely justified

### 3. **Memory Optimization for Similarity Matrix** 🔧
- **Problem:** 89K × 89K cosine similarity matrix requires 60GB+ memory
- **Solution:** Sample to 5,000 tracks for content-based filtering (~200MB achievable)
- **Trade-off:** Reduced recommendation scope but maintained matrix computability
- **Lesson:** Scale data pragmatically; not all models run on full datasets

### 4. **Hybrid Architecture > Single Model** 🎯
- **Decision:** Combine content-based + collaborative + ensemble classifier
- **Benefit:** 
  - Content-based handles new tracks
  - Collaborative catches user preferences
  - Classifier combines signals
- **Result:** More robust than any single approach

### 5. **Hyperparameter Tuning with Bayesian Optimization** ⚡
- **Decision:** Use Optuna instead of grid/random search
- **Benefit:** 30 trials found better params than 100 random trials in 50 min
- **Key params:** max_depth=12, learning_rate=0.15, n_estimators=100
- **Lesson:** Bayesian optimization scales better than brute force for continuous spaces

---

## 📁 File Structure

```
spotify-recsys/
├── README.md                          # Project documentation
├── CI_CD.md                           # GitHub Actions guide
├── TESTING.md                         # Test coverage guide
├── DOCKER.md                          # Docker deployment guide
├── requirements.txt                   # Python dependencies
├── pytest.ini                         # Test configuration
├── .flake8                            # Linting configuration
├── pyproject.toml                     # Black, isort, ruff settings
│
├── setup.py                           # Package installation
│
├── src/                               # Main source code
│   ├── __init__.py
│   ├── data/                          # Data pipeline
│   │   ├── __init__.py
│   │   ├── loader.py                  # Load CSV & analyze
│   │   ├── quality.py                 # 5-step validation gate
│   │   ├── cleaner.py                 # Data cleaning pipeline
│   │   └── run_data_pipeline.py       # Orchestrator
│   │
│   ├── features/                      # Feature engineering
│   │   ├── __init__.py
│   │   ├── engineering.py             # Create 12 engineered features
│   │   └── run_features.py            # Orchestrator
│   │
│   └── models/                        # ML models
│       ├── __init__.py
│       ├── trainer.py                 # Content & collaborative filtering
│       ├── baseline.py                # Baseline classifier
│       ├── compare_models.py          # 5-model comparison (5-fold CV)
│       ├── tuning.py                  # Optuna hyperparameter tuning
│       └── run_training.py            # MLflow-integrated training
│
├── app/                               # Web UI
│   ├── __init__.py
│   └── streamlit_app.py               # 4-page portfolio dashboard
│
├── tests/                             # Test suite (50+ tests)
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_data_quality.py           # Data validation tests
│   ├── test_features.py               # Feature engineering tests
│   └── test_model.py                  # Model prediction tests
│
├── data/                              # Data directories
│   ├── raw/                           # Original CSV files
│   └── processed/                     # Cleaned & engineered datasets
│
├── models/                            # Saved models
│   ├── baseline.pkl                   # LogisticRegression baseline
│   ├── tuned_model.pkl                # Optimized GradientBoosting
│   ├── production_model.pkl           # Production-ready model
│   ├── best_params.json               # Optuna best hyperparameters
│   └── model_comparison.pkl           # 5-model comparison results
│
├── results/                           # Analysis & metrics
│   ├── model_comparison.csv           # Model scores table
│   ├── tuning_metrics.json            # Optuna trial metrics
│   ├── tuning_trials.csv              # All 30 trial results
│   └── production_metadata.json       # Production model metadata
│
├── notebooks/                         # Jupyter notebooks (optional)
│   └── analysis.ipynb                 # EDA & exploration
│
├── .github/                           # GitHub configuration
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI/CD pipeline
│
├── Dockerfile                         # Docker image definition
├── docker-compose.yml                 # Multi-container orchestration
│
└── mlruns/                            # MLflow experiment tracking
    └── [auto-generated MLflow runs]

```

### Key Directories Explained

- **`src/`** — Production code (data → features → models)
- **`app/`** — Streamlit web dashboard
- **`tests/`** — 50+ pytest test cases
- **`data/processed/`** — Cleaned & engineered datasets
- **`models/`** — Saved model files (.pkl) + metadata
- **`results/`** — Metrics, comparison results, tuning trials
- **`.github/workflows/`** — GitHub Actions CI/CD automation
- **`mlruns/`** — MLflow experiment tracking (auto-created)

---

## 📈 Performance Metrics

### Training Pipeline
- **Data Loading:** 2 seconds (89,740 tracks)
- **Data Cleaning:** 3 seconds
- **Feature Engineering:** 5 seconds
- **Baseline Training:** 2 seconds
- **Model Comparison:** 8 minutes (5 models × 5-fold CV)
- **Hyperparameter Tuning:** 50-60 minutes (30 Optuna trials)
- **Total End-to-End:** ~1 hour 15 minutes

### Model Inference
- **Single Track Prediction:** 5ms
- **Batch (1000 tracks):** 500ms
- **Streamlit Dashboard Load:** <1 second (cached)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Run tests: `python -m pytest tests/ -v`
5. Run linting: `black src/ app/ && isort src/ app/`
6. Push to branch: `git push origin feature/your-feature`
7. Open pull request

---

## 📝 License

MIT License — See LICENSE file for details

---

## 👤 Author

**Data Scientist & ML Engineer**

- 📧 Email: [purnank18@gmail.com](mailto:purnank18@gmail.com)
- 🐙 GitHub: [purnankgogarkar](https://github.com/purnankgogarkar)
- 📦 Project: [Spotify Recommendation Engine](https://github.com/purnankgogarkar/mymlproject)

---

## 🙏 Acknowledgments

- Spotify dataset from Kaggle
- scikit-learn & Optuna communities
- Streamlit for dashboard framework
- GitHub Actions for CI/CD automation

---

## 📞 Support

For issues, questions, or feedback:
1. Check [GitHub Issues](https://github.com/purnankgogarkar/mymlproject/issues)
2. Email: purnank18@gmail.com for direct contact
3. Reference relevant code sections

---

**Last Updated:** April 30, 2026  
**Status:** Production Ready ✅  
**Version:** 1.0.0

### Quick Docker commands
```bash
# Build image
docker build -t spotify-recsys:latest .

# Run with volume mounts
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  spotify-recsys:latest
```

## Next
1. Set up data pipeline
2. Gather Spotify dataset
3. Feature engineering
4. Model prototypes
5. Evaluation + blend
