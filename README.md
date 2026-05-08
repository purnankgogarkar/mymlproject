# MyMLProject - Generic ML Template Framework

A production-ready, end-to-end machine learning project template with interactive Streamlit dashboard, automated preprocessing, 20+ model algorithms, and comprehensive testing.

## 🎯 Project Vision

Build a **reusable, enterprise-grade ML template** that handles the entire ML pipeline from raw data to production-ready model export, without requiring users to write complex code.

## 📁 Project Structure

```
mymlproject/
├── generic-ml-template/          # Main ML application (source code)
│   ├── app/                      # Streamlit web application
│   │   ├── pages/                # Multi-page dashboard (6 pages)
│   │   ├── utils/                # UI utilities and widgets
│   │   └── streamlit_app.py      # Main entry point
│   ├── src/                      # Core ML pipeline
│   │   ├── data/                 # Data loading & preprocessing
│   │   ├── models/               # Model training & evaluation
│   │   ├── features/             # Feature engineering
│   │   ├── config/               # Configuration management
│   │   └── export/               # Model export & equation extraction
│   ├── tests/                    # Comprehensive test suite (470+ tests)
│   ├── config/                   # Configuration templates
│   ├── .planning/                # Project roadmap & phase planning
│   ├── README.md                 # Detailed project documentation
│   ├── SKILLS.md                 # Technologies & capabilities per phase
│   └── ROADMAP.md                # Development phases & milestones
└── README.md                     # This file - project overview
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
cd generic-ml-template
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app/streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

## 📊 Complete ML Workflow (6 Pages)

| Step | Page | Purpose |
|------|------|---------|
| 1️⃣ | **Upload Data** | Load CSV/Excel with auto-detection & profiling |
| 2️⃣ | **Explore Data** | Interactive data analysis (distributions, correlations, missing values) |
| 3️⃣ | **Clean Data** | Missing values, encoding, scaling, outlier detection |
| 4️⃣ | **Configure Model** | Select algorithm, tune hyperparameters |
| 5️⃣ | **Train Model** | Real-time training with progress monitoring |
| 6️⃣ | **Results** | Metrics, equations, feature importance, model export |

## ✨ Key Features

### 🔄 Data Pipeline
- **Auto type detection** — Identifies numeric, categorical, datetime columns
- **Interactive profiling** — Statistical summaries and quality indicators
- **Data cleaning** — Missing values, encoding, scaling, outlier removal
- **Feature engineering** — Auto and custom transformations

### 🤖 Model Support (20+ Algorithms)
**Classification:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, Naïve Bayes, KNN, Neural Network, XGBoost, LightGBM

**Regression:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR, Neural Network, XGBoost, LightGBM, Polynomial Regression

### 📐 Equation Extraction
- **Regression equations** — Human-readable mathematical formulas for linear/tree models
- **Feature importance equations** — Normalized coefficients as contribution equations
- **Format:** Supports Linear, Ridge, Lasso, Decision Trees, Random Forests

### 📈 Results & Export
- **Comprehensive metrics** — Accuracy, F1, RMSE, AUC, Precision, Recall, Confusion Matrix, etc.
- **Cross-validation** — 5-fold CV with visualization
- **Model export** — `model.pkl`, `config.yaml`, `report.json`, equations
- **Real file downloads** — Timestamped exports ready for deployment

### 🧪 Testing
- **470+ unit tests** — Full test coverage for data, models, config, preprocessing
- **Pytest framework** — Fast, comprehensive test execution
- **Coverage reporting** — Pytest-cov integration

## 📊 Development Status

| Phase | Status | Features |
|-------|--------|----------|
| 1-4 | ✅ Complete | Data pipeline, preprocessing, model training, config system (244 tests) |
| 5 | ✅ Complete | Streamlit dashboard, data cleaning, equations, export (206 tests) |
| 6-8 | ⏳ Planned | Production Flask API, Docker, CI/CD, package distribution |

**Total Tests Passing:** 470/470 ✅

## 🛠️ Technologies Used

| Layer | Technologies |
|-------|--------------|
| **Data** | Pandas, NumPy |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Web UI** | Streamlit, Plotly |
| **Config** | PyYAML, Optuna (HPO), MLflow (tracking) |
| **Testing** | Pytest, Pytest-cov |
| **Export** | Pickle, JSON, YAML |

## 📚 Documentation

For detailed information:
- **[generic-ml-template/README.md](generic-ml-template/README.md)** — Complete technical documentation
- **[generic-ml-template/SKILLS.md](generic-ml-template/SKILLS.md)** — Detailed breakdown of technologies per phase
- **[generic-ml-template/.planning/](generic-ml-template/.planning/)** — Phase roadmaps and planning artifacts

## 🧪 Running Tests

```bash
cd generic-ml-template

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov=app tests/

# Run specific test file
pytest tests/test_data_loader.py -v
```

## 🔗 GitHub Repository

https://github.com/purnankgogarkar/mymlproject

## 📝 License

MIT License - See LICENSE file in generic-ml-template/

## 👤 Project Author

Created as a comprehensive machine learning framework for rapid prototyping and production deployment.

---

**Last Updated:** May 2026 | **Status:** Phase 5 Complete ✅
