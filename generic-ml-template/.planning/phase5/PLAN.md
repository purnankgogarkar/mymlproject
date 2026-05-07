# Phase 5: Streamlit Interactive Dashboard

**Duration:** 3 days (Days 9-11)  
**Goal:** Build interactive web UI for the entire ML pipeline with real-time visualizations  
**Estimated Tests:** 30+ tests  
**Tech Stack:** Streamlit, Plotly, Pandas, Session State Management  

---

## 📋 Phase Overview

### Problem Statement
- Current framework requires Python coding to use → not accessible to non-technical stakeholders
- Data scientists can't quickly iterate on different models/configs
- No visual feedback during training → hard to understand model behavior
- Results are scattered across terminal output, configs, MLflow → hard to explore

### Solution
Build a **multi-page Streamlit dashboard** with:
1. **Homepage** — Project overview and quick start
2. **Upload Data** — CSV/Excel upload with preview
3. **Explore Data** — Interactive data profiling and statistics
4. **Configure Model** — YAML config builder with visual editor
5. **Train Model** — Real-time training progress and metrics
6. **Results** — Model comparison, feature importance, predictions

---

## 🏗️ Architecture

```
app/
├── streamlit_app.py          # Main entry point
├── pages/
│   ├── 01_home.py            # Homepage with overview
│   ├── 02_upload_data.py     # Data upload & preview
│   ├── 03_explore_data.py    # Data profiling dashboard
│   ├── 04_configure_model.py # Model configuration builder
│   ├── 05_train_model.py     # Training monitor
│   └── 06_results.py         # Results visualization
├── utils/
│   ├── session_state.py      # Session state management
│   ├── visualizations.py     # Plotly charts
│   ├── config_builder.py     # UI for YAML config
│   ├── data_widgets.py       # Data upload/preview widgets
│   └── model_widgets.py      # Model training widgets
└── assets/
    ├── logo.png
    ├── styles.css
    └── config_templates.yaml

tests/
├── test_streamlit_app.py     # App structure tests
├── test_session_state.py     # Session state tests
├── test_visualizations.py    # Chart generation tests
└── test_config_builder.py    # Config builder tests
```

---

## 🎯 Detailed Component Specs

### 1. Main App (`app/streamlit_app.py`)

**Purpose:** App entry point, navigation, session state initialization

**Features:**
```python
import streamlit as st
from streamlit_multipage import MultiPage

# App configuration
st.set_page_config(
    page_title="Generic ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None
```

**UI Layout:**
- Top: Logo + title + version badge
- Sidebar:
  - Navigation menu (6 pages)
  - Current state indicator (data loaded? model trained?)
  - Settings panel (theme, advanced options)
  - Export options (config, results, model)
- Main: Page content
- Footer: Version, github link, documentation link

**Test Coverage:**
- App initialization
- Session state setup
- Page navigation
- State transitions

---

### 2. Home Page (`app/pages/01_home.py`)

**Purpose:** Project overview, feature showcase, quick start guide

**Content:**
```markdown
# 🤖 Generic ML Dashboard

Welcome to your AI-powered ML pipeline builder!

## What You Can Do

### 1. 📤 Upload Your Data
Load CSV or Excel files with automatic type detection

### 2. 📊 Explore Data
Interactive statistics, correlations, and distributions

### 3. ⚙️ Configure Model
Visual config builder (no coding required)

### 4. 🚀 Train Model
Real-time progress monitoring with metrics

### 5. 📈 Compare Results
Side-by-side model comparison and visualization

## Quick Start
1. Click "Upload Data" in sidebar
2. Select your CSV/Excel file
3. Configure model in "Configure Model" page
4. Click "Train Model" and watch progress
5. View results in "Results" page

## Features
- ✅ Auto data profiling
- ✅ Multiple model support (20+ algorithms)
- ✅ Cross-validation with metrics
- ✅ Feature importance visualization
- ✅ Model comparison
- ✅ Experiment tracking (MLflow)
- ✅ Hyperparameter tuning (Optuna)
```

**UI Elements:**
- Welcome banner with logo
- Feature cards (4 columns, 3 rows)
- Quick start steps with icons
- Links to docs and examples
- Recent experiments (if available)

**Test Coverage:**
- Page rendering
- Content display
- Link functionality

---

### 3. Upload Data Page (`app/pages/02_upload_data.py`)

**Purpose:** File upload with preview and validation

**Features:**

```python
st.title("📤 Upload Data")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Supported formats: CSV, Excel"
)

if uploaded_file:
    # Load with DataLoader
    loader = DataLoader(uploaded_file)
    df = loader.load()
    
    # Store in session
    st.session_state.data = df
    
    # Display preview
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Display profile
    with st.expander("📊 Data Profile"):
        profile = loader.profile()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", profile['n_rows'])
        with col2:
            st.metric("Columns", profile['n_cols'])
        with col3:
            st.metric("Missing %", f"{profile['missing_percent']:.1f}%")
        with col4:
            st.metric("Duplicates", profile['n_duplicates'])
    
    # Display data types
    with st.expander("🔍 Data Types"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numeric Columns**")
            for col in profile['numeric_cols']:
                st.write(f"  • {col}")
        with col2:
            st.write("**Categorical Columns**")
            for col in profile['categorical_cols']:
                st.write(f"  • {col}")
    
    # Column selection for target
    st.subheader("🎯 Select Target Column")
    target_col = st.selectbox(
        "Which column do you want to predict?",
        options=df.columns,
        help="This will be your model's target variable"
    )
    st.session_state.target_col = target_col
```

**UI Components:**
- File uploader widget
- Data preview table (first 10 rows, scrollable)
- Data profile metrics (rows, columns, missing %, duplicates)
- Data type display (numeric vs categorical)
- Target column selector

**Test Coverage:**
- File upload
- Data loading
- Profile calculation
- Preview rendering
- Session state updates

---

### 4. Explore Data Page (`app/pages/03_explore_data.py`)

**Purpose:** Interactive data profiling and visualization

**Features:**

```python
st.title("📊 Explore Data")

if st.session_state.data is None:
    st.warning("⚠️ Please upload data first in 'Upload Data' page")
    st.stop()

df = st.session_state.data

# Create explorer
explorer = DataExplorer(df, target_col=st.session_state.target_col)
analysis = explorer.analyze()

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Distributions",
    "🔗 Correlations",
    "📉 Missing Data",
    "⚡ Statistics",
    "💡 Recommendations"
])

with tab1:
    # Distribution plots
    cols = st.multiselect(
        "Select columns to visualize",
        options=df.columns,
        default=df.columns[:5]
    )
    
    for col in cols:
        fig = create_distribution_plot(df, col)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Correlation heatmap
    fig = create_correlation_heatmap(df[df.select_dtypes(include='number').columns])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Missing data visualization
    fig = create_missing_data_plot(df)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Statistics table
    stats = df.describe().T
    st.dataframe(stats, use_container_width=True)

with tab5:
    # Model recommendations
    st.subheader("Recommended Models")
    recommendations = explorer.recommend_models()
    for rec in recommendations:
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.write(f"**{rec['model']}**")
        with col2:
            st.write(f"*{rec['score']:.2f}*")
        with col3:
            st.write(rec['reason'])
```

**Visualizations:**
- Distribution plots (histogram + KDE)
- Correlation heatmap
- Missing data chart (% missing per column)
- Box plots (outlier detection)
- Scatter plots (numeric relationships)
- Categorical frequency charts

**Test Coverage:**
- Data validation
- Chart generation
- Tab rendering
- Recommendation display

---

### 5. Configure Model Page (`app/pages/04_configure_model.py`)

**Purpose:** Visual config builder for ML pipeline

**Features:**

```python
st.title("⚙️ Configure Model")

if st.session_state.data is None:
    st.warning("⚠️ Please upload data first")
    st.stop()

# Tabs for different config sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Data",
    "🔧 Preprocessing",
    "✨ Features",
    "🤖 Model",
    "📊 Evaluation"
])

with tab1:
    st.subheader("Data Configuration")
    
    test_size = st.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Fraction of data to use for testing"
    )
    
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        help="For reproducibility"
    )
    
    cv_folds = st.slider(
        "Cross-validation folds",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of CV folds for training"
    )

with tab2:
    st.subheader("Preprocessing Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_strategy = st.selectbox(
            "Missing value strategy",
            options=['mean', 'median', 'mode', 'forward_fill', 'drop', 'auto'],
            help="How to handle NaN values"
        )
    
    with col2:
        encoding = st.selectbox(
            "Categorical encoding",
            options=['one-hot', 'label', 'auto'],
            help="How to encode categorical variables"
        )
    
    with col3:
        scaling = st.selectbox(
            "Feature scaling",
            options=['standard', 'minmax', 'robust', 'none'],
            help="How to scale numeric features"
        )

with tab3:
    st.subheader("Feature Engineering Configuration")
    
    auto_generate = st.checkbox(
        "Auto-generate features",
        value=True,
        help="Automatically create new features"
    )
    
    if auto_generate:
        transformations = st.multiselect(
            "Feature transformations",
            options=['log', 'sqrt', 'square', 'cube', 'reciprocal', 'exp'],
            default=['log', 'sqrt', 'square'],
            help="Math transforms to apply"
        )
        
        interaction_features = st.checkbox(
            "Generate interaction features",
            value=True
        )
        
        polynomial_degree = st.slider(
            "Polynomial degree",
            min_value=1,
            max_value=3,
            value=2,
            help="Degree for polynomial features"
        )

with tab4:
    st.subheader("Model Configuration")
    
    problem_type = st.radio(
        "Problem type",
        options=['classification', 'regression'],
        help="Auto-detected from data"
    )
    
    model_name = st.selectbox(
        "Model selection",
        options=get_available_models(problem_type),
        help="Choose a model"
    )
    
    st.write("**Hyperparameters**")
    default_params = get_model_defaults(problem_type, model_name)
    
    params = {}
    for param_name, param_value in default_params.items():
        if isinstance(param_value, int):
            params[param_name] = st.number_input(
                param_name,
                value=param_value,
                step=1
            )
        elif isinstance(param_value, float):
            params[param_name] = st.number_input(
                param_name,
                value=param_value,
                step=0.01,
                format="%.3f"
            )

with tab5:
    st.subheader("Evaluation Configuration")
    
    cv_folds = st.slider(
        "CV folds for evaluation",
        min_value=2,
        max_value=10,
        value=5
    )
    
    st.write("**Metrics**")
    if problem_type == 'classification':
        metrics = st.multiselect(
            "Select metrics",
            options=['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
            default=['accuracy', 'f1', 'auc_roc']
        )
    else:
        metrics = st.multiselect(
            "Select metrics",
            options=['rmse', 'mae', 'r2', 'mape'],
            default=['rmse', 'mae', 'r2']
        )

# Save config button
if st.button("💾 Save Configuration"):
    config = build_config_dict(
        test_size, random_state, cv_folds,
        missing_strategy, encoding, scaling,
        auto_generate, transformations, interaction_features, polynomial_degree,
        problem_type, model_name, params,
        metrics
    )
    st.session_state.config = config
    st.success("✅ Configuration saved!")
    st.balloons()

# Load example config
with st.expander("📂 Load Example Config"):
    example = st.selectbox(
        "Choose example",
        options=['iris_classification', 'housing_regression', 'titanic']
    )
    if st.button("Load Example"):
        config = load_example_config(example)
        st.session_state.config = config
        st.success(f"✅ Loaded {example}")
```

**UI Components:**
- 5 tabs (Data, Preprocessing, Features, Model, Evaluation)
- Sliders for numeric ranges
- Dropdowns for categorical choices
- Checkboxes for toggles
- Multi-select for feature lists
- Number inputs for parameters
- Save/Load buttons
- Example config templates

**Test Coverage:**
- Config building
- Parameter validation
- UI rendering
- Config serialization

---

### 6. Train Model Page (`app/pages/05_train_model.py`)

**Purpose:** Model training with real-time progress monitoring

**Features:**

```python
st.title("🚀 Train Model")

if st.session_state.config is None:
    st.warning("⚠️ Please configure model first")
    st.stop()

# Model info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Problem Type", st.session_state.config['model']['type'])
with col2:
    st.metric("Model", st.session_state.config['model']['name'])
with col3:
    st.metric("CV Folds", st.session_state.config['evaluation']['cv_folds'])

# Advanced options
with st.expander("⚡ Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        use_mlflow = st.checkbox("Track with MLflow", value=False)
        experiment_name = st.text_input(
            "Experiment name",
            value="default_experiment"
        ) if use_mlflow else None
    
    with col2:
        use_optuna = st.checkbox("Tune hyperparameters with Optuna", value=False)
        n_trials = st.number_input(
            "Number of trials",
            min_value=10,
            value=50
        ) if use_optuna else None

# Training button
if st.button("▶️ Start Training", key="train_button"):
    # Progress bar and real-time metrics
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    
    try:
        # Initialize trainer
        X = st.session_state.data.drop(columns=[st.session_state.target_col])
        y = st.session_state.data[st.session_state.target_col]
        
        trainer = GenericTrainer(
            X, y,
            problem_type=st.session_state.config['model']['type'],
            cv_folds=st.session_state.config['evaluation']['cv_folds']
        )
        
        # Train
        status_text.text("🔄 Training model...")
        progress_bar.progress(30)
        
        trainer.train(
            model_name=st.session_state.config['model']['name'],
            **st.session_state.config['model']['hyperparams']
        )
        
        progress_bar.progress(60)
        status_text.text("📊 Computing metrics...")
        
        # Evaluate
        evaluator = Evaluator(
            y_true=y,
            y_pred=trainer.predict(X),
            y_pred_proba=trainer.predict_proba(X) if 'classification' in st.session_state.config['model']['type'] else None,
            problem_type=st.session_state.config['model']['type']
        )
        
        evaluator.evaluate()
        
        progress_bar.progress(90)
        status_text.text("✨ Finalizing...")
        
        # Store results
        st.session_state.model = trainer
        st.session_state.results = {
            'evaluator': evaluator,
            'trainer': trainer,
            'metrics': evaluator.get_metrics()
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Training complete!")
        
        # Display metrics
        metrics = evaluator.get_metrics()
        col1, col2, col3, col4 = st.columns(4)
        for i, (metric_name, metric_value) in enumerate(list(metrics.items())[:4]):
            with [col1, col2, col3, col4][i]:
                st.metric(metric_name.upper(), f"{metric_value:.4f}")
        
        st.success("✅ Model trained successfully!")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        status_text.text(f"Error: {str(e)}")
```

**UI Components:**
- Model info metrics
- Advanced options (MLflow, Optuna)
- Training progress bar
- Real-time status updates
- Metrics display
- Success/error messages

**Test Coverage:**
- Training pipeline
- Error handling
- Progress tracking
- Results storage

---

### 7. Results Page (`app/pages/06_results.py`)

**Purpose:** Model results visualization and comparison

**Features:**

```python
st.title("📈 Results")

if st.session_state.results is None:
    st.warning("⚠️ Please train a model first")
    st.stop()

evaluator = st.session_state.results['evaluator']
trainer = st.session_state.results['trainer']
metrics = st.session_state.results['metrics']

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Metrics",
    "🔍 Feature Importance",
    "📉 Predictions",
    "💾 Export"
])

with tab1:
    st.subheader("Model Performance Metrics")
    
    # Metrics table
    metrics_df = pd.DataFrame([metrics])
    st.dataframe(metrics_df.T, use_container_width=True)
    
    # Classification-specific metrics
    if 'confusion_matrix' in metrics:
        st.subheader("Confusion Matrix")
        fig = create_confusion_matrix_plot(metrics['confusion_matrix'])
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC curve for classification
    if 'fpr' in metrics:
        st.subheader("ROC Curve")
        fig = create_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc_roc'])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Feature Importance")
    
    # Get feature importance
    try:
        importance = trainer.get_feature_importance()
        
        # Plot
        fig = create_feature_importance_plot(importance)
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(importance.sort_values(ascending=False), use_container_width=True)
    except:
        st.info("ℹ️ Feature importance not available for this model")

with tab3:
    st.subheader("Prediction Analysis")
    
    # Predictions dataframe
    X_sample = st.session_state.data.drop(columns=[st.session_state.target_col]).head(100)
    y_pred = trainer.predict(X_sample)
    
    pred_df = X_sample.copy()
    pred_df['prediction'] = y_pred
    pred_df['actual'] = st.session_state.data[st.session_state.target_col].head(100).values
    
    st.dataframe(pred_df, use_container_width=True)

with tab4:
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export metrics as CSV
        metrics_csv = metrics_df.T.to_csv()
        st.download_button(
            "📥 Download Metrics (CSV)",
            data=metrics_csv,
            file_name="metrics.csv"
        )
    
    with col2:
        # Export predictions as CSV
        pred_csv = pred_df.to_csv()
        st.download_button(
            "📥 Download Predictions (CSV)",
            data=pred_csv,
            file_name="predictions.csv"
        )
    
    with col3:
        # Export config as YAML
        import yaml
        config_yaml = yaml.dump(st.session_state.config)
        st.download_button(
            "📥 Download Config (YAML)",
            data=config_yaml,
            file_name="config.yaml"
        )
```

**Visualizations:**
- Metrics table
- Confusion matrix (classification)
- ROC curve (classification)
- Feature importance bar chart
- Predictions table
- Export buttons (CSV, YAML)

**Test Coverage:**
- Results display
- Visualization rendering
- Export functionality

---

## 📁 Session State Management (`app/utils/session_state.py`)

**Purpose:** Centralized session state management

```python
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class AppState:
    """Central app state management"""
    
    # Data state
    data: Optional[Any] = None
    target_col: Optional[str] = None
    
    # Configuration state
    config: Optional[Dict[str, Any]] = None
    
    # Model state
    model: Optional[Any] = None
    trainer: Optional[Any] = None
    
    # Results state
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    
    # UI state
    current_page: str = "home"
    show_advanced: bool = False
    
    def initialize(self):
        """Initialize session state in Streamlit"""
        for key, value in self.__dict__.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def reset(self):
        """Reset all state"""
        for key in self.__dict__.keys():
            st.session_state[key] = None
    
    def reset_results(self):
        """Reset results only"""
        st.session_state.results = None
        st.session_state.metrics = None

def get_state() -> AppState:
    """Get or create app state"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
        st.session_state.app_state.initialize()
    return st.session_state.app_state
```

**Test Coverage:**
- State initialization
- State updates
- State reset

---

## 🎨 Visualizations (`app/utils/visualizations.py`)

**Purpose:** Reusable Plotly charts

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_distribution_plot(df, column):
    """Create distribution plot"""
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Distribution of {column}",
        labels={column: column}
    )
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    corr = df.corr()
    fig = go.Figure(
        data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns)
    )
    fig.update_layout(title="Feature Correlation Matrix")
    return fig

def create_feature_importance_plot(importance_series):
    """Create feature importance bar chart"""
    fig = px.bar(
        y=importance_series.index,
        x=importance_series.values,
        orientation='h',
        title="Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    return fig

def create_confusion_matrix_plot(cm):
    """Create confusion matrix heatmap"""
    fig = go.Figure(
        data=go.Heatmap(z=cm, text=cm, texttemplate="%{text}")
    )
    fig.update_layout(title="Confusion Matrix")
    return fig

def create_roc_curve(fpr, tpr, auc_score):
    """Create ROC curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC={auc_score:.3f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash')
    ))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    return fig
```

**Test Coverage:**
- Chart generation
- Data validation
- Plotly output

---

## 🧪 Test Strategy

### Unit Tests (15 tests)
- `test_streamlit_app.py` — App initialization, navigation
- `test_session_state.py` — State management
- `test_config_builder.py` — Config building

### Integration Tests (10 tests)
- `test_data_upload_flow.py` — Upload → Preview → Profile
- `test_training_flow.py` — Config → Train → Results
- `test_export_flow.py` — Results → Export

### UI Tests (5+ tests)
- Page rendering
- Widget interactions
- Error handling

**Total: 30+ tests**

---

## 🚀 Implementation Order

### Day 1: Core Infrastructure
1. Create app structure
2. Implement session state management
3. Build main app entry point
4. Create home page
5. **Tests: 5**

### Day 2: Data Handling & Exploration
1. Build upload data page
2. Build explore data page
3. Implement visualization utilities
4. Create data widgets
5. **Tests: 10**

### Day 3: Model Configuration & Training
1. Build configure model page
2. Build train model page
3. Build results page
4. Implement export functionality
5. **Tests: 15+**

---

## 📋 Definition of Done

- ✅ All 6 pages implemented and functional
- ✅ Session state management working
- ✅ All visualizations rendering correctly
- ✅ File upload and download working
- ✅ Training pipeline integrated
- ✅ Results exported to CSV/YAML
- ✅ 30+ tests passing
- ✅ Error handling and validation complete
- ✅ Documentation and comments added
- ✅ Responsive design (mobile-friendly)
- ✅ README updated with Phase 5 info

---

## 📊 Success Criteria

### Functionality
- [x] User can upload CSV/Excel files
- [x] Data is displayed with preview and statistics
- [x] All data exploration visualizations work
- [x] Config builder creates valid YAML configs
- [x] Model trains successfully with progress tracking
- [x] Results display all relevant metrics
- [x] Export to CSV/YAML works

### Performance
- [x] App loads in < 3 seconds
- [x] Data upload/preview in < 2 seconds
- [x] Training progress updates every 1 second

### Quality
- [x] 30+ tests passing
- [x] All error cases handled gracefully
- [x] Clear error messages for users
- [x] Responsive design works on mobile

---

## 💾 Deliverables

- ✅ Complete Streamlit app with 6 pages
- ✅ Session state management
- ✅ Plotly visualization utilities
- ✅ Config builder with validation
- ✅ 30+ tests with high coverage
- ✅ Updated README with UI screenshots
- ✅ Example configs for quick start
- ✅ API integration (use existing modules)

---

## 🔗 Dependencies on Previous Phases

**Phase 1-4 Required:**
- DataLoader & DataExplorer (Phase 1)
- Preprocessor & FeatureEngineer (Phase 2)
- GenericTrainer & Evaluator (Phase 3)
- ConfigLoader & MLflowTracker (Phase 4)

**No new dependencies** — Pure UI layer on top of existing modules!

---

## 📝 Notes

- Streamlit handles page routing automatically (pages/ folder)
- Session state persists between reruns
- Use `st.cache_data` for expensive operations
- All paths are relative (data/, configs/, mlruns/)
- Mobile-responsive CSS for Streamlit
