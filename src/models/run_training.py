"""
MLflow-integrated training pipeline for Spotify Recommendation Engine.

Trains baseline and tuned models, logs all metrics/params to MLflow,
and saves the production model.

Usage:
    python src/models/run_training.py
    
    Then start MLflow UI:
    mlflow ui --host 127.0.0.1 --port 5000
    
    Visit http://localhost:5000 to see all runs
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    print("❌ MLflow not installed. Install with: pip install mlflow")
    exit(1)


def load_and_prepare_data(filepath="data/processed/spotify_features.csv", test_size=0.2, random_state=42):
    """Load and prepare data for training."""
    print(f"📥 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find target (highest variance)
    variances = df[numeric_cols].var().sort_values(ascending=False)
    target_col = variances.index[0]
    print(f"✓ Target: {target_col}")
    
    # Binarize for classification
    y = (df[target_col] > df[target_col].median()).astype(int)
    print(f"✓ Binary classification: {target_col} > median")
    print(f"  Class distribution: {(y == 0).sum():,} / {(y == 1).sum():,}")
    
    # Get features (exclude target and metadata)
    feature_cols = [col for col in numeric_cols 
                   if col not in {target_col, 'track_id', 'track_name'}]
    X = df[feature_cols].copy()
    
    print(f"✓ Features: {X.shape[1]} numeric features")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✓ Train/Test: {X_train.shape[0]:,} / {X_test.shape[0]:,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def compute_metrics(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Compute train and test metrics."""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities (for AUC-ROC)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        has_proba = False
    
    # Build metrics dict
    metrics = {
        # Train metrics
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        
        # Test metrics
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    # Add AUC-ROC if available
    if has_proba:
        metrics['train_auc'] = roc_auc_score(y_train, y_train_proba)
        metrics['test_auc'] = roc_auc_score(y_test, y_test_proba)
    
    return metrics


def train_baseline_model(X_train, X_test, y_train, y_test):
    """Train baseline LogisticRegression model and log to MLflow."""
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL (LogisticRegression)")
    print("="*70)
    
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        # Define params
        params = {
            'model_type': 'LogisticRegression',
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs',
        }
        
        # Log params
        mlflow.log_params(params)
        
        # Train model
        model = LogisticRegression(max_iter=params['max_iter'], 
                                   random_state=params['random_state'],
                                   solver=params['solver'])
        model.fit(X_train, y_train)
        print("✓ Model trained")
        
        # Compute metrics
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test, "Baseline")
        mlflow.log_metrics(metrics)
        
        # Print metrics
        print(f"\n📊 Baseline Model Metrics:")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"   Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"   Train F1:       {metrics['train_f1']:.4f}")
        print(f"   Test F1:        {metrics['test_f1']:.4f}")
        if 'test_auc' in metrics:
            print(f"   Test AUC-ROC:   {metrics['test_auc']:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "baseline_model")
        
        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/baseline_logged.pkl")
        print("✓ Baseline model logged to MLflow and saved")
        
        return model, metrics


def train_tuned_model(X_train, X_test, y_train, y_test):
    """Train tuned GradientBoosting model and log to MLflow."""
    print("\n" + "="*70)
    print("TRAINING TUNED MODEL (GradientBoosting)")
    print("="*70)
    
    # Load best hyperparameters
    params_file = "models/best_params.json"
    if not os.path.exists(params_file):
        print(f"❌ {params_file} not found. Run src/models/tuning.py first.")
        return None, None
    
    with open(params_file, 'r') as f:
        best_params = json.load(f)
    
    print(f"✓ Loaded best parameters from {params_file}")
    print(f"  Best params: {best_params}")
    
    with mlflow.start_run(run_name="tuned_gradient_boosting"):
        # Prepare params for logging (convert types)
        log_params = {f"gb_{k}": str(v) for k, v in best_params.items()}
        mlflow.log_params(log_params)
        
        # Train model with best params
        model = GradientBoostingClassifier(**best_params)
        model.fit(X_train, y_train)
        print("✓ Model trained with best hyperparameters")
        
        # Compute metrics
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test, "Tuned")
        mlflow.log_metrics(metrics)
        
        # Print metrics
        print(f"\n📊 Tuned Model Metrics:")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"   Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"   Train F1:       {metrics['train_f1']:.4f}")
        print(f"   Test F1:        {metrics['test_f1']:.4f}")
        if 'test_auc' in metrics:
            print(f"   Test AUC-ROC:   {metrics['test_auc']:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "tuned_model")
        
        # Save locally
        joblib.dump(model, "models/tuned_logged.pkl")
        print("✓ Tuned model logged to MLflow and saved")
        
        return model, metrics


def save_production_model(baseline_model, tuned_model, baseline_metrics, tuned_metrics):
    """Select best model and save as production model."""
    print("\n" + "="*70)
    print("SELECTING PRODUCTION MODEL")
    print("="*70)
    
    # Compare test F1 scores
    baseline_f1 = baseline_metrics['test_f1']
    tuned_f1 = tuned_metrics['test_f1']
    
    print(f"Baseline F1:  {baseline_f1:.4f}")
    print(f"Tuned F1:     {tuned_f1:.4f}")
    
    if tuned_f1 > baseline_f1:
        production_model = tuned_model
        best_name = "Tuned GradientBoosting"
        improvement = ((tuned_f1 - baseline_f1) / baseline_f1) * 100
        print(f"\n✅ Selected: {best_name} (+{improvement:.2f}% improvement)")
    else:
        production_model = baseline_model
        best_name = "Baseline LogisticRegression"
        print(f"\n✅ Selected: {best_name}")
    
    # Save production model
    prod_path = "models/production_model.pkl"
    joblib.dump(production_model, prod_path)
    print(f"💾 Production model saved to {prod_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_name,
        'baseline_f1': baseline_f1,
        'tuned_f1': tuned_f1,
        'selected_f1': tuned_f1 if tuned_f1 > baseline_f1 else baseline_f1,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open("models/production_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("💾 Metadata saved to models/production_metadata.json")
    
    return production_model


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("SPOTIFY RECOMMENDATION ENGINE - MLflow Training Pipeline")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Set MLflow tracking URI (defaults to ./mlruns locally)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = "spotify_recsys"
    
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass  # Experiment already exists
    
    mlflow.set_experiment(experiment_name)
    print(f"✓ MLflow experiment: {experiment_name}")
    print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Load data
    print("\n" + "-"*70)
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    # Train baseline
    print("\n" + "-"*70)
    baseline_model, baseline_metrics = train_baseline_model(X_train, X_test, y_train, y_test)
    
    # Train tuned
    print("\n" + "-"*70)
    tuned_model, tuned_metrics = train_tuned_model(X_train, X_test, y_train, y_test)
    
    if tuned_model is None:
        print("⚠️  Tuned model training skipped. Using baseline as production model.")
        production_model = baseline_model
        os.makedirs("models", exist_ok=True)
        joblib.dump(production_model, "models/production_model.pkl")
    else:
        # Select and save production model
        print("\n" + "-"*70)
        production_model = save_production_model(baseline_model, tuned_model, 
                                                 baseline_metrics, tuned_metrics)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("✅ Next steps:")
    print("   1. Start MLflow UI:")
    print("      mlflow ui --host 127.0.0.1 --port 5000")
    print("   2. Open http://localhost:5000 in your browser")
    print("   3. View all runs, metrics, and artifacts")
    print("\n📁 Models saved:")
    print("   - models/production_model.pkl (selected for deployment)")
    print("   - models/baseline_logged.pkl (baseline reference)")
    print("   - models/tuned_logged.pkl (tuned reference)")
    print("\n🎯 Production model ready for deployment!")


if __name__ == "__main__":
    main()
