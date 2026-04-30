"""
Hyperparameter Tuning with Optuna for Spotify Recommendation Engine.

Tunes the best model (GradientBoostingClassifier) using Optuna.
Runs 30 trials with 5-fold cross-validation to find optimal hyperparameters.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("❌ Optuna not installed. Install with: pip install optuna")
    exit(1)


def load_and_prepare_data(filepath="data/processed/spotify_features.csv", test_size=0.2, random_state=42):
    """
    Load and prepare data for tuning.
    
    Args:
        filepath: Path to features CSV
        test_size: Test set fraction
        random_state: Random seed
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
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


def objective(trial, X_train, y_train, cv_folds=5):
    """
    Optuna objective function for GradientBoosting hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
    
    Returns:
        float: Mean cross-validation F1 score
    """
    # Suggest hyperparameters (scikit-learn GradientBoostingClassifier)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 100, step=10),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42,
    }
    
    # Train model with cross-validation
    model = GradientBoostingClassifier(**params)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    
    return scores.mean()


def run_tuning(n_trials=30, cv_folds=5):
    """
    Run Optuna hyperparameter tuning.
    
    Args:
        n_trials: Number of tuning trials
        cv_folds: Number of cross-validation folds per trial
    
    Returns:
        tuple: (study, best_params, trial_history)
    """
    # Load data
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None, None, None, None
    
    print("\n" + "="*70)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Trials: {n_trials}")
    print(f"CV Folds: {cv_folds}")
    print(f"Optimization metric: F1 Score")
    
    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='gbc_tuning'
    )
    
    # Run tuning
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, cv_folds),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best params
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"\n🏆 Best Trial: #{study.best_trial.number}")
    print(f"   CV F1 Score: {best_score:.4f}")
    print(f"\n📊 Best Hyperparameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"   {param:25s}: {value:.6f}")
        else:
            print(f"   {param:25s}: {value}")
    
    return study, best_params, X_train, X_test, y_train, y_test


def train_final_model(best_params, X_train, y_train):
    """
    Train final model with best hyperparameters on full training set.
    
    Args:
        best_params: Best hyperparameters from tuning
        X_train: Training features
        y_train: Training target
    
    Returns:
        model: Trained model
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    model = GradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train)
    print("✓ Model trained on full training set")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate final model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
    }
    
    print(f"\n📈 Test Set Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    return metrics


def save_results(best_params, metrics, model, study):
    """
    Save tuning results and final model.
    
    Args:
        best_params: Best hyperparameters
        metrics: Final evaluation metrics
        model: Trained model
        study: Optuna study object
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Save best parameters
    params_file = "models/best_params.json"
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n💾 Best parameters saved to {params_file}")
    
    # Save final model
    model_file = "models/tuned_model.pkl"
    joblib.dump(model, model_file)
    print(f"💾 Tuned model saved to {model_file}")
    
    # Save metrics
    metrics_file = "results/tuning_metrics.json"
    metrics_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'best_cv_f1': study.best_value,
        'test_metrics': metrics,
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_with_timestamp, f, indent=2)
    print(f"💾 Metrics saved to {metrics_file}")
    
    # Save trial history
    trials_file = "results/tuning_trials.csv"
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_file, index=False)
    print(f"💾 Trial history saved to {trials_file}")


if __name__ == "__main__":
    print("🚀 Spotify Recommendation Engine - Hyperparameter Tuning\n")
    
    # Run tuning
    study, best_params, X_train, X_test, y_train, y_test = run_tuning(n_trials=30, cv_folds=5)
    
    if best_params is None:
        print("\n❌ Tuning failed. Exiting.")
        exit(1)
    
    # Train final model
    try:
        model = train_final_model(best_params, X_train, y_train)
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        exit(1)
    
    # Evaluate
    try:
        metrics = evaluate_model(model, X_test, y_test)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        exit(1)
    
    # Save results
    try:
        save_results(best_params, metrics, model, study)
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        exit(1)
    
    print("\n✅ Hyperparameter tuning complete!")
    print("   Tuned model ready for production")
