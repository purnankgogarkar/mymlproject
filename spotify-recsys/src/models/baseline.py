"""
Baseline Model for Spotify Recommendation Engine.

Trains and evaluates baseline models (Logistic Regression for classification,
Linear Regression for regression) on engineered features.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')


def detect_task_and_target(df):
    """
    Detect if task is classification or regression, and identify target column.
    
    Args:
        df: DataFrame with features
    
    Returns:
        tuple: (target_col, is_classification, y)
    """
    # Look for explicit target column
    candidate_targets = [col for col in df.columns 
                        if col.lower() in ['genre', 'mood', 'target', 'popularity', 'class']]
    
    if candidate_targets:
        target_col = candidate_targets[0]
        y = df[target_col]
        
        # Detect if classification or regression
        if y.dtype == 'object' or (y.nunique() < 20 and y.nunique() < len(y) * 0.5):
            is_classification = True
        else:
            is_classification = False
        
        return target_col, is_classification, y
    
    # Try energy-related features
    energy_candidates = [col for col in df.columns if 'energy' in col.lower()]
    if energy_candidates:
        print(f"⚠️  No explicit target column found. Using '{energy_candidates[0]}' as pseudo-target")
        target_col = energy_candidates[0]
        y = df[target_col]
        
        # Binarize for classification if continuous
        if y.nunique() > 10:
            print(f"    Creating binary classification: {target_col} > median")
            target_col = f'{target_col}_high'
            y = (df[energy_candidates[0]] > df[energy_candidates[0]].median()).astype(int)
        
        return target_col, True, y
    
    # Try highest-variance numeric feature as pseudo-target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        target_col = variances.index[0]
        print(f"⚠️  No explicit target column found. Using '{target_col}' (highest variance) as pseudo-target")
        print(f"    Creating binary classification: {target_col} > median")
        target_col_bin = f'{target_col}_high'
        y = (df[target_col] > df[target_col].median()).astype(int)
        return target_col_bin, True, y
    
    raise ValueError("Cannot determine target: no suitable features found")


def load_data(filepath="data/processed/spotify_features.csv"):
    """
    Load engineered features from CSV.
    
    Args:
        filepath: Path to features CSV
    
    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Features file not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data: identify target, remove non-numeric features, split train/test.
    
    Args:
        df: DataFrame with features
        test_size: Test set fraction (default 0.2 = 80/20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, target_col, is_classification)
    """
    print("\n📊 Preparing data...")
    
    # Detect task and target
    target_col, is_classification, y = detect_task_and_target(df)
    print(f"✓ Task: {'Classification' if is_classification else 'Regression'}")
    print(f"✓ Target: {target_col}")
    print(f"  Unique values: {y.nunique()}")
    
    # Get numeric features (exclude target and metadata columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target-related columns
    cols_to_exclude = {'track_id', 'track_name', target_col}
    # Also exclude original columns that might have been binarized
    for col in df.columns:
        if '_high' in col or '_low' in col:
            cols_to_exclude.add(col)
    
    X_cols = [col for col in numeric_cols if col not in cols_to_exclude]
    X = df[X_cols].copy()
    
    print(f"✓ Features: {X.shape[1]} numeric features")
    print(f"  Sample features: {', '.join(X_cols[:5])}{'...' if len(X_cols) > 5 else ''}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if is_classification else None
    )
    
    print(f"✓ Train/Test split:")
    print(f"  Train: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test:  {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test, target_col, is_classification


def train_baseline(X_train, y_train, is_classification):
    """
    Train baseline model (LogisticRegression or LinearRegression).
    
    Args:
        X_train: Training features
        y_train: Training target
        is_classification: Whether task is classification
    
    Returns:
        model: Trained model
    """
    print("\n🤖 Training baseline model...")
    
    if is_classification:
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        print("✓ LogisticRegression (default settings)")
    else:
        model = LinearRegression()
        print("✓ LinearRegression (default settings)")
    
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    return model


def evaluate_classification(model, X_test, y_test):
    """
    Evaluate classification model.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n📈 Classification Evaluation:")
    print("="*50)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    # AUC-ROC only for binary classification
    if len(np.unique(y_test)) == 2:
        metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    return metrics


def evaluate_regression(model, X_test, y_test):
    """
    Evaluate regression model.
    
    Args:
        model: Trained regression model
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n📈 Regression Evaluation:")
    print("="*50)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
    }
    
    # Print results
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return metrics


def save_model(model, filepath="models/baseline.pkl"):
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained model
        filepath: Path to save model
    
    Returns:
        str: Path where model was saved
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\n💾 Model saved to {filepath}")
    return filepath


def print_summary(metrics, is_classification, model_type):
    """
    Print baseline evaluation summary.
    
    Args:
        metrics: Dictionary of evaluation metrics
        is_classification: Whether task is classification
        model_type: Name of model (e.g., 'LogisticRegression')
    """
    print("\n" + "="*50)
    print("BASELINE MODEL SUMMARY")
    print("="*50)
    print(f"\nModel: {model_type}")
    print(f"Task: {'Classification' if is_classification else 'Regression'}")
    
    print(f"\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric:15s}: {value:.4f}")
    
    print("\n✅ Baseline trained and saved!")


if __name__ == "__main__":
    print("🚀 Spotify Recommendation Engine - Baseline Model\n")
    
    # Load data
    try:
        df = load_data("data/processed/spotify_features.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Run feature pipeline first: python src/features/run_features.py")
        exit(1)
    
    # Prepare data
    try:
        X_train, X_test, y_train, y_test, target_col, is_classification = prepare_data(
            df,
            test_size=0.2,
            random_state=42
        )
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        exit(1)
    
    # Train baseline
    try:
        model = train_baseline(X_train, y_train, is_classification)
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        exit(1)
    
    # Evaluate
    try:
        if is_classification:
            metrics = evaluate_classification(model, X_test, y_test)
            model_type = "LogisticRegression"
        else:
            metrics = evaluate_regression(model, X_test, y_test)
            model_type = "LinearRegression"
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        exit(1)
    
    # Save model
    try:
        save_model(model, "models/baseline.pkl")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        exit(1)
    
    # Summary
    print_summary(metrics, is_classification, model_type)
    
    print("\n📁 Next steps:")
    print("   • Advanced models: src/models/advanced.py")
    print("   • Hyperparameter tuning: src/models/tuning.py")
    print("   • Production API: app/api.py")
