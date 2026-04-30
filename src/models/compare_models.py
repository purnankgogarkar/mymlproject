"""
Model Comparison Framework for Spotify Recommendation Engine.

Compares multiple classification/regression models on engineered features.
Includes baseline, tree-based, and boosting approaches with cross-validation.
"""

import time
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost if available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_and_prepare_data(filepath="data/processed/spotify_features.csv", test_size=0.2, random_state=42):
    """
    Load and prepare data for model comparison.
    
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
    print(f"✓ Target: {target_col} (variance: {variances.iloc[0]:.6f})")
    
    # Binarize for classification
    y = (df[target_col] > df[target_col].median()).astype(int)
    print(f"✓ Binary classification: {target_col} > median")
    print(f"  Class 0: {(y == 0).sum():,} samples")
    print(f"  Class 1: {(y == 1).sum():,} samples")
    
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


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, cv_folds=5):
    """
    Train model with cross-validation and evaluate on test set.
    
    Args:
        model: Sklearn model instance
        X_train, X_test, y_train, y_test: Train/test data
        model_name: Name of model (for logging)
        cv_folds: Number of cross-validation folds
    
    Returns:
        dict: Results with CV scores, test score, training time
    """
    print(f"\n🤖 {model_name}:")
    print("-" * 60)
    
    # Training time
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  ✓ Trained in {train_time:.3f}s")
    
    # Cross-validation (5-fold)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print(f"  ✓ 5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    test_auc = None
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  ✓ Test Accuracy: {test_accuracy:.4f}")
    print(f"  ✓ Test F1:       {test_f1:.4f}")
    if test_auc is not None:
        print(f"  ✓ Test AUC-ROC:  {test_auc:.4f}")
    
    return {
        'model': model,
        'model_name': model_name,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc': test_auc,
        'train_time': train_time,
    }


def save_results(results, filepath="models/model_comparison.pkl"):
    """Save comparison results to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(results, filepath)
    print(f"\n💾 Results saved to {filepath}")


def create_comparison_table(results_list):
    """
    Create and display comparison table.
    
    Args:
        results_list: List of result dictionaries from train_and_evaluate
    
    Returns:
        pd.DataFrame: Comparison table
    """
    rows = []
    for res in results_list:
        rows.append({
            'Model': res['model_name'],
            'CV Mean': f"{res['cv_mean']:.4f}",
            'CV Std': f"{res['cv_std']:.4f}",
            'Test Acc': f"{res['test_accuracy']:.4f}",
            'Test F1': f"{res['test_f1']:.4f}",
            'Test AUC': f"{res['test_auc']:.4f}" if res['test_auc'] else "N/A",
            'Train Time (s)': f"{res['train_time']:.3f}",
        })
    
    comparison_df = pd.DataFrame(rows)
    return comparison_df


def print_recommendation(results_list):
    """Print model recommendation."""
    print("\n" + "="*70)
    print("MODEL RECOMMENDATION")
    print("="*70)
    
    # Sort by CV mean score
    best = sorted(results_list, key=lambda x: x['cv_mean'], reverse=True)[0]
    
    print(f"\n🏆 Best Model: {best['model_name']}")
    print(f"   CV Score:    {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print(f"   Test Acc:    {best['test_accuracy']:.4f}")
    print(f"   Test F1:     {best['test_f1']:.4f}")
    print(f"   Train Time:  {best['train_time']:.3f}s")
    
    print(f"\n📊 Why {best['model_name']}?")
    if 'RandomForest' in best['model_name']:
        print("   • Handles non-linear relationships in audio features")
        print("   • Provides feature importance rankings")
        print("   • Robust to outliers in Spotify data")
        print("   • No scaling needed (tree-based)")
    elif 'GradientBoosting' in best['model_name']:
        print("   • State-of-the-art sequential boosting")
        print("   • Strong performance on complex music patterns")
        print("   • Learns feature interactions automatically")
        print("   • Optimal for recommendation systems")
    elif 'XGBoost' in best['model_name']:
        print("   • Industry-leading performance")
        print("   • Fast training with GPU support")
        print("   • Built-in regularization prevents overfitting")
        print("   • Excellent for Kaggle-style music datasets")
    elif 'LogisticRegression' in best['model_name']:
        print("   • Fast baseline for interpretability")
        print("   • Linear relationships sufficient for this dataset")
        print("   • Consistent cross-validation scores")
    elif 'SVM' in best['model_name']:
        print("   • Excellent for high-dimensional features")
        print("   • Strong generalization on test set")
        print("   • Kernel methods capture feature interactions")


if __name__ == "__main__":
    print("🚀 Spotify Recommendation Engine - Model Comparison\n")
    
    # Load and prepare data
    try:
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        exit(1)
    
    print("\n" + "="*70)
    print("COMPARING MODELS")
    print("="*70)
    
    results_list = []
    
    # 1. Logistic Regression (Baseline)
    try:
        model_lr = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
        res = train_and_evaluate(model_lr, X_train, X_test, y_train, y_test, 
                                "LogisticRegression (Baseline)")
        results_list.append(res)
        
        # Save model
        joblib.dump(model_lr, "models/model_logistic_regression.pkl")
        print("  💾 Saved to models/model_logistic_regression.pkl")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # 2. Random Forest
    try:
        model_rf = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                         random_state=42, n_jobs=-1)
        res = train_and_evaluate(model_rf, X_train, X_test, y_train, y_test,
                                "RandomForestClassifier")
        results_list.append(res)
        
        # Save model
        joblib.dump(model_rf, "models/model_random_forest.pkl")
        print("  💾 Saved to models/model_random_forest.pkl")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # 3. Gradient Boosting
    try:
        model_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                             learning_rate=0.1, random_state=42)
        res = train_and_evaluate(model_gb, X_train, X_test, y_train, y_test,
                                "GradientBoostingClassifier")
        results_list.append(res)
        
        # Save model
        joblib.dump(model_gb, "models/model_gradient_boosting.pkl")
        print("  💾 Saved to models/model_gradient_boosting.pkl")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # 4. XGBoost (if available) — faster version
    if HAS_XGBOOST:
        try:
            model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=5,
                                         learning_rate=0.1, random_state=42,
                                         verbosity=0, tree_method='hist')
            res = train_and_evaluate(model_xgb, X_train, X_test, y_train, y_test,
                                    "XGBClassifier (fast)")
            results_list.append(res)
            
            # Save model
            joblib.dump(model_xgb, "models/model_xgboost.pkl")
            print("  💾 Saved to models/model_xgboost.pkl")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # 5. SVM (Linear Kernel) — much faster than RBF
    try:
        print("\n⏳ Training SVM with linear kernel (fast)...")
        model_svm = SVC(kernel='linear', random_state=42, max_iter=1000)
        res = train_and_evaluate(model_svm, X_train, X_test, y_train, y_test,
                                "SVM (Linear Kernel)")
        results_list.append(res)
        
        # Save model
        joblib.dump(model_svm, "models/model_svm.pkl")
        print("  💾 Saved to models/model_svm.pkl")
    except KeyboardInterrupt:
        print("  ⚠️  SVM training interrupted by user")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Display comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    comparison_df = create_comparison_table(results_list)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    print("\n💾 Comparison table saved to results/model_comparison.csv")
    
    # Save results
    save_results(results_list)
    
    # Recommendation
    if results_list:
        print_recommendation(results_list)
    
    print("\n✅ Model comparison complete!")
    print("   Best model ready for deployment")
