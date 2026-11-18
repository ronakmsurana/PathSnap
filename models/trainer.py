"""
PathSnap Multi-Model Training and Benchmark Engine
Trains and rigorously benchmarks 6 distinct models across 5 algorithm families:
1. Logistic Regression (Linear/Probabilistic)
2. Gaussian Naive Bayes (Probabilistic Generative)
3. Decision Tree Classifier (Tree-based)
4. Random Forest Classifier (Ensemble Bagging)
5. LightGBM Classifier (Gradient Boosting)
6. XGBoost Classifier (Gradient Boosting)

Enforces GroupKFold cross-validation on the 80 training trips (zero intra-trip leakage)
and evaluates on the independent 10 student-testing trips.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score

from evaluation.metrics import evaluate_model_performance, compute_route_continuity
from data_processing.loader import load_trip


def load_datasets(data_dir: str = "data/processed"):
    train_path = os.path.join(data_dir, "train_dataset.npz")
    test_path = os.path.join(data_dir, "student_test_dataset.npz")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Processed datasets not found in {data_dir}. Run dataset_builder first.")

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    return train_data, test_data


def get_model_definitions():
    """
    Defines 6 diverse models across 5 distinct algorithm families.
    """
    return [
        {
            "name": "Logistic Regression",
            "family": "Linear and probabilistic",
            "config": "C=1.0, penalty=l2, max_iter=500, with StandardScaler",
            "model": LogisticRegression(C=1.0, penalty="l2", max_iter=500, random_state=42),
            "requires_scaling": True
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Linear and probabilistic",
            "config": "var_smoothing=1e-9",
            "model": GaussianNB(),
            "requires_scaling": True
        },
        {
            "name": "Decision Tree",
            "family": "Tree-based",
            "config": "max_depth=12, min_samples_split=20, criterion=gini",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_split=20, random_state=42),
            "requires_scaling": False
        },
        {
            "name": "Random Forest",
            "family": "Ensemble",
            "config": "n_estimators=100, max_depth=14, max_features=sqrt, n_jobs=-1",
            "model": RandomForestClassifier(n_estimators=100, max_depth=14, max_features="sqrt", random_state=42, n_jobs=-1),
            "requires_scaling": False
        },
        {
            "name": "LightGBM",
            "family": "Boosting",
            "config": "n_estimators=150, lr=0.08, num_leaves=31, n_jobs=-1",
            "model": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),
            "requires_scaling": False
        },
        {
            "name": "XGBoost",
            "family": "Boosting",
            "config": "n_estimators=120, lr=0.1, max_depth=6, eval_metric=logloss, n_jobs=-1",
            "model": xgb.XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=6, eval_metric="logloss", random_state=42, n_jobs=-1),
            "requires_scaling": False
        }
    ]


def run_group_cv(model, X, y, groups, n_splits=5, requires_scaling=False) -> float:
    """
    Executes group-aware cross-validation ensuring trips are never split across folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train_f, y_train_f = X[train_idx], y[train_idx]
        X_val_f, y_val_f = X[val_idx], y[val_idx]

        if requires_scaling:
            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(X_train_f)
            X_val_f = scaler.transform(X_val_f)

        model.fit(X_train_f, y_train_f)
        y_val_pred = model.predict(X_val_f)
        score = f1_score(y_val_f, y_val_pred, average="macro", zero_division=0)
        f1_scores.append(score)

    return float(np.mean(f1_scores))


def train_and_benchmark(
    data_dir: str = "data/processed",
    output_dir: str = "results",
    artifacts_dir: str = "artifacts",
    cv_sample_size: int = 250000
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Loading train and student-test datasets...")
    train_data, test_data = load_datasets(data_dir)

    X_train = train_data["X"]
    y_train = train_data["y"]
    groups_train = train_data["groups"]

    X_test = test_data["X"]
    y_test = test_data["y"]
    groups_test = test_data["groups"]
    point_idx_test = test_data["point_indices"]
    arc_ids_test = test_data["arc_ids"]

    # Pre-fit scaler for algorithms that require normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))

    # Load sample trip arc lookup for route continuity calculation
    sample_trip = load_trip("../Student folder/student_test", "00000056")
    arcs_lookup = sample_trip.arcs

    # Stratified/grouped subsample for rapid 5-fold GroupKFold cross-validation
    if len(X_train) > cv_sample_size:
        np.random.seed(42)
        unique_groups = np.unique(groups_train)
        # Select representative trips for CV
        selected_trips = np.random.choice(unique_groups, size=min(30, len(unique_groups)), replace=False)
        cv_mask = np.isin(groups_train, selected_trips)
        X_cv, y_cv, groups_cv = X_train[cv_mask], y_train[cv_mask], groups_train[cv_mask]
        print(f"Using representative group partition for CV: {len(y_cv):,} samples across {len(selected_trips)} trips.")
    else:
        X_cv, y_cv, groups_cv = X_train, y_train, groups_train

    models_info = get_model_definitions()
    results = []

    print(f"\n=======================================================")
    print(f"Starting Benchmark of {len(models_info)} Models Across Algorithm Families")
    print(f"=======================================================\n")

    for item in models_info:
        name = item["name"]
        family = item["family"]
        config = item["config"]
        model = item["model"]
        scaling = item["requires_scaling"]

        print(f">>> Evaluating: {name} ({family})")

        # 1. 5-Fold Group-aware Cross-Validation
        t0_cv = time.time()
        cv_macro_f1 = run_group_cv(model, X_cv, y_cv, groups_cv, n_splits=5, requires_scaling=scaling)
        print(f"    5-Fold GroupKFold CV Macro F1: {cv_macro_f1:.4f} (CV time: {time.time() - t0_cv:.1f}s)")

        # 2. Fit full model on all 80 training trips
        t0_fit = time.time()
        X_tr = X_train_scaled if scaling else X_train
        model.fit(X_tr, y_train)
        train_time = time.time() - t0_fit
        print(f"    Fitted on {len(X_tr):,} training samples in: {train_time:.2f}s")

        # 3. Save model artifact
        artifact_path = os.path.join(artifacts_dir, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, artifact_path)

        # 4. Predict on student_test partition
        X_te = X_test_scaled if scaling else X_test
        t0_inf = time.time()
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_te)[:, 1]
        else:
            scores = model.decision_function(X_te)
            probs = 1.0 / (1.0 + np.exp(-scores))
        inf_time = time.time() - t0_inf
        inf_latency_per_1k = (inf_time / len(X_te)) * 1000.0 * 1000.0  # ms per 1000 observations

        # 5. Evaluate all comprehensive metrics
        perf = evaluate_model_performance(
            model_name=name,
            algorithm_family=family,
            key_config=config,
            y_true=y_test,
            y_probs=probs,
            groups=groups_test,
            point_indices=point_idx_test,
            arc_ids=arc_ids_test,
            arcs_lookup=arcs_lookup,
            cv_macro_f1=cv_macro_f1,
            training_time=train_time,
            inference_time_ms=inf_latency_per_1k
        )
        results.append(perf)

        print(f"    Student-Test Top-1 Accuracy: {perf['top1_accuracy'] * 100:.2f}%")
        print(f"    Student-Test Macro F1:       {perf['student_test_macro_f1']:.4f}")
        print(f"    Log Loss:                    {perf['log_loss']:.4f}")
        print(f"    Inference Latency:           {perf['inference_time_ms_per_1k']:.2f} ms/1k obs\n")

    # Build Consolidated Comparison Table (Deliverables Sec 3.1)
    df_results = pd.DataFrame(results)
    
    # Save full results JSON and CSV
    results_csv = os.path.join(output_dir, "model_comparison.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\nSaved consolidated model comparison table to: {results_csv}")

    # Generate Markdown Table for terminal display
    summary_cols = [
        "model", "algorithm_family", "key_configuration",
        "cv_macro_f1", "top1_accuracy", "student_test_macro_f1",
        "log_loss", "inference_time_ms_per_1k"
    ]
    print("\n--- CONSOLIDATED MODEL COMPARISON TABLE ---")
    print(df_results[summary_cols].to_markdown(index=False))

    return df_results


if __name__ == "__main__":
    train_and_benchmark()
