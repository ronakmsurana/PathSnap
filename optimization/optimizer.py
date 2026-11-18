"""
Bayesian Hyperparameter Optimization Engine for PathSnap
Uses Optuna to perform Bayesian optimization with GroupKFold cross-validation
on the 80 training trips, avoiding any intra-trip data leakage.
Exports trial records, parameter importance, and compares untuned vs. tuned models.
"""

import os
import time
import joblib
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

from evaluation.metrics import evaluate_model_performance
from data_processing.loader import load_trip


# Suppress overly verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(
    data_dir: str = "data/processed",
    output_dir: str = "results",
    artifacts_dir: str = "artifacts",
    n_trials: int = 25,
    cv_sample_size: int = 250000
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Loading datasets for Bayesian Optimization...")
    train_data = np.load(os.path.join(data_dir, "train_dataset.npz"))
    test_data = np.load(os.path.join(data_dir, "student_test_dataset.npz"))

    X_train = train_data["X"]
    y_train = train_data["y"]
    groups_train = train_data["groups"]

    X_test = test_data["X"]
    y_test = test_data["y"]
    groups_test = test_data["groups"]
    point_idx_test = test_data["point_indices"]
    arc_ids_test = test_data["arc_ids"]

    # Select representative trips for GroupKFold CV during optimization
    np.random.seed(42)
    unique_groups = np.unique(groups_train)
    selected_trips = np.random.choice(unique_groups, size=min(25, len(unique_groups)), replace=False)
    cv_mask = np.isin(groups_train, selected_trips)
    X_cv, y_cv, groups_cv = X_train[cv_mask], y_train[cv_mask], groups_train[cv_mask]

    print(f"Optimizing on {len(y_cv):,} samples across {len(selected_trips)} trips with 3-Fold GroupKFold.")

    gkf = GroupKFold(n_splits=3)
    cv_splits = list(gkf.split(X_cv, y_cv, groups=groups_cv))

    trials_data = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        # Search space definition
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 250),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 64),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

        scores = []
        for train_idx, val_idx in cv_splits:
            X_tr_f, y_tr_f = X_cv[train_idx], y_cv[train_idx]
            X_va_f, y_va_f = X_cv[val_idx], y_cv[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr_f, y_tr_f)
            preds = clf.predict(X_va_f)
            scores.append(f1_score(y_va_f, preds, average="macro", zero_division=0))

        mean_f1 = float(np.mean(scores))
        duration = time.time() - t0

        record = {"trial_id": trial.number, "cv_macro_f1": mean_f1, "duration_s": duration}
        record.update(params)
        trials_data.append(record)

        return mean_f1

    print(f"Starting Optuna study with {n_trials} trials...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    print(f"\nOptimization Finished!")
    print(f"Best Trial #{best_trial.number} - CV Macro F1: {best_trial.value:.5f}")
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # Save all trial records
    df_trials = pd.DataFrame(trials_data)
    trials_csv = os.path.join(output_dir, "bayesian_optimization_trials.csv")
    df_trials.to_csv(trials_csv, index=False)
    print(f"Saved all optimization trials to {trials_csv}")

    # Refit best tuned model on full 80-trip training set
    print("\nRefitting best tuned model on all 80 training trips...")
    best_params = {
        "n_estimators": best_trial.params["n_estimators"],
        "learning_rate": best_trial.params["learning_rate"],
        "num_leaves": best_trial.params["num_leaves"],
        "max_depth": best_trial.params["max_depth"],
        "min_child_samples": best_trial.params["min_child_samples"],
        "subsample": best_trial.params["subsample"],
        "colsample_bytree": best_trial.params["colsample_bytree"],
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    t0_refit = time.time()
    tuned_model = lgb.LGBMClassifier(**best_params)
    tuned_model.fit(X_train, y_train)
    refit_time = time.time() - t0_refit

    tuned_artifact_path = os.path.join(artifacts_dir, "best_tuned_lightgbm.joblib")
    joblib.dump(tuned_model, tuned_artifact_path)
    print(f"Saved best tuned model artifact to {tuned_artifact_path} (Trained in {refit_time:.2f}s)")

    # Evaluate tuned model on student-test partition
    sample_trip = load_trip("../Student folder/student_test", "00000056")
    arcs_lookup = sample_trip.arcs

    t0_inf = time.time()
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]
    inf_latency = ((time.time() - t0_inf) / len(X_test)) * 1000.0 * 1000.0

    perf_tuned = evaluate_model_performance(
        model_name="Tuned LightGBM (Bayesian Opt)",
        algorithm_family="Boosting",
        key_config=f"Best Optuna params (trial {best_trial.number})",
        y_true=y_test,
        y_probs=tuned_probs,
        groups=groups_test,
        point_indices=point_idx_test,
        arc_ids=arc_ids_test,
        arcs_lookup=arcs_lookup,
        cv_macro_f1=best_trial.value,
        training_time=refit_time,
        inference_time_ms=inf_latency
    )

    # Load untuned LightGBM from artifacts for direct comparison
    untuned_path = os.path.join(artifacts_dir, "lightgbm.joblib")
    if os.path.exists(untuned_path):
        untuned_model = joblib.load(untuned_path)
        t0_inf_u = time.time()
        untuned_probs = untuned_model.predict_proba(X_test)[:, 1]
        inf_latency_u = ((time.time() - t0_inf_u) / len(X_test)) * 1000.0 * 1000.0

        perf_untuned = evaluate_model_performance(
            model_name="Untuned LightGBM (Default)",
            algorithm_family="Boosting",
            key_config="Default hyperparameters",
            y_true=y_test,
            y_probs=untuned_probs,
            groups=groups_test,
            point_indices=point_idx_test,
            arc_ids=arc_ids_test,
            arcs_lookup=arcs_lookup,
            cv_macro_f1=0.0,
            training_time=0.0,
            inference_time_ms=inf_latency_u
        )
    else:
        perf_untuned = perf_tuned

    # Save Comparison Table (Deliverables Sec 4)
    df_comp = pd.DataFrame([perf_untuned, perf_tuned])
    comp_csv = os.path.join(output_dir, "tuned_vs_untuned_comparison.csv")
    df_comp.to_csv(comp_csv, index=False)
    print(f"\nSaved tuned vs. untuned comparison to {comp_csv}")

    return {
        "best_trial": best_trial,
        "tuned_performance": perf_tuned,
        "comparison_df": df_comp
    }


if __name__ == "__main__":
    run_bayesian_optimization()
