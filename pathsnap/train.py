"""Compare five models with grouped CV, then tune the selected family with Optuna TPE."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .data import FEATURES


SEED = 352
TRAIN_PER_TRIP = 350
FOLDS = 3
MODELS = {
    "Logistic Regression": ("Linear", {"C": 1.0, "max_iter": 120}),
    "Gaussian Naive Bayes": ("Probabilistic", {"var_smoothing": 1e-9}),
    "Decision Tree": ("Tree", {"max_depth": 9, "min_samples_leaf": 30}),
    "Random Forest": ("Bagging", {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 16}),
    "Histogram Gradient Boosting": ("Boosting", {
        "learning_rate": 0.1, "max_iter": 90, "max_leaf_nodes": 23, "min_samples_leaf": 40,
    }),
}


def read_split(cache, split):
    trips = {}
    for path in sorted((cache / split).glob("*.npz")):
        with np.load(path) as f:
            trips[path.stem] = {name: f[name] for name in f.files}
    if not trips:
        raise ValueError(f"Run pathsnap.prepare first. No cached trips in {cache / split}")
    return trips


def training_rows(trips, trip_ids):
    matrix, labels = [], []
    for trip_id in trip_ids:
        data = trips[trip_id]
        aligned = np.flatnonzero(data["labels"] >= 0)
        rng = np.random.default_rng(SEED + int(trip_id))
        sample = np.sort(rng.choice(aligned, min(len(aligned), TRAIN_PER_TRIP), replace=False))
        arcs = data["arcs"][sample]
        truth = data["labels"][sample, None]
        # Include only observations with a positive candidate during fitting.
        valid = (arcs >= 0) & (arcs == truth).any(axis=1)[:, None]
        matrix.append(data["features"][sample][valid])
        labels.append((arcs == truth)[valid])
    return np.concatenate(matrix), np.concatenate(labels)


def instantiate(name, config):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**config, random_state=SEED))
    if name == "Gaussian Naive Bayes":
        return make_pipeline(StandardScaler(), GaussianNB(**config))
    if name == "Decision Tree":
        return DecisionTreeClassifier(**config, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(**config, n_jobs=2, random_state=SEED)
    return HistGradientBoostingClassifier(**config, early_stopping=False, random_state=SEED)


def score_candidates(model, arc, features):
    scores = np.full(arc.shape, -1, dtype=np.float32)
    valid = arc >= 0
    scores[valid] = model.predict_proba(features[valid])[:, 1]
    return scores


def evaluate(model, trips, trip_ids, baseline=False):
    truths, scores, correct, hits = [], [], [], []
    predictions = {}
    inference = 0.0
    for trip_id in trip_ids:
        data = trips[trip_id]
        arc, y = data["arcs"], data["labels"]
        start = perf_counter()
        if baseline:
            proba = np.where(arc >= 0, 0.0, -1.0)
            proba[:, 0] = 1.0
        else:
            proba = score_candidates(model, arc, data["features"])
        slot = np.argmax(proba, axis=1)
        inference += perf_counter() - start
        predictions[trip_id] = proba
        aligned = y >= 0
        valid = (arc >= 0) & aligned[:, None]
        truths.append((arc == y[:, None])[valid])
        scores.append(proba[valid])
        correct.append(arc[np.arange(len(slot)), slot][aligned] == y[aligned])
        hits.append((arc == y[:, None]).any(axis=1)[aligned])
    y = np.concatenate(truths).astype(np.int32)
    prob = np.concatenate(scores)
    predicted = prob >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], average="macro", zero_division=0)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "macro_precision": precision, "macro_recall": recall, "macro_f1": f1,
        "accuracy": float(np.mean(y == predicted)),
        "top1": float(np.mean(np.concatenate(correct))),
        "miss_rate": float(1 - np.mean(np.concatenate(hits))),
        "log_loss": None if baseline else log_loss(y, prob.astype(np.float64), labels=[0, 1]),
        "inference_s": inference, "aligned_observations": sum(len(x) for x in correct),
        "confusion_matrix": matrix.tolist(),
    }, predictions


def cv_model(name, config, trips, folds, fit_metrics=False):
    fold_results = []
    for fit_ids, validation_ids, x, y in folds:
        model = instantiate(name, config)
        start = perf_counter()
        model.fit(x, y)
        duration = perf_counter() - start
        metrics, _ = evaluate(model, trips, validation_ids)
        metrics["fit_s"] = duration
        if fit_metrics:
            labels = model.predict(x)
            metrics["fit_sample_macro_f1"] = precision_recall_fscore_support(
                y, labels, average="macro", zero_division=0)[2]
        fold_results.append(metrics)
    return fold_results


def search(trial, name, default):
    settings = dict(default)
    if name == "Logistic Regression":
        settings["C"] = trial.suggest_float("C", 0.005, 50.0, log=True)
    elif name == "Gaussian Naive Bayes":
        settings["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)
    elif name in ("Decision Tree", "Random Forest"):
        settings.update(max_depth=trial.suggest_int("max_depth", 6, 18),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 70, log=True))
        if name == "Random Forest":
            settings["n_estimators"] = trial.suggest_int("n_estimators", 40, 100, step=20)
    else:
        settings.update(
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
            max_iter=trial.suggest_int("max_iter", 70, 170, step=10),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 47, step=4),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 100, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-5, 3.0, log=True),
        )
    return settings


def mean_metric(folds, key):
    return float(np.mean([fold[key] for fold in folds]))


def save_predictions(folder, name, predictions):
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / (name.lower().replace(" ", "_") + ".npz")
    np.savez(file, **predictions)


def train(cache, results, artifacts, n_trials):
    results.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    trips = read_split(cache, "train")
    student = read_split(cache, "student_test")
    ids = np.array(sorted(trips))
    folds = []
    for fit, valid in GroupKFold(n_splits=FOLDS).split(ids, groups=ids):
        fit_ids, valid_ids = ids[fit].tolist(), ids[valid].tolist()
        folds.append((fit_ids, valid_ids, *training_rows(trips, fit_ids)))
    (results / "cv_splits.json").write_text(json.dumps([
        {"fit": fit, "validation": validation} for fit, validation, _, _ in folds
    ], indent=2) + "\n")
    x, y = training_rows(trips, ids)
    summary = {"seed": SEED, "training_observations_per_trip_cap": TRAIN_PER_TRIP,
               "candidate_fit_rows": len(y), "models": {}}
    evaluation, predictions = evaluate(None, student, sorted(student), baseline=True)
    summary["baseline"] = evaluation
    save_predictions(cache / "scores", "Nearest Segment", predictions)
    for name, (family, config) in MODELS.items():
        cv = cv_model(name, config, trips, folds, fit_metrics=True)
        model = instantiate(name, config)
        start = perf_counter()
        model.fit(x, y)
        fit_time = perf_counter() - start
        evaluation, predictions = evaluate(model, student, sorted(student))
        save_predictions(cache / "scores", name, predictions)
        summary["models"][name] = {
            "family": family, "config": config, "cv": cv,
            "cv_macro_f1": mean_metric(cv, "macro_f1"),
            "cv_fit_sample_macro_f1": mean_metric(cv, "fit_sample_macro_f1"),
            "student": evaluation, "full_fit_s": fit_time,
        }
        print(f"{name}: CV macro-F1 {mean_metric(cv, 'macro_f1'):.4f}, "
              f"student Top-1 {evaluation['top1']:.4f}", flush=True)

    winner = max(summary["models"], key=lambda name: summary["models"][name]["cv_macro_f1"])
    defaults = MODELS[winner][1]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4))
    start = perf_counter()

    def objective(trial):
        settings = search(trial, winner, defaults)
        metrics = cv_model(winner, settings, trips, folds)
        trial.set_user_attr("fold_scores", [m["macro_f1"] for m in metrics])
        score = mean_metric(metrics, "macro_f1")
        print(f"TPE trial {trial.number + 1}/{n_trials}: {score:.4f}", flush=True)
        return score

    study.optimize(objective, n_trials=n_trials)
    duration = perf_counter() - start
    tuned = {**defaults, **study.best_params}
    cv = cv_model(winner, tuned, trips, folds, fit_metrics=True)
    model = instantiate(winner, tuned)
    start = perf_counter()
    model.fit(x, y)
    fit_time = perf_counter() - start
    evaluation, predictions = evaluate(model, student, sorted(student))
    save_predictions(cache / "scores", "Tuned", predictions)
    summary["selected"] = winner
    summary["tuned"] = {
        "config": tuned, "cv": cv, "cv_macro_f1": mean_metric(cv, "macro_f1"),
        "full_fit_s": fit_time, "student": evaluation, "search_s": duration,
        "trial_count": n_trials, "best_trial": study.best_trial.number,
    }
    with (results / "optimization_trials.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["trial", "macro_f1", "duration_s", "fold_scores", "parameters"])
        for trial in study.trials:
            writer.writerow([trial.number, trial.value, trial.duration.total_seconds(),
                             json.dumps(trial.user_attrs["fold_scores"]), json.dumps(trial.params)])
    use_tuned = summary["tuned"]["cv_macro_f1"] >= summary["models"][winner]["cv_macro_f1"]
    summary["deployed_configuration"] = "tuned" if use_tuned else "untuned"
    if not use_tuned:
        model = instantiate(winner, defaults).fit(x, y)
    joblib.dump(model, artifacts / "model.joblib", compress=3)
    (artifacts / "model_config.json").write_text(json.dumps({
        "model": winner, "configuration": tuned if use_tuned else defaults,
        "features": FEATURES, "seed": SEED, "variant": summary["deployed_configuration"],
    }, indent=2) + "\n")
    (results / "model_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/features"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        train(args.cache, args.results, args.artifacts, args.trials)
