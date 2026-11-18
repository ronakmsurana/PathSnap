"""Refit the deployed classifier on each saved split and score all aligned observations."""
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from .train import evaluate, instantiate, read_split, training_rows


def audit():
    results = Path("results")
    folds = json.loads((results / "cv_splits.json").read_text())
    model_config = json.loads(Path("artifacts/model_config.json").read_text())
    trips = read_split(Path(".cache/features"), "train")
    rows = []
    for index, fold in enumerate(folds, 1):
        model = instantiate(model_config["model"], model_config["configuration"])
        model.fit(*training_rows(trips, fold["fit"]))
        row = {"fold": index}
        for partition in ("fit", "validation"):
            scores, _ = evaluate(model, trips, fold[partition])
            row[partition] = {key: scores[key] for key in ("macro_f1", "top1", "aligned_observations")}
        rows.append(row)
    (results / "overfitting_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        audit()
