"""Tune decoder weights on training trips and compare full routes on student testing."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .data import GAP_S, generate_candidates, load_reference, load_trip
from .decoder import build_graph, decode, lcs_length, reconstruct, transition_lattice
from .reference import observation_labels
from .train import instantiate, read_split, score_candidates, training_rows


def route_metrics(route, matched, labels, runs, connected, steps):
    aligned = labels >= 0
    sequence = np.asarray([arc for run in runs for arc in run], dtype=np.int32)
    # Known reference discontinuities are left in the ordered comparison.
    common = int(lcs_length(sequence, route))
    eligible = steps[1:] <= GAP_S
    return {
        "correct": int(np.sum((matched == labels) & aligned)),
        "aligned": int(np.sum(aligned)),
        "connected": int(np.sum(connected & eligible)),
        "eligible": int(np.sum(eligible)),
        "gps_gaps": int(np.sum(~eligible)),
        "route_breaks": int(np.sum(~connected & eligible)),
        "lcs_arcs": common, "predicted_arcs": len(sequence), "reference_arcs": len(route),
    }


def summary(rows):
    totals = {key: sum(row[key] for row in rows) for key in (
        "correct", "aligned", "connected", "eligible", "gps_gaps", "route_breaks",
        "lcs_arcs", "predicted_arcs", "reference_arcs", "inference_s",
    )}
    return {
        **totals,
        "top1": totals["correct"] / totals["aligned"] if totals["aligned"] else None,
        "continuity": totals["connected"] / totals["eligible"] if totals["eligible"] else None,
        "sequence_f1": 2 * totals["lcs_arcs"] / (totals["predicted_arcs"] + totals["reference_arcs"]),
    }


def tune_decoder(root, cache, results, model_config):
    cv = json.loads((results / "cv_splits.json").read_text())[0]
    trips = read_split(cache, "train")
    model = instantiate(model_config["model"], model_config["configuration"]).fit(
        *training_rows(trips, cv["fit"]))
    settings = {(beta, weight): [0, 0, 0, 0] for beta in (8.0, 20.0, 40.0)
                for weight in (0.5, 1.2, 2.5)}
    for trip_id in cv["validation"]:
        path = root / "train" / trip_id
        trip = load_trip(path)
        candidates = generate_candidates(trip)
        stored = trips[trip_id]
        if not np.array_equal(candidates.arc, stored["arcs"]):
            raise ValueError(f"Stale candidate cache for {trip_id}. Rerun pathsnap.prepare.")
        proba = score_candidates(model, candidates.arc, candidates.features)
        distances, steps, displacement = transition_lattice(trip, candidates, build_graph(trip))
        observations = np.arange(len(steps))
        aligned = stored["labels"] >= 0
        for (beta, weight), measures in settings.items():
            slots = decode(proba, distances, steps, displacement, beta, weight)
            arcs = candidates.arc[observations, slots]
            connected = np.isfinite(distances[observations[1:], slots[:-1], slots[1:]])
            measures[0] += int(np.sum((arcs == stored["labels"]) & aligned))
            measures[1] += int(aligned.sum())
            measures[2] += int(connected.sum())
            measures[3] += int(np.sum(steps[1:] <= GAP_S))
    trials = [{"beta_m": beta, "transition_weight": weight,
               "top1": hits / aligned, "continuity": connected / eligible}
              for (beta, weight), (hits, aligned, connected, eligible) in settings.items()]
    best = max(trials, key=lambda row: (row["top1"], row["continuity"]))
    (results / "decoder_trials.json").write_text(json.dumps({
        "fit_trip_ids": cv["fit"], "validation_trip_ids": cv["validation"], "trials": trials,
    }, indent=2) + "\n")
    print(f"Decoder beta={best['beta_m']}, weight={best['transition_weight']}", flush=True)
    return {"beta_m": best["beta_m"], "transition_weight": best["transition_weight"]}


def run(root, cache, results, artifacts):
    config_path = artifacts / "model_config.json"
    config = json.loads(config_path.read_text())
    config["decoder"] = tune_decoder(root, cache, results, config)
    model_results = json.loads((results / "model_metrics.json").read_text())
    names = ["Nearest Segment", *model_results["models"], "Tuned"]
    selected = "Tuned" if config["variant"] == "tuned" else config["model"]
    files = {name: np.load(cache / "scores" / (name.lower().replace(" ", "_") + ".npz"))
             for name in names}
    rows = []
    try:
        for directory in sorted((root / "student_test").iterdir()):
            if not directory.is_dir() or directory.name not in files[selected].files:
                continue
            trip_id = directory.name
            trip = load_trip(directory)
            route = load_reference(directory, trip)
            labels, _ = observation_labels(trip, route)
            candidates = generate_candidates(trip)
            graph = build_graph(trip)
            start = perf_counter()
            distances, steps, displacement = transition_lattice(trip, candidates, graph)
            lattice_s = perf_counter() - start
            for name in [*names, "Selected + Viterbi"]:
                start = perf_counter()
                if name == "Selected + Viterbi":
                    slots = decode(files[selected][trip_id], distances, steps, displacement,
                                   config["decoder"]["beta_m"], config["decoder"]["transition_weight"])
                else:
                    slots = files[name][trip_id].argmax(axis=1)
                chosen, runs, connected = reconstruct(trip, candidates, slots, distances, steps, graph)
                duration = perf_counter() - start
                rows.append({"model": name, "trip": trip_id, "inference_s": duration,
                             "routing_lattice_s": lattice_s,
                             **route_metrics(route, chosen, labels, runs, connected, steps)})
            print(f"Evaluated routes for {trip_id}", flush=True)
    finally:
        for file in files.values():
            file.close()
    with (results / "route_metrics_by_trip.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    data = {name: summary([row for row in rows if row["model"] == name])
            for name in [*names, "Selected + Viterbi"]}
    data["Selected + Viterbi"]["lattice_s"] = sum(
        row["routing_lattice_s"] for row in rows if row["model"] == "Selected + Viterbi")
    (results / "route_metrics.json").write_text(json.dumps(data, indent=2) + "\n")
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Student folder"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/features"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data, args.cache, args.results, args.artifacts)
