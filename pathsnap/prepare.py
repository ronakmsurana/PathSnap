"""Inventory data and cache development features for repeatable experiments."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .data import GAP_S, generate_candidates, load_reference, load_trip
from .reference import observation_labels


def prepare(root, cache, results):
    inventory = []
    cache.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    for split in ("train", "student_test"):
        destination = cache / split
        destination.mkdir(exist_ok=True)
        for directory in sorted((root / split).iterdir()):
            if not directory.is_dir() or not directory.name.isdecimal():
                continue
            trip = load_trip(directory)
            route = load_reference(directory, trip)
            candidates = generate_candidates(trip)
            labels, error = observation_labels(trip, route)
            path = destination / f"{directory.name}.npz"
            np.savez(path, arcs=candidates.arc, features=candidates.features, labels=labels)
            steps = np.diff(trip.track[:, 2])
            route_edges = trip.arcs[route]
            node_count = len(trip.nodes)
            graph = csr_matrix((np.ones(len(trip.arcs), dtype=bool), tuple(trip.arcs.T)),
                               shape=(node_count, node_count))
            present = np.unique(trip.arcs)
            inventory.append({
                "split": split, "trip": directory.name,
                "observations": len(trip.track), "nodes": node_count, "arcs": len(trip.arcs),
                "reference_arcs": len(route), "weak_components": connected_components(
                    graph, directed=False, return_labels=False),
                "isolated_nodes": node_count - len(present),
                "zero_length_arcs": int(np.sum(trip.length < 1e-6)),
                "file_bytes": sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
                "reference_breaks": int(np.sum(route_edges[:-1, 1] != route_edges[1:, 0])),
                "repeat_reference_arcs": len(route) - len(np.unique(route)),
                "max_gap_s": float(steps.max(initial=0)), "gaps_over_30s": int(np.sum(steps > GAP_S)),
                "one_second_steps": int(np.sum(steps == 1)),
                "min_lon": float(trip.track[:, 0].min()), "max_lon": float(trip.track[:, 0].max()),
                "min_lat": float(trip.track[:, 1].min()), "max_lat": float(trip.track[:, 1].max()),
                "nonfinite": 0, "aligned": int(np.sum(labels >= 0)),
                "alignment_median_error_m": float(np.median(error)),
                "candidate_hits": int(np.sum((candidates.arc == labels[:, None]).any(axis=1) & (labels >= 0))),
                "nearest_fallback": int(np.sum(candidates.outside_radius)),
            })
        count = sum(row["split"] == split for row in inventory)
        print(f"Prepared {count} {split} trips", flush=True)
        if count == 0:
            raise ValueError(f"No trips in {root / split}")
    with (results / "dataset_inventory.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(inventory[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory)
    summaries = {}
    for split in ("train", "student_test"):
        rows = [row for row in inventory if row["split"] == split]
        totals = {key: sum(row[key] for row in rows) for key in
                  ("observations", "aligned", "candidate_hits", "nearest_fallback",
                   "nonfinite", "reference_breaks", "gaps_over_30s", "one_second_steps")}
        summaries[split] = {"trips": len(rows), **totals,
                            "max_gap_s": max(row["max_gap_s"] for row in rows),
                            "candidate_miss_rate": 1 - totals["candidate_hits"] / totals["aligned"]}
    (results / "dataset_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Student folder"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/features"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    prepare(args.data, args.cache, args.results)
