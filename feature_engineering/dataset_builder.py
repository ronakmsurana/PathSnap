"""
Dataset Builder for PathSnap
Extracts candidate features and ground-truth alignments for all trips in Train and Student-Test partitions.
Saves processed NumPy datasets and metadata to `data/processed/` with GroupKFold trip IDs.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

from data_processing.loader import load_trip, list_trip_ids
from feature_engineering.candidate_generator import SpatialCandidateGenerator
from feature_engineering.extractor import CandidateFeatureExtractor, FEATURE_COLUMNS


def build_partition_dataset(
    partition_name: str,
    partition_dir: str,
    output_dir: str = "data/processed",
    search_radius_meters: float = 45.0,
    max_trips: int = None
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    generator = SpatialCandidateGenerator(search_radius_meters=search_radius_meters)
    extractor = CandidateFeatureExtractor()

    trip_ids = list_trip_ids(partition_dir)
    if max_trips is not None:
        trip_ids = trip_ids[:max_trips]

    print(f"--- Building dataset for {partition_name} ({len(trip_ids)} trips) ---")

    all_X = []
    all_y = []
    all_groups = []
    all_point_idx = []
    all_arc_ids = []
    all_ranks = []
    all_times = []

    trip_stats = []
    t_start = time.time()

    for idx, tid in enumerate(trip_ids):
        t0 = time.time()
        trip = load_trip(partition_dir, tid, load_route=True)
        cands_list, stats = generator.generate_trip_candidates(trip, align_route=True)
        X_trip, y_trip, meta_trip = extractor.extract_features_for_trip(trip, cands_list, include_label=True)

        all_X.append(X_trip)
        all_y.append(y_trip)
        all_groups.extend([tid] * len(y_trip))
        all_point_idx.extend([m["point_idx"] for m in meta_trip])
        all_arc_ids.extend([m["arc_id"] for m in meta_trip])
        all_ranks.extend([m["rank"] for m in meta_trip])
        all_times.extend([m["gps_time"] for m in meta_trip])

        trip_stats.append({
            "partition": partition_name,
            "trip_id": tid,
            "observations": stats["total_observations"],
            "candidates_total": stats["total_candidates"],
            "avg_candidates_per_obs": stats["avg_candidates_per_obs"],
            "miss_count": stats["candidate_misses"],
            "miss_rate_pct": stats["candidate_miss_rate"] * 100.0,
            "extract_time_sec": time.time() - t0
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_ids):
            print(f"Processed {idx + 1}/{len(trip_ids)} trips... (Elapsed: {time.time() - t_start:.1f}s)")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    groups_all = np.array(all_groups)
    point_idx_all = np.array(all_point_idx, dtype=np.int32)
    arc_ids_all = np.array(all_arc_ids, dtype=np.int32)
    ranks_all = np.array(all_ranks, dtype=np.int16)
    times_all = np.array(all_times, dtype=np.float64)

    save_path = os.path.join(output_dir, f"{partition_name}_dataset.npz")
    np.savez_compressed(
        save_path,
        X=X_all,
        y=y_all,
        groups=groups_all,
        point_indices=point_idx_all,
        arc_ids=arc_ids_all,
        ranks=ranks_all,
        gps_times=times_all,
        feature_names=np.array(FEATURE_COLUMNS)
    )

    df_stats = pd.DataFrame(trip_stats)
    stats_csv = os.path.join("results", f"candidate_stats_{partition_name}.csv")
    df_stats.to_csv(stats_csv, index=False)

    print(f"Saved {partition_name} dataset to {save_path}:")
    print(f"  Total samples (candidates): {len(y_all):,}")
    print(f"  Positive labels: {y_all.sum():,} ({y_all.mean() * 100:.2f}%)")
    print(f"  Overall Candidate Miss Rate: {df_stats['miss_count'].sum() / df_stats['observations'].sum() * 100:.2f}%")
    print(f"  Candidate stats saved to {stats_csv}")

    return {
        "X": X_all,
        "y": y_all,
        "groups": groups_all,
        "stats": df_stats
    }


def main():
    root = "../Student folder"
    build_partition_dataset("train", os.path.join(root, "train"))
    build_partition_dataset("student_test", os.path.join(root, "student_test"))


if __name__ == "__main__":
    main()
