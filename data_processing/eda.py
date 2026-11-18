"""
PathSnap Exploratory Data Analysis (EDA) Module
Calculates comprehensive dataset statistics, data-quality metrics,
and road network graph characteristics for Train and Student-Test partitions.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from data_processing.loader import load_trip, list_trip_ids


def haversine_distance_meters(lon1, lat1, lon2, lat2):
    """Compute Haversine distance between two coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def run_eda(dataset_root: str, output_dir: str = "results") -> pd.DataFrame:
    """
    Performs full exploratory data analysis across training and student_test partitions.
    Saves summary CSVs and visual plots to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    partitions = {
        "train": os.path.join(dataset_root, "train"),
        "student_test": os.path.join(dataset_root, "student_test")
    }

    trip_records = []

    for part_name, part_path in partitions.items():
        if not os.path.exists(part_path):
            print(f"Warning: Partition path not found: {part_path}")
            continue

        trip_ids = list_trip_ids(part_path)
        print(f"Analyzing {part_name}: {len(trip_ids)} trips found.")

        for tid in trip_ids:
            trip = load_trip(part_path, tid, load_route=True)

            # GPS Track statistics
            track = trip.track
            n_points = len(track)
            lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]

            # Missing or invalid checks
            nan_count = np.isnan(track).sum()
            inf_count = np.isinf(track).sum()

            # Time intervals
            dt = np.diff(ts)
            dt_mean = float(np.mean(dt)) if len(dt) > 0 else 0.0
            dt_median = float(np.median(dt)) if len(dt) > 0 else 0.0
            dt_max = float(np.max(dt)) if len(dt) > 0 else 0.0
            dt_is_1s = float(np.mean(np.isclose(dt, 1.0, atol=0.05))) if len(dt) > 0 else 0.0
            large_gaps = int(np.sum(dt > 3.0))

            # Spatial extent
            min_lon, max_lon = float(lon.min()), float(lon.max())
            min_lat, max_lat = float(lat.min()), float(lat.max())

            # Distance and speeds
            dists = haversine_distance_meters(lon[:-1], lat[:-1], lon[1:], lat[1:])
            total_dist_m = float(np.sum(dists))
            speeds_mps = np.where(dt > 0, dists / np.maximum(dt, 0.001), 0.0)
            avg_speed_kmh = float(np.mean(speeds_mps) * 3.6) if len(speeds_mps) > 0 else 0.0
            max_speed_kmh = float(np.max(speeds_mps) * 3.6) if len(speeds_mps) > 0 else 0.0

            # Road Network statistics
            n_nodes = len(trip.nodes)
            n_arcs = len(trip.arcs)

            # Route statistics
            n_route_arcs = len(trip.route) if trip.route is not None else 0

            trip_records.append({
                "partition": part_name,
                "trip_id": tid,
                "gps_points": n_points,
                "duration_sec": float(ts[-1] - ts[0]) if n_points > 1 else 0.0,
                "dt_mean": dt_mean,
                "dt_median": dt_median,
                "dt_max": dt_max,
                "pct_1s_sampling": dt_is_1s * 100.0,
                "large_gaps_count": large_gaps,
                "total_dist_meters": total_dist_m,
                "avg_speed_kmh": avg_speed_kmh,
                "max_speed_kmh": max_speed_kmh,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "nodes_count": n_nodes,
                "arcs_count": n_arcs,
                "route_arcs_count": n_route_arcs,
                "nan_values": nan_count,
                "inf_values": inf_count
            })

    df_summary = pd.DataFrame(trip_records)
    csv_path = os.path.join(output_dir, "eda_trip_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved EDA trip summary to {csv_path}")

    # Generate EDA plots
    sns.set_theme(style="whitegrid")
    
    # Plot 1: GPS Points per Trip by Partition
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_summary, x="gps_points", hue="partition", bins=25, kde=True, palette="tab10")
    plt.title("Distribution of GPS Points per Trip")
    plt.xlabel("Number of GPS Observations")
    plt.ylabel("Trip Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "eda_gps_points_distribution.png"), dpi=200)
    plt.close()

    # Plot 2: Average Speed Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_summary, x="avg_speed_kmh", hue="partition", bins=25, kde=True, palette="tab10")
    plt.title("Distribution of Average Vehicle Speeds (km/h)")
    plt.xlabel("Average Speed (km/h)")
    plt.ylabel("Trip Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "eda_speed_distribution.png"), dpi=200)
    plt.close()

    # Plot 3: Network Size (Nodes vs Arcs)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_summary, x="nodes_count", y="arcs_count", hue="partition", s=80, alpha=0.8)
    plt.title("Road Network Scale per Trip (Nodes vs. Arcs)")
    plt.xlabel("Node Count")
    plt.ylabel("Arc Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "eda_network_scale.png"), dpi=200)
    plt.close()

    print(f"Generated EDA plots in {os.path.join(output_dir, 'plots')}")
    return df_summary


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "../Student folder"
    run_eda(root)
