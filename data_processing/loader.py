"""
PathSnap Dataset Loader
Handles loading and basic structural validation of trip files:
- .track: longitude, latitude, timestamp
- .nodes: longitude, latitude
- .arcs:  from_node_id, to_node_id
- .route: sequence of ground-truth traversed arc_ids
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TripData:
    trip_id: str
    track: np.ndarray        # shape: (N, 3) -> [lon, lat, timestamp]
    nodes: np.ndarray        # shape: (V, 2) -> [lon, lat]
    arcs: np.ndarray         # shape: (E, 2) -> [from_node, to_node]
    route: Optional[np.ndarray] = None  # shape: (R,) -> sequence of arc_ids


def load_trip(trip_dir: str, trip_id: str, load_route: bool = True) -> TripData:
    """Load all 4 files for a single trip."""
    track_path = os.path.join(trip_dir, trip_id, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, trip_id, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, trip_id, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, trip_id, f"{trip_id}.route")

    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not os.path.exists(arcs_path):
        raise FileNotFoundError(f"Arcs file not found: {arcs_path}")

    track = np.loadtxt(track_path, dtype=float)
    if track.ndim == 1 and len(track) == 3:
        track = track.reshape(1, 3)

    nodes = np.loadtxt(nodes_path, dtype=float)
    if nodes.ndim == 1 and len(nodes) == 2:
        nodes = nodes.reshape(1, 2)

    arcs = np.loadtxt(arcs_path, dtype=int)
    if arcs.ndim == 1 and len(arcs) == 2:
        arcs = arcs.reshape(1, 2)

    route = None
    if load_route and os.path.exists(route_path):
        try:
            route = np.loadtxt(route_path, dtype=int)
            if route.ndim == 0:
                route = np.array([int(route)], dtype=int)
        except Exception:
            route = None

    return TripData(
        trip_id=trip_id,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route
    )


def list_trip_ids(partition_dir: str) -> List[str]:
    """List all valid trip IDs in a partition directory sorted."""
    if not os.path.exists(partition_dir):
        return []
    trips = []
    for item in os.listdir(partition_dir):
        item_path = os.path.join(partition_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            trips.append(item)
    return sorted(trips)


def load_partition(partition_dir: str, load_route: bool = True) -> Dict[str, TripData]:
    """Load all trips in a given partition directory."""
    trip_ids = list_trip_ids(partition_dir)
    data = {}
    for tid in trip_ids:
        data[tid] = load_trip(partition_dir, tid, load_route=load_route)
    return data
