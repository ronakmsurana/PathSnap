"""Per-trip input, geometry and candidate features. Reference routes load separately."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from shapely import STRtree, linestrings, points


FEATURES = (
    "distance_m", "distance_above_nearest_m", "heading_agreement",
    "short_heading_agreement", "speed_mps", "arc_length_m", "projection_fraction",
    "start_overrun_m", "end_overrun_m", "nearby_road_count", "elapsed_s",
    "heading_support_m",
)
CANDIDATES = 10
RADIUS_M = 65.0
GAP_S = 30.0


def read(directory, ext, columns, dtype):
    directory = Path(directory)
    file = directory / f"{directory.name}.{ext}"
    array = np.loadtxt(file, dtype=dtype, ndmin=2)
    if array.shape[1] != columns or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"Expected nonempty finite {columns}-column data in {file}")
    return array


@dataclass
class Trip:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    gps: np.ndarray
    xy: np.ndarray
    start: np.ndarray
    vector: np.ndarray
    length: np.ndarray


def load_trip(directory):
    track = read(directory, "track", 3, float)
    nodes = read(directory, "nodes", 2, float)
    arcs = read(directory, "arcs", 2, np.int32)
    if np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError(f"Timestamps must increase in {directory}")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Invalid node ID in {directory}")
    for coords in (track[:, :2], nodes):
        if np.any(np.abs(coords) > [180, 90]):
            raise ValueError(f"Invalid GPS coordinates in {directory}")
    lat = float(np.median(track[:, 1]))
    lon = float(np.median(track[:, 0]))
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def project(coords):
        return np.column_stack(transform.transform(coords[:, 0], coords[:, 1]))

    gps, xy = project(track), project(nodes)
    start = xy[arcs[:, 0]]
    vector = xy[arcs[:, 1]] - start
    return Trip(track, nodes, arcs, gps, xy, start, vector, np.linalg.norm(vector, axis=1))


def load_reference(directory, trip):
    reference = read(directory, "route", 1, np.int32)[:, 0]
    if np.any(reference < 0) or np.any(reference >= len(trip.arcs)):
        raise ValueError(f"Invalid arc ID in {directory}")
    return reference


def motion(trip):
    time = trip.track[:, 2]
    steps = np.diff(time, prepend=time[0])
    displacement = np.linalg.norm(np.diff(trip.gps, axis=0, prepend=trip.gps[:1]), axis=1)
    breaks = np.r_[0, np.flatnonzero(steps > GAP_S), len(time)]
    speed = np.divide(displacement, steps, out=np.zeros_like(steps),
                      where=(steps > 0) & (steps <= GAP_S))
    headings = []
    for seconds in (3.0, 1.0):
        heading = np.zeros_like(trip.gps)
        for start, end in zip(breaks[:-1], breaks[1:]):
            timestamps = time[start:end]
            before = np.maximum(0, timestamps.searchsorted(timestamps - seconds))
            after = np.minimum(end - start - 1,
                               timestamps.searchsorted(timestamps + seconds, side="right") - 1)
            heading[start:end] = trip.gps[start + after] - trip.gps[start + before]
        support = np.linalg.norm(heading, axis=1)
        unit = heading / np.maximum(support[:, None], 3.0)
        headings.append((unit, support))
    return steps, displacement, speed, headings


def project_pairs(trip, observations, arcs):
    lengths = trip.length[arcs]
    offsets = trip.gps[observations] - trip.start[arcs]
    raw_fraction = np.sum(offsets * trip.vector[arcs], axis=1) / np.maximum(lengths**2, 1e-12)
    fraction = np.clip(raw_fraction, 0.0, 1.0)
    distance = np.linalg.norm(offsets - trip.vector[arcs] * fraction[:, None], axis=1)
    return distance, fraction, raw_fraction


@dataclass
class Candidates:
    arc: np.ndarray
    fraction: np.ndarray
    features: np.ndarray
    outside_radius: np.ndarray


def generate_candidates(trip):
    valid = np.flatnonzero(trip.length > 1e-6)
    if not len(valid):
        raise ValueError("Road network has no nonzero road segments")
    tree = STRtree(linestrings(trip.xy[trip.arcs[valid]]))
    gps = points(trip.gps)
    obs, tree_ids = tree.query(gps, predicate="dwithin", distance=RADIUS_M)
    counts = np.bincount(obs, minlength=len(gps))
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        nearest_obs, nearest_arcs = tree.query_nearest(gps[missing])
        obs = np.r_[obs, missing[nearest_obs]]
        tree_ids = np.r_[tree_ids, nearest_arcs]
    arcs = valid[tree_ids]
    distance, fraction, raw_fraction = project_pairs(trip, obs, arcs)
    direction = trip.vector[arcs] / trip.length[arcs, None]
    steps, _, speed, [(heading, support), (short_heading, _)] = motion(trip)
    agreement = (direction * heading[obs]).sum(axis=1)
    short_agreement = (direction * short_heading[obs]).sum(axis=1)

    order = np.lexsort((arcs, -agreement, distance, obs))
    sorted_obs = obs[order]
    new_group = np.r_[True, sorted_obs[1:] != sorted_obs[:-1]]
    starts = np.maximum.accumulate(np.where(new_group, np.arange(len(order)), 0))
    rank = np.arange(len(order)) - starts
    keep = rank < CANDIDATES
    indices = order[keep]
    rows, cols = obs[indices], rank[keep]
    arc = np.full((len(gps), CANDIDATES), -1, dtype=np.int32)
    arc[rows, cols] = arcs[indices]
    projected = np.zeros_like(arc, dtype=np.float32)
    projected[rows, cols] = fraction[indices]
    nearest_distances = distance[order[new_group]]
    length = trip.length[arcs[indices]]
    raw = raw_fraction[indices]
    features = np.zeros((*arc.shape, len(FEATURES)), dtype=np.float32)
    features[rows, cols] = np.column_stack((
        distance[indices], distance[indices] - nearest_distances[rows],
        agreement[indices], short_agreement[indices], speed[rows], length,
        fraction[indices], np.maximum(0, -raw) * length,
        np.maximum(0, raw - 1) * length, counts[rows],
        np.minimum(steps[rows], GAP_S), support[rows],
    ))
    return Candidates(arc, projected, features, counts == 0)
