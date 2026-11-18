"""
Feature Extraction for Map Matching Candidates
Extracts spatial, kinematic, and topological features for (GPS observation, candidate arc) pairs.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from shapely.geometry import LineString, Point

from data_processing.loader import TripData
from feature_engineering.candidate_generator import ObservationCandidates
from feature_engineering.projection import LocalMetricProjector, compute_bearing_deg, angle_difference_deg


FEATURE_COLUMNS = [
    "dist_to_segment",
    "projection_ratio",
    "segment_length",
    "dist_to_start_node",
    "dist_to_end_node",
    "dist_to_midpoint",
    "gps_speed_mps",
    "heading_diff_deg",
    "heading_cos_sim",
    "spatial_likelihood",
    "rank_by_dist",
    "is_closest",
    "speed_length_ratio"
]


class CandidateFeatureExtractor:
    def __init__(self, sigma_meters: float = 12.0):
        self.sigma = float(sigma_meters)

    def extract_features_for_trip(
        self,
        trip: TripData,
        candidates_list: List[ObservationCandidates],
        include_label: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Extracts feature vectors and metadata for all candidate pairs in a trip.
        Returns:
            X: np.ndarray of shape (num_total_candidates, num_features)
            y: np.ndarray of shape (num_total_candidates,) with binary labels (1 = true arc, 0 = distractor)
            metadata: list of dicts with (trip_id, point_idx, arc_id, gps_time) for reconstruction
        """
        track = trip.track
        nodes = trip.nodes
        arcs = trip.arcs

        ref_lon = float(np.mean(track[:, 0]))
        ref_lat = float(np.mean(track[:, 1]))
        projector = LocalMetricProjector(ref_lon, ref_lat)

        track_xy = projector.forward(track[:, 0], track[:, 1])
        node_xy = projector.forward(nodes[:, 0], nodes[:, 1])

        # Precompute arc geometries and orientations
        u_nodes = arcs[:, 0]
        v_nodes = arcs[:, 1]
        A = node_xy[u_nodes]
        B = node_xy[v_nodes]
        AB = B - A
        arc_lengths = np.hypot(AB[:, 0], AB[:, 1])
        arc_lengths_safe = np.maximum(arc_lengths, 1e-4)
        arc_bearings = np.degrees(np.arctan2(AB[:, 0], AB[:, 1])) % 360.0
        arc_midpoints = (A + B) / 2.0

        # Precompute GPS trajectory kinematics
        n_obs = len(track)
        gps_bearings = np.zeros(n_obs, dtype=float)
        gps_speeds = np.zeros(n_obs, dtype=float)

        for t in range(n_obs):
            if t == 0:
                if n_obs > 1:
                    dt = max(0.01, track[1, 2] - track[0, 2])
                    dx = track_xy[1, 0] - track_xy[0, 0]
                    dy = track_xy[1, 1] - track_xy[0, 1]
                    dist = np.hypot(dx, dy)
                    gps_speeds[t] = dist / dt
                    gps_bearings[t] = compute_bearing_deg(dx, dy)
            else:
                dt = max(0.01, track[t, 2] - track[t - 1, 2])
                dx = track_xy[t, 0] - track_xy[t - 1, 0]
                dy = track_xy[t, 1] - track_xy[t - 1, 1]
                dist = np.hypot(dx, dy)
                gps_speeds[t] = dist / dt
                if dist > 0.5:
                    gps_bearings[t] = compute_bearing_deg(dx, dy)
                else:
                    gps_bearings[t] = gps_bearings[t - 1]

        rows = []
        labels = []
        meta_list = []

        for obs in candidates_list:
            t = obs.point_idx
            pt_xy = track_xy[t]
            true_arc = obs.true_arc_id
            v_gps = gps_speeds[t]
            b_gps = gps_bearings[t]

            for rank, arc_id in enumerate(obs.candidate_arc_ids):
                a_pt = A[arc_id]
                b_pt = B[arc_id]
                ab = AB[arc_id]
                l2 = np.dot(ab, ab)
                seg_len = arc_lengths[arc_id]

                # Orthogonal projection factor t_proj
                if l2 > 0:
                    t_proj = max(0.0, min(1.0, float(np.dot(pt_xy - a_pt, ab) / l2)))
                else:
                    t_proj = 0.0

                proj_pt = a_pt + t_proj * ab
                dist_seg = float(np.hypot(pt_xy[0] - proj_pt[0], pt_xy[1] - proj_pt[1]))
                dist_start = float(np.hypot(pt_xy[0] - a_pt[0], pt_xy[1] - a_pt[1]))
                dist_end = float(np.hypot(pt_xy[0] - b_pt[0], pt_xy[1] - b_pt[1]))
                dist_mid = float(np.hypot(pt_xy[0] - arc_midpoints[arc_id, 0], pt_xy[1] - arc_midpoints[arc_id, 1]))

                # Heading difference
                b_arc = arc_bearings[arc_id]
                h_diff = angle_difference_deg(b_gps, b_arc) if v_gps > 1.0 else 0.0
                h_cos = np.cos(np.radians(h_diff))

                # Spatial emission likelihood
                spatial_like = float(np.exp(- (dist_seg ** 2) / (2.0 * (self.sigma ** 2))))

                # Kinematic traversal ratio
                speed_ratio = float(v_gps / arc_lengths_safe[arc_id])

                is_closest = 1.0 if rank == 0 else 0.0

                feature_vec = [
                    dist_seg,
                    t_proj,
                    seg_len,
                    dist_start,
                    dist_end,
                    dist_mid,
                    float(v_gps),
                    float(h_diff),
                    float(h_cos),
                    spatial_like,
                    float(rank),
                    is_closest,
                    speed_ratio
                ]

                rows.append(feature_vec)
                meta_list.append({
                    "trip_id": trip.trip_id,
                    "point_idx": t,
                    "arc_id": arc_id,
                    "rank": rank,
                    "dist_seg": dist_seg,
                    "gps_time": obs.gps_time
                })

                if include_label:
                    label = 1 if (true_arc is not None and arc_id == true_arc) else 0
                    labels.append(label)

        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32) if include_label else np.zeros(len(rows), dtype=np.int32)
        return X, y, meta_list
