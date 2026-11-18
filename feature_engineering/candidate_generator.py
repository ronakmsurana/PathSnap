"""
Spatial Candidate Generation for Map Matching
Uses R-tree spatial indexing (Shapely STRtree) to rapidly query candidate road segments
for noisy GPS observations and performs ground-truth route sequence alignment.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import shapely
from shapely.geometry import LineString, Point

from data_processing.loader import TripData
from feature_engineering.projection import LocalMetricProjector


@dataclass
class ObservationCandidates:
    point_idx: int
    gps_time: float
    gps_lon: float
    gps_lat: float
    gps_x: float
    gps_y: float
    candidate_arc_ids: List[int]
    true_arc_id: Optional[int] = None
    is_miss: bool = False


class SpatialCandidateGenerator:
    def __init__(self, search_radius_meters: float = 45.0, fallback_k: int = 3):
        self.search_radius = float(search_radius_meters)
        self.fallback_k = int(fallback_k)

    def generate_trip_candidates(
        self,
        trip: TripData,
        align_route: bool = True
    ) -> Tuple[List[ObservationCandidates], Dict[str, float]]:
        """
        Generates candidate road segments for each GPS observation in a trip.
        Returns:
            list of ObservationCandidates
            dictionary with candidate generation statistics (miss rate, avg candidates, etc.)
        """
        track = trip.track
        nodes = trip.nodes
        arcs = trip.arcs
        route = trip.route

        # Set up local metric projection
        ref_lon = float(np.mean(track[:, 0]))
        ref_lat = float(np.mean(track[:, 1]))
        projector = LocalMetricProjector(ref_lon, ref_lat)

        track_xy = projector.forward(track[:, 0], track[:, 1])
        node_xy = projector.forward(nodes[:, 0], nodes[:, 1])

        # Construct LineString geometries for all road segments (arcs)
        arc_lines = []
        for u, v in arcs:
            arc_lines.append(LineString([node_xy[u], node_xy[v]]))

        # Build spatial R-tree index over road network segments
        tree = shapely.STRtree(arc_lines)

        # Align ground truth route if available and requested
        true_arc_per_obs = [None] * len(track)
        if align_route and route is not None and len(route) > 0:
            true_arc_per_obs = self._align_route_to_track(
                track_xy, route, arcs, node_xy
            )

        # Generate candidates for each GPS point
        obs_candidates_list = []
        total_misses = 0
        total_candidates = 0

        for t in range(len(track)):
            pt = Point(track_xy[t, 0], track_xy[t, 1])
            query_buf = pt.buffer(self.search_radius)
            cands = list(tree.query(query_buf))

            if len(cands) == 0:
                # Fallback: query nearest road segments
                nearest_indices = tree.nearest(pt)
                cands = [int(nearest_indices)] if np.ndim(nearest_indices) == 0 else list(nearest_indices)

            # Sort candidate arcs by exact distance to point
            cand_dists = [pt.distance(arc_lines[c]) for c in cands]
            sorted_indices = np.argsort(cand_dists)
            sorted_cands = [int(cands[i]) for i in sorted_indices]

            true_arc = true_arc_per_obs[t]
            is_miss = False
            if true_arc is not None:
                if true_arc not in sorted_cands:
                    is_miss = True
                    total_misses += 1

            total_candidates += len(sorted_cands)

            obs_candidates_list.append(ObservationCandidates(
                point_idx=t,
                gps_time=track[t, 2],
                gps_lon=track[t, 0],
                gps_lat=track[t, 1],
                gps_x=track_xy[t, 0],
                gps_y=track_xy[t, 1],
                candidate_arc_ids=sorted_cands,
                true_arc_id=true_arc,
                is_miss=is_miss
            ))

        miss_rate = (total_misses / len(track)) if len(track) > 0 else 0.0
        avg_cands = (total_candidates / len(track)) if len(track) > 0 else 0.0

        stats = {
            "total_observations": len(track),
            "total_candidates": total_candidates,
            "avg_candidates_per_obs": avg_cands,
            "candidate_misses": total_misses,
            "candidate_miss_rate": miss_rate
        }

        return obs_candidates_list, stats

    def _align_route_to_track(
        self,
        track_xy: np.ndarray,
        route: np.ndarray,
        arcs: np.ndarray,
        node_xy: np.ndarray
    ) -> List[int]:
        """
        Monotonically aligns the ground-truth sequence of traversed road segments (.route)
        to the continuous sequence of GPS observations (.track).
        """
        route_lines = []
        for r in route:
            u, v = arcs[r]
            route_lines.append(LineString([node_xy[u], node_xy[v]]))

        n_pts = len(track_xy)
        n_route = len(route)
        matched_arcs = [None] * n_pts

        curr_r_idx = 0
        lookahead = 25
        lookbehind = 2

        for t in range(n_pts):
            pt = Point(track_xy[t, 0], track_xy[t, 1])
            low = max(0, curr_r_idx - lookbehind)
            high = min(n_route, curr_r_idx + lookahead)
            best_idx = curr_r_idx
            best_dist = float("inf")

            for r_idx in range(low, high):
                d = pt.distance(route_lines[r_idx])
                if d < best_dist:
                    best_dist = d
                    best_idx = r_idx

            curr_r_idx = best_idx
            matched_arcs[t] = int(route[curr_r_idx])

        return matched_arcs
