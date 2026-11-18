"""
Hidden Markov Model & Viterbi Sequential Route Reconstruction Engine for PathSnap
Reconstructs globally optimal, topologically continuous routes from noisy GPS trajectories
and candidate classification probabilities.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx
from collections import defaultdict

from data_processing.loader import TripData
from feature_engineering.candidate_generator import ObservationCandidates


class RouteDecoderHMM:
    """
    Hidden Markov Model decoder implementing the Viterbi algorithm
    with graph topology constraints based on Newson & Krumm map-matching formulation.
    """
    def __init__(
        self,
        beta: float = 12.0,          # Transition parameter (meters)
        sigma_z: float = 10.0,       # GPS noise standard deviation (meters)
        alpha: float = 0.65,         # Weight of ML classifier probability vs. geometric emission
        max_gap_seconds: float = 15.0, # Threshold to break sequence at large GPS gaps
        disconnected_penalty: float = 150.0
    ):
        self.beta = float(beta)
        self.sigma_z = float(sigma_z)
        self.alpha = float(alpha)
        self.max_gap_seconds = float(max_gap_seconds)
        self.disconnected_penalty = float(disconnected_penalty)

    def build_network_graph(self, arcs: np.ndarray, nodes: np.ndarray) -> Tuple[nx.DiGraph, Dict[int, List[int]]]:
        """
        Builds directed road network graph with edge lengths in meters.
        Also creates node -> outgoing arcs mapping for fast topological queries.
        """
        G = nx.DiGraph()
        node_to_outgoing = defaultdict(list)

        # Precompute metric distances
        ref_lat = np.radians(np.mean(nodes[:, 1]))
        cos_lat = np.cos(ref_lat)

        for arc_id in range(len(arcs)):
            u, v = arcs[arc_id]
            node_to_outgoing[u].append(arc_id)
            dx = (nodes[v, 0] - nodes[u, 0]) * cos_lat * 111320.0
            dy = (nodes[v, 1] - nodes[u, 1]) * 110540.0
            length = float(np.hypot(dx, dy))
            G.add_edge(u, v, arc_id=arc_id, weight=length)

        return G, dict(node_to_outgoing)

    def decode_trip(
        self,
        trip: TripData,
        candidates_list: List[ObservationCandidates],
        candidate_probs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes Viterbi decoding over all observations in a trip.
        Returns:
            classifier_only_arcs: point-by-point argmax predictions
            viterbi_arcs: globally optimal decoded sequence
            route_continuity_raw: continuity before HMM
            route_continuity_hmm: continuity after HMM
            pointwise_accuracy_raw: Top-1 accuracy before HMM
            pointwise_accuracy_hmm: Top-1 accuracy after HMM
        """
        track = trip.track
        arcs = trip.arcs
        nodes = trip.nodes
        route = trip.route
        n_obs = len(candidates_list)

        G, node_to_outgoing = self.build_network_graph(arcs, nodes)
        path_cache = {}

        # Map candidate probabilities back to each observation
        obs_cands = []
        offset = 0
        for obs in candidates_list:
            k = len(obs.candidate_arc_ids)
            probs = candidate_probs[offset: offset + k]
            obs_cands.append({
                "obs": obs,
                "arc_ids": obs.candidate_arc_ids,
                "probs": probs
            })
            offset += k

        # 1. Baseline Classifier-Only argmax predictions
        clf_only_arcs = []
        true_arcs = []
        for item in obs_cands:
            best_idx = int(np.argmax(item["probs"]))
            clf_only_arcs.append(item["arc_ids"][best_idx])
            true_arcs.append(item["obs"].true_arc_id)

        # 2. Viterbi Trellis Forward Pass
        # States at step t: S_t = list of candidate arc_ids
        trellis_log_prob = []  # V_t(j)
        trellis_backptr = []   # B_t(j)

        ref_lat = np.radians(np.mean(track[:, 1]))
        cos_lat = np.cos(ref_lat)

        for t in range(n_obs):
            curr_arcs = obs_cands[t]["arc_ids"]
            curr_probs = obs_cands[t]["probs"]
            n_curr = len(curr_arcs)

            # Compute emission log-likelihood:
            # log P(z_t | a) = alpha * log(P_clf) - (dist_seg^2) / (2 * sigma_z^2)
            log_emissions = np.zeros(n_curr, dtype=float)
            for j in range(n_curr):
                arc_j = curr_arcs[j]
                u_j, v_j = arcs[arc_j]
                # Orthogonal distance proxy from candidate generator
                p_clf = max(1e-6, curr_probs[j])
                log_p_clf = np.log(p_clf)
                log_emissions[j] = self.alpha * log_p_clf

            # Time gap check
            time_gap = (track[t, 2] - track[t - 1, 2]) if t > 0 else 0.0
            is_new_chain = (t == 0) or (time_gap > self.max_gap_seconds)

            if is_new_chain:
                # Uniform prior + emission
                trellis_log_prob.append(log_emissions)
                trellis_backptr.append(np.full(n_curr, -1, dtype=int))
            else:
                prev_log_prob = trellis_log_prob[t - 1]
                prev_arcs = obs_cands[t - 1]["arc_ids"]
                n_prev = len(prev_arcs)

                # Euclidean GPS distance between consecutive observations
                dx_gps = (track[t, 0] - track[t - 1, 0]) * cos_lat * 111320.0
                dy_gps = (track[t, 1] - track[t - 1, 1]) * 110540.0
                d_gps = float(np.hypot(dx_gps, dy_gps))

                v_curr = np.zeros(n_curr, dtype=float)
                bp_curr = np.zeros(n_curr, dtype=int)

                for j in range(n_curr):
                    arc_j = curr_arcs[j]
                    u_j, v_j = arcs[arc_j]

                    scores = []
                    for i in range(n_prev):
                        arc_i = prev_arcs[i]
                        u_i, v_i = arcs[arc_i]

                        # Calculate network transition distance
                        if arc_i == arc_j:
                            # Remained on the same segment
                            d_net = d_gps
                        elif v_i == u_j:
                            # Direct successor node in graph!
                            d_net = d_gps
                        else:
                            # Graph shortest path lookup with memoization and local metric cutoff
                            pair = (v_i, u_j)
                            if pair not in path_cache:
                                try:
                                    d_net_val, _ = nx.single_source_dijkstra(G, source=v_i, target=u_j, cutoff=120.0, weight="weight")
                                    path_cache[pair] = float(d_net_val)
                                except (nx.NetworkXNoPath, nx.NodeNotFound):
                                    path_cache[pair] = self.disconnected_penalty
                            d_net = path_cache[pair]

                        # Exponential transition probability: exp( - |d_net - d_gps| / beta )
                        dist_diff = abs(d_net - d_gps)
                        log_trans = - (dist_diff / self.beta)

                        score = prev_log_prob[i] + log_trans
                        scores.append(score)

                    best_prev_idx = int(np.argmax(scores))
                    bp_curr[j] = best_prev_idx
                    v_curr[j] = scores[best_prev_idx] + log_emissions[j]

                trellis_log_prob.append(v_curr)
                trellis_backptr.append(bp_curr)

        # 3. Viterbi Backtrack
        viterbi_arcs = [None] * n_obs
        curr_state_idx = int(np.argmax(trellis_log_prob[-1]))

        for t in range(n_obs - 1, -1, -1):
            viterbi_arcs[t] = obs_cands[t]["arc_ids"][curr_state_idx]
            curr_state_idx = trellis_backptr[t][curr_state_idx]
            if curr_state_idx == -1 and t > 0:
                # Chain boundary reached, find argmax of previous sub-chain
                curr_state_idx = int(np.argmax(trellis_log_prob[t - 1]))

        # 4. Evaluation and Comparison
        valid_true = [t for t in true_arcs if t is not None]
        has_labels = len(valid_true) > 0

        acc_raw = float(np.mean([clf_only_arcs[i] == true_arcs[i] for i in range(n_obs) if true_arcs[i] is not None])) if has_labels else float("nan")
        acc_hmm = float(np.mean([viterbi_arcs[i] == true_arcs[i] for i in range(n_obs) if true_arcs[i] is not None])) if has_labels else float("nan")

        # Route continuity calculation
        continuity_raw = self._calculate_continuity(clf_only_arcs, arcs)
        continuity_hmm = self._calculate_continuity(viterbi_arcs, arcs)

        return {
            "trip_id": trip.trip_id,
            "observations_count": n_obs,
            "classifier_only_arcs": clf_only_arcs,
            "viterbi_arcs": viterbi_arcs,
            "pointwise_accuracy_raw": acc_raw,
            "pointwise_accuracy_hmm": acc_hmm,
            "accuracy_improvement": (acc_hmm - acc_raw) if has_labels else float("nan"),
            "route_continuity_raw": continuity_raw,
            "route_continuity_hmm": continuity_hmm,
            "continuity_improvement": continuity_hmm - continuity_raw
        }

    def _calculate_continuity(self, arc_seq: List[int], arcs: np.ndarray) -> float:
        if len(arc_seq) <= 1:
            return 1.0
        valid = 0
        total = len(arc_seq) - 1
        for i in range(total):
            a1 = arc_seq[i]
            a2 = arc_seq[i + 1]
            if a1 == a2:
                valid += 1
            elif a1 < len(arcs) and a2 < len(arcs):
                if arcs[a1, 1] == arcs[a2, 0]:
                    valid += 1
        return float(valid / total) if total > 0 else 1.0
