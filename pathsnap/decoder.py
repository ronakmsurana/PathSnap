"""Directed routing and Viterbi decoding over candidate road segments."""
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from numba import njit

from .data import GAP_S, motion


@dataclass
class Graph:
    offsets: np.ndarray
    edge_ids: np.ndarray
    destinations: np.ndarray
    lengths: np.ndarray


def build_graph(trip):
    sources, destinations = trip.arcs.T
    order = np.argsort(sources, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(sources, minlength=len(trip.nodes)))]
    return Graph(offsets, order, destinations, trip.length)


@njit(cache=True)
def route_distances(arcs, positions, steps, endpoints, lengths, offsets, edge_ids):
    """Compute pairwise transition lengths within a time-dependent search budget."""
    rows, width = arcs.shape
    output = np.full((rows, width, width), np.inf, dtype=np.float32)
    visited = np.zeros(len(offsets) - 1, dtype=np.int64)
    distance = np.empty(len(visited))
    token = 0
    for t in range(1, rows):
        if steps[t] > GAP_S:
            continue
        budget = 32.0 + 48.0 * steps[t]
        for i in range(width):
            source_arc = arcs[t - 1, i]
            if source_arc < 0:
                continue
            source = endpoints[source_arc, 1]
            tail = lengths[source_arc] - positions[t - 1, i]
            token += 1
            distance[source] = 0.0
            visited[source] = token
            heap = [(0.0, np.int32(source))]
            while len(heap):
                dist, node = heappop(heap)
                if dist > distance[node]:
                    continue
                for pointer in range(offsets[node], offsets[node + 1]):
                    edge = edge_ids[pointer]
                    target = endpoints[edge, 1]
                    new_distance = dist + lengths[edge]
                    if tail + new_distance <= budget and (
                        visited[target] != token or new_distance < distance[target]
                    ):
                        distance[target] = new_distance
                        visited[target] = token
                        heappush(heap, (new_distance, np.int32(target)))
            for j in range(width):
                target_arc = arcs[t, j]
                if target_arc < 0:
                    continue
                target = endpoints[target_arc, 0]
                delta = positions[t, j] - positions[t - 1, i]
                # Small backward shifts on a single arc can be GPS jitter.
                if source_arc == target_arc and delta >= -12.0:
                    value = max(0.0, delta)
                elif visited[target] == token:
                    value = tail + distance[target] + positions[t, j]
                else:
                    continue
                if value <= budget:
                    output[t, i, j] = value
    return output


def transition_lattice(trip, candidates, graph):
    steps, displacement, _, _ = motion(trip)
    positions = candidates.fraction * trip.length[np.maximum(candidates.arc, 0)]
    distances = route_distances(candidates.arc, positions, steps, trip.arcs,
                                trip.length, graph.offsets, graph.edge_ids)
    return distances, steps, displacement


@njit(cache=True)
def viterbi(emissions, distances, steps, displacement, beta, weight):
    rows, width = emissions.shape
    parents = np.full((rows, width), -1, dtype=np.int32)
    end_best = np.empty(rows, dtype=np.int32)
    previous = emissions[0].copy()
    end_best[0] = np.argmax(previous)
    for t in range(1, rows):
        current = np.full(width, -np.inf)
        if steps[t] <= GAP_S:
            for j in range(width):
                if not np.isfinite(emissions[t, j]):
                    continue
                for i in range(width):
                    length = distances[t, i, j]
                    if not np.isfinite(length) or not np.isfinite(previous[i]):
                        continue
                    mismatch = abs(length - displacement[t])
                    score = previous[i] - weight * mismatch / (beta + 3.0 * steps[t]) + emissions[t, j]
                    if score > current[j]:
                        current[j] = score
                        parents[t, j] = i
        if not np.any(np.isfinite(current)):
            current = emissions[t].copy()
        end_best[t] = np.argmax(current)
        previous = current - current[end_best[t]]
    slots = np.empty(rows, dtype=np.int32)
    slot = end_best[-1]
    for t in range(rows - 1, -1, -1):
        slots[t] = slot
        if t:
            slot = parents[t, slot] if parents[t, slot] >= 0 else end_best[t - 1]
    return slots


def decode(probabilities, distances, steps, displacement, beta, weight):
    valid = probabilities >= 0
    scores = np.where(valid, np.maximum(probabilities, 1e-6), 0.0)
    emissions = np.full(scores.shape, -np.inf)
    np.log(scores / scores.sum(axis=1, keepdims=True), out=emissions, where=valid)
    return viterbi(emissions, distances, steps, displacement, beta, weight)


def shortest_bridge(graph, source, target, budget):
    best = {source: 0.0}
    parent = {}
    heap = [(0.0, int(source))]
    while heap:
        distance, node = heappop(heap)
        if distance > best[node]:
            continue
        if node == target:
            edges = []
            while node != source:
                node, edge = parent[node]
                edges.append(int(edge))
            return edges[::-1]
        for pointer in range(graph.offsets[node], graph.offsets[node + 1]):
            edge = graph.edge_ids[pointer]
            dest = int(graph.destinations[edge])
            total = distance + graph.lengths[edge]
            if total <= budget and total < best.get(dest, np.inf):
                best[dest] = total
                parent[dest] = (node, edge)
                heappush(heap, (total, dest))
    raise RuntimeError("A previously viable transition could not be reconstructed")


def reconstruct(trip, candidates, slots, distances, steps, graph):
    row = np.arange(len(slots))
    chosen = candidates.arc[row, slots]
    fractions = candidates.fraction[row, slots]
    connected = np.isfinite(distances[row[1:], slots[:-1], slots[1:]])
    runs = [[int(chosen[0])]]
    for t, (left, right, joins) in enumerate(zip(chosen[:-1], chosen[1:], connected), 1):
        if not joins:
            runs.append([int(right)])
        elif left != right or (fractions[t] - fractions[t - 1]) * trip.length[left] < -12.0:
            bridge = shortest_bridge(graph, trip.arcs[left, 1], trip.arcs[right, 0],
                                     33.0 + 48.0 * steps[t])
            runs[-1].extend([*bridge, int(right)])
    return chosen, runs, connected


@njit(cache=True)
def lcs_length(left, right):
    """Count matching arc IDs that occur in the same order, including loop revisits."""
    previous = np.zeros(len(right) + 1, dtype=np.int32)
    current = previous.copy()
    for a in left:
        for j in range(len(right)):
            if a == right[j]:
                current[j + 1] = previous[j] + 1
            else:
                current[j + 1] = max(previous[j + 1], current[j])
        previous, current = current, previous
    return previous[-1]
