"""Derive approximate observation labels from a chronological reference route."""
import numpy as np
from numba import njit

from .data import GAP_S, motion


@njit(cache=True)
def align_positions(gps, start, vector, length, position, heading, steps, displacement):
    width = min(12, len(start))
    indices = np.zeros((len(gps), width), dtype=np.int32)
    offsets = np.zeros((len(gps), width))
    distances = np.empty((len(gps), width))
    costs = np.empty((len(gps), width))
    previous = np.zeros(width)
    parents = np.zeros((len(gps), width), dtype=np.int32)
    for t in range(len(gps)):
        distances[t] = np.inf
        for j in range(len(start)):
            dx = gps[t, 0] - start[j, 0]
            dy = gps[t, 1] - start[j, 1]
            fraction = 0.0
            if length[j] > 0:
                fraction = min(1.0, max(0.0, (dx * vector[j, 0] + dy * vector[j, 1]) / length[j]**2))
            dist = (dx - fraction * vector[j, 0])**2 + (dy - fraction * vector[j, 1])**2
            if dist >= distances[t, width - 1]:
                continue
            slot = width - 1
            while slot > 0 and dist < distances[t, slot - 1]:
                distances[t, slot] = distances[t, slot - 1]
                indices[t, slot] = indices[t, slot - 1]
                offsets[t, slot] = offsets[t, slot - 1]
                slot -= 1
            distances[t, slot] = dist
            indices[t, slot] = j
            offsets[t, slot] = position[j] + fraction * length[j]
        for k in range(width):
            j = indices[t, k]
            agreement = 0.0
            if length[j] > 0:
                agreement = np.dot(vector[j], heading[t]) / length[j]
            emission = np.sqrt(distances[t, k]) / 10.0 + 0.7 * (1.0 - agreement)
            if t == 0:
                costs[t, k] = emission
                continue
            best = np.inf
            for p in range(width):
                advance = offsets[t, k] - offsets[t - 1, p]
                backtrack = max(0.0, -advance - 8.0) / 8.0
                if steps[t] > GAP_S:
                    transition = min(8.0, max(0.0, advance) / 500.0) + backtrack
                else:
                    mismatch = abs(max(0.0, advance) - displacement[t])
                    transition = min(20.0, mismatch / (10.0 + 3.0 * steps[t])) + min(backtrack, 30.0)
                    # Prefer progressing through the route at shared endpoints.
                    if indices[t, k] < indices[t - 1, p]:
                        transition += 0.25
                score = previous[p] + transition
                if score < best:
                    best = score
                    parents[t, k] = p
            costs[t, k] = emission + best
        previous = costs[t] - np.min(costs[t])
    slot = np.argmin(previous)
    labels = np.empty(len(gps), dtype=np.int32)
    error = np.empty(len(gps))
    for t in range(len(gps) - 1, -1, -1):
        labels[t] = indices[t, slot]
        error[t] = np.sqrt(distances[t, slot])
        slot = parents[t, slot]
    return labels, error


def observation_labels(trip, route):
    length = np.ascontiguousarray(trip.length[route])
    jumps = np.linalg.norm(trip.start[route][1:] - trip.xy[trip.arcs[route[:-1], 1]], axis=1)
    positions = np.r_[0.0, np.cumsum(length[:-1] + jumps)]
    steps, displacement, _, headings = motion(trip)
    labels, error = align_positions(trip.gps, trip.start[route], trip.vector[route],
                                    length, positions, headings[0][0], steps, displacement)
    result = route[labels].copy()
    result[error > 100.0] = -1
    return result, error
