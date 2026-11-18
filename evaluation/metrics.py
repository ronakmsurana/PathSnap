"""
Evaluation Metrics for PathSnap
Computes all required metrics defined in Hackathon Guidelines:
- Cross-Validation Macro F1-score
- Student-Test Macro F1-score
- Observation-level Top-1 Accuracy
- Route Continuity
- Log Loss
- Precision, Recall, Confusion Matrix
- Training & Inference Time
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix
)


def compute_observation_top1(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    groups: np.ndarray,
    point_indices: np.ndarray,
    arc_ids: np.ndarray
) -> Tuple[float, List[int], List[int]]:
    """
    Computes Observation-level Top-1 Accuracy:
    For each GPS observation, select the candidate with the highest predicted score
    and compare it with the reference road segment.
    """
    # Create unique observation keys: trip_id + "_" + point_idx
    obs_keys = [f"{g}_{p}" for g, p in zip(groups, point_indices)]
    
    df = pd.DataFrame({
        "obs_key": obs_keys,
        "y_true": y_true,
        "prob": y_probs,
        "arc_id": arc_ids
    })

    # Find the candidate with maximum predicted probability for each observation
    idx_max = df.groupby("obs_key")["prob"].idxmax()
    top_candidates = df.loc[idx_max]

    # For each observation, check if true arc had y_true == 1
    # Top-1 accuracy is whether the selected top candidate is indeed the true arc
    top1_acc = float(top_candidates["y_true"].mean())
    predicted_arcs = top_candidates["arc_id"].tolist()
    
    return top1_acc, predicted_arcs, top_candidates["y_true"].tolist()


def compute_route_continuity(
    predicted_arcs: List[int],
    arcs_lookup: np.ndarray
) -> float:
    """
    Calculates Route Continuity:
    The fraction of consecutive GPS observation predictions that form valid
    topological transitions in the road network graph (same arc or connecting node).
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    valid_transitions = 0
    total_transitions = len(predicted_arcs) - 1

    for i in range(total_transitions):
        a1 = predicted_arcs[i]
        a2 = predicted_arcs[i + 1]

        if a1 == a2:
            # Vehicle stayed on the same road segment
            valid_transitions += 1
        elif a1 < len(arcs_lookup) and a2 < len(arcs_lookup):
            # Check if to_node of a1 connects to from_node of a2
            to_node_1 = arcs_lookup[a1, 1]
            from_node_2 = arcs_lookup[a2, 0]
            if to_node_1 == from_node_2:
                valid_transitions += 1

    return float(valid_transitions / total_transitions) if total_transitions > 0 else 1.0


def evaluate_model_performance(
    model_name: str,
    algorithm_family: str,
    key_config: str,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    groups: np.ndarray,
    point_indices: np.ndarray,
    arc_ids: np.ndarray,
    arcs_lookup: np.ndarray = None,
    cv_macro_f1: float = 0.0,
    training_time: float = 0.0,
    inference_time_ms: float = 0.0
) -> Dict[str, Any]:
    """
    Consolidates all metrics required for the Model Comparison Table.
    """
    # Binary predictions using 0.5 threshold for candidate-level metrics
    y_pred_binary = (y_probs >= 0.5).astype(int)

    cand_acc = float(accuracy_score(y_true, y_pred_binary))
    macro_f1 = float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0))
    pos_f1 = float(f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0))
    precision = float(precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0))

    try:
        # Clip probabilities to avoid infinite log loss
        eps = 1e-15
        probs_clipped = np.clip(y_probs, eps, 1 - eps)
        loss = float(log_loss(y_true, probs_clipped))
    except Exception:
        loss = float("nan")

    # Observation Top-1 Accuracy
    top1_acc, pred_arcs, _ = compute_observation_top1(
        y_true, y_probs, groups, point_indices, arc_ids
    )

    # Route Continuity
    continuity = float("nan")
    if arcs_lookup is not None:
        continuity = compute_route_continuity(pred_arcs, arcs_lookup)

    cm = confusion_matrix(y_true, y_pred_binary).tolist()

    return {
        "model": model_name,
        "algorithm_family": algorithm_family,
        "key_configuration": key_config,
        "cv_macro_f1": cv_macro_f1,
        "top1_accuracy": top1_acc,
        "route_continuity": continuity,
        "student_test_macro_f1": macro_f1,
        "candidate_precision": precision,
        "candidate_recall": recall,
        "candidate_f1": pos_f1,
        "log_loss": loss,
        "training_time_s": training_time,
        "inference_time_ms_per_1k": inference_time_ms,
        "confusion_matrix": cm
    }
