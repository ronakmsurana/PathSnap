"""
HMM Viterbi Benchmark Script for PathSnap
Evaluates Classifier-Only vs. Classifier + HMM/Viterbi route reconstruction
across all 10 student-testing trips, measuring accuracy gains and route continuity improvements.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from data_processing.loader import load_trip, list_trip_ids
from feature_engineering.candidate_generator import SpatialCandidateGenerator
from feature_engineering.extractor import CandidateFeatureExtractor
from hmm.decoder import RouteDecoderHMM


def benchmark_hmm_on_student_test(
    data_dir: str = "../Student folder/student_test",
    artifacts_dir: str = "artifacts",
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load trained model
    model_path = os.path.join(artifacts_dir, "best_tuned_lightgbm.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(artifacts_dir, "lightgbm.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(artifacts_dir, "random_forest.joblib")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    generator = SpatialCandidateGenerator(search_radius_meters=45.0)
    extractor = CandidateFeatureExtractor()
    decoder = RouteDecoderHMM(beta=12.0, sigma_z=10.0, alpha=0.65)

    trip_ids = list_trip_ids(data_dir)
    print(f"Benchmarking HMM on {len(trip_ids)} student-testing trips...")

    results = []
    t_start_all = time.time()

    for idx, tid in enumerate(trip_ids):
        t0 = time.time()
        trip = load_trip(data_dir, tid, load_route=True)
        cands_list, cand_stats = generator.generate_trip_candidates(trip, align_route=True)
        X, _, meta = extractor.extract_features_for_trip(trip, cands_list, include_label=False)

        # Get classifier candidate probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            scores = model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-scores))

        # Decode sequence with HMM Viterbi
        dec_res = decoder.decode_trip(trip, cands_list, probs)
        duration_ms = (time.time() - t0) * 1000.0

        results.append({
            "trip_id": tid,
            "observations": dec_res["observations_count"],
            "miss_rate_pct": cand_stats["candidate_miss_rate"] * 100.0,
            "raw_top1_acc_pct": dec_res["pointwise_accuracy_raw"] * 100.0,
            "hmm_top1_acc_pct": dec_res["pointwise_accuracy_hmm"] * 100.0,
            "acc_improvement_pct": dec_res["accuracy_improvement"] * 100.0,
            "raw_continuity_pct": dec_res["route_continuity_raw"] * 100.0,
            "hmm_continuity_pct": dec_res["route_continuity_hmm"] * 100.0,
            "continuity_gain_pct": dec_res["continuity_improvement"] * 100.0,
            "latency_ms": duration_ms
        })

        print(f"Trip {tid}: Acc {dec_res['pointwise_accuracy_raw']*100:.1f}% -> {dec_res['pointwise_accuracy_hmm']*100:.1f}% | Continuity {dec_res['route_continuity_raw']*100:.1f}% -> {dec_res['route_continuity_hmm']*100:.1f}% (+{dec_res['continuity_improvement']*100:.1f}%) in {duration_ms:.0f}ms")

    df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, "hmm_viterbi_evaluation.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n=======================================================")
    print(f"HMM Route Reconstruction Benchmark Summary (10 Trips)")
    print(f"=======================================================")
    print(f"Mean Top-1 Accuracy:  Classifier-Only: {df['raw_top1_acc_pct'].mean():.2f}% -> HMM+Viterbi: {df['hmm_top1_acc_pct'].mean():.2f}%")
    print(f"Mean Route Continuity: Classifier-Only: {df['raw_continuity_pct'].mean():.2f}% -> HMM+Viterbi: {df['hmm_continuity_pct'].mean():.2f}% (+{df['continuity_gain_pct'].mean():.2f}%)")
    print(f"Mean Inference Latency: {df['latency_ms'].mean():.1f} ms/trip")
    print(f"Saved evaluation metrics to: {out_csv}")

    return df


if __name__ == "__main__":
    benchmark_hmm_on_student_test()
