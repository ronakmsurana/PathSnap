"""
PathSnap FastAPI Backend (Laptop 1 Server)
Hosts the trained machine learning models, spatial index, and HMM/Viterbi route reconstruction engine.
Receives evaluation requests from Laptop 2, performs real-time inference,
computes server-side accuracy against ground-truth reference data, and streams console logs.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_processing.loader import load_trip, list_trip_ids
from feature_engineering.candidate_generator import SpatialCandidateGenerator
from feature_engineering.extractor import CandidateFeatureExtractor
from hmm.decoder import RouteDecoderHMM


# Configure formatted terminal logging (Mandatory for evaluator inspection)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | Laptop 1 API >> %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pathsnap_api")

app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description="Backend API hosting trained ML models and HMM Viterbi route reconstruction",
    version="1.0.0"
)

# Enable CORS for cross-laptop communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and component state
DATA_DIR = os.environ.get("PATHSNAP_DATA_DIR", "../Student folder")
ARTIFACTS_DIR = os.environ.get("PATHSNAP_ARTIFACTS_DIR", "artifacts")

MODEL = None
SCALER = None
CANDIDATE_GEN = None
FEATURE_EXTRACTOR = None
HMM_DECODER = None


@app.on_event("startup")
def load_system_artifacts():
    global MODEL, SCALER, CANDIDATE_GEN, FEATURE_EXTRACTOR, HMM_DECODER

    logger.info("Initializing PathSnap Inference Pipeline...")

    # Load top model (prefer tuned lightgbm, fallback to lightgbm, random forest, or logistic regression)
    candidate_model_files = [
        "best_tuned_lightgbm.joblib",
        "lightgbm.joblib",
        "random_forest.joblib",
        "decision_tree.joblib",
        "logistic_regression.joblib"
    ]

    for mf in candidate_model_files:
        mpath = os.path.join(ARTIFACTS_DIR, mf)
        if os.path.exists(mpath):
            MODEL = joblib.load(mpath)
            logger.info(f"Loaded primary inference model: {mf}")
            break

    if MODEL is None:
        logger.warning("No serialized model artifact found in artifacts/. Place a trained .joblib file.")

    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
        logger.info("Loaded feature scaler.")

    CANDIDATE_GEN = SpatialCandidateGenerator(search_radius_meters=45.0)
    FEATURE_EXTRACTOR = CandidateFeatureExtractor()
    HMM_DECODER = RouteDecoderHMM(beta=12.0, sigma_z=10.0, alpha=0.65)
    logger.info("Pipeline components initialized successfully. System ready for requests.")


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "server": "Laptop 1 (PathSnap Backend)",
        "model_loaded": MODEL is not None,
        "model_type": type(MODEL).__name__ if MODEL else None
    }


@app.get("/trips")
def get_available_trips():
    """Lists available evaluation trips prioritizing evaluator_test, then student_test, then train."""
    possible_dirs = [
        os.environ.get("PATHSNAP_EVAL_DIR", ""),
        "evaluator_test",
        "../evaluator_test",
        "../ML Hackathon/evaluator_test",
        os.path.join(DATA_DIR, "evaluator_test"),
        os.path.join(DATA_DIR, "student_test"),
        os.path.join(DATA_DIR, "train")
    ]
    eval_dir = None
    for d in possible_dirs:
        if d and os.path.exists(d) and any(f.isdigit() for f in os.listdir(d)):
            eval_dir = os.path.abspath(d)
            break

    trips = list_trip_ids(eval_dir) if eval_dir else []
    return {
        "partition": os.path.basename(eval_dir) if eval_dir else "unknown",
        "total_trips": len(trips),
        "trips": trips
    }


@app.post("/predict/{trip_id}")
def predict_and_evaluate_trip(trip_id: str, request: Request):
    """
    Main evaluation endpoint:
    Receives trip_id from Laptop 2, processes it through complete prediction pipeline,
    computes evaluation accuracy against server-side ground truth, and logs in CMD.
    """
    client_ip = request.client.host if request.client else "unknown"
    t_start = time.time()
    logger.info(f"Received evaluation request for Trip ID: '{trip_id}' from Client IP: {client_ip}")

    # Locate trip directory (check evaluator_test first, then student_test, then train)
    possible_roots = [
        os.environ.get("PATHSNAP_EVAL_DIR", ""),
        "evaluator_test",
        "../evaluator_test",
        "../ML Hackathon/evaluator_test",
        os.path.join(DATA_DIR, "evaluator_test"),
        os.path.join(DATA_DIR, "student_test"),
        os.path.join(DATA_DIR, "train")
    ]
    trip_path = None
    for pr in possible_roots:
        if pr and os.path.exists(os.path.join(pr, trip_id)):
            trip_path = pr
            break

    if not os.path.exists(os.path.join(trip_path, trip_id)):
        logger.error(f"Trip ID '{trip_id}' not found in dataset partitions!")
        raise HTTPException(status_code=404, detail=f"Trip ID '{trip_id}' not found.")

    try:
        # 1. Load Trip Data (Ground truth is retained strictly server-side)
        trip = load_trip(trip_path, trip_id, load_route=True)
        n_obs = len(trip.track)

        # 2. Candidate Generation
        cands_list, cand_stats = CANDIDATE_GEN.generate_trip_candidates(trip, align_route=True)

        # 3. Feature Extraction
        X, _, meta = FEATURE_EXTRACTOR.extract_features_for_trip(trip, cands_list, include_label=False)

        # 4. Model Classification Inference
        if MODEL is not None:
            if SCALER is not None and getattr(MODEL, "__class__", None).__name__ in ["LogisticRegression", "GaussianNB"]:
                X_scaled = SCALER.transform(X)
                probs = MODEL.predict_proba(X_scaled)[:, 1]
            elif hasattr(MODEL, "predict_proba"):
                probs = MODEL.predict_proba(X)[:, 1]
            else:
                scores = MODEL.decision_function(X)
                probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            # Fallback distance-based baseline
            probs = np.array([float(m["dist_seg"]) for m in meta])
            probs = 1.0 / (1.0 + probs)

        # 5. HMM and Viterbi Sequential Route Reconstruction
        hmm_result = HMM_DECODER.decode_trip(trip, cands_list, probs)

        total_latency_ms = (time.time() - t_start) * 1000.0

        # Build reconstructed coordinate polylines for UI map rendering
        reconstructed_arcs = hmm_result["viterbi_arcs"]
        coords_reconstructed = []
        for arc_id in reconstructed_arcs:
            if arc_id < len(trip.arcs):
                u, v = trip.arcs[arc_id]
                coords_reconstructed.append([
                    [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])],
                    [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]
                ])

        gps_trajectory = [[float(p[1]), float(p[0])] for p in trip.track]

        # Log detailed metrics to console for evaluator inspection
        logger.info(f"Trip '{trip_id}' Reconstructed Successfully in {total_latency_ms:.1f} ms:")
        logger.info(f"   Observations:             {n_obs}")
        logger.info(f"   Candidate Miss Rate:      {cand_stats['candidate_miss_rate'] * 100:.2f}%")
        logger.info(f"   Observation Top-1 Acc:    {hmm_result['pointwise_accuracy_hmm'] * 100:.2f}% (Baseline Clf: {hmm_result['pointwise_accuracy_raw'] * 100:.2f}%)")
        logger.info(f"   Route Continuity:         {hmm_result['route_continuity_hmm'] * 100:.2f}% (Baseline Clf: {hmm_result['route_continuity_raw'] * 100:.2f}%)")
        logger.info(f"   Continuity Gain:         +{hmm_result['continuity_improvement'] * 100:.2f}%")

        return {
            "status": "success",
            "trip_id": trip_id,
            "observations_count": n_obs,
            "candidate_miss_rate": cand_stats["candidate_miss_rate"],
            "top1_accuracy": hmm_result["pointwise_accuracy_hmm"],
            "baseline_top1_accuracy": hmm_result["pointwise_accuracy_raw"],
            "route_continuity": hmm_result["route_continuity_hmm"],
            "baseline_route_continuity": hmm_result["route_continuity_raw"],
            "continuity_gain": hmm_result["continuity_improvement"],
            "reconstructed_arc_count": len(reconstructed_arcs),
            "reconstructed_arcs": reconstructed_arcs,
            "gps_trajectory": gps_trajectory,
            "reconstructed_segments": coords_reconstructed,
            "inference_time_ms": total_latency_ms
        }

    except Exception as e:
        logger.error(f"Error processing trip '{trip_id}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
