"""FastAPI for live PathSnap inference on Laptop 1."""
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from .data import FEATURES, generate_candidates, load_reference, load_trip
from .decoder import build_graph, decode, reconstruct, transition_lattice
from .reference import observation_labels
from .route_eval import route_metrics, summary
from .train import score_candidates


logger = logging.getLogger("uvicorn.error")


def read_model(artifacts):
    config = json.loads((artifacts / "model_config.json").read_text())
    if tuple(config["features"]) != FEATURES or "decoder" not in config:
        raise RuntimeError("Model configuration does not match the inference features or is missing decoder settings")
    return joblib.load(artifacts / "model.joblib"), config


def predict(trip, model, config):
    candidates = generate_candidates(trip)
    proba = score_candidates(model, candidates.arc, candidates.features)
    graph = build_graph(trip)
    distances, steps, displacement = transition_lattice(trip, candidates, graph)
    settings = config["decoder"]
    slots = decode(proba, distances, steps, displacement,
                   settings["beta_m"], settings["transition_weight"])
    chosen, runs, connected = reconstruct(trip, candidates, slots, distances, steps, graph)
    return candidates, chosen, runs, connected, steps


def latlon_routes(trip, runs):
    lines = []
    for run in runs:
        arcs = trip.arcs[run]
        nodes = np.r_[arcs[0, 0], arcs[:, 1]]
        lines.append(trip.nodes[nodes, ::-1].tolist())
    return lines


def trip_directories(root):
    if not root.is_dir():
        raise RuntimeError(f"Trip directory does not exist: {root}")
    return {
        p.name: p for p in sorted(root.iterdir())
        if p.is_dir() and re.fullmatch(r"[0-9]+", p.name)
        and all((p / f"{p.name}.{ext}").is_file() for ext in ("track", "nodes", "arcs"))
    }


def create_app():
    @asynccontextmanager
    async def startup(app):
        mode = os.getenv("PATHSNAP_MODE", "student")
        if mode not in ("student", "evaluator"):
            raise RuntimeError("PATHSNAP_MODE must be student or evaluator")
        default = "Student folder/student_test" if mode == "student" else "evaluator_test"
        root = Path(os.getenv("PATHSNAP_TRIPS", default))
        app.state.trips = trip_directories(root)
        if len(app.state.trips) != 10:
            raise RuntimeError(f"Expected 10 complete trips in {root}, found {len(app.state.trips)}")
        app.state.references = Path(os.getenv("PATHSNAP_REFERENCES", str(root)))
        app.state.mode = mode
        app.state.model, app.state.config = read_model(Path(os.getenv("PATHSNAP_ARTIFACTS", "artifacts")))
        yield

    app = FastAPI(title="PathSnap", version="1.0", lifespan=startup)
    origins = os.getenv("PATHSNAP_UI_ORIGINS", "http://127.0.0.1:8001,http://localhost:8001")
    app.add_middleware(CORSMiddleware, allow_origins=[value.strip() for value in origins.split(",")],
                       allow_methods=["GET"])
    app.add_middleware(GZipMiddleware, minimum_size=1500)

    @app.get("/health")
    def health():
        return {"status": "ready", "mode": app.state.mode, "model": app.state.config["model"]}

    @app.get("/trips")
    def trips():
        return {"mode": app.state.mode, "trips": list(app.state.trips)}

    @app.get("/match/{trip_id}")
    def match(trip_id: str):
        directory = app.state.trips.get(trip_id)
        if directory is None:
            raise HTTPException(status_code=404, detail="Trip not found")
        started = perf_counter()
        try:
            trip = load_trip(directory)
            candidates, chosen, runs, connected, steps = predict(trip, app.state.model, app.state.config)
        except ValueError as exc:
            logger.error("Invalid trip input for %s: %s", trip_id, exc)
            raise HTTPException(status_code=422, detail="The trip inputs could not be loaded or matched") from exc
        inference = perf_counter() - started

        evaluation = None
        reference_dir = app.state.references / trip_id
        if (reference_dir / f"{trip_id}.route").is_file():
            try:
                reference = load_reference(reference_dir, trip)
                labels, _ = observation_labels(trip, reference)
                metrics = summary([{"inference_s": inference, **route_metrics(
                    reference, chosen, labels, runs, connected, steps)}])
                evaluation = {"top1_accuracy": metrics["top1"], "aligned_observations": metrics["aligned"],
                              "route_continuity": metrics["continuity"], "sequence_f1": metrics["sequence_f1"]}
            except ValueError:
                logger.exception("Invalid server-side reference for trip %s", trip_id)
        roads = np.unique(candidates.arc[candidates.arc >= 0])
        logger.info("PathSnap %s trip=%s fixes=%d route_parts=%d inference=%.3fs scored=%s",
                    app.state.mode, trip_id, len(chosen), len(runs), inference, evaluation is not None)
        return {
            "trip_id": trip_id, "mode": app.state.mode, "model": app.state.config["model"],
            "observations_summary": {"count": len(chosen), "trajectory_parts": len(runs),
                                     "nearest_fallback": int(candidates.outside_radius.sum())},
            "inference_seconds": inference, "evaluation": evaluation, "matched_arc_ids": chosen.tolist(),
            "gps_latlon": trip.track[:, [1, 0]].tolist(), "route_latlon": latlon_routes(trip, runs),
            "nearby_roads_latlon": trip.nodes[trip.arcs[roads]][:, :, ::-1].tolist(),
        }

    return app


app = create_app()
