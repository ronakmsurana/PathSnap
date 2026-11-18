# PathSnap

PathSnap reconstructs a vehicle route from noisy GPS readings and a directed road network. A classifier scores nearby road segments, then Viterbi decoding selects a sequence that respects the network. The browser on Laptop 2 requests a new inference from Laptop 1 each time a trip is selected.

## Install and reproduce

Run these commands from the repository root with Python 3.12. The model artifact is tied to the scikit-learn version in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the supplied data at `Student folder/train/<trip_id>/` and `Student folder/student_test/<trip_id>/`. Each trip needs files named `<trip_id>.track`, `<trip_id>.nodes`, `<trip_id>.arcs`, and `<trip_id>.route`. The public repository excludes raw data and instructor-provided PDFs. Keep the reserved evaluator data on Laptop 1.

```bash
python -m pathsnap.prepare
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pathsnap.train
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pathsnap.route_eval
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pathsnap.visualize
```

By default the pipeline compares five models with three trip-grouped folds, logs twelve Optuna TPE trials, fits the selected configuration, and tunes nine decoder configurations on one held-out training fold. It writes the model and its decoder settings to `artifacts/`, experiment results to `results/`, and plots and the interactive Folium map to `report/`. The visualization command accepts a different development trip via `--trip`.

## Run the demo

Run the API on Laptop 1. The default mode loads the supplied ten student-test trips. Existing model artifacts can be used directly.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

On Laptop 2, serve the interface from its local copy of this repository. This laptop needs only Python and the `web/` folder.

```bash
python3 -m http.server 8001 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8001`. Enter `http://<Laptop-1-LAN-IP>:8000` in **API address**, connect, choose a trip, and select **Reconstruct route**. Keep the API terminal visible for the evaluator. Both laptops need to be able to reach Laptop 1 on TCP port 8000. For a single-machine preview, use the default API address in the UI. Map tiles use an internet connection. The proposed route, nearby road segments, and points also render over a blank background when tiles are unavailable.

For final evaluation, start the API in evaluator mode after the ten reserved trip folders have been provided.

```bash
PATHSNAP_MODE=evaluator PATHSNAP_TRIPS=/path/to/evaluator_test \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Optionally set `PATHSNAP_REFERENCES` to a separate server-only directory with the same trip subdirectories and `.route` files. Either way, the API loads each reference after the route has been predicted. The response includes only aggregate scores, GPS inputs and road geometry, and the predictions. An unavailable reference yields a route with no evaluation scores.

The default browser origins are the two loopback addresses on port 8001. If you serve the UI from another origin, set `PATHSNAP_UI_ORIGINS` on Laptop 1 to a comma-separated list of the actual UI origins before starting the API. On Windows PowerShell, set environment variables with `$env:PATHSNAP_MODE="evaluator"` and `$env:PATHSNAP_TRIPS="C:\path\to\evaluator_test"` before running Uvicorn.

| Request | Response |
| --- | --- |
| `GET /health` | Mode and model readiness |
| `GET /trips` | Ten available trip IDs |
| `GET /match/{trip_id}` | Fresh inference, map layers and aggregate scores |
| `GET /docs` | OpenAPI browser |

Run `python scripts/check_api.py http://<Laptop-1-LAN-IP>:8000` from Laptop 2 for a smoke check. The script expects all ten trips to have usable reference files.

## How the results are measured

- Each trip is projected into local metres. Candidate generation uses a Shapely spatial index over directed segments and keeps up to ten candidates within 65 m. If the radius query comes up empty, the closest segment is used and flagged.
- Supplied route entries describe traversal order. The reference-only alignment scores nearby route occurrences using projection distance, direction and chronological progress. It excludes observations that align more than 100 m from the reference. Repeated segment occurrences have distinct positions in that alignment. The resulting observation labels are approximate.
- All methods see the same features and partitions. Each training trip contributes up to 350 sampled aligned observations, dropping samples that lack a positive candidate. Validation and student metrics use all aligned observations, including candidate misses for observation Top-1. Candidate-level precision, recall, macro F1, confusion matrices and log loss score binary predictions over candidate rows. CV model selection uses mean macro F1. The final fit uses the 80 training trips.
- Viterbi uses log-normalized classifier probabilities as emissions and penalizes the difference between feasible directed path length and observed GPS displacement. Directed Dijkstra searches respect a time-based distance budget. The decoder restarts after a gap above 30 seconds or when no transition from the current sequence is feasible. Each connected section is drawn separately.
- Continuity is the fraction of adjacent observation pairs with feasible transitions, excluding gaps above 30 seconds. Ordered route F1 is twice the longest common subsequence length of predicted and reference arc IDs divided by their combined lengths. This measures order and penalizes extra arcs, including false revisits.

`results/dataset_inventory.csv` records the input inventory. `results/model_metrics.json`, `results/optimization_trials.csv`, `results/decoder_trials.json` and `results/route_metrics_by_trip.csv` preserve the measured results. The method for weighting directed distance mismatch follows the formulation in [Newson and Krumm (2009)](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/). The team should review, reproduce, and be able to explain the implementation and its experimental outcomes for the viva.

To collect the local report evidence, run Uvicorn with output saved to `report/api_server.log` and save the API smoke-check output to `report/api_check.log`. With the UI and server running and the Folium HTML already generated, run `geckodriver --port 4444`, then execute:

```bash
python scripts/capture_demo.py
python scripts/build_report.py
```

For the report cover, supply `report/team.json` as an array of four objects with `name` and `srn` keys. Pass the public repository URL to the report builder with `--repo`. The PDF and its evidence are separate from the GitHub source deliverable. During the physical demo, capture the receiving laptop's terminal after inference requests and include that evidence according to the submission guide.
