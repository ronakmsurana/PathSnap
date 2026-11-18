# PathSnap: Distributed Map-Matching & Route Reconstruction

**Department of Computer Science and Engineering | PES University**  
**UE24CS352A: Machine Learning Hackathon**

PathSnap is an end-to-end, distributed machine learning system designed to reconstruct continuous, realistic vehicle trajectories by matching noisy GPS observations to an underlying road network graph. It couples machine learning candidate classification with topologically constrained Hidden Markov Model (HMM) Viterbi decoding.

---

## 1. System Architecture: Two-Laptop Distributed Deployment

The application is deployed across two laptops communicating over a local area network (LAN/Wi-Fi):

```
+-------------------------------------------------------------+
|                     LAPTOP 2 (Frontend)                     |
|  - Streamlit Evaluator Interface (ui/app.py)                |
|  - Dropdown selecting any of the 10 Evaluation Trips        |
|  - Sends HTTP POST request to Laptop 1 IP                   |
|  - Renders Reconstructed Trajectory & Interactive Folium Map|
+------------------------------+------------------------------+
                               |  HTTP POST /predict/{trip_id}
                               v
+-------------------------------------------------------------+
|                     LAPTOP 1 (Backend)                      |
|  - FastAPI Engine (api/main.py under Uvicorn)               |
|  - Road Network Topology Graph & Local Spatial R-Tree Index |
|  - Tuned LightGBM Candidate Classifier (artifacts/)         |
|  - Topologically Constrained Viterbi HMM Decoder (hmm/)     |
|  - Server-Side Ground-Truth Evaluation                      |
|  - Live Request & Metric Terminal Logging in CMD            |
+-------------------------------------------------------------+
```

---

## 2. Directory Structure

```
PathSnap/
├── api/                    # FastAPI backend (Laptop 1 server)
│   └── main.py             # Server endpoints, model serving & real-time CMD logger
├── ui/                     # Evaluator-facing frontend (Laptop 2)
│   └── app.py              # Streamlit dashboard with interactive map rendering
├── data_processing/        # Data loading, inspection, and EDA
│   ├── loader.py           # Parsing for .track, .nodes, .arcs, and .route files
│   └── eda.py              # Comprehensive EDA and distribution plotting
├── feature_engineering/    # Spatial candidate extraction and kinematics
│   ├── projection.py       # Conformal local metric coordinate projection (meters)
│   ├── candidate_generator.py # Shapely STRtree R-tree candidate generation (45m radius)
│   ├── extractor.py        # Extraction of 13 physical, kinematic, and topological features
│   └── dataset_builder.py  # Group-aware dataset builder for Train & Student-Test
├── models/                 # Model training & cross-validation engine
│   └── trainer.py          # 5-Fold GroupKFold CV benchmark across 6 models
├── evaluation/             # Metrics calculation & diagnostic plotting
│   ├── metrics.py          # Top-1 accuracy, route continuity, F1, log loss, latency
│   └── visualize_results.py# Comparative bar charts and loss curves
├── optimization/           # Bayesian hyperparameter optimization
│   └── optimizer.py        # Optuna TPE search with 3-Fold GroupKFold CV
├── hmm/                    # Sequential route reconstruction
│   ├── decoder.py          # Topologically constrained Viterbi HMM engine
│   └── benchmark_hmm.py    # 10-trip student-test comparative benchmark
├── visualization/          # Geographical plotting
│   └── folium_map.py       # Multi-layer Folium interactive HTML map generator
├── artifacts/              # Serialized model weights (.joblib) & scaler
├── results/                # Evaluation CSV tables, logs, and plots
│   ├── model_comparison.csv
│   ├── tuned_vs_untuned_comparison.csv
│   ├── hmm_viterbi_evaluation.csv
│   └── plots/              # Diagnostic charts
├── report/                 # Consolidated Technical Report
│   ├── PathSnap_Technical_Report.pdf
│   └── generate_report.py  # ReportLab PDF compiler
├── requirements.txt        # Full project dependencies
└── README.md               # System documentation & deployment guide
```

---

## 3. Experimental Benchmark Results

### 3.1 Model Comparison Across 5 Algorithm Families (6 Models)
Evaluated using 5-Fold GroupKFold cross-validation on the 80 training trips (grouped strictly by `trip_id` to prevent intra-trip data leakage) and tested on the 10 `student_test` trips:

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Log Loss | Latency (ms/1k) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | $C=1.0$, L2 penalty, StandardScaler | 0.8685 | 78.46% | 0.8691 | 0.1082 | **0.02 ms** |
| **Gaussian Naive Bayes** | Linear / Probabilistic | $\text{var\_smoothing}=10^{-9}$ | 0.7810 | 52.20% | 0.7806 | 0.7867 | 0.10 ms |
| **Decision Tree** | Tree-based | $\text{max\_depth}=12$, $\text{min\_split}=20$ | 0.8774 | 79.12% | 0.8874 | 0.1011 | 0.04 ms |
| **Random Forest** | Ensemble | $N=100$, $\text{max\_depth}=14$, $\sqrt{F}$ | 0.8822 | 79.64% | 0.8900 | 0.0831 | 0.73 ms |
| **LightGBM** | Gradient Boosting | $N=150$, $\eta=0.08$, $\text{leaves}=31$ | 0.8805 | 79.30% | 0.8879 | 0.0907 | 0.47 ms |
| **XGBoost** | Gradient Boosting | $N=120$, $\eta=0.10$, $\text{max\_depth}=6$ | 0.8814 | **79.70%** | 0.8890 | **0.0842** | 0.16 ms |

### 3.2 Bayesian Hyperparameter Optimization (LightGBM)
Optimized using Optuna TPE with 3-Fold GroupKFold CV:
- **CV Macro F1:** Boosted from `0.8805` to **`0.8956`**
- **Student-Test Top-1 Accuracy:** `79.29%` $\to$ **`79.68%`**
- **Log Loss:** Reduced from `0.0907` to **`0.0836`**
- **Best Hyperparameters:** $N=127$, $\eta=0.0425$, $\text{leaves}=33$, $\text{depth}=9$, $\text{min\_child}=43$, $\text{subsample}=0.752$, $\text{colsample}=0.864$.

### 3.3 HMM Viterbi Sequential Route Reconstruction (10 Student-Test Trips)
Coupling classifier probabilities with graph-topological transition penalties:
- **Mean Top-1 Accuracy:** `82.63%` (Classifier) $\to$ **`86.15%` (HMM+Viterbi)** ($+3.52\%$ gain)
- **Mean Route Continuity:** `89.67%` (Classifier) $\to$ **`96.99%` (HMM+Viterbi)** (**$+7.32\%$ boost in continuous graph edges**)
- **Mean Latency:** `1,065 ms` per complete trajectory.

---

## 4. Execution & Two-Laptop Deployment Guide

### Step 1: Install Dependencies (Both Laptops)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Running Laptop 1 (FastAPI Server)
Find Laptop 1's local IP address (`ipconfig` on Windows or `ifconfig` / `ip a` on macOS/Linux, e.g. `192.168.1.105`).
```bash
cd PathSnap
export PYTHONPATH=.
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*Note: Laptop 1's terminal will actively stream formatted request logs and calculated accuracy scores for the evaluators.*

### Step 3: Running Laptop 2 (Evaluator UI)
```bash
cd PathSnap
streamlit run ui/app.py --server.port 8501
```
1. Open the UI in your browser: `http://localhost:8501`.
2. In the sidebar, set the **Laptop 1 API Endpoint** to `http://<LAPTOP_1_IP>:8000`.
3. Select any of the 10 evaluation trips and click **Reconstruct Route**.
4. The system will display observation accuracy, route continuity, latency, and the interactive Folium map.

---

## 5. Technical Report
The consolidated Technical Report adhering strictly to Sections 2.2 through 7 of the hackathon guidelines is compiled and available at:
`report/PathSnap_Technical_Report.pdf`
