# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to an underlying road network graph.

## System Architecture: Two-Laptop Distributed Deployment

- **Laptop 1 (Model & API Backend):**
  - Hosts trained ML classifiers, road network graph representation, and HMM/Viterbi sequential route decoder.
  - Implements a FastAPI backend that receives evaluation requests, executes inference, logs request details in real-time, and computes server-side evaluation metrics.
- **Laptop 2 (Evaluator UI Frontend):**
  - Hosts an interactive user interface (Streamlit) that displays evaluation trip options.
  - Submits selected trips over the local network to Laptop 1 and visualizes the reconstructed routes, metrics, and multi-layer Folium interactive maps.

---

## Repository Structure

```
PathSnap/
├── api/                    # FastAPI backend endpoints, server-side evaluation & request handling
├── ui/                     # Evaluator-facing frontend interface
├── data_processing/        # Data loading, inspection, cleaning, and preprocessing
├── feature_engineering/    # Spatial indexing, candidate generation, and feature extraction
├── models/                 # Model training scripts across multiple algorithm families
├── evaluation/             # Model comparison, cross-validation, metrics, and confusion matrices
├── optimization/           # Bayesian hyperparameter optimization (Optuna)
├── hmm/                    # Hidden Markov Model & Viterbi route reconstruction engine
├── visualization/          # Folium map generation and geographical visualization scripts
├── artifacts/              # Serialized model weights, scalers, and configuration files
├── results/                # Evaluation logs, metric tables (CSV), and experimental records
├── report/                 # Technical report and documentation
├── requirements.txt        # Project dependencies
└── README.md               # System documentation & deployment guide
```

---

## Installation & Setup

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Laptop 1 (FastAPI Server)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Running Laptop 2 (Evaluator UI)
```bash
streamlit run ui/app.py --server.port 8501
```
Configure Laptop 2 with Laptop 1's local IP address (e.g., `http://192.168.x.x:8000`).
