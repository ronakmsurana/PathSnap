# PathSnap: Phase 1 Review Deliverable
**UE24CS352A: Machine Learning Hackathon | PES University**  
**Team Members:** Ashith Rao K, Ronak M Surana, Arati, Ruturaj Rajaram Shedbale

---

## Review 1 Overview: Pre-Processing, Feature Engineering & Baseline Models

This branch contains our completed **Review 1** milestone deliverables:
1. **Dataset Exploration & Pre-Processing:** Local metric projection (converting angular coordinates to Cartesian meters) and R-tree spatial indexing.
2. **Spatial Candidate Generation & Miss Rate Analysis:** 45-meter candidate road segment retrieval with adaptive nearest fallback and candidate miss rate auditing.
3. **Feature Engineering:** Extraction of 13 physical, kinematic, and topological features per candidate arc.
4. **Baseline Model Experimentation:** 5-Fold Group-aware Cross-Validation (`GroupKFold` strictly grouped by `trip_id`) across 6 models spanning 5 algorithm families.
5. **Interactive Geographical Visualizations:** Multi-layer Folium maps rendering raw GPS points, road network geometry, and ground-truth route.

---

## 1. Data Pre-Processing & Spatial Indexing

### 1.1 Local Metric Coordinate Projection
GPS observations are recorded in angular geographic units (degrees longitude and latitude). Standard Euclidean calculations in degrees distort physical distances and fail to preserve true spatial proportions.
To resolve this, we implemented a local Conformal / Equirectangular metric projection centered at the trajectory mean coordinates:
$$x = R \cdot (\lambda - \lambda_0) \cos(\phi_0)$$
$$y = R \cdot (\phi - \phi_0)$$
where $R = 6,371,000\text{ meters}$. This projects angular coordinates into local Cartesian meter coordinates with sub-millimeter geometric accuracy across urban extents.

### 1.2 R-Tree Spatial Indexing & Candidate Generation
Road network arcs are compiled into Shapely `LineString` geometries and indexed using an R-tree spatial index (`shapely.STRtree`). For each GPS point:
- Candidate road segments within a **45-meter search radius** are retrieved.
- If no segment lies within 45m (due to severe GPS noise or signal gaps), an adaptive fallback queries the **$k=3$ nearest road segments**.
- Ground-truth traversed road segments from `.route` are aligned to continuous GPS points using a monotonic forward-backward search window.

### 1.3 Candidate Miss Rate Audit (Guideline Requirement)
- **Train partition (80 trips):** 10.30% Candidate Miss Rate.
- **Student-test partition (10 trips):** 3.47% Candidate Miss Rate.
- **Root Cause Identified during EDA:** In urban canyons and high-speed turns, multipath GPS drift occasionally exceeds 45m (up to 569m observed in trip 00000002), causing the true road segment to lie outside the 45m radius. This empirical finding justifies using machine learning feature scoring and sequential decoding.

---

## 2. Feature Engineering: 13 Features & Rationale

For every candidate road segment $(A, B)$ associated with a GPS observation $P_t$, we extract 13 spatial, kinematic, and topological features:

| Feature Name | Definition / Formula | Engineering Rationale |
| :--- | :--- | :--- |
| `dist_to_segment` | Orthogonal Euclidean distance in meters from $P_t$ to line segment $(A, B)$. | Primary spatial proximity indicator. True segments have lower perpendicular distance. |
| `projection_ratio` | Scalar $t_{\text{proj}} = \text{clamp}\left(\frac{(P - A) \cdot (B - A)}{\|B - A\|^2}, 0, 1\right)$. | Identifies where the vehicle projects along the road segment (near start $t \approx 0$, middle, or end $t \approx 1$). |
| `segment_length` | Length of candidate arc in meters $\|B - A\|$. | Road classification proxy; distinguishes long highways from short intersection links. |
| `dist_to_start_node` | Distance from $P_t$ to arc start node $A$ in meters. | Captures segment entry proximity and intersection transitions. |
| `dist_to_end_node` | Distance from $P_t$ to arc end node $B$ in meters. | Captures segment exit proximity and forward road progression. |
| `dist_to_midpoint` | Distance from $P_t$ to arc midpoint $(A + B) / 2$. | Centrality metric reducing endpoint boundary bias. |
| `gps_speed_mps` | Instantaneous vehicle speed $\|P_t - P_{t-1}\| / \Delta t$ in m/s. | Kinematic state indicating whether vehicle is cruising, stopping, or turning. |
| `heading_diff_deg` | Absolute angular difference between GPS heading vector and candidate arc azimuth $[0^\circ, 180^\circ]$. | Crucial discriminator: penalizes candidate segments in opposite direction (one-ways) or perpendicular cross-streets. |
| `heading_cos_sim` | $\cos(\Delta \theta) \in [-1, 1]$. | Smooth continuous alignment score between motion and road direction. |
| `spatial_likelihood` | Calibrated Gaussian emission score $\exp\left(-\frac{d_{\text{perp}}^2}{2\sigma^2}\right)$ with $\sigma=12\text{m}$. | Models GPS measurement error decay based on sensor noise characteristics. |
| `rank_by_dist` | Rank index of candidate sorted by perpendicular distance ($0, 1, 2, \dots$). | Relative spatial standing among competing local candidates. |
| `is_closest` | Binary indicator ($1$ if candidate has rank 0, else $0$). | Fast baseline flag indicating nearest road segment. |
| `speed_length_ratio` | $\text{gps\_speed} / \text{segment\_length}$. | Traversal feasibility ratio; penalizes unrealistic traversal dynamics. |

---

## 3. Baseline Model Benchmarking (5 Algorithm Families)

We evaluated 6 models across 5 distinct algorithm families. To strictly prevent intra-trip data leakage, cross-validation was conducted using **5-Fold GroupKFold strictly grouped by `trip_id` on the 80 training trips** (observations from the same trip never appear across both train and validation folds):

| Model | Algorithm Family | Key Configuration | 5-Fold CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Binary Log Loss | Latency (ms/1k) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | $C=1.0$, L2, StandardScaler | `0.8685` | 78.46% | 0.8691 | 0.1082 | **0.02 ms** |
| **Gaussian Naive Bayes** | Linear / Probabilistic | $\text{var\_smoothing}=10^{-9}$ | `0.7810` | 52.20% | 0.7806 | 0.7867 | 0.10 ms |
| **Decision Tree** | Tree-based | $\text{max\_depth}=12, \text{min\_split}=20$ | `0.8774` | 79.12% | 0.8874 | 0.1011 | 0.04 ms |
| **Random Forest** | Ensemble Bagging | $N=100, \text{depth}=14, \sqrt{F}$ | `0.8822` | 79.64% | 0.8900 | **0.0831** | 0.73 ms |
| **LightGBM** | Gradient Boosting | $N=150, \eta=0.08, \text{leaves}=31$ | `0.8805` | 79.30% | 0.8879 | 0.0907 | 0.47 ms |
| **XGBoost** | Gradient Boosting | $N=120, \eta=0.10, \text{depth}=6$ | `0.8814` | **79.70%** | 0.8890 | 0.0842 | 0.16 ms |

### Key Experimental Insights:
1. **Tree Ensembles (Random Forest, XGBoost, LightGBM)** achieve the highest observation-level Top-1 accuracy (~79.7%) and lowest log loss (~0.083), effectively capturing non-linear interactions between distance, heading difference, and vehicle speed.
2. **Linear Models (Logistic Regression)** provide an extremely fast baseline (0.02 ms/1k) with respectable accuracy (78.5%), but cannot model angular-spatial non-linearities.
3. **Gaussian Naive Bayes** performs poorly (52.2% Top-1, 0.7867 log loss) due to its strong independence assumption, which fails on correlated spatial features (`dist_to_segment`, `dist_to_start_node`, `spatial_likelihood`).

---

## 4. Repository Structure (Review 1 Deliverable)

```
PathSnap/
├── data_processing/        # Data loading, inspection, and EDA
│   ├── loader.py           # Parsing for .track, .nodes, .arcs, and .route
│   └── eda.py              # Comprehensive EDA and distribution plotting
├── feature_engineering/    # Spatial candidate extraction and feature engineering
│   ├── projection.py       # Conformal local metric coordinate projection (meters)
│   ├── candidate_generator.py # Shapely STRtree R-tree candidate generator (45m radius)
│   ├── extractor.py        # 13 physical, kinematic, and topological features
│   └── dataset_builder.py  # Group-aware dataset builder for Train & Student-Test
├── models/                 # Model training & cross-validation engine
│   └── trainer.py          # 5-Fold GroupKFold CV benchmark across 6 models
├── evaluation/             # Metrics calculation & diagnostic plotting
│   ├── metrics.py          # Top-1 accuracy, route continuity, F1, log loss, latency
│   └── visualize_results.py# Comparative bar charts and loss curves
├── visualization/          # Geographical plotting
│   └── folium_map.py       # Multi-layer Folium interactive map generator (Esri tiles)
├── artifacts/              # Serialized baseline model weights (.joblib) & scaler
├── results/                # Recorded evaluation CSV tables, logs, and plots
│   ├── eda_trip_summary.csv
│   ├── candidate_stats_train.csv
│   ├── candidate_stats_student_test.csv
│   ├── model_comparison.csv
│   ├── folium_maps/        # Interactive HTML map
│   └── plots/              # Diagnostic charts
├── requirements.txt        # Full dependencies
└── README.md               # Review 1 documentation & verification guide
```

---

## 5. Verification Commands for Evaluators

### 1. Run Dataset Exploration & Summary Stats
```bash
python -m data_processing.eda "../Student folder"
```

### 2. View Model Benchmarking Results
```bash
python -m models.trainer
```

### 3. Open Interactive Geographical Map
```bash
open "results/folium_maps/trip_00000005.html"
```
*(Displays raw GPS points, road network segments, and ground-truth route with layer controls).*
