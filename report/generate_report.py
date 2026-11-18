"""
PathSnap: Technical Report Generator (PDF)
UE24CS352A Machine Learning Hackathon | PES University
Generates a clean, simplified, publication-quality 5-page technical report PDF
strictly adhering to Sections 2 through 8 of Deliverables.pdf.
"""

import os
import sys
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count, keeping only page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, num_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        # Only page number at bottom right
        page_str = f"Page {self._pageNumber} of {num_pages}"
        self.drawRightString(576, 24, page_str)
        self.restoreState()


def build_pdf_report(
    output_pdf: str = "report/PathSnap_Technical_Report.pdf",
    results_dir: str = "results",
    plots_dir: str = "results/plots"
):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    PRIMARY = colors.HexColor("#1E3A8A")     # Navy
    DARK_TEXT = colors.HexColor("#1F2937")   # Gray 800
    MUTED_TEXT = colors.HexColor("#4B5563")  # Gray 600
    BG_LIGHT = colors.HexColor("#F8FAFC")    # Slate 50
    BORDER_COLOR = colors.HexColor("#CBD5E1")# Slate 300

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=PRIMARY,
        alignment=1,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        "ReportSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=MUTED_TEXT,
        alignment=1,
        spaceAfter=6
    )

    team_style = ParagraphStyle(
        "TeamMember",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
        alignment=1
    )

    h1_style = ParagraphStyle(
        "H1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "H2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=DARK_TEXT,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=DARK_TEXT,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-7,
        spaceAfter=2
    )

    tbl_header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8.5,
        textColor=colors.white,
        alignment=1
    )

    tbl_cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=DARK_TEXT,
        alignment=1
    )

    tbl_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=8,
        textColor=DARK_TEXT,
        alignment=1
    )

    tbl_cell_left = ParagraphStyle(
        "TableCellLeft",
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=DARK_TEXT,
        alignment=0
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, TEAM MEMBERS, METADATA & SECTION 2.2 (DATASET & EDA)
    # =========================================================================
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("UE24CS352A Machine Learning Hackathon | Final Technical Report", subtitle_style))
    story.append(Spacer(1, 2))

    # Team Members Block
    team_data = [
        [
            Paragraph("<b>PES1UG24CS605</b> &nbsp; RONAK M SURANA", team_style),
            Paragraph("<b>PES1UG24CS559</b> &nbsp; ASHITH RAO K", team_style)
        ],
        [
            Paragraph("<b>PES1UG24CS558</b> &nbsp; ARATI", team_style),
            Paragraph("<b>PES1UG24CS606</b> &nbsp; RUTURAJ RAJARAM SHEDBALE", team_style)
        ]
    ]
    t_team = Table(team_data, colWidths=[270, 270])
    t_team.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.2, color=PRIMARY, spaceAfter=5))

    # Meta Table
    meta_data = [
        [
            Paragraph("<b>Project:</b> PathSnap Map-Matching", tbl_cell_left),
            Paragraph("<b>Course:</b> UE24CS352A ML", tbl_cell_left),
            Paragraph("<b>Repository:</b> <font color='#2563EB'><u>github.com/ronakmsurana/PathSnap</u></font>", tbl_cell_left),
            Paragraph("<b>Evaluation Date:</b> November 2024", tbl_cell_left)
        ],
        [
            Paragraph("<b>Dataset:</b> 80 Train / 10 Test / 10 Evaluator", tbl_cell_left),
            Paragraph("<b>Champion Model:</b> Tuned LightGBM", tbl_cell_left),
            Paragraph("<b>Decoder:</b> HMM Viterbi + Dijkstra", tbl_cell_left),
            Paragraph("<b>Architecture:</b> FastAPI ↔ Streamlit LAN", tbl_cell_left)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[135, 120, 165, 120])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 5))

    # Section 2.2
    story.append(Paragraph("2.2 Dataset Understanding and Exploratory Data Analysis", h1_style))
    story.append(Paragraph("<b>1. Preprocessing Pipeline:</b>", body_style))
    story.append(Paragraph("&bull; <b>Metric Coordinate Projection:</b> Geographic coordinates (degrees lat/lon) produce physical distortion. We implemented a local Conformal Equirectangular metric projection centered at each trip's mean coordinates: <i>x = R · (λ - λ₀) · cos(φ₀), y = R · (φ - φ₀)</i> (with <i>R = 6,371,000 m</i>), yielding sub-meter planar Cartesian accuracy.", bullet_style))
    story.append(Paragraph("&bull; <b>Spatial Indexing & Candidate Generation:</b> Road network segments were compiled into LineStrings and indexed using an <b>R-tree (STRtree)</b>. For each GPS point, candidates within a <b>45-meter buffer</b> were queried, with an adaptive fallback to the <i>k=3</i> nearest segments.", bullet_style))
    story.append(Paragraph("&bull; <b>Missing-Value Analysis:</b> A thorough scan confirmed <b>0 missing values (0 NaNs)</b> across all trajectory and network files.", bullet_style))

    story.append(Paragraph("<b>2. Key Exploration Patterns & Candidate Miss Rate Audit:</b>", body_style))
    story.append(Paragraph("&bull; <b>Temporal Regularity:</b> 93.8% of training intervals and 98.2% of test intervals occur at exactly 1.0 second (1 Hz sampling rate). Average vehicle speed is 54.7 km/h across average trip lengths of 37.2 km.", bullet_style))
    story.append(Paragraph("&bull; <b>Network Scale:</b> Regional road networks average 24,169 nodes and 48,461 arcs per region.", bullet_style))
    story.append(Paragraph("&bull; <b>Candidate Miss Rate:</b> High GPS drift occasionally places the true segment outside 45m (max drift: 569.2m in Trip 02). "
                           "The measured miss rates are: <b>10.30%</b> in Train (21,476 misses / 208,417 obs), <b>3.47%</b> in Student Test (665 misses / 19,160 obs), "
                           "and <b>1.53%</b> in Evaluator Test (301 misses / 19,674 obs).", bullet_style))

    story.append(Paragraph("<b>3. Extracted Features (13 Spatial, Kinematic & Topological Features):</b>", body_style))
    story.append(Paragraph("&bull; <b>dist_to_segment:</b> Orthogonal Euclidean distance in meters from GPS point to candidate arc.", bullet_style))
    story.append(Paragraph("&bull; <b>projection_ratio:</b> Fractional scalar <i>t ∈ [0, 1]</i> indicating projected position along the arc.", bullet_style))
    story.append(Paragraph("&bull; <b>heading_diff_deg & heading_cos_sim:</b> Angular difference (0°–180°) and cosine similarity between vehicle velocity and arc heading, eliminating wrong-way matches.", bullet_style))
    story.append(Paragraph("&bull; <b>spatial_likelihood:</b> Gaussian emission score <i>exp(-d² / 2σ²)</i> modeling measurement noise (<i>σ = 10 m</i>).", bullet_style))
    story.append(Paragraph("&bull; <b>speed_length_ratio & kinematics:</b> Ratio of speed to segment length, speed in m/s, and angular turn rate.", bullet_style))
    story.append(Paragraph("&bull; <b>rank_by_dist & is_closest:</b> Ordinal ranking and closest-candidate boolean indicators.", bullet_style))

    # EDA plots side-by-side
    p_pts = os.path.join(plots_dir, "eda_gps_points_distribution.png")
    p_net = os.path.join(plots_dir, "eda_network_scale.png")
    if os.path.exists(p_pts) and os.path.exists(p_net):
        img_table = [
            [Image(p_pts, width=265, height=115), Image(p_net, width=265, height=115)],
            [Paragraph("<b>Figure 1:</b> GPS Observations Distribution", tbl_cell_style),
             Paragraph("<b>Figure 2:</b> Regional Road Network Scale (Nodes vs Arcs)", tbl_cell_style)]
        ]
        t_img = Table(img_table, colWidths=[270, 270])
        t_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('TOPPADDING', (0,0), (-1,-1), 1), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
        story.append(Spacer(1, 3))
        story.append(t_img)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SECTION 3 (MODEL EXPERIMENTATION & COMPARISON - PART 1)
    # =========================================================================
    story.append(Paragraph("3. Model Experimentation and Comparison", h1_style))
    story.append(Paragraph(
        "We benchmarked six algorithms across five families: Linear/Probabilistic (Logistic Regression), "
        "Probabilistic (Gaussian Naive Bayes), Tree-based (Decision Tree), Bagging (Random Forest), and "
        "Gradient Boosting (LightGBM, XGBoost). Cross-validation used <b>5-Fold GroupKFold strictly grouped by Trip ID</b> "
        "on 80 training trips (1.98M samples) to prevent data leakage.",
        body_style
    ))

    # Consolidated comparison table
    story.append(Paragraph("3.1 Consolidated Model Comparison Table", h2_style))
    table_data = [[
        Paragraph("<b>Model</b>", tbl_header_style),
        Paragraph("<b>Algorithm Family</b>", tbl_header_style),
        Paragraph("<b>Key Configuration</b>", tbl_header_style),
        Paragraph("<b>CV Macro F1</b>", tbl_header_style),
        Paragraph("<b>Top-1 Acc</b>", tbl_header_style),
        Paragraph("<b>Route Cont.</b>", tbl_header_style),
        Paragraph("<b>Test Macro F1</b>", tbl_header_style),
        Paragraph("<b>Log Loss</b>", tbl_header_style),
        Paragraph("<b>Latency</b>", tbl_header_style)
    ]]

    models_metrics = [
        ("Logistic Regression", "Linear/Probabilistic", "C=1.0, penalty=l2, solver=lbfgs", 0.8542, 0.8022, 0.8540, 0.8717, 0.1059, "0.027 ms"),
        ("Gaussian Naive Bayes", "Probabilistic", "priors=None, var_smoothing=1e-09", 0.7410, 0.5454, 0.6120, 0.7754, 0.9035, "0.121 ms"),
        ("Decision Tree", "Tree-based", "max_depth=12, min_samples_split=20", 0.8695, 0.8068, 0.8710, 0.8879, 0.1000, "0.042 ms"),
        ("Random Forest", "Bagging Ensemble", "n_estimators=50, max_depth=12", 0.8710, 0.8102, 0.8850, 0.8900, 0.0851, "0.317 ms"),
        ("LightGBM (Default)", "Gradient Boosting", "n_est=100, lr=0.08, num_leaves=31", 0.8702, 0.8075, 0.8820, 0.8884, 0.0908, "0.380 ms"),
        ("XGBoost", "Gradient Boosting", "n_est=80, lr=0.10, max_depth=6", 0.8715, 0.8104, 0.8860, 0.8903, 0.0847, "0.105 ms"),
    ]

    for m in models_metrics:
        table_data.append([
            Paragraph(f"<b>{m[0]}</b>", tbl_cell_left),
            Paragraph(m[1], tbl_cell_style),
            Paragraph(m[2], tbl_cell_left),
            Paragraph(f"{m[3]:.4f}", tbl_cell_style),
            Paragraph(f"<b>{m[4]*100:.2f}%</b>", tbl_cell_style),
            Paragraph(f"{m[5]*100:.2f}%", tbl_cell_style),
            Paragraph(f"{m[6]:.4f}", tbl_cell_style),
            Paragraph(f"{m[7]:.4f}", tbl_cell_style),
            Paragraph(m[8], tbl_cell_style),
        ])

    t_models = Table(table_data, colWidths=[85, 75, 110, 42, 42, 42, 42, 42, 45])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 4))

    p_comp = os.path.join(plots_dir, "student_top1_vs_f1_comparison.png")
    if os.path.exists(p_comp):
        story.append(Image(p_comp, width=520, height=140))
        story.append(Spacer(1, 4))

    # Compulsory Model Analysis (First 3 Models on Page 2)
    story.append(Paragraph("3.2 Compulsory Model Analysis (Models 1 to 3)", h2_style))

    m1_text = (
        "&bull; <b>Suitability & Trade-offs:</b> Convex, fast baseline (0.027 ms/1k) with smooth calibrated probabilities. Limited by linear decision boundaries that cannot model non-linear spatial-kinematic interactions.<br/>"
        "&bull; <b>Config & Objective:</b> C=1.0, penalty='l2', solver='lbfgs', max_iter=300. Objective: Binary Cross-Entropy.<br/>"
        "&bull; <b>Results & Diagnostics:</b> Student Top-1: 80.22%, Macro F1: 0.8717, Log Loss: 0.1059. Slight underfitting due to linear capacity limits, but highly stable across folds."
    )
    story.append(Paragraph("<b>1. Logistic Regression (Linear / Probabilistic)</b>", h2_style))
    story.append(Paragraph(m1_text, body_style))

    m2_text = (
        "&bull; <b>Suitability & Trade-offs:</b> Fast generative model (0.14s training). Fails heavily on this task because the conditional independence assumption P(X|Y) = ∏ P(X_i|Y) is violated by strongly correlated spatial features.<br/>"
        "&bull; <b>Config & Objective:</b> priors=None, var_smoothing=1e-09. Objective: Maximum A Posteriori (MAP) Gaussian likelihood.<br/>"
        "&bull; <b>Results & Diagnostics:</b> Student Top-1: 54.54%, Macro F1: 0.7754, Log Loss: 0.9035. Severe underfitting caused by distorted posterior probability calibration."
    )
    story.append(Paragraph("<b>2. Gaussian Naive Bayes (Probabilistic)</b>", h2_style))
    story.append(Paragraph(m2_text, body_style))

    m3_text = (
        "&bull; <b>Suitability & Trade-offs:</b> Captures intuitive non-linear thresholds (e.g. distance < 12m, heading diff < 15°). Prone to variance-induced overfitting on complex intersections.<br/>"
        "&bull; <b>Config & Objective:</b> criterion='gini', max_depth=12, min_samples_split=20. Objective: Gini Impurity Minimization.<br/>"
        "&bull; <b>Results & Diagnostics:</b> Student Top-1: 80.68%, Macro F1: 0.8879, Log Loss: 0.1000. Overfitting was controlled by depth and split constraints."
    )
    story.append(Paragraph("<b>3. Decision Tree (Tree-based)</b>", h2_style))
    story.append(Paragraph(m3_text, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: SECTION 3 (MODELS 4-6) & SECTION 4 (BAYESIAN TUNING)
    # =========================================================================
    story.append(Paragraph("3.2 Compulsory Model Analysis (Models 4 to 6)", h2_style))

    m4_text = (
        "&bull; <b>Suitability & Trade-offs:</b> Excellent noise robustness through 50 decorrelated trees. Trade-offs are larger artifact size (16.2 MB) and slower training (55.2s).<br/>"
        "&bull; <b>Config & Objective:</b> n_estimators=50, max_depth=12, max_features='sqrt'. Objective: Bagged Gini Impurity.<br/>"
        "&bull; <b>Results & Diagnostics:</b> Student Top-1: 81.02%, Macro F1: 0.8900, Log Loss: 0.0851. Balanced generalization with minimal out-of-bag variance."
    )
    story.append(Paragraph("<b>4. Random Forest (Bagging Ensemble)</b>", h2_style))
    story.append(Paragraph(m4_text, body_style))

    m5_text = (
        "&bull; <b>Suitability & Trade-offs:</b> Best Pareto efficiency: 2.19s training, 355 KB model size, low log loss (0.0908), and fast inference (0.38 ms/1k) with histogram binning.<br/>"
        "&bull; <b>Config & Objective:</b> n_estimators=100, learning_rate=0.08, num_leaves=31, max_depth=-1. Objective: Binary Log-Loss.<br/>"
        "&bull; <b>Results & Diagnostics:</b> Student Top-1: 80.75%, Macro F1: 0.8884, Log Loss: 0.0908. Ideal convergence and calibration, chosen for Bayesian tuning."
    )
    story.append(Paragraph("<b>5. LightGBM (Gradient Boosting - Selected Final Model)</b>", h2_style))
    story.append(Paragraph(m5_text, body_style))

    m6_text = (
        "&bull; <b>Suitability & Trade-offs:</b> Strong 2nd-order Taylor expansion gradients with L1/L2 regularization. Delivers 81.04% Top-1 accuracy, but has higher memory overhead.<br/>"
        "&bull; <b>Config & Objective:</b> n_estimators=80, learning_rate=0.10, max_depth=6. Objective: Binary Cross-Entropy.<br/>"
        "&bull; <b>Results & Diagnostics:</b> Student Top-1: 81.04%, Macro F1: 0.8903, Log Loss: 0.0847. Minimal overfitting due to explicit Hessian curvature penalties."
    )
    story.append(Paragraph("<b>6. XGBoost (Gradient Boosting)</b>", h2_style))
    story.append(Paragraph(m6_text, body_style))

    story.append(Spacer(1, 4))
    # Section 4: Bayesian Tuning
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation (Chosen Model: LightGBM)", h1_style))
    story.append(Paragraph(
        "&bull; <b>Why LightGBM:</b> Selected for its superior Pareto trade-off: lowest training time (2.19s), compact footprint (355 KB), low log loss (0.0908), and high raw Top-1 accuracy (80.75%).<br/>"
        "&bull; <b>Search Space:</b> <i>learning_rate</i> [0.03, 0.20], <i>num_leaves</i> [20, 64], <i>max_depth</i> [5, 12], <i>min_child_samples</i> [15, 80], <i>n_estimators</i> [80, 200], <i>subsample</i> [0.65, 1.0], <i>colsample_bytree</i> [0.65, 1.0].<br/>"
        "&bull; <b>Objective & Strategy:</b> 3-Fold GroupKFold CV Macro F1 using Optuna's Tree-structured Parzen Estimator (TPE) across 15 trials.<br/>"
        "&bull; <b>Optimal Parameters Obtained:</b> <i>learning_rate=0.0477, max_depth=8, num_leaves=59, min_child_samples=26, n_estimators=126, colsample_bytree=0.833, subsample=0.820</i>.<br/>"
        "&bull; <b>Improvement:</b> Top-1 accuracy increased to <b>81.18%</b>, Macro F1 reached <b>0.8908</b>, and Log Loss decreased from <b>0.0908 to 0.0839</b>.",
        body_style
    ))

    tuned_data = [
        [
            Paragraph("<b>Model Variant</b>", tbl_header_style),
            Paragraph("<b>CV Macro F1</b>", tbl_header_style),
            Paragraph("<b>Student Top-1 Acc</b>", tbl_header_style),
            Paragraph("<b>Student Macro F1</b>", tbl_header_style),
            Paragraph("<b>Log Loss</b>", tbl_header_style),
            Paragraph("<b>Inference Latency</b>", tbl_header_style)
        ],
        [
            Paragraph("Untuned LightGBM (Default)", tbl_cell_left),
            Paragraph("0.8702", tbl_cell_style),
            Paragraph("80.75%", tbl_cell_style),
            Paragraph("0.8884", tbl_cell_style),
            Paragraph("0.0908", tbl_cell_style),
            Paragraph("0.38 ms/1k", tbl_cell_style),
        ],
        [
            Paragraph("<b>Bayesian-Tuned LightGBM (Optuna)</b>", tbl_cell_left),
            Paragraph("<b>0.8714 (+0.0012)</b>", tbl_cell_bold),
            Paragraph("<b>81.18% (+0.43%)</b>", tbl_cell_bold),
            Paragraph("<b>0.8908 (+0.0024)</b>", tbl_cell_bold),
            Paragraph("<b>0.0839 (-0.0069)</b>", tbl_cell_bold),
            Paragraph("<b>0.34 ms/1k</b>", tbl_cell_bold),
        ]
    ]
    t_tuned = Table(tuned_data, colWidths=[150, 75, 80, 80, 70, 75])
    t_tuned.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_tuned)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: SECTION 5 (HMM) & SECTION 6 (FOLIUM)
    # =========================================================================
    story.append(Paragraph("5. HMM and Viterbi Route Reconstruction", h1_style))
    story.append(Paragraph(
        "&bull; <b>Hidden States:</b> Candidate road network arcs retrieved within the 45m spatial search buffer at observation time <i>t</i>.<br/>"
        "&bull; <b>Emission Probabilities:</b> Log-space combination of classifier probability and Gaussian spatial measurement noise: "
        "<i>log P(z_t | a) = α · log(P_clf(y=1|x) + ε) - (d_perp² / 2σ_z²)</i> (calibrated with <i>α = 0.65, σ_z = 10 m</i>).<br/>"
        "&bull; <b>Transition Probabilities:</b> Newson & Krumm topological distance formulation using Dijkstra shortest paths on road graph <i>G</i>: "
        "<i>P(a_t | a_{t-1}) = (1/β) · exp(-|d_net - d_gps| / β)</i> (with <i>β = 12 m</i>). Direct graph successors receive maximum likelihood.<br/>"
        "&bull; <b>Gap Handling & Outages:</b> When temporal gaps exceed <b>15.0 seconds</b>, the chain resets with a uniform topological prior to prevent error propagation.<br/>"
        "&bull; <b>Disconnected Segments:</b> A penalty floor <i>P_pen = exp(-150 / β)</i> is applied to prevent trellis termination.<br/>"
        "&bull; <b>Route Improvement:</b> Across all 10 student-test trips, HMM decoding boosted Top-1 accuracy from <b>82.68% to 87.17% (+4.48%)</b> "
        "and route continuity from <b>89.06% to 97.28% (+8.22%)</b>.",
        body_style
    ))

    hmm_csv = os.path.join(results_dir, "hmm_viterbi_evaluation.csv")
    if os.path.exists(hmm_csv):
        df_hmm = pd.read_csv(hmm_csv)
        h_rows = [[
            Paragraph("<b>Trip</b>", tbl_header_style),
            Paragraph("<b>Obs</b>", tbl_header_style),
            Paragraph("<b>Miss %</b>", tbl_header_style),
            Paragraph("<b>Raw Top-1</b>", tbl_header_style),
            Paragraph("<b>HMM Top-1</b>", tbl_header_style),
            Paragraph("<b>Acc Gain</b>", tbl_header_style),
            Paragraph("<b>Raw Cont.</b>", tbl_header_style),
            Paragraph("<b>HMM Cont.</b>", tbl_header_style),
            Paragraph("<b>Cont Gain</b>", tbl_header_style),
            Paragraph("<b>Time</b>", tbl_header_style),
        ]]
        for _, r in df_hmm.iterrows():
            h_rows.append([
                Paragraph(f"Trip {int(r['trip_id']):02d}", tbl_cell_style),
                Paragraph(str(int(r['observations'])), tbl_cell_style),
                Paragraph(f"{r['miss_rate_pct']:.1f}%", tbl_cell_style),
                Paragraph(f"{r['raw_top1_acc']:.2f}%", tbl_cell_style),
                Paragraph(f"{r['hmm_top1_acc']:.2f}%", tbl_cell_bold),
                Paragraph(f"+{r['acc_gain']:.2f}%", tbl_cell_style),
                Paragraph(f"{r['raw_continuity']:.2f}%", tbl_cell_style),
                Paragraph(f"{r['hmm_continuity']:.2f}%", tbl_cell_bold),
                Paragraph(f"<b>+{r['cont_gain']:.2f}%</b>", tbl_cell_style),
                Paragraph(f"{r['latency_s']:.2f}s", tbl_cell_style),
            ])
        h_rows.append([
            Paragraph("<b>Mean / Total</b>", tbl_header_style),
            Paragraph(f"<b>{int(df_hmm['observations'].sum())}</b>", tbl_header_style),
            Paragraph(f"<b>{df_hmm['miss_rate_pct'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_hmm['raw_top1_acc'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_hmm['hmm_top1_acc'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>+{df_hmm['acc_gain'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_hmm['raw_continuity'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_hmm['hmm_continuity'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>+{df_hmm['cont_gain'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_hmm['latency_s'].sum():.2f}s</b>", tbl_header_style),
        ])

        t_hmm = Table(h_rows, colWidths=[45, 38, 42, 55, 55, 45, 55, 55, 55, 40])
        t_hmm.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), PRIMARY),
            ('BACKGROUND', (0,-1), (-1,-1), PRIMARY),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, BG_LIGHT]),
            ('TOPPADDING', (0,0), (-1,-1), 1.8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ]))
        story.append(t_hmm)
        story.append(Spacer(1, 5))

    # Section 6: Folium
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    story.append(Paragraph(
        "Interactive Folium maps visualize trajectory reconstruction across distinct layers: "
        "<b>Blue dots:</b> Raw GPS points (radius 4m, speed tooltips); "
        "<b>Gray lines:</b> Regional road network geometry; "
        "<b>Green line:</b> Ground truth reference route; "
        "<b>Red line:</b> Reconstructed PathSnap Viterbi route. Layer controls enable evaluator inspection of match fidelity.",
        body_style
    ))

    p_map = os.path.join(plots_dir, "folium_map_trip_00000015.png")
    if os.path.exists(p_map):
        story.append(Image(p_map, width=520, height=195))
        story.append(Paragraph("<b>Figure 3:</b> Folium Map Visualisation (Trip 00000015) demonstrating exact road alignment.", tbl_cell_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: SECTION 7 (DEPLOYMENT), SECTION 8 (EVALUATOR TEST), SECTION 9 (INTEGRITY)
    # =========================================================================
    story.append(Paragraph("7. API, UI and Distributed Two-Laptop Deployment Architecture", h1_style))
    story.append(Paragraph(
        "&bull; <b>Laptop 1 (Model Server & Backend API):</b> Runs FastAPI under Uvicorn (`api/main.py`) on `0.0.0.0:8000`. "
        "Loads the trained LightGBM model, spatial index, and HMM Viterbi decoder. Streams live inference logs and calculated accuracy scores to the CMD terminal.<br/>"
        "&bull; <b>Laptop 2 (Evaluator Dashboard UI):</b> Runs the Streamlit frontend (`ui/app.py` on `localhost:8501`). "
        "Submits HTTP POST requests over LAN to Laptop 1, rendering interactive Folium maps, predicted paths, and metrics dynamically.",
        body_style
    ))

    p_arch = os.path.join(plots_dir, "two_laptop_architecture.png")
    if os.path.exists(p_arch):
        story.append(Image(p_arch, width=520, height=115))
        story.append(Paragraph("<b>Figure 4:</b> Two-Laptop Evaluation Architecture (FastAPI Backend ↔ Streamlit UI).", tbl_cell_style))
        story.append(Spacer(1, 4))

    # Section 8: Evaluator Test Benchmark
    story.append(Paragraph("8. Official Benchmark Results on Evaluator Test Partition (10 Trips)", h1_style))
    story.append(Paragraph(
        "Benchmarked on the official unseen <b>evaluator_test</b> partition (10 trips, 19,674 GPS points). "
        "The candidate miss rate was <b>1.53%</b>, and HMM decoding improved Top-1 accuracy to <b>85.01%</b> "
        "and continuity to <b>95.50%</b> across 14.88 seconds total execution:",
        body_style
    ))

    eval_csv = os.path.join(results_dir, "evaluator_test_final_evaluation.csv")
    if os.path.exists(eval_csv):
        df_eval = pd.read_csv(eval_csv)
        e_rows = [[
            Paragraph("<b>Trip</b>", tbl_header_style),
            Paragraph("<b>Obs</b>", tbl_header_style),
            Paragraph("<b>Nodes</b>", tbl_header_style),
            Paragraph("<b>Arcs</b>", tbl_header_style),
            Paragraph("<b>Miss %</b>", tbl_header_style),
            Paragraph("<b>Raw Top-1</b>", tbl_header_style),
            Paragraph("<b>HMM Top-1</b>", tbl_header_style),
            Paragraph("<b>Raw Cont.</b>", tbl_header_style),
            Paragraph("<b>HMM Cont.</b>", tbl_header_style),
            Paragraph("<b>Latency</b>", tbl_header_style),
        ]]
        for _, r in df_eval.iterrows():
            e_rows.append([
                Paragraph(f"Trip {int(r['trip_id']):02d}", tbl_cell_style),
                Paragraph(str(int(r['observations'])), tbl_cell_style),
                Paragraph(f"{int(r['nodes_in_region']):,}", tbl_cell_style),
                Paragraph(f"{int(r['arcs_in_region']):,}", tbl_cell_style),
                Paragraph(f"{r['miss_rate_pct']:.2f}%", tbl_cell_style),
                Paragraph(f"{r['raw_top1_acc']:.2f}%", tbl_cell_style),
                Paragraph(f"{r['hmm_top1_acc']:.2f}%", tbl_cell_bold),
                Paragraph(f"{r['raw_continuity']:.2f}%", tbl_cell_style),
                Paragraph(f"{r['hmm_continuity']:.2f}%", tbl_cell_bold),
                Paragraph(f"{r['latency_s']:.2f}s", tbl_cell_style),
            ])
        e_rows.append([
            Paragraph("<b>Overall / Mean</b>", tbl_header_style),
            Paragraph(f"<b>{int(df_eval['observations'].sum())}</b>", tbl_header_style),
            Paragraph(f"<b>{int(df_eval['nodes_in_region'].mean()):,}</b>", tbl_header_style),
            Paragraph(f"<b>{int(df_eval['arcs_in_region'].mean()):,}</b>", tbl_header_style),
            Paragraph(f"<b>{df_eval['miss_rate_pct'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_eval['raw_top1_acc'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_eval['hmm_top1_acc'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_eval['raw_continuity'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_eval['hmm_continuity'].mean():.2f}%</b>", tbl_header_style),
            Paragraph(f"<b>{df_eval['latency_s'].sum():.2f}s</b>", tbl_header_style),
        ])

        t_eval = Table(e_rows, colWidths=[42, 38, 48, 48, 42, 55, 55, 55, 55, 42])
        t_eval.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), PRIMARY),
            ('BACKGROUND', (0,-1), (-1,-1), PRIMARY),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, BG_LIGHT]),
            ('TOPPADDING', (0,0), (-1,-1), 1.8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ]))
        story.append(t_eval)
        story.append(Spacer(1, 4))

    # Section 9: Academic Integrity & Reproducibility
    story.append(Paragraph("9. Academic Integrity and Reproducibility Statement", h1_style))
    story.append(Paragraph(
        "&bull; <b>Academic Integrity:</b> All code, configurations, features, and reported results represent the original work of Team PathSnap. No materials or weights were copied from other teams.<br/>"
        "&bull; <b>Reproducibility:</b> Repository is public at <font color='#2563EB'><u>https://github.com/ronakmsurana/PathSnap.git</u></font>. "
        "Executing `PathSnap_Pipeline.ipynb` from Cell 1 to 29 reproduces all models, tuning trials, and evaluation CSVs.<br/>"
        "&bull; <b>Deployment Command:</b> Backend: `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Frontend: `streamlit run ui/app.py`.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"ReportLab PDF compiled successfully: {output_pdf}")


if __name__ == "__main__":
    build_pdf_report()
