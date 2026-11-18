"""Build the PathSnap technical report from saved experiment results and demo evidence."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


INK = colors.HexColor("#112d3b")
TEAL = colors.HexColor("#086d68")
MUTED = colors.HexColor("#476069")
PAPER = colors.HexColor("#eef5f1")

STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=43, leading=47, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontSize=17, leading=23, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=24, leading=31, textColor=INK,
                         spaceAfter=12),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=TEAL,
                         spaceBefore=10, spaceAfter=4, keepWithNext=True),
    "body": ParagraphStyle("body", fontSize=10, leading=14.5, textColor=INK, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontSize=8.8, leading=12, textColor=MUTED, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontSize=8.1, leading=10.4, textColor=INK, alignment=TA_LEFT),
    "header": ParagraphStyle("header", fontName="Helvetica-Bold", fontSize=8.1, leading=10.4,
                             textColor=colors.white),
    "terminal": ParagraphStyle("terminal", fontName="Courier", fontSize=7.7, leading=11,
                                textColor=colors.white),
}


MODEL_DETAILS = {
    "Logistic Regression": ("A linear log-odds baseline tests whether features consistently push scores "
                            "in one direction. Standardized inputs reduce scale effects. It minimizes "
                            "regularized binary log loss. A linear decision boundary can miss interactions "
                            "between heading, speed and segment distance."),
    "Gaussian Naive Bayes": ("Gaussian likelihoods give a fast probability baseline. The classifier uses "
                            "class-conditional means and variances with additive variance smoothing. "
                            "Distance and distance above the nearest candidate are correlated, and raw "
                            "distance distributions are skewed. Those assumptions can hurt probability quality."),
    "Decision Tree": ("Threshold splits can use combinations of heading, distance and speed. The tree "
                       "minimizes Gini impurity and uses a depth cap and a minimum leaf size. A single tree's "
                       "piecewise-constant scores can be sensitive to individual splits."),
    "Random Forest": ("Bagged trees capture feature interactions and average leaf probabilities to reduce "
                       "variance. Each tree uses Gini splits, with restricted depth and leaf size. More "
                       "trees increase fitting and prediction cost."),
    "Histogram Gradient Boosting": ("Sequential trees optimize binary log loss on binned features. Binning "
                                    "reduces split search cost and leaf limits constrain complexity. The model "
                                    "can learn that heading matters more when recent motion is consistent."),
}


def pct(value):
    return f"{value:.2%}"


def params(config):
    return ", ".join(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                     for key, value in config.items())


def read(path):
    return json.loads(path.read_text())


def page(canvas, document):
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#c7dcd2"))
    canvas.line(45, 40, width - 45, 40)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(45, 27, "PathSnap  /  UE24CS352A  /  PES University")
    canvas.drawRightString(width - 45, 27, f"{document.page:02d}")


def build(results, evidence, members_path, repo):
    inventory = read(results / "dataset_summary.json")
    metrics = read(results / "model_metrics.json")
    routes = read(results / "route_metrics.json")
    api_passed = "All 10 live trip requests passed." in (evidence / "api_check.log").read_text()
    if not api_passed:
        raise ValueError("API verification log has no success marker")
    log = (evidence / "api_server.log").read_text().splitlines()
    receipt = [line for line in log if "PathSnap student trip=" in line or '"GET /match/' in line]
    receipt = [line for line in receipt if "404 Not Found" not in line][:8]
    if not receipt:
        raise ValueError("Missing inference receipt from API server log")
    with (results / "route_metrics_by_trip.csv").open() as file:
        per_trip = defaultdict(dict)
        for row in csv.DictReader(file):
            per_trip[row["trip"]][row["model"]] = row
    tuned = metrics["tuned"]
    winner = metrics["selected"]
    deployed = "Tuned" if metrics["deployed_configuration"] == "tuned" else winner
    untuned = metrics["models"][winner]
    only = routes[deployed]
    with_viterbi = routes["Selected + Viterbi"]
    members = [] if not members_path.exists() else read(members_path)
    if members and (len(members) != 4 or any(not item["name"] or not item["srn"] for item in members)):
        raise ValueError("The team roster must have four names and SRNs")
    story = []

    def p(text, style="body"):
        story.append(Paragraph(text, STYLES[style]))

    def table(headers, rows, widths=None):
        items = [[Paragraph(escape(str(v)), STYLES["header"]) for v in headers]]
        items += [[Paragraph(escape(str(v)), STYLES["cell"]) for v in row] for row in rows]
        t = Table(items, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend((t, Spacer(1, 10)))

    def image(name, width, max_height):
        path = evidence / name
        photo = Image(str(path))
        scale = min(width / photo.imageWidth, max_height / photo.imageHeight)
        photo.drawWidth = photo.imageWidth * scale
        photo.drawHeight = photo.imageHeight * scale
        story.extend((photo, Spacer(1, 7)))

    def new_page(title):
        story.append(PageBreak())
        p(title, "h1")

    p("PES UNIVERSITY · DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", "caption")
    story.append(Spacer(1, 55))
    p("PathSnap", "title")
    p("Distributed map matching and route reconstruction", "subtitle")
    story.append(Spacer(1, 25))
    p("UE24CS352A Machine Learning Hackathon · Technical Report", "h2")
    if members:
        table(["Team member", "SRN"], [[m["name"], m["srn"]] for m in members], [315, 177])
    else:
        p("Draft · team member names and SRNs pending", "caption")
    if repo:
        p(f'Repository: <a href="{escape(repo)}" color="#086d68">{escape(repo)}</a>')
    p("Experimental result", "h2")
    p(f"Across 10 student-test trips, the deployed classifier reached {pct(only['top1'])} "
      f"observation Top-1 before decoding and {pct(with_viterbi['top1'])} after Viterbi. "
      f"Feasible transition continuity changed from {pct(only['continuity'])} to "
      f"{pct(with_viterbi['continuity'])}. Ordered route F1 changed from {pct(only['sequence_f1'])} "
      f"to {pct(with_viterbi['sequence_f1'])}.")
    p("Data scope", "h2")
    p("All results use the supplied 80 training and 10 student-test trips. Reserved evaluator "
      "trips were unavailable in the development package. The live evidence uses the student-test split.")
    p("Assistance disclosure", "h2")
    p("Codex assisted with implementation, experiment execution, debugging, and assembling the report "
      "from recorded outputs. Experiment logs and commands accompany the source for reproduction.")

    new_page("2. Dataset and exploratory analysis")
    p("Each .track row contains longitude, latitude and a timestamp. .nodes rows index road coordinates. "
      ".arcs rows index directed edges and refer to node rows. .route lists those arc IDs in the supplied "
      "traversal order. All development files parsed with finite coordinates, increasing timestamps and valid IDs.")
    table(["Split", "Trips", "GPS fixes", "Aligned", "Gaps >30 s", "Largest gap", "Candidate miss"],
          [[name, v["trips"], f'{v["observations"]:,}', f'{v["aligned"]:,}',
            v["gaps_over_30s"], f'{v["max_gap_s"]:,.0f} s', pct(v["candidate_miss_rate"])]
           for name, v in inventory.items()],
          [77, 43, 73, 72, 67, 76, 84])
    image("dataset_summary.png", 492, 184)
    p("The trip sizes vary widely. Training trip 00000090 has a 153,064 s gap, and the inventory found "
      "13 non-adjacent reference arc pairs in training and one in student testing. The full CSV also "
      "records weak components, isolated nodes, coordinate bounds and zero-length arcs.")
    p("Observation labels and features", "h2")
    p("A reference-only dynamic program aligns observations to up to twelve nearby reference occurrences. "
      "Its cost combines distance, heading and progress along the ordered route. It allows limited "
      "backtracking from GPS jitter and weakens the progress cost after long timestamp gaps. "
      "At distances above 100 m, the observation is left unlabelled. Both metrics and training use these "
      "approximate labels, and results may favor behavior similar to the alignment heuristic.")
    p("A trip-centered azimuthal equidistant projection supplies local metres. A spatial index queries directed "
      "segments within 65 m, keeps the closest ten and falls back to the closest arc when the radius query "
      "is empty. Offset and overrun features describe where the point projects on the segment. Heading "
      "agreement is computed at two time scales. Speed, elapsed time, heading support, local road count "
      "and arc length give context. All features are computed from GPS and the road graph.")

    new_page("3. Model experimentation")
    p(f"Three GroupKFold folds use whole training trips. Fitting samples at most "
      f"{metrics['training_observations_per_trip_cap']} aligned observations per trip, drops fitting observations "
      f"without a positive candidate, and uses {metrics['candidate_fit_rows']:,} candidate rows in the full "
      f"fit. Validation and student testing score all aligned observations, including misses for Top-1. "
      f"Candidate classification metrics use a 0.5 probability cutoff. Inference times sum candidate scoring "
      f"over the ten student trips. Macro F1 weights both classes equally despite the many incorrect candidates per fix. {winner} led on this criterion.")
    table(["Model / family / key setting", "CV F1", "Top-1", "Cont.", "Macro P / R / F1", "Log loss", "Infer. s"],
          [[f"{name} / {v['family']} / {next(iter(v['config']))}={next(iter(v['config'].values()))}", f"{v['cv_macro_f1']:.3f}", pct(v["student"]["top1"]),
            pct(routes[name]["continuity"]),
            " / ".join(f"{v['student'][m]:.3f}" for m in ("macro_precision", "macro_recall", "macro_f1")),
            f"{v['student']['log_loss']:.3f}", f"{v['student']['inference_s']:.2f}"]
           for name, v in metrics["models"].items()],
          [114, 43, 52, 51, 108, 62, 62])
    baseline = metrics["baseline"]
    p(f"Nearest-segment baseline reached {pct(baseline['top1'])} Top-1 and "
      f"{pct(routes['Nearest Segment']['continuity'])} continuity. It selects a geometry rank, so "
      "probability log loss is omitted.")
    new_page("3.1 Model comparison across trips")
    image("model_comparison.png", 492, 360)
    p("Candidate F1 and log loss evaluate binary match scores. Top-1 evaluates which arc wins within "
      "each observation, so the metrics can move in different directions.", "caption")

    audit_file = results / "overfitting_audit.json"
    if audit_file.exists():
        audit = read(audit_file)
        p("Generalization on unseen trips", "h2")
        p("The deployed configuration was refit on each saved split. Both sides were scored on all "
          "aligned observations from their respective trips using the same candidate rules.")
        table(["Fold", "Fit macro F1", "Held-out F1", "Fit Top-1", "Held-out Top-1"],
              [[r["fold"], f'{r["fit"]["macro_f1"]:.4f}', f'{r["validation"]["macro_f1"]:.4f}',
                pct(r["fit"]["top1"]), pct(r["validation"]["top1"])] for r in audit],
              [52, 110, 110, 110, 110])
        p("These folds also guided hyperparameter selection. Fold variation describes sensitivity "
          "to the trip mix. The reserved evaluation remains the independent check.", "caption")

    new_page("3.2 Per-model findings")
    for name, v in metrics["models"].items():
        s = v["student"]
        fit = v["cv_fit_sample_macro_f1"]
        valid = v["cv_macro_f1"]
        gap = fit - valid
        interpretation = (
            f"The fit-minus-validation difference is {gap:+.3f}. The comparison mixes sampling effects "
            "and generalization because fitting omits candidate misses and caps observations per trip."
        )
        p(name, "h2")
        p(f"{MODEL_DETAILS[name]} Settings: {escape(params(v['config']))}. "
          f"Student Top-1 {pct(s['top1'])}, macro F1 {s['macro_f1']:.3f}, log loss {s['log_loss']:.3f}. "
          f"Mean fit F1 {fit:.3f}, mean validation F1 {valid:.3f}. {interpretation} "
          f"Full fit took {v['full_fit_s']:.2f} s.")
    p("See model_metrics.json for the held-out fold scores, candidate-row accuracy, and confusion "
      "matrices. The comparison figure uses the measured outcomes of all five classifiers.", "caption")

    new_page("4. Bayesian optimisation")
    p(f"Optuna TPE proposed {tuned['trial_count']} configurations for {winner}. "
      f"The first four trials initialized the sampler. Each trial maximized mean candidate macro F1 across "
      f"the same three grouped folds. The recorded search took {tuned['search_s']:.1f} s. "
      f"The best logged trial index was {tuned['best_trial']} (zero-based).")
    if winner == "Histogram Gradient Boosting":
        table(["Parameter", "Search range", "Best observed value"], [
            ["learning_rate", "0.03–0.25, log", f"{tuned['config']['learning_rate']:.5f}"],
            ["max_iter", "70–170, steps of 10", tuned["config"]["max_iter"]],
            ["max_leaf_nodes", "15–47, steps of 4", tuned["config"]["max_leaf_nodes"]],
            ["min_samples_leaf", "20–100, log", tuned["config"]["min_samples_leaf"]],
            ["l2_regularization", "0.00001–3, log", f"{tuned['config']['l2_regularization']:.5g}"],
        ], [175, 175, 142])
    else:
        p(f"Winning values: {escape(params(tuned['config']))}. Search spaces are declared in pathsnap/train.py.")
    table(["Config", "CV macro F1", "Student Top-1", "Student macro F1", "Log loss", "Infer. s"],
          [[name, f'{v["cv_macro_f1"]:.4f}', pct(v["student"]["top1"]),
            f'{v["student"]["macro_f1"]:.4f}', f'{v["student"]["log_loss"]:.4f}',
            f'{v["student"]["inference_s"]:.2f}']
           for name, v in [("Untuned", untuned), ("Tuned", tuned)]],
          [64, 88, 89, 101, 75, 75])
    p(f"Tuning shifted CV macro F1 by {tuned['cv_macro_f1'] - untuned['cv_macro_f1']:+.4f} "
      f"and student Top-1 by {(tuned['student']['top1'] - untuned['student']['top1']) * 100:+.2f} "
      f"percentage points. The deployed artifact uses the {metrics['deployed_configuration']} "
      "configuration based on grouped CV. Trial parameters, durations and fold scores are saved in "
      "optimization_trials.csv.")

    new_page("5. HMM and Viterbi reconstruction")
    p("The state at one GPS fix is a directed candidate arc and its projected position. Emission scores "
      "are the log of the classifier scores normalized over valid states. For each feasible pair, the "
      "transition penalty is weight × |road distance − GPS displacement| / (beta + 3 × elapsed seconds). "
      "The road distance joins the remainder of the previous arc, a directed Dijkstra bridge and the prefix "
      "of the next arc. Searches are bounded by 32 m plus 48 m per elapsed second. Small backward shifts up "
      "to 12 m on the same road are treated as jitter.")
    p("Viterbi chooses the best predecessor plus its transition and the current emission, then backtracks. "
      "A break starts a new run after 30 s or when every predecessor is infeasible. Empty radius queries "
      "use a nearest-arc fallback. Route geometry includes graph edges between matched arcs, and breaks "
      "have no drawn connector. Nine decoder configurations were measured on a held-out training fold.")
    image("decoder_comparison.png", 492, 220)
    p(f"On student testing, short-gap route breaks changed from {only['route_breaks']:,} to "
      f"{with_viterbi['route_breaks']:,}; {with_viterbi['gps_gaps']} long gaps were excluded from "
      "the continuity denominator. Ordered route F1 uses longest common subsequence length, retaining "
      "repeat traversals and penalizing extra predicted arcs.")
    changes = []
    for trip, row in per_trip.items():
        before, after = row[deployed], row["Selected + Viterbi"]
        delta = (int(after["correct"]) - int(before["correct"])) / int(after["aligned"])
        changes.append((delta, trip))
    worst, trip = min(changes)
    best, best_trip = max(changes)
    p(f"The largest Top-1 change was {best * 100:+.2f} percentage points on {best_trip}. "
      f"The smallest was {worst * 100:+.2f} points on {trip}. Feasibility and the "
      "shortest-distance assumption can favor the wrong branch when the GPS track is ambiguous. "
      "Per-trip route metrics preserve these differences.")

    new_page("6. Geographical visualisation")
    image("folium_map.png", 492, 445)
    p("Folium overlay for student-test trip 00000076. Orange dots are raw GPS fixes, grey segments "
      "are nearby roads, purple is the reference traversal, and teal is the predicted route. The "
      "layer control toggles these four views.", "caption")
    p("The development map uses references for visual inspection. The live evaluator interface receives "
      "predictions and aggregate server scores. The interactive HTML is generated by pathsnap.visualize.")

    new_page("7. API and two-laptop demo")
    image("demo_ui.png", 492, 370)
    p("Browser capture from a student-test request to the live FastAPI server.", "caption")
    p("Laptop 1 runs FastAPI and predicts from GPS and road geometry. It loads the reference after "
      "prediction and returns aggregate scores. Laptop 2 serves the UI locally, requests the trip "
      "list and sends the selected trip ID. The test below used two local processes on one machine. "
      "The reserved-data and physical two-laptop evaluation still require an onsite run.")
    block = Table([[Paragraph("<br/>".join(escape(line) for line in receipt), STYLES["terminal"])]],
                  colWidths=[492])
    block.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INK),
                               ("TOPPADDING", (0, 0), (-1, -1), 9),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    story.append(block)
    p("Receiving API server terminal transcript (student-test mode). All ten advertised trips returned "
      "HTTP 200 with a route and scores. An out-of-list trip returned HTTP 404.", "caption")
    p("Method reference: Newson and Krumm, Hidden Markov Map Matching Through Noise and Sparseness "
      '(2009), <a href="https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/" '
      'color="#086d68">Microsoft Research</a>.', "caption")

    report = SimpleDocTemplate(str(evidence / "PathSnap_Technical_Report.pdf"), pagesize=A4,
                               topMargin=47, bottomMargin=52, leftMargin=45, rightMargin=45,
                               title="PathSnap Technical Report",
                               author=", ".join(m["name"] for m in members) if members else "")
    report.build(story, onFirstPage=page, onLaterPages=page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("report"))
    parser.add_argument("--members", type=Path, default=Path("report/team.json"))
    parser.add_argument("--repo", default="")
    args = parser.parse_args()
    build(args.results, args.output, args.members, args.repo)
