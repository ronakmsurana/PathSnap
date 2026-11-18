"""Create a Folium map for a development trip and plot measured experiment results."""
import argparse
import csv
import json
from pathlib import Path

import folium
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .api import latlon_routes, predict, read_model
from .data import load_reference, load_trip


COLORS = {"roads": "#778d8b", "reference": "#6a3d9a", "prediction": "#087b79", "gps": "#c44d21"}


def folium_map(directory, target, artifacts):
    model, config = read_model(artifacts)
    trip = load_trip(directory)
    _, _, predicted, _, _ = predict(trip, model, config)
    reference = load_reference(directory, trip)
    edges = trip.arcs[reference]
    splits = np.flatnonzero(edges[:-1, 1] != edges[1:, 0]) + 1
    reference_routes = latlon_routes(trip, np.split(reference, splits))
    gps = trip.track[:, [1, 0]]
    map_ = folium.Map(location=np.median(gps, axis=0).tolist(), tiles=None,
                      prefer_canvas=True, control_scale=True)
    # Bundle Leaflet so the exported development map opens offline.
    vendor = Path(__file__).resolve().parents[1] / "web" / "vendor"
    map_.default_js = []
    map_.default_css = []
    css = (vendor / "leaflet.css").read_text()
    js = (vendor / "leaflet.js").read_text().split("//# sourceMappingURL")[0]
    map_.get_root().header.add_child(folium.Element(f"<style>{css} .leaflet-container {{ background: #edf1ee }}</style><script>{js}</script>"))
    ends = trip.xy[trip.arcs]
    lower = trip.gps.min(axis=0) - 350
    upper = trip.gps.max(axis=0) + 350
    intersects = ((ends.max(axis=1) >= lower) & (ends.min(axis=1) <= upper)).all(axis=1)
    context = trip.nodes[trip.arcs[intersects]][:, :, ::-1]
    layers = [folium.FeatureGroup(name=name).add_to(map_) for name in (
        "Nearby road network", "Reference route", "Predicted route", "Raw GPS",
    )]
    folium.PolyLine(context.tolist(), color=COLORS["roads"], weight=1.2, opacity=0.45).add_to(layers[0])
    folium.PolyLine(reference_routes, color=COLORS["reference"], weight=8, opacity=0.80).add_to(layers[1])
    folium.PolyLine(latlon_routes(trip, predicted), color=COLORS["prediction"],
                    weight=4, opacity=0.95).add_to(layers[2])
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2.0, stroke=False, fill=True,
                            fill_color=COLORS["gps"], fill_opacity=0.42).add_to(layers[3])
    folium.LayerControl(collapsed=False).add_to(map_)
    map_.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()], padding=(22, 22))
    legend = """
    <div style="font:14px system-ui;position:fixed;bottom:25px;right:15px;z-index:9999;
                border-radius:8px;background:white;padding:10px 14px;line-height:1.55">
      <b>PathSnap · Student test</b><br>
      <span style="color:#c44d21">●</span> Raw GPS<br>
      <span style="color:#778d8b">━</span> Nearby roads<br>
      <span style="color:#6a3d9a">━</span> Reference route<br>
      <span style="color:#087b79">━</span> Predicted route
    </div>
    """
    map_.get_root().html.add_child(folium.Element(legend))
    target.parent.mkdir(parents=True, exist_ok=True)
    map_.save(target)


def plots(results, output):
    output.mkdir(parents=True, exist_ok=True)
    all_metrics = json.loads((results / "model_metrics.json").read_text())
    models = list(all_metrics["models"].values())
    labels = ["Logistic\nRegression", "Gaussian\nNaive Bayes", "Decision\nTree",
              "Random\nForest", "Histogram\nBoosting"]
    x = np.arange(len(labels))
    fig, axs = plt.subplots(3, 1, figsize=(9.2, 8.5), layout="constrained")
    axs[0].bar(x, [m["cv_macro_f1"] for m in models], color="#087b79")
    axs[0].set_title("Grouped cross-validation · candidate macro F1", loc="left", fontweight="bold")
    axs[0].set_ylim(0.0, 1.03)
    axs[1].bar(x - .18, [m["student"]["top1"] for m in models], .36,
               color="#087b79", label="Observation Top-1")
    axs[1].bar(x + .18, [m["student"]["macro_f1"] for m in models], .36,
               color="#6a3d9a", label="Candidate macro F1")
    axs[1].set_ylim(0.0, 1.03)
    axs[1].set_title("Student test · classification", loc="left", fontweight="bold")
    axs[1].legend(loc="lower right")
    loss = [m["student"]["log_loss"] for m in models]
    axs[2].bar(x, loss, color="#94624b")
    axs[2].set_title("Student test · candidate log loss (lower is better)", loc="left", fontweight="bold")
    axs[2].set_ylim(0.0, max(loss) * 1.2)
    for ax in axs:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
        for bars in ax.containers:
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)

    with (results / "dataset_inventory.csv").open() as file:
        rows = list(csv.DictReader(file))
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.1), layout="constrained")
    for split, label, color in (("train", "Train", "#087b79"), ("student_test", "Student test", "#6a3d9a")):
        group = [r for r in rows if r["split"] == split]
        axs[0].hist([int(r["observations"]) for r in group], bins=np.geomspace(250, 20000, 15),
                    histtype="step", linewidth=2, color=color, label=label)
        axs[1].scatter([int(r["arcs"]) for r in group], [float(r["max_gap_s"]) for r in group],
                       s=24, color=color, label=label, alpha=.8)
    axs[0].set(xscale="log", xlabel="GPS fixes per trip", ylabel="Trips")
    axs[1].set(xscale="log", yscale="log", xlabel="Directed arcs per trip", ylabel="Largest GPS gap (seconds)")
    axs[0].legend()
    fig.savefig(output / "dataset_summary.png", dpi=160)
    plt.close(fig)

    route = json.loads((results / "route_metrics.json").read_text())
    selected = "Tuned" if all_metrics["deployed_configuration"] == "tuned" else all_metrics["selected"]
    fig, ax = plt.subplots(figsize=(8.8, 3.5), layout="constrained")
    x = np.arange(3)
    for offset, name, label, color in ((-.2, selected, "Classifier only", "#6a3d9a"),
                                      (.2, "Selected + Viterbi", "Classifier + Viterbi", "#087b79")):
        bars = ax.bar(x + offset, [route[name][key] for key in ("top1", "continuity", "sequence_f1")],
                      width=.4, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_xticks(x, ["Observation Top-1", "Continuity", "Ordered route F1"])
    ax.set_ylim(0, 1.09)
    ax.legend(loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output / "decoder_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", type=Path, default=Path("Student folder/student_test/00000076"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("report"))
    args = parser.parse_args()
    plots(args.results, args.output)
    folium_map(args.trip, args.output / "trip_map.html", args.artifacts)
