"""
PathSnap Geographical Visualization using Folium
Generates interactive, multi-layer maps for:
1. Single trips with raw GPS, road network, reference route, and predicted route
2. Multi-trip interactive explorer with automatic camera fly/zoom and trip isolation
3. Standalone interactive HTML Trip Navigator Hub
"""

import os
import json
import folium
from folium import plugins
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List, Dict, Tuple

from data_processing.loader import TripData, load_trip, list_trip_ids
from feature_engineering.candidate_generator import SpatialCandidateGenerator
from feature_engineering.extractor import CandidateFeatureExtractor


# Distinct palette for multi-trip overlays
TRIP_COLORS = [
    {"gps": "#1E88E5", "route": "#2E7D32", "name": "Blue / Green"},
    {"gps": "#E53935", "route": "#8E24AA", "name": "Red / Purple"},
    {"gps": "#FB8C00", "route": "#00897B", "name": "Orange / Teal"},
    {"gps": "#3949AB", "route": "#D81B60", "name": "Indigo / Pink"},
    {"gps": "#00ACC1", "route": "#F4511E", "name": "Cyan / Coral"},
    {"gps": "#7CB342", "route": "#5E35B1", "name": "Lime / Violet"},
    {"gps": "#8D6E63", "route": "#1565C0", "name": "Brown / Blue"},
    {"gps": "#F06292", "route": "#2E7D32", "name": "Rose / Forest"}
]

ESRI_STREET_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_STREET_ATTR = "Tiles &copy; Esri &mdash; Street Map"

ESRI_TOPO_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_TOPO_ATTR = "Tiles &copy; Esri &mdash; Topographic"


def predict_trip_route(
    trip: TripData,
    model_path: str = "artifacts/lightgbm.joblib"
) -> List[int]:
    """
    Runs spatial candidate generation, feature extraction, and ML scoring
    using the trained LightGBM model to predict the reconstructed sequence of road arcs.
    """
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} not found. Skipping prediction.")
        return []

    gen = SpatialCandidateGenerator(search_radius_meters=45.0, fallback_k=3)
    candidates_list, _ = gen.generate_trip_candidates(trip, align_route=False)

    extractor = CandidateFeatureExtractor()
    X, _, metadata = extractor.extract_features_for_trip(trip, candidates_list, include_label=False)

    if len(X) == 0:
        return []

    model = joblib.load(model_path)
    probs = model.predict_proba(X)[:, 1]

    meta_df = pd.DataFrame(metadata)
    meta_df["prob"] = probs

    best_rows = meta_df.loc[meta_df.groupby("point_idx")["prob"].idxmax()].sort_values("point_idx")
    raw_arcs = best_rows["arc_id"].tolist()

    # Deduplicate sequential duplicates to form continuous road segment sequence
    reconstructed_route = []
    for arc in raw_arcs:
        if len(reconstructed_route) == 0 or reconstructed_route[-1] != arc:
            reconstructed_route.append(int(arc))

    return reconstructed_route


def create_trip_folium_map(
    trip: TripData,
    predicted_arcs: Optional[List[int]] = None,
    max_network_arcs_to_render: int = 1500,
    output_html_path: Optional[str] = None
) -> folium.Map:
    """
    Builds an interactive Folium map for a single trip.
    """
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=ESRI_STREET_URL,
        attr=ESRI_STREET_ATTR
    )

    folium.TileLayer(
        tiles=ESRI_TOPO_URL,
        attr=ESRI_TOPO_ATTR,
        name="Esri Topographic",
        overlay=False,
        control=True
    ).add_to(fmap)

    fg_road_network = folium.FeatureGroup(name="Road Network (Sub-graph)", show=False)
    fg_raw_gps = folium.FeatureGroup(name=f"Trip {trip.trip_id}: Raw GPS Observations", show=True)
    fg_ground_truth = folium.FeatureGroup(name=f"Trip {trip.trip_id}: Ground-Truth Route (Green)", show=True)
    fg_predicted = folium.FeatureGroup(name=f"Trip {trip.trip_id}: Predicted Route (Red)", show=True)

    min_lon, max_lon = track[:, 0].min() - 0.015, track[:, 0].max() + 0.015
    min_lat, max_lat = track[:, 1].min() - 0.015, track[:, 1].max() + 0.015

    rendered_arcs = 0
    for arc_idx in range(len(arcs)):
        u, v = arcs[arc_idx]
        p_u = nodes[u]
        p_v = nodes[v]
        if (min_lon <= p_u[0] <= max_lon and min_lat <= p_u[1] <= max_lat) or \
           (min_lon <= p_v[0] <= max_lon and min_lat <= p_v[1] <= max_lat):
            folium.PolyLine(
                locations=[[p_u[1], p_u[0]], [p_v[1], p_v[0]]],
                color="#888888",
                weight=1.5,
                opacity=0.6,
                tooltip=f"Arc ID: {arc_idx}"
            ).add_to(fg_road_network)
            rendered_arcs += 1
            if rendered_arcs >= max_network_arcs_to_render:
                break

    gps_coords = [[row[1], row[0]] for row in track]
    folium.PolyLine(
        locations=gps_coords,
        color="#1E88E5",
        weight=2,
        opacity=0.5,
        dash_array="4, 6"
    ).add_to(fg_raw_gps)

    step_sample = max(1, len(track) // 250)
    for i in range(0, len(track), step_sample):
        folium.CircleMarker(
            location=[track[i, 1], track[i, 0]],
            radius=3.5,
            color="#0D47A1",
            fill=True,
            fill_color="#2196F3",
            fill_opacity=0.8,
            popup=f"Trip: {trip.trip_id} | GPS #{i} | Time: {track[i, 2]:.1f}s"
        ).add_to(fg_raw_gps)

    if route is not None and len(route) > 0:
        for arc_id in route:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                folium.PolyLine(
                    locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                    color="#2E7D32",
                    weight=5,
                    opacity=0.85,
                    tooltip=f"Ground Truth Arc: {arc_id}"
                ).add_to(fg_ground_truth)

    has_predictions = False
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        has_predictions = True
        for arc_id in predicted_arcs:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                folium.PolyLine(
                    locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                    color="#D32F2F",
                    weight=4.5,
                    opacity=0.9,
                    dash_array="6, 8",
                    tooltip=f"Predicted Road Arc: {arc_id}"
                ).add_to(fg_predicted)

    fg_road_network.add_to(fmap)
    fg_raw_gps.add_to(fmap)
    fg_ground_truth.add_to(fmap)
    if has_predictions:
        fg_predicted.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    plugins.Fullscreen(position="topright").add_to(fmap)

    # Dynamic legend line for predicted route
    if has_predictions:
        if route is not None and len(route) > 0:
            overlap = len(set(route).intersection(set(predicted_arcs)))
            overlap_pct = (overlap / len(set(route))) * 100.0
            pred_legend_line = f'<i style="background: #D32F2F; width: 14px; height: 4px; display: inline-block;"></i> <b>Predicted Route</b> ({len(predicted_arcs)} arcs | <span style="color:#2E7D32; font-weight:700;">{overlap_pct:.1f}% match</span>)<br>'
        else:
            pred_legend_line = f'<i style="background: #D32F2F; width: 14px; height: 4px; display: inline-block;"></i> <b>Predicted Route</b> ({len(predicted_arcs)} arcs)<br>'
    else:
        pred_legend_line = '<i style="color: #94a3b8; font-size: 11px;">⚠️ Predicted Route: Not rendered in this view</i><br>'

    legend_html = f'''
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 280px; min-height: 140px; 
        background-color: white; border: 2px solid #cbd5e1; z-index: 9999; font-size: 12px;
        padding: 12px; border-radius: 8px; box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
    <div style="font-weight: 700; font-size: 13px; color: #1e3a8a; margin-bottom: 6px;">PathSnap: Trip {trip.trip_id}</div>
    <i style="background: #2196F3; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> Noisy GPS Observations ({len(track)} pts)<br>
    <i style="background: #888888; width: 14px; height: 3px; display: inline-block;"></i> Local Road Network<br>
    <i style="background: #2E7D32; width: 14px; height: 4px; display: inline-block;"></i> Ground-Truth Reference ({len(route) if route is not None else 0} arcs)<br>
    {pred_legend_line}
    <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 6px 0;">
    <small style="color: #64748b;">Toggle layers on/off via control panel (top right)</small>
    </div>
    '''
    fmap.get_root().html.add_child(folium.Element(legend_html))

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        fmap.save(output_html_path)
        print(f"Saved Folium map to: {output_html_path}")

    return fmap


def create_multi_trip_folium_map(
    trips: List[TripData],
    output_html_path: str = "results/folium_maps/multi_trip_comparison.html"
) -> folium.Map:
    """
    Builds a single interactive Folium map containing multiple trips simultaneously,
    with an interactive controller that automatically flies/zooms to any selected trip
    and isolates its trajectory.
    """
    if len(trips) == 0:
        raise ValueError("At least one trip must be provided.")

    # Calculate overall bounding center
    all_lats = []
    all_lons = []
    for trip in trips:
        all_lats.extend(trip.track[:, 1])
        all_lons.extend(trip.track[:, 0])

    initial_trip = trips[0]
    center_lat = float(np.mean(initial_trip.track[:, 1]))
    center_lon = float(np.mean(initial_trip.track[:, 0]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=ESRI_STREET_URL,
        attr=ESRI_STREET_ATTR
    )

    folium.TileLayer(
        tiles=ESRI_TOPO_URL,
        attr=ESRI_TOPO_ATTR,
        name="Esri Topographic",
        overlay=False,
        control=True
    ).add_to(fmap)

    map_var_name = fmap.get_name()
    trip_metadata = {}
    options_html = ""

    for idx, trip in enumerate(trips):
        color_scheme = TRIP_COLORS[idx % len(TRIP_COLORS)]
        gps_color = color_scheme["gps"]
        route_color = color_scheme["route"]

        # Only the first trip is visible initially to keep view clean
        is_first = (idx == 0)
        fg_gps = folium.FeatureGroup(name=f"Trip {trip.trip_id} (GPS Dots)", show=is_first)
        fg_route = folium.FeatureGroup(name=f"Trip {trip.trip_id} (True Route)", show=is_first)

        # Plot GPS polyline
        gps_coords = [[row[1], row[0]] for row in trip.track]
        folium.PolyLine(
            locations=gps_coords,
            color=gps_color,
            weight=2.5,
            opacity=0.6,
            dash_array="4, 6"
        ).add_to(fg_gps)

        # Plot sampled GPS points
        sample_step = max(1, len(trip.track) // 200)
        for i in range(0, len(trip.track), sample_step):
            folium.CircleMarker(
                location=[trip.track[i, 1], trip.track[i, 0]],
                radius=3.5,
                color=gps_color,
                fill=True,
                fill_color=gps_color,
                fill_opacity=0.85,
                popup=f"Trip {trip.trip_id} | GPS #{i} | Time: {trip.track[i, 2]:.1f}s"
            ).add_to(fg_gps)

        # Plot Ground Truth Route
        if trip.route is not None and len(trip.route) > 0:
            for arc_id in trip.route:
                if arc_id < len(trip.arcs):
                    u, v = trip.arcs[arc_id]
                    folium.PolyLine(
                        locations=[[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]],
                        color=route_color,
                        weight=5,
                        opacity=0.85,
                        tooltip=f"Trip {trip.trip_id} | Arc {arc_id}"
                    ).add_to(fg_route)

        fg_gps.add_to(fmap)
        fg_route.add_to(fmap)

        min_lat = float(trip.track[:, 1].min())
        max_lat = float(trip.track[:, 1].max())
        min_lon = float(trip.track[:, 0].min())
        max_lon = float(trip.track[:, 0].max())

        trip_metadata[trip.trip_id] = {
            "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
            "center": [float(np.mean(trip.track[:, 1])), float(np.mean(trip.track[:, 0]))],
            "gps_var": fg_gps.get_name(),
            "route_var": fg_route.get_name(),
            "obs_count": len(trip.track),
            "arcs_count": len(trip.route) if trip.route is not None else 0,
            "color": route_color
        }

        options_html += f'<option value="{trip.trip_id}">Trip {trip.trip_id} ({len(trip.track)} pts)</option>\n'

    global_bounds = [
        [float(min(all_lats)), float(min(all_lons))],
        [float(max(all_lats)), float(max(all_lons))]
    ]

    folium.LayerControl(collapsed=True).add_to(fmap)
    plugins.Fullscreen(position="topright").add_to(fmap)

    # Inject interactive floating selector control with auto-fly/zoom functionality
    control_widget = f'''
    <div id="tripControlPanel" style="
        position: fixed; 
        top: 20px; left: 60px; z-index: 9999;
        background: rgba(255, 255, 255, 0.96); 
        border-radius: 10px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
        padding: 14px 18px; 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        min-width: 290px;
        border: 1px solid #cbd5e1;
    ">
        <div style="font-weight: 700; font-size: 15px; color: #1e3a8a; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
            <span>📍</span> <span>PathSnap: Select & Go to Trip</span>
        </div>
        
        <div style="margin-bottom: 10px;">
            <select id="tripSelectDropdown" onchange="autoFlyToTrip(this.value)" style="
                width: 100%; padding: 8px 10px; font-size: 13px; font-weight: 600; 
                border-radius: 6px; border: 1.5px solid #2563eb; outline: none; background: white; cursor: pointer; color: #1e293b;
            ">
                <option value="all">🌐 Show & Zoom to All Trips</option>
                {options_html}
            </select>
        </div>

        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 12px; color: #475569;">
            <input type="checkbox" id="isolateTripCheck" checked onchange="toggleIsolation()" style="cursor: pointer;">
            <label for="isolateTripCheck" style="cursor: pointer; user-select: none;">Isolate selected trip (hide others)</label>
        </div>

        <div id="tripInfoBadge" style="
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; font-size: 12px; color: #334155;
        ">
            <b>Selected:</b> Trip {trips[0].trip_id}<br>
            <b>Observations:</b> {len(trips[0].track)} GPS points<br>
            <b>Route Arcs:</b> {len(trips[0].route) if trips[0].route is not None else 0} segments
        </div>
    </div>

    <script>
    var tripMeta = {json.dumps(trip_metadata)};
    var globalBounds = {json.dumps(global_bounds)};

    function autoFlyToTrip(tid) {{
        var map = {map_var_name};
        var isolate = document.getElementById("isolateTripCheck").checked;
        var infoBadge = document.getElementById("tripInfoBadge");

        if (tid === "all") {{
            map.flyToBounds(globalBounds, {{ padding: [40, 40], duration: 1.5 }});
            // Show all trip layers
            for (var key in tripMeta) {{
                var g = window[tripMeta[key].gps_var];
                var r = window[tripMeta[key].route_var];
                if (g && !map.hasLayer(g)) map.addLayer(g);
                if (r && !map.hasLayer(r)) map.addLayer(r);
            }}
            infoBadge.innerHTML = "<b>Viewing All Trips</b><br>Displaying multiple trajectories simultaneously.";
        }} else {{
            var meta = tripMeta[tid];
            if (!meta) return;

            // Fly camera smoothly to the selected trip's bounding box
            map.flyToBounds(meta.bounds, {{ padding: [60, 60], duration: 1.5 }});

            // Handle visibility and isolation
            for (var key in tripMeta) {{
                var g = window[tripMeta[key].gps_var];
                var r = window[tripMeta[key].route_var];
                if (key === tid) {{
                    if (g && !map.hasLayer(g)) map.addLayer(g);
                    if (r && !map.hasLayer(r)) map.addLayer(r);
                }} else if (isolate) {{
                    if (g && map.hasLayer(g)) map.removeLayer(g);
                    if (r && map.hasLayer(r)) map.removeLayer(r);
                }} else {{
                    if (g && !map.hasLayer(g)) map.addLayer(g);
                    if (r && !map.hasLayer(r)) map.addLayer(r);
                }}
            }}

            infoBadge.innerHTML = "<b style='color:" + meta.color + "'>Trip " + tid + "</b><br>" +
                                  "<b>Observations:</b> " + meta.obs_count + " GPS points<br>" +
                                  "<b>Route Arcs:</b> " + meta.arcs_count + " segments";
        }}
    }}

    function toggleIsolation() {{
        var dropdown = document.getElementById("tripSelectDropdown");
        autoFlyToTrip(dropdown.value);
    }}

    // Set dropdown default to first trip
    window.addEventListener('DOMContentLoaded', function() {{
        var dropdown = document.getElementById("tripSelectDropdown");
        if (dropdown && dropdown.options.length > 1) {{
            dropdown.selectedIndex = 1;
            autoFlyToTrip(dropdown.value);
        }}
    }});
    </script>
    '''
    fmap.get_root().html.add_child(folium.Element(control_widget))

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    fmap.save(output_html_path)
    print(f"Saved Multi-Trip Interactive Map with Auto-Fly to: {output_html_path}")
    return fmap


def create_trip_navigator_html(
    trip_ids: List[str],
    output_html_path: str = "results/folium_maps/trip_navigator.html"
):
    """
    Creates an interactive Trip Navigator hub with a dropdown menu to seamlessly
    switch between any trip map right inside the browser.
    """
    options_html = "\n".join([f'<option value="trip_{tid}.html">Trip {tid}</option>' for tid in trip_ids])

    hub_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PathSnap | Interactive Trip Navigator</title>
    <style>
        body, html {{
            margin: 0; padding: 0; height: 100%; width: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex; flex-direction: column; overflow: hidden; background: #f1f5f9;
        }}
        .header {{
            background: #1e3a8a; color: white; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15); z-index: 100;
        }}
        .header h1 {{ margin: 0; font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px; }}
        .controls {{ display: flex; align-items: center; gap: 12px; }}
        select, button {{
            padding: 8px 14px; font-size: 14px; border-radius: 6px; border: 1px solid #cbd5e1; outline: none; cursor: pointer;
        }}
        select {{ background: white; font-weight: 500; min-width: 160px; }}
        button {{
            background: #2563eb; color: white; border: none; font-weight: 600; transition: background 0.2s;
        }}
        button:hover {{ background: #1d4ed8; }}
        .multi-btn {{
            background: #059669; margin-left: 8px;
        }}
        .multi-btn:hover {{ background: #047857; }}
        #mapFrame {{ flex: 1; border: none; width: 100%; height: 100%; background: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ PathSnap Interactive Trip Navigator</h1>
        <div class="controls">
            <label for="tripSelect" style="font-size: 14px; font-weight: 500;">Select Trip:</label>
            <select id="tripSelect" onchange="loadTrip(this.value)">
                {options_html}
            </select>
            <button class="multi-btn" onclick="loadTrip('multi_trip_comparison.html')">Multi-Trip Auto-Fly Map</button>
        </div>
    </div>
    <iframe id="mapFrame" src="trip_{trip_ids[0]}.html"></iframe>

    <script>
        function loadTrip(file) {{
            document.getElementById('mapFrame').src = file;
        }}
    </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w") as f:
        f.write(hub_html)
    print(f"Saved Interactive Trip Navigator Hub to: {output_html_path}")


def generate_all_visualizations(
    trip_ids: Optional[List[str]] = None,
    dataset_dirs: Optional[List[str]] = None,
    output_dir: str = "results/folium_maps",
    model_path: str = "artifacts/lightgbm.joblib"
):
    """
    Renders all individual trip maps with ML predicted routes,
    the multi-trip comparison map, and the trip navigator portal.
    """
    if trip_ids is None:
        trip_ids = ['00000000', '00000005', '00000007', '00000009', '00000052', '00000056', '00000076', '00000091']

    if dataset_dirs is None:
        dataset_dirs = ['../Student folder/student_test', '../Student folder/train', 'Student folder/student_test', 'Student folder/train']

    os.makedirs(output_dir, exist_ok=True)
    loaded_trips = []

    for tid in trip_ids:
        trip = None
        for d in dataset_dirs:
            candidate_path = os.path.join(d, tid)
            if os.path.exists(candidate_path):
                try:
                    trip = load_trip(d, tid, load_route=True)
                    break
                except Exception:
                    pass

        if trip is None:
            print(f"Skipping trip {tid}: directory not found.")
            continue

        loaded_trips.append(trip)
        print(f"--> Generating map for Trip {tid} ({len(trip.track)} GPS observations)...")
        # Run ML model prediction
        pred_route = predict_trip_route(trip, model_path=model_path)
        out_file = os.path.join(output_dir, f"trip_{tid}.html")
        create_trip_folium_map(trip, predicted_arcs=pred_route, output_html_path=out_file)

    # Generate multi-trip auto-fly map
    if len(loaded_trips) > 0:
        multi_map_path = os.path.join(output_dir, "multi_trip_comparison.html")
        create_multi_trip_folium_map(loaded_trips[:5], output_html_path=multi_map_path)

    # Generate navigator hub
    available_tids = [t.trip_id for t in loaded_trips]
    hub_path = os.path.join(output_dir, "trip_navigator.html")
    create_trip_navigator_html(available_tids, output_html_path=hub_path)
    print("All Folium visualization maps regenerated successfully with ML Predicted Routes!")


if __name__ == "__main__":
    generate_all_visualizations()
