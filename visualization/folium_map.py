"""
PathSnap Geographical Visualization using Folium
Generates interactive, multi-layer maps displaying:
1. Raw GPS observations (blue points / trajectory)
2. Road network candidate geometry (gray segments)
3. Reference ground truth route (green polyline)
4. Predicted route (red polyline, when available)
"""

import os
import folium
from folium import plugins
import numpy as np
from typing import Optional, List, Tuple

from data_processing.loader import TripData


def create_trip_folium_map(
    trip: TripData,
    predicted_arcs: Optional[List[int]] = None,
    max_network_arcs_to_render: int = 1500,
    output_html_path: Optional[str] = None
) -> folium.Map:
    """
    Builds an interactive Folium map for a given trip.
    """
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    # Calculate center coordinate (lat, lon)
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    ESRI_STREET = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=ESRI_STREET,
        attr="Tiles &copy; Esri &mdash; Street Map"
    )

    # Feature groups for toggleable layers
    fg_road_network = folium.FeatureGroup(name="Road Network (Sub-graph)", show=False)
    fg_raw_gps = folium.FeatureGroup(name="Raw GPS Trajectory (Observations)", show=True)
    fg_ground_truth = folium.FeatureGroup(name="Ground-Truth Route (Actual)", show=True)
    fg_predicted = folium.FeatureGroup(name="Predicted Reconstructed Route", show=True)

    # 1. Road Network Layer (render subset around trajectory to maintain high browser performance)
    min_lon, max_lon = track[:, 0].min() - 0.01, track[:, 0].max() + 0.01
    min_lat, max_lat = track[:, 1].min() - 0.01, track[:, 1].max() + 0.01

    rendered_arcs = 0
    for arc_idx in range(len(arcs)):
        u, v = arcs[arc_idx]
        p_u = nodes[u]
        p_v = nodes[v]
        # Bounding box check
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

    # 2. Raw GPS Observations Layer (Blue dots + subtle trail)
    gps_coords = [[row[1], row[0]] for row in track]
    folium.PolyLine(
        locations=gps_coords,
        color="#1E88E5",
        weight=2,
        opacity=0.5,
        dash_array="4, 6"
    ).add_to(fg_raw_gps)

    # Add circles for individual GPS readings (sample if very large)
    step_sample = max(1, len(track) // 300)
    for i in range(0, len(track), step_sample):
        folium.CircleMarker(
            location=[track[i, 1], track[i, 0]],
            radius=3,
            color="#0D47A1",
            fill=True,
            fill_color="#2196F3",
            fill_opacity=0.8,
            popup=f"GPS #{i} | Time: {track[i, 2]:.1f}s"
        ).add_to(fg_raw_gps)

    # 3. Reference Ground Truth Route Layer (Green line)
    if route is not None and len(route) > 0:
        gt_segments = []
        for arc_id in route:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                gt_segments.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
        
        for seg in gt_segments:
            folium.PolyLine(
                locations=seg,
                color="#2E7D32",  # Forest Green
                weight=5,
                opacity=0.85,
                tooltip="Ground Truth Route"
            ).add_to(fg_ground_truth)

    # 4. Predicted Route Layer (Red line)
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        for arc_id in predicted_arcs:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                folium.PolyLine(
                    locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                    color="#D32F2F",  # Vivid Red
                    weight=4,
                    opacity=0.9,
                    dash_array="6, 8",
                    tooltip=f"Predicted Arc ID: {arc_id}"
                ).add_to(fg_predicted)

    # Add layers to map
    fg_road_network.add_to(fmap)
    fg_raw_gps.add_to(fmap)
    fg_ground_truth.add_to(fmap)
    if predicted_arcs is not None:
        fg_predicted.add_to(fmap)

    # Add layer controls and full screen button
    folium.LayerControl(collapsed=False).add_to(fmap)
    plugins.Fullscreen(position="topright").add_to(fmap)

    # Add floating legend
    legend_html = '''
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 230px; height: 130px; 
        background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
        padding: 10px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    ">
    <b>PathSnap Route Legend</b><br>
    <i style="background: #2196F3; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> Noisy GPS Points<br>
    <i style="background: #888888; width: 14px; height: 3px; display: inline-block;"></i> Road Network Segments<br>
    <i style="background: #2E7D32; width: 14px; height: 4px; display: inline-block;"></i> Ground-Truth Reference<br>
    <i style="background: #D32F2F; width: 14px; height: 4px; display: inline-block;"></i> Predicted Route<br>
    </div>
    '''
    fmap.get_root().html.add_child(folium.Element(legend_html))

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        fmap.save(output_html_path)
        print(f"Interactive Folium map saved to: {output_html_path}")

    return fmap
