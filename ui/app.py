"""
PathSnap Evaluator Interface (Laptop 2 Frontend)
Interactive dashboard built with Streamlit for live evaluation:
1. Connects to Laptop 1 FastAPI backend over LAN/WiFi
2. Displays the ten reserved evaluation trips
3. Sends selected trip to Laptop 1 and visualizes reconstructed route, continuity, and Folium map
"""

import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd


st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar: Server Connection Settings
st.sidebar.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/navigation.svg", width=50)
st.sidebar.title("PathSnap Deployment")
st.sidebar.markdown("**Two-Laptop Distributed Setup**")

backend_url = st.sidebar.text_input(
    "Laptop 1 API Endpoint",
    value="http://localhost:8000",
    help="Enter the IP address of Laptop 1 (e.g. http://192.168.1.105:8000)"
)

# Test Backend Connection
server_online = False
model_info = "Unknown"
try:
    health_resp = requests.get(f"{backend_url}/health", timeout=2.5)
    if health_resp.status_code == 200:
        data = health_resp.json()
        server_online = True
        model_info = data.get("model_type", "Active")
        st.sidebar.success(f"Connected to Laptop 1\n\nModel: {model_info}")
    else:
        st.sidebar.error("Laptop 1 returned error status")
except Exception:
    st.sidebar.warning("Laptop 1 Offline. Check network or start API.")

st.sidebar.markdown("---")
st.sidebar.markdown("### System Specifications")
st.sidebar.markdown("- **Architecture:** FastAPI + Streamlit Distributed")
st.sidebar.markdown("- **Candidate Search:** 45m Geodesic R-Tree")
st.sidebar.markdown("- **Route Decoder:** Topologically Constrained Viterbi HMM")


# Main Page Header
st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.markdown(
    "Reconstructs continuous, realistic vehicle road trajectories from noisy GPS observations "
    "using machine learning classification and topologically constrained Hidden Markov Model (HMM) Viterbi decoding."
)

st.markdown("---")

# Fetch Trips from Laptop 1
default_trips = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

trips_list = default_trips
if server_online:
    try:
        trips_resp = requests.get(f"{backend_url}/trips", timeout=2.0)
        if trips_resp.status_code == 200:
            trips_list = trips_resp.json().get("trips", default_trips)
    except Exception:
        pass

# Trip Selection Controls
col1, col2 = st.columns([3, 1])
with col1:
    selected_trip = st.selectbox(
        "Select an Evaluation Trip to Reconstruct:",
        options=trips_list,
        index=0
    )
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Reconstruct Route", use_container_width=True, type="primary")

if run_btn:
    if not server_online:
        st.error(f"Cannot connect to Laptop 1 backend at `{backend_url}`. Please verify backend server is running.")
    else:
        with st.spinner(f"Sending Trip '{selected_trip}' to Laptop 1 for end-to-end inference..."):
            try:
                res = requests.post(f"{backend_url}/predict/{selected_trip}", timeout=60.0)
                if res.status_code == 200:
                    payload = res.json()
                    st.success(f"Trip '{selected_trip}' successfully reconstructed in {payload['inference_time_ms']:.1f} ms!")

                    # Metric Columns
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Observation Top-1 Accuracy</div>
                            <div class="metric-value">{payload['top1_accuracy'] * 100:.1f}%</div>
                            <small style="color: green;">Baseline: {payload['baseline_top1_accuracy'] * 100:.1f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Route Continuity</div>
                            <div class="metric-value">{payload['route_continuity'] * 100:.1f}%</div>
                            <small style="color: green;">+{payload['continuity_gain'] * 100:.1f}% via HMM</small>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">GPS Observations</div>
                            <div class="metric-value">{payload['observations_count']}</div>
                            <small>Candidate Miss: {payload['candidate_miss_rate'] * 100:.2f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">End-to-End Latency</div>
                            <div class="metric-value">{payload['inference_time_ms']:.0f} ms</div>
                            <small>Backend Processing</small>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### Interactive Geographical Route Visualization")

                    # Generate Interactive Folium Map
                    gps_pts = payload["gps_trajectory"]
                    reconstructed_segs = payload["reconstructed_segments"]

                    if len(gps_pts) > 0:
                        center_lat = sum(p[0] for p in gps_pts) / len(gps_pts)
                        center_lon = sum(p[1] for p in gps_pts) / len(gps_pts)

                        esri_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
                        esri_attr = "Tiles &copy; Esri &mdash; Street Map"
                        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles=esri_url, attr=esri_attr)

                        # Layer 1: Raw GPS Points
                        fg_gps = folium.FeatureGroup(name="Noisy GPS Observations (Blue)", show=True)
                        for i in range(0, len(gps_pts), max(1, len(gps_pts) // 250)):
                            folium.CircleMarker(
                                location=gps_pts[i],
                                radius=3,
                                color="#1E88E5",
                                fill=True,
                                fill_color="#42A5F5",
                                fill_opacity=0.7
                            ).add_to(fg_gps)
                        fg_gps.add_to(fmap)

                        # Layer 2: Reconstructed Route
                        fg_recon = folium.FeatureGroup(name="Reconstructed Route (Red Polyline)", show=True)
                        for seg in reconstructed_segs:
                            folium.PolyLine(
                                locations=seg,
                                color="#D32F2F",
                                weight=5,
                                opacity=0.85
                            ).add_to(fg_recon)
                        fg_recon.add_to(fmap)

                        folium.LayerControl(collapsed=False).add_to(fmap)

                        st_folium(fmap, width=1100, height=550)

                else:
                    st.error(f"Inference failed with status {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Failed to communicate with Laptop 1 API: {e}")
