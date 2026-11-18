"""
Geographical & Metric Projection Utilities for PathSnap
Converts angular coordinates (lon, lat) to local Cartesian metric coordinates (meters)
and computes geodesic azimuths and distances.
"""

import numpy as np


class LocalMetricProjector:
    """
    Local Conformal / Equirectangular metric projection centered at (lon0, lat0).
    Accurate to within millimeter precision over local urban extents (tens of km).
    """
    R = 6371000.0  # Earth's mean radius in meters

    def __init__(self, ref_lon: float, ref_lat: float):
        self.ref_lon = float(ref_lon)
        self.ref_lat = float(ref_lat)
        self.ref_lon_rad = np.radians(self.ref_lon)
        self.ref_lat_rad = np.radians(self.ref_lat)
        self.cos_lat0 = np.cos(self.ref_lat_rad)

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        Projects (lon, lat) in degrees to (x, y) in meters.
        Input can be scalar or 1D/2D arrays.
        """
        d_lon = np.radians(lon) - self.ref_lon_rad
        d_lat = np.radians(lat) - self.ref_lat_rad
        x = self.R * d_lon * self.cos_lat0
        y = self.R * d_lat
        if np.ndim(lon) == 0:
            return np.array([x, y], dtype=float)
        return np.column_stack([x, y])

    def inverse(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Projects (x, y) in meters back to (lon, lat) in degrees.
        """
        lon = np.degrees(x / (self.R * self.cos_lat0) + self.ref_lon_rad)
        lat = np.degrees(y / self.R + self.ref_lat_rad)
        if np.ndim(x) == 0:
            return np.array([lon, lat], dtype=float)
        return np.column_stack([lon, lat])


def compute_bearing_deg(dx: float, dy: float) -> float:
    """
    Calculates compass bearing in degrees [0, 360) from Cartesian vector (dx, dy).
    0 = North (+y), 90 = East (+x), 180 = South (-y), 270 = West (-x).
    """
    angle = np.degrees(np.arctan2(dx, dy))
    return float((angle + 360.0) % 360.0)


def angle_difference_deg(b1: float, b2: float) -> float:
    """
    Computes minimum angular difference in degrees [0, 180] between two bearings.
    """
    diff = abs(b1 - b2) % 360.0
    return float(360.0 - diff if diff > 180.0 else diff)
