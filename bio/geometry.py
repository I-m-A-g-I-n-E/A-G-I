import numpy as np
import torch
from typing import Dict, Tuple

# --- 1. The Rosetta Stone: Musical Intervals -> Torsion Angles ---
# Maps the most fundamental musical intervals to their corresponding (phi, psi) angles.
# This is the core of our geometric decoder.
INTERVAL_TO_TORSION: Dict[str, Tuple[float, float]] = {
    "Perfect 5th": (-57.0, -47.0),   # Most stable interval -> Alpha Helix
    "Minor 2nd":   (-139.0, 135.0),  # Tense, structured interval -> Beta Sheet
    "Major 3rd":   (60.0, 60.0),     # Bright, open interval -> Left-handed Helix
    "Unison":      (-75.0, 160.0),   # Static interval -> Polyproline II Helix
    # Add a default for unclassified/dissonant intervals
    "Dissonant":   (-60.0, -140.0)   # A common loop/turn conformation
}

# --- 2. Idealized Backbone Geometry ---
# Standard bond lengths (Angstroms) and angles (degrees) for the polypeptide chain.
IDEAL_GEOMETRY = {
    "N-CA": 1.45,
    "CA-C": 1.52,
    "C-N": 1.33,
    "CA-C-N_angle": 116.0,
    "C-N-CA_angle": 122.0,
    "N-CA-C_angle": 111.0,
}

def place_next_atom(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                    length: float, angle: float, torsion: float) -> np.ndarray:
    """
    Places the 4th atom (p4) given the coordinates of the previous three (p1, p2, p3)
    and the ideal geometry (bond length, angle, and torsion angle).
    This is a standard Natural Extension Reference Frame (NERF) algorithm.
    """
    # Convert angles (degrees -> radians)
    angle_rad = np.deg2rad(angle)
    torsion_rad = np.deg2rad(torsion)

    # Define local reference frame at p3
    # e1: along the bond from p2 to p3
    v12 = p2 - p1
    v23 = p3 - p2
    e1 = v23 / (np.linalg.norm(v23) + 1e-8)

    # Normal to the plane (p1, p2, p3)
    n = np.cross(v12, v23)
    n_norm = np.linalg.norm(n)

    if n_norm < 1e-8:
        # Degenerate case: choose an arbitrary orthogonal to e1
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, e1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e2 = np.cross(e1, tmp)
        e2 = e2 / (np.linalg.norm(e2) + 1e-8)
    else:
        # e2 lies in the plane orthogonal to e1 and aligned with the previous plane
        e2 = np.cross(e1, n)
        e2 = e2 / (np.linalg.norm(e2) + 1e-8)

    # e3 completes right-handed frame
    e3 = np.cross(e2, e1)

    # Local coordinates of new atom relative to p3
    d = float(length)
    x = -d * np.cos(angle_rad)
    y =  d * np.sin(angle_rad) * np.cos(torsion_rad)
    z =  d * np.sin(angle_rad) * np.sin(torsion_rad)

    p4 = p3 + x * e1 + y * e2 + z * e3
    return p4
