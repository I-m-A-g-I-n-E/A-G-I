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
    # Convert angles to radians
    angle_rad = np.deg2rad(angle)
    torsion_rad = np.deg2rad(torsion)

    # Calculate vectors
    v21 = p1 - p2
    v23 = p3 - p2
    v21 = v21 / np.linalg.norm(v21)
    v23 = v23 / np.linalg.norm(v23)

    # Calculate normal vector for rotation
    normal = np.cross(v21, v23)
    normal = normal / np.linalg.norm(normal)

    # Create rotation matrix
    # Rotate v21 around normal to get initial position for p4
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec((angle_rad - np.pi) * normal)
    p4_initial = p2 + rot.apply(v21) * length

    # Apply torsion rotation
    rot_torsion = R.from_rotvec(torsion_rad * v23)
    p4 = p2 + rot_torsion.apply(p4_initial - p2)

    return p4
