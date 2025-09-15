import numpy as np
from typing import List, Tuple

# 1. The Scale: Allowed Torsion Angle Bins (Ramachandran Plot)
SCALE_TABLE = {
    # Alpha-helix region (φ≈-60, ψ≈-45). Provide a small palette of allowed notes.
    'helix': np.array([
        [-63.0, -42.0],
        [-60.0, -45.0],
        [-57.0, -47.0],
        [-55.0, -50.0],
    ], dtype=np.float32),
    # Beta-sheet region (φ≈-135, ψ≈135). Provide a small palette.
    'sheet': np.array([
        [-150.0, 150.0],
        [-139.0, 135.0],
        [-130.0, 130.0],
        [-120.0, 140.0],
    ], dtype=np.float32),
    # Loop/turn region with expanded palette to increase geometric flexibility
    'loop':  np.array([
        [60.0, 60.0],     # Left-handed helix
        [-75.0, 145.0],   # Polyproline II
        [-60.0, -140.0],  # Generic turn
        [-90.0, 120.0],   # Extended beta region (increases separation)
        [80.0, 0.0],      # Left-handed helix variant
        [-100.0, 100.0],  # Bridge region
        [-50.0, -30.0],   # Alpha-R region
        [50.0, 50.0],     # Positive phi region
    ], dtype=np.float32),
}


def snap_to_scale(mode: str, phi_raw: float, psi_raw: float) -> Tuple[float, float]:
    """Snaps a raw torsion angle pair to the nearest allowed note in the scale."""
    allowed_bins = SCALE_TABLE[mode]
    raw_point = np.array([phi_raw, psi_raw])

    # Find the closest allowed bin
    distances = np.linalg.norm(allowed_bins - raw_point, axis=1)
    best_bin_idx = int(np.argmin(distances))

    return tuple(allowed_bins[best_bin_idx])


# 2. The Meter: Enforce Periodicity (Conceptual Stub)
def enforce_meter(mode: str, torsions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Refines a list of torsions to enforce periodic structure.
    This is a complex step. For now, we will implement a simple smoothing filter.
    """
    if mode == 'helix' and len(torsions) > 4:
        # Simple moving average to smooth helical parameters
        smoothed: List[Tuple[float, float]] = [torsions[0], torsions[1]]
        for i in range(2, len(torsions) - 2):
            avg_phi = float(np.mean([t[0] for t in torsions[i-2:i+3]]))
            avg_psi = float(np.mean([t[1] for t in torsions[i-2:i+3]]))
            smoothed.append((avg_phi, avg_psi))
        smoothed.extend(torsions[-2:])
        return smoothed
    return torsions  # Pass-through for sheet/loop for now
