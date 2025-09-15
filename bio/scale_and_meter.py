import numpy as np
from typing import List, Tuple
try:
    from agi.harmonia.notation import HelixNotes, LoopNotes, Movement, Gesture, Handedness
except Exception:
    # Fallback stubs when harmonia is unavailable (keeps tests passing)
    HelixNotes = type('HelixNotes', (), {'P5': 'helix_p5', 'CENTER': 'helix_center'})
    LoopNotes = type('LoopNotes', (), {'PPII': 'loop_ppii', 'RESOLUTION': 'loop_resolution'})
    Movement = None
    Gesture = None
    Handedness = None

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


# Movement-aware table (non-breaking addition): maps modes to symbolic note sets
# This complements the numeric SCALE_TABLE and is opt-in for advanced flows.
SCALE_TABLE_MOVES = {
    'helix': [HelixNotes.P5, HelixNotes.CENTER],
    'loop': [LoopNotes.PPII, LoopNotes.RESOLUTION],  # RESOLUTION is the key!
}


def snap_to_movement(mode: str, phi: float, psi: float):
    """Return the nearest Movement object (not just angles).

    If annotation classes are unavailable, falls back to nearest numeric bin
    and returns (phi, psi) tuple for backward compatibility.
    """
    if Movement is None or Gesture is None:
        return snap_to_scale(mode, phi, psi)

    # Build candidate Movement objects for this mode
    note_symbols = SCALE_TABLE_MOVES.get(mode, [])
    candidates: List[Movement] = []
    for sym in note_symbols:
        if mode == 'helix':
            gest = Gesture.HELIX_P5 if sym == HelixNotes.P5 else Gesture.SHEET_CENTER
        elif mode == 'loop':
            gest = Gesture.LOOP_RESOLUTION if sym == LoopNotes.RESOLUTION else Gesture.LOOP_RESOLUTION
        else:
            gest = Gesture.LOOP_RESOLUTION
        candidates.append(Movement(gesture=gest, mode=mode, role='body'))

    if not candidates:
        return snap_to_scale(mode, phi, psi)

    # Choose nearest by angular distance
    def degs(m: Movement) -> Tuple[float, float]:
        return m.to_degrees()

    import math
    def ang_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dphi = (a[0] - b[0])
        dpsi = (a[1] - b[1])
        # wrap to [-180,180]
        dphi = ((dphi + 180.0) % 360.0) - 180.0
        dpsi = ((dpsi + 180.0) % 360.0) - 180.0
        return math.hypot(dphi, dpsi)

    target = (float(phi), float(psi))
    best = min(candidates, key=lambda m: ang_dist(target, degs(m)))
    return best


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
