import torch
import numpy as np
from bio.devices import svd_safe
from typing import List, Tuple

def estimate_key_and_modes(composition_vectors: torch.Tensor, sequence: str) -> Tuple[torch.Tensor, List[str]]:
    """
    Estimates a global kore vector (principal harmonic direction) and per-residue
    structural modes (helix/sheet/loop) using simple amino acid propensities.

    Args:
        composition_vectors: (L, 48) or (1, 48) torch tensor of composition vectors.
        sequence: Protein sequence as a string of one-letter amino acid codes.

    Returns:
        global_kore: (48,) torch tensor principal component.
        modes: list[str] of length len(sequence) with values in {"helix","sheet","loop"}.
    """
    # Ensure 2D for SVD
    if composition_vectors.ndim == 1:
        X = composition_vectors.unsqueeze(0)
    else:
        X = composition_vectors     

    # 1. Determine Global Kore (The Tonic Drone): top right singular vector
    # Use svd_safe to accommodate MPS/CPU/CUDA differences
    U, S, Vh = svd_safe(X, full_matrices=False)
    global_kore = Vh[0, :]

    # 2. Determine Local Modes (Helix, Sheet, Loop) via simple propensities
    helix_prone = set(['A', 'L', 'M', 'E', 'K', 'R', 'Q'])
    sheet_prone = set(['V', 'I', 'Y', 'F', 'W', 'T'])

    modes: List[str] = []
    for aa in sequence:
        if aa in helix_prone:
            modes.append('helix')
        elif aa in sheet_prone:
            modes.append('sheet')
        else:
            modes.append('loop')

    return global_kore, modes
