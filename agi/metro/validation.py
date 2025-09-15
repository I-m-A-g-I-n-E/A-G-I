"""Quantitative validation utilities: TM-score and RMSD via tmtools."""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import tmtools  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    tmtools = None


def compare_structures(pdb_generated_path: str, pdb_reference_path: str) -> Dict:
    """Calculates TM-score and RMSD between two PDB files using tmtools.

    Returns a dictionary with keys: tm_score, rmsd, alignment_length.
    If tmtools is unavailable, returns a dict with null metrics and a warning string.
    """
    if tmtools is None:  # graceful degradation
        return {
            "tm_score": None,
            "rmsd": None,
            "alignment_length": None,
            "warning": "tmtools not installed; install from requirements.txt to enable validation",
        }

    coords_gen, _ = tmtools.io.get_coords(pdb_generated_path)
    coords_ref, _ = tmtools.io.get_coords(pdb_reference_path)
    # Use first model, clip to shared length
    min_len = min(len(coords_gen[0]), len(coords_ref[0]))
    res = tmtools.tm_align(np.asarray(coords_gen[0][:min_len]), np.asarray(coords_ref[0][:min_len]))
    return {
        "tm_score": float(res.tm_score),
        "rmsd": float(res.rmsd),
        "alignment_length": int(min_len),
    }
