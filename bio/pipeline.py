#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .composer import HarmonicPropagator
from .conductor import Conductor
from .key_estimator import estimate_key_and_modes
from .sonifier import TrinitySonifier
from .utils import ensure_dir_for, save_json


# -------------------------
# Composition
# -------------------------

def compose_sequence(
    sequence: str,
    *,
    samples: int = 1,
    variability: float = 0.0,
    seed: Optional[int] = None,
    window_jitter: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose an AA sequence into an ensemble of 48D windows, returning mean and certainty.

    Returns:
        mean_composition: [W,48] float32 torch tensor
        harmonic_certainty: [W] float32 torch tensor
    """
    all_runs: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(max(1, samples)):
            sample_seed = (seed + i) if seed is not None else None
            composer = HarmonicPropagator(
                n_layers=4,
                variability=variability,
                seed=sample_seed,
                window_jitter=window_jitter,
            )
            comp_vecs = composer(sequence)  # (num_windows, 48)
            all_runs.append(comp_vecs)

    # Align varying window counts (due to jitter) by truncating to the minimum
    min_windows = min(cv.shape[0] for cv in all_runs)
    all_runs_aligned = [cv[:min_windows] for cv in all_runs]
    ensemble = torch.stack(all_runs_aligned, dim=0)  # (samples, W, 48)
    mean_composition = ensemble.mean(dim=0)          # (W, 48)
    variance_composition = ensemble.var(dim=0, unbiased=False)       # (W, 48)
    harmonic_certainty = (1.0 - variance_composition.mean(dim=-1)).clamp(0.0, 1.0)  # (W,)

    return mean_composition.to(torch.float32), harmonic_certainty.to(torch.float32)


# -------------------------
# Structure generation & refinement
# -------------------------

def conduct_backbone(
    composition_vectors: torch.Tensor,
    sequence: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Conductor]:
    """Generate N-CA-C backbone and torsions from composition vectors.

    Returns backbone, phi, psi, modes, and the Conductor used.
    """
    conductor = Conductor()
    backbone, phi, psi, modes = conductor.build_backbone(composition_vectors, sequence=sequence)
    return backbone, phi, psi, modes, conductor


def refine_backbone(
    conductor: Conductor,
    backbone: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    modes: List[str],
    sequence: str,
    *,
    max_iters: int = 150,
    step_deg: float = 2.0,
    seed: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    **refine_kwargs,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Refine torsions to reduce dissonance and clashes.

    Returns (refined_torsions, refined_backbone).
    """
    init_torsions = [(float(phi[i]), float(psi[i])) for i in range(len(phi))]
    try:
        refined_torsions, refined_backbone = conductor.refine_torsions(
            init_torsions, modes, sequence,
            max_iters=max_iters, step_deg=step_deg, seed=seed, weights=weights, **refine_kwargs,
        )
    except TypeError:
        # Backward compatibility: conductor.refine_torsions may not accept advanced kwargs
        refined_torsions, refined_backbone = conductor.refine_torsions(
            init_torsions, modes, sequence,
            max_iters=max_iters, step_deg=step_deg, seed=seed, weights=weights,
        )
    return refined_torsions, refined_backbone


def quality_report(
    conductor: Conductor,
    backbone: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    modes: List[str],
) -> Dict:
    return conductor.quality_check(backbone, phi, psi, modes)


# -------------------------
# Sonification
# -------------------------

def estimate_kore(
    composition_vectors: torch.Tensor,
    sequence: str,
) -> torch.Tensor:
    kore, _ = estimate_key_and_modes(composition_vectors, sequence)
    return kore


def sonify_3ch(
    composition_vectors: torch.Tensor,
    kore_vector: torch.Tensor,
    certainty: torch.Tensor,
    dissonance: torch.Tensor,
    *,
    bpm: float = 96.0,
    stride_ticks: int = 16,
    sample_rate: int = 48000,
    tonic_hz: float = 220.0,
    center_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    cw = center_weights or {"kore": 1.5, "cert": 1.0, "diss": 2.5}
    son = TrinitySonifier(sample_rate=sample_rate, bpm=bpm, tonic_hz=tonic_hz, stride_ticks=stride_ticks)
    wave = son.sonify_composition_3ch(composition_vectors, kore_vector, certainty, dissonance, cw)
    return wave


def save_wav(wave: np.ndarray, path: str, sample_rate: int = 48000) -> None:
    ensure_dir_for(path)
    # Reuse TrinitySonifier for safe normalization and writing
    son = TrinitySonifier(sample_rate=sample_rate)
    son.save_wav(wave, path)


# -------------------------
# Helpers
# -------------------------

def dissonance_scalar_to_vec(value: float, length: int) -> torch.Tensor:
    return torch.tensor([float(value)] * int(length), dtype=torch.float32)
