#!/usr/bin/env python3
"""
Protein Folding utilities for the 48-Manifold immune analog.

This module provides lightweight, dependency-free adapters to map protein
folding windows to the 48-atom manifold used by `immunity.py`.

Design goals:
- Keep it torch-only (no external bio libs), with simple synthetic generators
  so the pipeline runs out-of-the-box.
- Provide clear extension points to swap in real structural features later
  (Ramachandran, contact-map consistency, SASA, DSSP, etc.).

Exports:
- featurize_window(x):
    Accepts either a [48] vector (assumed already per-residue wholeness)
    or a [48, 3] coordinate path. Returns a normalized [48] per-residue vector.
- compute_folding_score(peptides):
    Aggregates a [48] vector to a scalar folding_score (mean in [0,1]).
- generate_synthetic_windows(...):
    Produces simple synthetic windows for self / foreign / misfolded cases
    aligned to the thymus education patterns used by `immunity.py`.
 - educate_thymus_from_synthetics(immune, windows):
     Feeds self windows to the thymus using the same featurization signature
     that will be used during presentation, to avoid signature mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import torch

# Canonical 48 size
MHC_SIZE = 48

def _normalize_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    denom = (x_max - x_min).clamp_min(eps)
    return (x - x_min) / denom


def featurize_window(x: torch.Tensor) -> torch.Tensor:
    """
    Map an input window to a per-residue [48] vector in [0, 1].

    Accepted forms:
    - [48]: already per-residue wholeness-like values (any real range). We min-max normalize.
    - [48, 3]: toy 3D coordinates; we compute a simple smoothness/curvature proxy
      using second differences and invert/normalize to [0,1] (smoother = higher score).
    """
    if x.dim() == 1 and x.numel() == MHC_SIZE:
        v = x.to(dtype=torch.float32)
        return _normalize_minmax(v)
    if x.dim() == 2 and x.shape[0] == MHC_SIZE and x.shape[1] == 3:
        # Second-difference curvature magnitude per residue (toy proxy)
        coords = x.to(dtype=torch.float32)
        # Pad endpoints by replication for a simple 2nd-diff on edges
        c_prev = torch.vstack([coords[0:1], coords[:-1]])
        c_next = torch.vstack([coords[1:], coords[-1:]])
        second_diff = c_next - 2 * coords + c_prev  # [48,3]
        curv = second_diff.pow(2).sum(dim=1).sqrt()  # [48]
        # Invert: smoother (lower curvature) -> higher score
        curv_norm = _normalize_minmax(curv)
        score = 1.0 - curv_norm
        return score.clamp(0.0, 1.0)
    raise ValueError("featurize_window expects [48] or [48,3] tensor")


def compute_folding_score(peptides: torch.Tensor) -> float:
    """
    Aggregate a [48] per-residue vector to a scalar folding score in [0,1].
    Current policy: simple mean.
    """
    if peptides.dim() != 1 or peptides.numel() != MHC_SIZE:
        raise ValueError("compute_folding_score expects a [48] vector")
    return float(peptides.clamp(0.0, 1.0).mean().item())


@dataclass
class SyntheticWindow:
    epitope_id: str
    peptides: torch.Tensor  # [48] in [0,1]
    folding_score: float    # scalar in [0,1]
    kind: Literal["self", "foreign", "misfolded"]


def generate_synthetic_windows(
    n_self: int = 4,
    n_mis: int = 4,
    n_foreign: int = 4,
    seed: int = 42,
) -> List[SyntheticWindow]:
    """
    Create simple synthetic windows aligned to the thymus patterns used in
    `immunity.ImmuneSystem._create_thymus()` (bases: 0, 100, 200).

    - self: peptides derived from torch.arange(48)+base, normalized, high score
    - foreign: torch.arange(48)+offset (offset far away), normalized, high score
    - misfolded: random noise; penalized folding_score
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    windows: List[SyntheticWindow] = []

    bases = [0, 100, 200]
    for i in range(n_self):
        base = bases[i % len(bases)]
        raw = torch.arange(MHC_SIZE, dtype=torch.float32) + base
        pep = featurize_window(raw)
        fs = compute_folding_score(pep)
        windows.append(SyntheticWindow(
            epitope_id=f"self_{i}", peptides=pep, folding_score=fs, kind="self"
        ))

    for i in range(n_foreign):
        # Create a distinct shape so min-max normalization does not collapse it to the same ramp
        offset = 1000 + 37 * i
        idx = torch.arange(MHC_SIZE, dtype=torch.float32)
        # Sinusoidal modulation (different frequency/phase per i) + offset
        mod = 5.0 * torch.sin(2.0 * torch.pi * (idx / MHC_SIZE) * (1 + (i % 3)) + (i * 0.7))
        raw = idx + offset + mod
        pep = featurize_window(raw)
        fs = compute_folding_score(pep)
        windows.append(SyntheticWindow(
            epitope_id=f"foreign_{i}", peptides=pep, folding_score=fs, kind="foreign"
        ))

    for i in range(n_mis):
        noise = torch.randn((MHC_SIZE,), generator=g)
        pep = featurize_window(noise)
        # Down-weight folding_score to simulate misfold
        fs = max(0.0, compute_folding_score(pep) - 0.6)
        windows.append(SyntheticWindow(
            epitope_id=f"mis_{i}", peptides=pep, folding_score=fs, kind="misfolded"
        ))

    return windows


def educate_thymus_from_synthetics(immune_system, windows: List[SyntheticWindow]) -> int:
    """
    Add featurized self-window signatures to the thymus so later presentations
    of those windows are recognized as self. Returns the number of patterns added.

    immune_system: an instance of immunity.ImmuneSystem
    windows: list of SyntheticWindow
    """
    # Late import to avoid circular import at module load time
    from immunity import CellType  # type: ignore

    thymus = immune_system.thymus
    added = 0
    for w in windows:
        if w.kind == "self":
            # Use the same peptide vector that will be hashed at presentation time
            # (Antigen.compute_self_signature hashes the tensor bytes directly)
            thymus.check_tolerance(w.peptides)
            added += 1
    return added
