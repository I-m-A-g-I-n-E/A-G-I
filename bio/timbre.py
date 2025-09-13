#!/usr/bin/env python3
"""
Timbre vector engineering for amino acids on the 48D manifold.

Produces a fixed, canonical 48D vector for each of the 20 amino acids using
simple, interpretable mappings from physicochemical properties into structured
subspaces of the manifold (keven/kodd).
"""
from __future__ import annotations

import math
import torch
from typing import Dict, List

from .amino_acids import AMINO_ACIDS, AA_PROPERTIES
from manifold import MANIFOLD_DIM, keven, kodd


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.norm(x)
    if float(n.item()) < eps:
        return x
    return x / n


class TimbreGenerator:
    """Engineers a 48D timbre vector from amino acid properties.

    The generator precomputes a map from amino acid code -> 48D unit vector.
    """

    def __init__(self):
        # Precompute property stats (z-score parameters) across the 20 amino acids
        props: List[List[float]] = [AA_PROPERTIES[aa] for aa in AMINO_ACIDS]
        t = torch.tensor(props, dtype=torch.float32)  # (20,5)
        # Columns: [hydrophobicity, size, charge, helix, sheet]
        self.h_mean, self.h_std = t[:, 0].mean(), t[:, 0].std(unbiased=False) + 1e-6
        self.s_mean, self.s_std = t[:, 1].mean(), t[:, 1].std(unbiased=False) + 1e-6
        self.he_mean, self.he_std = t[:, 3].mean(), t[:, 3].std(unbiased=False) + 1e-6
        self.sh_mean, self.sh_std = t[:, 4].mean(), t[:, 4].std(unbiased=False) + 1e-6

        # Define interpretable biochemical clusters (overlapping by design)
        self.clusters = {
            "hydrophobics": set(list("ILVFM")),
            "aromatics": set(list("FWY")),
            "acidic": set(list("DE")),
            "basic": set(list("KRH")),
            "polar_uncharged": set(list("STNQ")),
            "special": set(list("GPC")),  # gly, pro, cys
            "small": set(list("AGST")),
        }

        self.cluster_keys = list(self.clusters.keys())  # fixed order

        self.timbre_map: Dict[str, torch.Tensor] = {aa: self.generate(aa) for aa in AMINO_ACIDS}

    def _generate_signal(self, value: float, length: int, freq: float = 1.0) -> torch.Tensor:
        """Generate a simple sinusoidal signal based on a scalar property.

        The amplitude scales with the magnitude of the value (normalized by 5.0)
        and the phase is offset by the value to differentiate amino acids.
        """
        # Use a fixed time base for determinism
        t = torch.linspace(0.0, 2.0 * math.pi * float(freq), steps=length, dtype=torch.float32)
        amp = float(value) / 5.0
        return torch.sin(t + float(value)) * amp

    def _z(self, x: float, mean: torch.Tensor, std: torch.Tensor) -> float:
        return float((torch.tensor(x, dtype=torch.float32) - mean) / std)

    def generate(self, amino_acid: str) -> torch.Tensor:
        """Generate a single 48D timbre vector for a given amino acid."""
        props = AA_PROPERTIES[amino_acid]
        hydrophobicity, size, charge, helix, sheet = props

        # Initialize the 48D vector
        vector = torch.zeros(MANIFOLD_DIM, dtype=torch.float32)

        # Map properties to specific subspaces of the manifold
        # keven (structure): hydrophobicity, size, helix-sheet differential
        kev_idx = keven.indices(MANIFOLD_DIM)
        # Normalize to balance scales across properties
        h_z = self._z(hydrophobicity, self.h_mean, self.h_std)
        s_z = self._z(size, self.s_mean, self.s_std)
        hs_diff_z = self._z(helix, self.he_mean, self.he_std) - self._z(sheet, self.sh_mean, self.sh_std)

        vector[kev_idx[0:8]] = self._generate_signal(h_z, 8, freq=1.0)
        vector[kev_idx[8:16]] = self._generate_signal(s_z, 8, freq=2.0)
        vector[kev_idx[16:24]] = self._generate_signal(hs_diff_z, 8, freq=3.0)

        # kodd (dynamics/tension): charge as localized impulse
        kod_idx = kodd.indices(MANIFOLD_DIM)

        # 1) Cluster multi-hot across the first slice of kodd (7 dims)
        # Each active cluster contributes a small, equal weight.
        cluster_weights = torch.zeros(7, dtype=torch.float32)
        for i, key in enumerate(self.cluster_keys):
            if amino_acid in self.clusters[key]:
                cluster_weights[i] = 1.0
        if cluster_weights.sum() > 0:
            cluster_weights = cluster_weights / cluster_weights.sum()
        vector[kod_idx[0:7]] = cluster_weights

        # 2) Charge encoding (two dedicated slots): negative -> slot 8, positive -> slot 9
        if charge < 0:
            vector[kod_idx[8]] = 1.5  # stronger impulse for charge
        elif charge > 0:
            vector[kod_idx[9]] = 1.5

        # 3) Aromaticity emphasis (slot 10)
        if amino_acid in self.clusters["aromatics"]:
            vector[kod_idx[10]] = 1.0

        # 4) Special cases (G, P, C) into slots 11-13
        specials = ["G", "P", "C"]
        for j, aa_sym in enumerate(specials):
            if amino_acid == aa_sym:
                vector[kod_idx[11 + j]] = 1.0

        return _unit(vector)


def get_cosine_similarity_vector(timbre_map: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a flattened 190-element vector of pairwise cosine similarities."""
    vectors = torch.stack([timbre_map[aa] for aa in AMINO_ACIDS], dim=0)  # (20,48)
    sim_matrix = torch.matmul(vectors, vectors.T)  # (20,20)

    vals = []
    n = len(AMINO_ACIDS)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(sim_matrix[i, j].item())
    return torch.tensor(vals, dtype=torch.float32)
