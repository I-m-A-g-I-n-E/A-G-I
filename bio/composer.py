#!/usr/bin/env python3
"""
Harmonic Propagator: compositional layers that resolve amino-acid note fields
into stable 48D composition vectors without attention or heavy matmuls.

Uses the reversible `Fractal48Layer` and a strict `kull`-like distillation rule
(2 keven + 2 kodd) to preserve only the most stable, on-grid information.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from manifold import MANIFOLD_DIM, Fractal48Layer, keven, kodd
from .timbre import TimbreGenerator


class HarmonicLayer(nn.Module):
    """
    A single layer of harmonic propagation. It is one section of the orchestra.
    It takes a field of notes and encourages them to settle into a more stable chord.
    Input expected shape: (B, C=48, H, W)
    """

    def __init__(self, channels: int = MANIFOLD_DIM):
        super().__init__()
        assert channels == MANIFOLD_DIM and channels % 48 == 0
        # The Reversible Mixer: allows notes to interact without information loss.
        self.mixer = Fractal48Layer(channels)
        # The Conductor's Baton: a simple activation to guide the harmony.
        self.activation = nn.GELU()

    def distill(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the `kull` operator's law: keep the most essential 2+2 core.
        This is computational apoptosis, pruning dissonance.
        Keeps the first two even and first two odd channel indices.
        """
        B, C, H, W = x.shape
        mask = torch.zeros(C, dtype=torch.bool, device=x.device)
        # Get the first 2 `keven` and first 2 `kodd` indices
        keven_indices = keven.indices(C)[:2]
        kodd_indices = kodd.indices(C)[:2]
        mask[keven_indices] = True
        mask[kodd_indices] = True
        return x * mask.view(1, C, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: mix, guide, and distill."""
        x = self.mixer(x)
        x = self.activation(x)
        x = self.distill(x)
        return x


class HarmonicPropagator(nn.Module):
    """
    The full Composer. It takes a sequence of amino acid 'notes' and propagates
    harmony through them, resolving them into Composition Vectors per window.
    Output shape: (num_windows, 48)
    """

    def __init__(self, n_layers: int = 4, timbre_generator: TimbreGenerator | None = None):
        super().__init__()
        self.timbre_generator = timbre_generator or TimbreGenerator()
        self.layers = nn.ModuleList([HarmonicLayer() for _ in range(n_layers)])

    @staticmethod
    def _validate_sequence(seq: str) -> str:
        seq = (seq or "").strip().upper()
        if not seq:
            raise ValueError("Empty amino acid sequence")
        # Basic validation: ensure only 20 canonical letters are present
        from .amino_acids import AMINO_ACIDS
        allowed = set(AMINO_ACIDS)
        bad = [c for c in seq if c not in allowed]
        if bad:
            raise ValueError(f"Invalid amino acids in sequence: {sorted(set(bad))}")
        return seq

    def _embed_sequence(self, aa_sequence: str) -> torch.Tensor:
        # 1. Timbre Embedding (The Sheet Music): (1, L, 48)
        notes: List[torch.Tensor] = [self.timbre_generator.timbre_map[aa] for aa in aa_sequence]
        x = torch.stack(notes, dim=0).unsqueeze(0)  # (1, L, 48)
        return x

    def forward(self, aa_sequence: str) -> torch.Tensor:
        """
        Takes a raw string of amino acids, composes it, and returns Composition Vectors.
        Output: (num_windows, 48)
        """
        aa_sequence = self._validate_sequence(aa_sequence)
        if len(aa_sequence) < MANIFOLD_DIM:
            raise ValueError("Sequence must be at least 48 amino acids long.")

        x = self._embed_sequence(aa_sequence)  # (1, L, 48)

        # 2. Windowing: overlapping 48-residue phrases with stride 16 (48/3)
        stride = MANIFOLD_DIM // 3
        windows = x.unfold(dimension=1, size=MANIFOLD_DIM, step=stride)  # (1, num_windows, 48, 48)
        # Reinterpret one 48 as channels, the other as spatial width
        # Target shape for mixer: (B=num_windows, C=48, H=1, W=48)
        B, num_windows, H, W = windows.shape  # B should be 1
        x = windows.reshape(B * num_windows, H, W)  # (num_windows, 48, 48)
        x = x.unsqueeze(2)  # (num_windows, 48, 1, 48)
        x = x  # channels=48, H=1, W=48

        # 3. Composition through reversible layers
        for layer in self.layers:
            x = layer(x)

        # 4. Resolve to Composition Vector: average across spatial dims (H and W)
        composition_vectors = x.mean(dim=(2, 3))  # (num_windows, 48)
        return composition_vectors
