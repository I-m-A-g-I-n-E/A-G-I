#!/usr/bin/env python3
"""
Harmonic Propagator: compositional layers that resolve amino-acid note fields
into stable 48D composition vectors without attention or heavy matmuls.

Uses the reversible `Fractal48Layer` and a strict `kull`-like distillation rule
(2 keven + 2 kodd) to preserve only the most stable, on-grid information.
"""
from __future__ import annotations

from typing import List, Optional

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

    def __init__(
        self,
        channels: int = MANIFOLD_DIM,
        *,
        pick_span_even: int = 8,
        pick_span_odd: int = 8,
        noise_std: float = 0.0,
        rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        assert channels == MANIFOLD_DIM and channels % 48 == 0
        # The Reversible Mixer: allows notes to interact without information loss.
        self.mixer = Fractal48Layer(channels)
        # The Conductor's Baton: a simple activation to guide the harmony.
        self.activation = nn.GELU()
        # Variability controls
        self.pick_span_even = int(max(2, pick_span_even))
        self.pick_span_odd = int(max(2, pick_span_odd))
        self.noise_std = float(max(0.0, noise_std))
        self.rng = rng

    def distill(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the `kull` operator's law: keep the most essential 2+2 core.
        This is computational apoptosis, pruning dissonance.
        Keeps the first two even and first two odd channel indices.
        """
        B, C, H, W = x.shape
        mask = torch.zeros(C, dtype=torch.bool, device=x.device)
        # Compute candidate pools (low-frequency band) then pick 2 from each
        kev_all = keven.indices(C)
        kod_all = kodd.indices(C)
        kev_pool = kev_all[: min(self.pick_span_even, kev_all.numel())]
        kod_pool = kod_all[: min(self.pick_span_odd, kod_all.numel())]

        # If pool smaller than 2, fall back to deterministic first two
        if kev_pool.numel() >= 2:
            # Randomly choose 2 distinct even indices
            kev_sel = torch.randperm(kev_pool.numel(), generator=self.rng, device=x.device)[:2]
            keven_indices = kev_pool[kev_sel]
        else:
            keven_indices = kev_all[:2]

        if kod_pool.numel() >= 2:
            kod_sel = torch.randperm(kod_pool.numel(), generator=self.rng, device=x.device)[:2]
            kodd_indices = kod_pool[kod_sel]
        else:
            kodd_indices = kod_all[:2]

        mask[keven_indices] = True
        mask[kodd_indices] = True
        return x * mask.view(1, C, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: mix, guide, and distill."""
        x = self.mixer(x)
        x = self.activation(x)
        if self.noise_std > 0.0:
            noise = torch.randn(x.shape, generator=self.rng, device=x.device, dtype=x.dtype) * self.noise_std
            x = x + noise
        x = self.distill(x)
        return x


class HarmonicPropagator(nn.Module):
    """
    The full Composer. It takes a sequence of amino acid 'notes' and propagates
    harmony through them, resolving them into Composition Vectors per window.
    Output shape: (num_windows, 48)
    """

    def __init__(
        self,
        n_layers: int = 4,
        timbre_generator: TimbreGenerator | None = None,
        *,
        variability: float = 0.0,
        seed: Optional[int] = None,
        window_jitter: bool = False,
    ):
        super().__init__()
        self.timbre_generator = timbre_generator or TimbreGenerator()
        # Seedable RNG
        self.rng = torch.Generator(device="cpu")
        if seed is not None:
            self.rng.manual_seed(int(seed))
        # Variability parameters
        v = float(max(0.0, min(1.0, variability)))
        # Map variability to spans and noise
        pick_span = 2 + int(round(v * 22))  # up to ~24 candidates
        noise_std = 0.0 if v == 0.0 else 0.02 * v
        self.window_jitter = bool(window_jitter)

        self.layers = nn.ModuleList([
            HarmonicLayer(
                pick_span_even=pick_span,
                pick_span_odd=pick_span,
                noise_std=noise_std,
                rng=self.rng,
            )
            for _ in range(n_layers)
        ])

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
        # Optional jitter: shift the start by an offset in [0, stride-1]
        if self.window_jitter and x.size(1) >= MANIFOLD_DIM + 1:
            offset = int(torch.randint(low=0, high=stride, size=(1,), generator=self.rng).item())
            x_eff = x[:, offset:]
            if x_eff.size(1) < MANIFOLD_DIM:
                x_eff = x  # fallback if too short after offset
        else:
            x_eff = x
        windows = x_eff.unfold(dimension=1, size=MANIFOLD_DIM, step=stride)  # (1, num_windows, 48, 48)
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
