#!/usr/bin/env python3
"""
Core components of the 48-Manifold system.

This module provides the foundational building blocks for creating systems
based on fractal reversibility and alphabetic resonance, including:
- The core `Fractal48Layer` for reversible operations.
- `RouterMode` for defining possibility/manifestation paths.
- `SixAxisState` for semantic coordinate systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Ensure MPS compatibility
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Canonical manifold dimensionality used across the project
MANIFOLD_DIM = 48

class RouterMode(Enum):
    """Defines the two primary routing modes in the 48-Manifold system."""
    W_POSSIBILITY = "W"  # Left hand, open future, possibility
    M_MANIFESTATION = "M"  # Right hand, crystallized present, manifestation

@dataclass
class SixAxisState:
    """Represents a point in the 6D semantic manifold."""
    who: torch.Tensor    # Identity/agency
    what: torch.Tensor   # Object/substance
    when: torch.Tensor   # Temporal/phase
    where: torch.Tensor  # Spatial/location
    why: torch.Tensor    # Causal/purpose
    how: torch.Tensor    # Method/process

    def to_tensor(self) -> torch.Tensor:
        """Stack all six axes into a single 6xD tensor."""
        return torch.stack([
            self.who, self.what, self.when,
            self.where, self.why, self.how
        ])

class Fractal48Layer(nn.Module):
    """
    Core 48-basis reversible layer.

    This layer implements a reversible transformation based on a 48-channel
    factorization (48 = 2^4 * 3), using space-to-depth and depth-to-space
    operations for perfect reversibility.
    """

    def __init__(self, channels: int = 48):
        super().__init__()
        assert channels % 48 == 0, "Channels must be 48-aligned"
        self.channels = channels

        # Reversible mixing matrices (det = ±1)
        self.keven_mix = nn.Parameter(torch.eye(channels))
        self.kodd_mix = nn.Parameter(torch.eye(channels))

        # Initialize near-orthogonal for better conditioning
        nn.init.orthogonal_(self.keven_mix)
        nn.init.orthogonal_(self.kodd_mix)

    def space_to_depth(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Generic space-to-depth using pixel_unshuffle."""
        return F.pixel_unshuffle(x, downscale_factor=factor)

    def depth_to_space(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Generic depth-to-space using pixel_shuffle."""
        return F.pixel_shuffle(x, upscale_factor=factor)

    def _apply_channel_mix(self, x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Apply a (48x48) mixing matrix across the channel dimension per 48-ch block."""
        B, C, H, W = x.shape
        assert C % self.channels == 0, "Channel dimension must be a multiple of base channels"
        n_blocks = C // self.channels
        x = x.view(B, n_blocks, self.channels, H, W)
        # TODO: Consider if `view` is better than `reshape` for performance.
        # `reshape` can sometimes copy data, while `view` guarantees it won't.
        # For now, this is functionally correct.
        x = torch.einsum('bnchw,cd->bndhw', x, mix)
        return x.view(B, C, H, W)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Forward or inverse pass through the 48-factorization."""
        if not inverse:
            # Encode: 48 -> 16 -> 8 -> 4 (via 3x then 2x space-to-depth)
            B, C, H, W = x.shape
            if (H % 3 == 0) and (W % 3 == 0):
                x = self.space_to_depth(x, 3)
                x = self._apply_channel_mix(x, self.kodd_mix)
            B, C, H, W = x.shape
            if (H % 2 == 0) and (W % 2 == 0):
                x = self.space_to_depth(x, 2)
                x = self._apply_channel_mix(x, self.keven_mix)
        else:
            # Decode: 4 -> 8 -> 16 -> 48
            # Note: The inverse operation uses the transpose of the mixing matrices.
            B, C, H, W = x.shape
            if (H * 2 <= 256 and W * 2 <= 256): # Heuristic to decide if we can apply the next step
                x = self._apply_channel_mix(x, self.keven_mix.T)
                x = self.depth_to_space(x, 2)
            B, C, H, W = x.shape
            if (H * 3 <= 256 and W * 3 <= 256):
                x = self._apply_channel_mix(x, self.kodd_mix.T)
                x = self.depth_to_space(x, 3)

        return x


# === Parity/selection utilities ==================================================
def _broadcast_mask(x: torch.Tensor, dim: int, mask_1d: torch.Tensor, *, keep: bool = True) -> torch.Tensor:
    """Broadcast a 1D boolean mask along `dim` to match x's shape.

    If keep=True the mask selects entries to keep (others zeroed).
    If keep=False the mask selects entries to kill (these zeroed).
    """
    if mask_1d.dtype != torch.bool:
        raise ValueError("mask_1d must be a boolean tensor")
    if mask_1d.numel() != x.size(dim):
        raise ValueError("mask length must match the size of x along the given dim")

    # Decide which positions are 1 vs 0
    use_mask = mask_1d if keep else ~mask_1d

    # Build broadcastable shape
    shape = [1] * x.dim()
    shape[dim] = use_mask.numel()
    bmask = use_mask.view(shape).to(device=x.device)
    # Convert to multiplicative mask in same dtype as x
    if x.is_floating_point():
        bmask = bmask.to(dtype=x.dtype)
    else:
        bmask = bmask.to(dtype=torch.float32)
    return x * bmask


class Kull:
    """Generic culling/selection utilities.

    Provides low-level helpers to keep or kill positions based on a boolean mask
    along an arbitrary tensor dimension. Higher-level units (KEven/KOdd) use this.
    """

    @staticmethod
    def keep(x: torch.Tensor, dim: int, mask_1d: torch.Tensor) -> torch.Tensor:
        """Keep positions where mask_1d is True; zero others."""
        return _broadcast_mask(x, dim, mask_1d, keep=True)

    @staticmethod
    def kill(x: torch.Tensor, dim: int, mask_1d: torch.Tensor) -> torch.Tensor:
        """Kill (zero) positions where mask_1d is True; keep others."""
        return _broadcast_mask(x, dim, mask_1d, keep=False)


class KEven:
    """Even-index selector utilities.

    Semantics:
    - keep(x): keep even indices (i % 2 == 0), zero the odd ones.
    - kill(x): zero the even indices, keep the odd ones.
    """

    @staticmethod
    def mask(length: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        return (idx % 2 == 0)

    @staticmethod
    def indices(length: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return 1D tensor of even indices in range(length)."""
        return torch.nonzero(KEven.mask(length, device=device), as_tuple=False).squeeze(-1)

    @staticmethod
    def keep(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        m = KEven.mask(x.size(dim), device=x.device)
        return Kull.keep(x, dim, m)

    @staticmethod
    def kill(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        m = KEven.mask(x.size(dim), device=x.device)
        return Kull.kill(x, dim, m)


class KOdd:
    """Odd-index selector utilities.

    Semantics:
    - keep(x): keep odd indices (i % 2 == 1), zero the even ones.
    - kill(x): zero the odd indices, keep the even ones.
    """

    @staticmethod
    def mask(length: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        return (idx % 2 == 1)

    @staticmethod
    def indices(length: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return 1D tensor of odd indices in range(length)."""
        return torch.nonzero(KOdd.mask(length, device=device), as_tuple=False).squeeze(-1)

    @staticmethod
    def keep(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        m = KOdd.mask(x.size(dim), device=x.device)
        return Kull.keep(x, dim, m)

    @staticmethod
    def kill(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        m = KOdd.mask(x.size(dim), device=x.device)
        return Kull.kill(x, dim, m)


# Lowercase aliases requested for educational/teaching compatibility
class kull:
    keep = staticmethod(Kull.keep)
    kill = staticmethod(Kull.kill)


class keven:
    mask = staticmethod(KEven.mask)
    keep = staticmethod(KEven.keep)
    kill = staticmethod(KEven.kill)


class kodd:
    mask = staticmethod(KOdd.mask)
    indices = staticmethod(KOdd.indices)
    keep = staticmethod(KOdd.keep)
    kill = staticmethod(KOdd.kill)


# Public API
__all__ = [
    "device",
    "MANIFOLD_DIM",
    "RouterMode",
    "SixAxisState",
    "Fractal48Layer",
    "Kull",
    "KEven",
    "KOdd",
    "kull",
    "keven",
    "kodd",
]
