"""
Measurement utilities for numerical tolerances and default weights.
"""
from __future__ import annotations
import torch
from typing import Literal

Purpose = Literal['reconstruction', 'comparison']

def tolerance_for(dtype: torch.dtype, purpose: Purpose = 'reconstruction') -> float:
    """Return an absolute tolerance based on dtype and purpose.
    Defaults chosen to preserve existing test behavior.
    """
    # Base table
    if purpose == 'reconstruction':
        if dtype == torch.float64:
            return 1e-10
        else:
            # Assume float32/float16 use same looser bound used in tests
            return 1e-5
    elif purpose == 'comparison':
        if dtype == torch.float64:
            return 1e-12
        else:
            return 1e-6
    else:
        # Fallback
        return 1e-5

# Default symbolic weights for dyadic/triadic contributions, available to tests if needed
DYADIC_WEIGHT: float = 1.0
TRIADIC_WEIGHT: float = 1.0
