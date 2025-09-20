"""
CRT phase mapping and parity calculations for the 48-manifold.
"""

from typing import Tuple
import numpy as np


def crt_phase48(x: int, y: int) -> Tuple[int, int]:
    """
    Map pixel coordinates to 48-phase CRT coordinates with parity.
    
    Args:
        x, y: Pixel coordinates
        
    Returns:
        (phi, parity): phi in [0, 47], parity in {0=keven, 1=kodd}
    """
    # CRT mapping: 48 = 16 × 3
    d = x % 16  # dyadic component (0-15)
    t = y % 3   # triadic component (0-2)
    phi = d * 3 + t  # phase index [0, 47]
    
    # Parity calculation
    parity = (x + y) & 1  # 0=keven (even), 1=kodd (odd)
    
    return phi, parity


def keven_kodd_mask(width: int, height: int) -> np.ndarray:
    """
    Generate boolean mask for keven/kodd parity across the canvas.
    
    Args:
        width, height: Canvas dimensions
        
    Returns:
        Boolean array where True=kodd, False=keven
    """
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    return ((x_coords + y_coords) & 1).astype(bool)


def phi_field(width: int, height: int) -> np.ndarray:
    """
    Generate the phi field across the canvas.
    
    Args:
        width, height: Canvas dimensions
        
    Returns:
        Integer array with phi values [0, 47]
    """
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    d = x_coords % 16
    t = y_coords % 3
    return (d * 3 + t).astype(np.int32)


def iters_for_phi(phi: int, base_iters: int, variation: float = 0.1) -> int:
    """
    Calculate iteration count based on phi for variation across the 48-manifold.
    
    Args:
        phi: Phase index [0, 47]
        base_iters: Base iteration count
        variation: Variation factor [0, 1]
        
    Returns:
        Adjusted iteration count
    """
    # Create smooth variation based on phi
    phase_factor = 1.0 + variation * np.sin(2 * np.pi * phi / 48)
    return max(1, int(base_iters * phase_factor))