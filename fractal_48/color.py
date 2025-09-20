"""
Color palette and parity-aware coloring for the 48-manifold.
"""

import numpy as np
import math
from typing import Tuple, Union


def hsl_to_srgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    """
    Convert HSL to sRGB with gamma correction.
    
    Args:
        h: Hue [0, 360)
        s: Saturation [0, 1]
        l: Lightness [0, 1]
        
    Returns:
        (r, g, b) in [0, 1] with gamma 2.2 correction
    """
    # Normalize hue to [0, 1]
    h = (h % 360.0) / 360.0
    
    # HSL to RGB conversion
    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p
    
    if s == 0:
        r = g = b = l  # achromatic
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    # Apply gamma correction (gamma = 2.2)
    gamma = 2.2
    r = pow(max(0, min(1, r)), 1/gamma)
    g = pow(max(0, min(1, g)), 1/gamma)
    b = pow(max(0, min(1, b)), 1/gamma)
    
    return r, g, b


def palette48(n_smooth: float, phi: int, parity: int, config) -> Tuple[float, float, float]:
    """
    Generate 48-phase aware color with parity gating.
    
    Args:
        n_smooth: Smooth escape time
        phi: Phase index [0, 47]
        parity: 0=keven, 1=kodd
        config: FractalConfig instance
        
    Returns:
        (r, g, b) color tuple
    """
    # Use cached hue table for performance
    cache = config.get_cache()
    hue = cache['hue_by_phi'][phi]
    
    # Base saturation and lightness from escape time
    # Map escape time to lightness with some contrast
    if n_smooth >= config.max_iters:
        # Interior points (didn't escape)
        saturation = 0.1
        lightness = 0.0
    else:
        # Exterior points (escaped)
        t = min(1.0, n_smooth / 100.0)  # Normalize escape time
        saturation = 0.7 + 0.3 * math.sin(2 * math.pi * t)
        lightness = 0.3 + 0.6 * t
    
    # Apply parity modulation (keven/kodd gating)
    if parity == 1:  # kodd
        saturation = max(0, min(1, saturation + config.delta_s))
        lightness = max(0, min(1, lightness + config.delta_l))
    else:  # keven
        saturation = max(0, min(1, saturation - config.delta_s))
        lightness = max(0, min(1, lightness - config.delta_l))
    
    return hsl_to_srgb(hue, saturation, lightness)


def palette_newton(basin: int, steps: int, phi: int, parity: int, config) -> Tuple[float, float, float]:
    """
    Generate color for Newton fractal basins.
    
    Args:
        basin: Root basin index (0, 1, 2)
        steps: Number of iteration steps
        phi: Phase index [0, 47]
        parity: 0=keven, 1=kodd
        config: FractalConfig instance
        
    Returns:
        (r, g, b) color tuple
    """
    # Use cached hue table and add basin offset
    cache = config.get_cache()
    base_hue_from_phi = cache['hue_by_phi'][phi]
    
    # Base hue by basin with 120° separation
    basin_hues = [0, 120, 240]  # Red, Green, Blue regions
    base_hue = (base_hue_from_phi + basin_hues[basin]) % 360.0
    
    # Lightness based on convergence speed
    if steps >= config.max_iters:
        lightness = 0.1  # Slow convergence
    else:
        lightness = 0.3 + 0.5 * (1.0 - min(1.0, steps / 50.0))
    
    saturation = 0.8
    
    # Apply parity modulation
    if parity == 1:  # kodd
        saturation = max(0, min(1, saturation + config.delta_s))
        lightness = max(0, min(1, lightness + config.delta_l))
    else:  # keven
        saturation = max(0, min(1, saturation - config.delta_s))
        lightness = max(0, min(1, lightness - config.delta_l))
    
    return hsl_to_srgb(base_hue, saturation, lightness)


def apply_smooth_coloring(n: int, z: complex, config) -> float:
    """
    Apply smooth coloring based on escape time.
    
    Args:
        n: Raw iteration count
        z: Final z value
        config: FractalConfig instance
        
    Returns:
        Smooth escape time value
    """
    if n >= config.max_iters:
        return float(config.max_iters)
    
    if abs(z) <= config.bailout:
        return float(n)
    
    # Smooth escape time calculation
    return n + 1 - math.log(math.log(abs(z), 2), 2)