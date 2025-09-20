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


def hsl_to_srgb_vectorized(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Vectorized HSL to sRGB conversion with gamma correction.
    
    Args:
        h: Hue array [0, 360)
        s: Saturation array [0, 1] 
        l: Lightness array [0, 1]
        
    Returns:
        RGB array of shape (*h.shape, 3) with values in [0, 1] and gamma 2.2 correction
    """
    # Normalize hue to [0, 1]
    h_norm = (h % 360.0) / 360.0
    
    # HSL to RGB conversion - vectorized
    def hue_to_rgb_vectorized(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.where(t < 0, t + 1, t)
        t = np.where(t > 1, t - 1, t)
        
        result = np.where(t < 1/6, p + (q - p) * 6 * t, 
                 np.where(t < 1/2, q,
                 np.where(t < 2/3, p + (q - p) * (2/3 - t) * 6, p)))
        return result
    
    # Handle achromatic case
    achromatic = (s == 0)
    
    # Compute q and p arrays
    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    
    # Compute RGB channels
    r = hue_to_rgb_vectorized(p, q, h_norm + 1/3)
    g = hue_to_rgb_vectorized(p, q, h_norm)
    b = hue_to_rgb_vectorized(p, q, h_norm - 1/3)
    
    # Handle achromatic pixels
    r = np.where(achromatic, l, r)
    g = np.where(achromatic, l, g) 
    b = np.where(achromatic, l, b)
    
    # Apply gamma correction (gamma = 2.2) as array ops
    gamma = 2.2
    r = np.power(np.clip(r, 0, 1), 1/gamma)
    g = np.power(np.clip(g, 0, 1), 1/gamma)
    b = np.power(np.clip(b, 0, 1), 1/gamma)
    
    # Stack into RGB array
    return np.stack([r, g, b], axis=-1).astype(np.float32)


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


def palette48_vectorized(n_smooth: np.ndarray, phi_map: np.ndarray, parity_map: np.ndarray, config) -> np.ndarray:
    """
    Vectorized 48-phase aware color generation with parity gating.
    
    Args:
        n_smooth: Array of smooth escape times
        phi_map: Array of phase indices [0, 47]
        parity_map: Array of parity values {0=keven, 1=kodd}
        config: FractalConfig instance
        
    Returns:
        RGB array of shape (*phi_map.shape, 3) with values in [0, 1]
    """
    # Get cached hue table for vectorized lookup
    cache = config.get_cache()
    hue_by_phi = cache['hue_by_phi']
    
    # Vectorized hue lookup
    hue = hue_by_phi[phi_map]
    
    # Base saturation and lightness from escape time
    interior_mask = (n_smooth >= config.max_iters)
    
    # Initialize arrays
    saturation = np.zeros_like(n_smooth, dtype=np.float32)
    lightness = np.zeros_like(n_smooth, dtype=np.float32)
    
    # Interior points (didn't escape)
    saturation[interior_mask] = 0.1
    lightness[interior_mask] = 0.0
    
    # Exterior points (escaped) - vectorized computation  
    exterior_mask = ~interior_mask
    t = np.minimum(1.0, n_smooth[exterior_mask] / 100.0)  # Normalize escape time
    saturation[exterior_mask] = 0.7 + 0.3 * np.sin(2 * np.pi * t)
    lightness[exterior_mask] = 0.3 + 0.6 * t
    
    # Apply parity modulation (keven/kodd gating) - vectorized
    kodd_mask = (parity_map == 1)
    keven_mask = (parity_map == 0)
    
    # kodd parity - add deltas
    saturation[kodd_mask] = np.clip(saturation[kodd_mask] + config.delta_s, 0, 1)
    lightness[kodd_mask] = np.clip(lightness[kodd_mask] + config.delta_l, 0, 1)
    
    # keven parity - subtract deltas  
    saturation[keven_mask] = np.clip(saturation[keven_mask] - config.delta_s, 0, 1)
    lightness[keven_mask] = np.clip(lightness[keven_mask] - config.delta_l, 0, 1)
    
    # Convert HSL to sRGB using vectorized function
    return hsl_to_srgb_vectorized(hue, saturation, lightness)


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


def palette_newton_vectorized(basin: np.ndarray, steps: np.ndarray, phi_map: np.ndarray, parity_map: np.ndarray, config) -> np.ndarray:
    """
    Vectorized Newton fractal basin coloring.
    
    Args:
        basin: Array of root basin indices (0, 1, 2)
        steps: Array of iteration steps
        phi_map: Array of phase indices [0, 47]
        parity_map: Array of parity values {0=keven, 1=kodd}
        config: FractalConfig instance
        
    Returns:
        RGB array of shape (*phi_map.shape, 3) with values in [0, 1]
    """
    # Get cached hue table for vectorized lookup
    cache = config.get_cache()
    hue_by_phi = cache['hue_by_phi']
    
    # Base hue by basin with 120° separation
    basin_hues = np.array([0, 120, 240], dtype=np.float32)  # Red, Green, Blue regions
    base_hue_from_phi = hue_by_phi[phi_map]
    base_hue = (base_hue_from_phi + basin_hues[basin]) % 360.0
    
    # Lightness based on convergence speed - vectorized
    slow_convergence = (steps >= config.max_iters)
    lightness = np.where(slow_convergence, 
                        0.1,  # Slow convergence
                        0.3 + 0.5 * (1.0 - np.minimum(1.0, steps / 50.0)))
    
    saturation = np.full_like(lightness, 0.8, dtype=np.float32)
    
    # Apply parity modulation - vectorized
    kodd_mask = (parity_map == 1)
    keven_mask = (parity_map == 0)
    
    # kodd parity - add deltas
    saturation[kodd_mask] = np.clip(saturation[kodd_mask] + config.delta_s, 0, 1)
    lightness[kodd_mask] = np.clip(lightness[kodd_mask] + config.delta_l, 0, 1)
    
    # keven parity - subtract deltas
    saturation[keven_mask] = np.clip(saturation[keven_mask] - config.delta_s, 0, 1)
    lightness[keven_mask] = np.clip(lightness[keven_mask] - config.delta_l, 0, 1)
    
    # Convert HSL to sRGB using vectorized function
    return hsl_to_srgb_vectorized(base_hue, saturation, lightness)


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


def apply_smooth_coloring_vectorized(n: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray, config) -> np.ndarray:
    """
    Vectorized smooth coloring based on escape time.
    
    Args:
        n: Array of raw iteration counts
        z_real: Array of final z real parts
        z_imag: Array of final z imaginary parts
        config: FractalConfig instance
        
    Returns:
        Array of smooth escape time values
    """
    # Handle interior points
    interior_mask = (n >= config.max_iters)
    
    # Compute magnitude
    z_mag = np.sqrt(z_real**2 + z_imag**2)
    
    # Initialize result array
    result = n.astype(np.float32)
    
    # Apply smooth coloring only to appropriate points
    valid_escape_mask = (z_mag > config.bailout) & ~interior_mask
    
    if np.any(valid_escape_mask):
        # Use the exact same formula as scalar version for consistency
        # n + 1 - math.log(math.log(abs(z), 2), 2)
        z_mag_valid = z_mag[valid_escape_mask]
        
        # Replicate the exact scalar computation to avoid numerical differences
        # math.log(abs(z), 2) = math.log(abs(z)) / math.log(2)
        # math.log(math.log(abs(z), 2), 2) = math.log(math.log(abs(z)) / math.log(2)) / math.log(2)
        
        log_z = np.log(z_mag_valid)  # natural log of |z|
        log_z_base2 = log_z / np.log(2)  # log base 2 of |z|
        log_log_z_base2 = np.log(log_z_base2) / np.log(2)  # log base 2 of log base 2 of |z|
        
        result[valid_escape_mask] = n[valid_escape_mask] + 1 - log_log_z_base2
    
    # Set interior points to max_iters
    result[interior_mask] = float(config.max_iters)
    
    return result