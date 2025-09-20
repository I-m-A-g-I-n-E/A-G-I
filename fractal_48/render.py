"""
Core rendering engine for fractal generation.
"""

import numpy as np
from typing import List, Optional
import math
import time

from .config import FractalConfig
from .crt import crt_phase48, phi_field, keven_kodd_mask, iters_for_phi
from .kernels import mandelbrot, julia, newton3, cphi, smooth_escape_time
from .color import palette48, palette_newton, apply_smooth_coloring
from .perms import apply_perm_for_frame


def render_frame(config: FractalConfig, frame_idx: int = 0) -> np.ndarray:
    """
    Render a single fractal frame according to the 48-manifold specifications.
    
    Args:
        config: Fractal configuration
        frame_idx: Frame index for animation (0-47)
        
    Returns:
        RGB image array of shape (H, W, 3) with values in [0, 1]
    """
    H, W = config.height, config.width
    img = np.zeros((H, W, 3), dtype=np.float32)
    
    # Use cached invariant maps for performance
    cache = config.get_cache()
    phi_map = cache['phi_map']
    parity_map = cache['parity_map']
    complex_coords = cache['complex_coords']
    
    # Render based on kernel type
    if config.kernel == "mandelbrot":
        img = _render_mandelbrot(complex_coords, phi_map, parity_map, config)
    elif config.kernel == "julia":
        img = _render_julia(complex_coords, phi_map, parity_map, config, frame_idx)
    elif config.kernel == "newton":
        img = _render_newton(complex_coords, phi_map, parity_map, config)
    else:
        raise ValueError(f"Unknown kernel: {config.kernel}")
    
    # Apply reversible permutation for animation
    if config.animate:
        img = apply_perm_for_frame(img, frame_idx, config)
    
    return img


def _render_mandelbrot(complex_coords: np.ndarray, phi_map: np.ndarray, 
                      parity_map: np.ndarray, config: FractalConfig) -> np.ndarray:
    """Render Mandelbrot set."""
    H, W = complex_coords.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    
    # Get cached iteration map
    cache = config.get_cache()
    iters_map = cache['iters_map']
    
    for y in range(H):
        for x in range(W):
            c = complex_coords[y, x]
            phi = phi_map[y, x]
            parity = int(parity_map[y, x])
            
            # Use cached iterations based on phi
            iters = iters_map[y, x]
            
            # Compute Mandelbrot iteration
            n, z = mandelbrot(c, iters, config.bailout)
            
            # Apply smooth coloring
            n_smooth = apply_smooth_coloring(n, z, config)
            
            # Generate color with parity gating
            r, g, b = palette48(n_smooth, phi, parity, config)
            img[y, x] = [r, g, b]
    
    return img


def _render_julia(complex_coords: np.ndarray, phi_map: np.ndarray,
                 parity_map: np.ndarray, config: FractalConfig, frame_idx: int) -> np.ndarray:
    """Render Julia set with 48-phase parameter sweep."""
    H, W = complex_coords.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    
    # Get cached iteration map
    cache = config.get_cache()
    iters_map = cache['iters_map']
    
    for y in range(H):
        for x in range(W):
            z0 = complex_coords[y, x]
            phi = phi_map[y, x]
            parity = int(parity_map[y, x])
            
            # Julia constant varies with phi and frame
            frame_theta = 2 * math.pi * frame_idx / config.loop_frames
            c = cphi(phi, config.julia_r, config.julia_theta + frame_theta)
            
            # Use cached iterations based on phi
            iters = iters_map[y, x]
            
            # Compute Julia iteration
            n, z = julia(z0, c, iters, config.bailout)
            
            # Apply smooth coloring
            n_smooth = apply_smooth_coloring(n, z, config)
            
            # Generate color with parity gating
            r, g, b = palette48(n_smooth, phi, parity, config)
            img[y, x] = [r, g, b]
    
    return img


def _render_newton(complex_coords: np.ndarray, phi_map: np.ndarray,
                  parity_map: np.ndarray, config: FractalConfig) -> np.ndarray:
    """Render Newton fractal for z^3 - 1."""
    H, W = complex_coords.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    
    # Get cached iteration map
    cache = config.get_cache()
    iters_map = cache['iters_map']
    
    for y in range(H):
        for x in range(W):
            z0 = complex_coords[y, x]
            phi = phi_map[y, x]
            parity = int(parity_map[y, x])
            
            # Use cached iterations based on phi
            iters = iters_map[y, x]
            
            # Compute Newton iteration
            steps, basin = newton3(z0, iters, eps=1e-6)
            
            # Generate color based on basin and convergence
            r, g, b = palette_newton(basin, steps, phi, parity, config)
            img[y, x] = [r, g, b]
    
    return img


def render_loop(config: FractalConfig) -> List[np.ndarray]:
    """
    Render a 48-frame animation loop.
    
    Args:
        config: Fractal configuration with animate=True
        
    Returns:
        List of 48 RGB image arrays
    """
    if not config.animate:
        return [render_frame(config, 0)]
    
    frames = []
    print(f"Rendering {config.loop_frames} frames...")
    
    start_time = time.time()
    for frame_idx in range(config.loop_frames):
        print(f"  Frame {frame_idx + 1}/{config.loop_frames}", end="\r")
        frame = render_frame(config, frame_idx)
        frames.append(frame)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s ({elapsed/config.loop_frames:.3f}s/frame)")
    
    return frames


def benchmark_render_performance(config: FractalConfig) -> dict:
    """
    Benchmark rendering performance for different settings.
    
    Args:
        config: Base configuration
        
    Returns:
        Performance metrics dictionary
    """
    metrics = {}
    
    # Single frame benchmark
    start_time = time.time()
    frame = render_frame(config, 0)
    single_frame_time = time.time() - start_time
    
    pixels_per_second = (config.width * config.height) / single_frame_time
    
    metrics.update({
        'single_frame_time': single_frame_time,
        'pixels_per_second': pixels_per_second,
        'estimated_48_frame_time': single_frame_time * 48,
        'width': config.width,
        'height': config.height,
        'kernel': config.kernel,
        'max_iters': config.max_iters
    })
    
    return metrics