"""
Configuration and validation for fractal rendering.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FractalConfig:
    """Configuration for fractal rendering with 48-manifold constraints."""
    
    # Canvas dimensions (must be 48-aligned)
    width: int = 1920
    height: int = 1152
    
    # Fractal parameters
    kernel: str = "mandelbrot"  # "mandelbrot", "julia", "newton"
    center: Tuple[float, float] = (-0.75, 0.0)
    scale: float = 600.0  # pixels per unit
    rotation: float = 0.0  # rotation in degrees
    
    # Iteration parameters
    max_iters: int = 1000
    bailout: float = 4.0
    
    # Julia-specific parameters
    julia_r: float = 0.4
    julia_theta: float = 0.0
    
    # Color parameters
    palette_mode: str = "smooth"
    base_hue: float = 210.0
    delta_s: float = 0.05  # saturation modulation for parity
    delta_l: float = 0.04  # lightness modulation for parity
    
    # Animation parameters
    animate: bool = False
    loop_frames: int = 48
    
    # Output parameters
    output_path: str = "fractal_48_output"
    
    def __post_init__(self):
        """Validate 48-alignment and other constraints."""
        if self.width % 48 != 0:
            raise ValueError(f"Width must be 48-aligned, got {self.width}")
        if self.height % 48 != 0:
            raise ValueError(f"Height must be 48-aligned, got {self.height}")
        
        if self.kernel not in ["mandelbrot", "julia", "newton"]:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        if self.loop_frames != 48:
            raise ValueError(f"Loop frames must be 48, got {self.loop_frames}")
    
    def pixel_to_complex(self, x: int, y: int) -> complex:
        """Convert pixel coordinates to complex plane coordinates."""
        # Center the coordinate system
        px = (x - self.width / 2) / self.scale
        py = (y - self.height / 2) / self.scale
        
        # Apply rotation if specified
        if self.rotation != 0.0:
            angle = np.radians(self.rotation)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            px_rot = px * cos_a - py * sin_a
            py_rot = px * sin_a + py * cos_a
            px, py = px_rot, py_rot
        
        # Translate to center
        return complex(px + self.center[0], py + self.center[1])