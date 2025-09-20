"""
Fractal-48: Colored fractal generator on the 2^4×3 (48) manifold.

This module provides an interactive and CLI-capable fractal renderer that rides the 48-manifold:
- 48-aligned canvas with CRT-based 48-phase scheduler and keven/kodd parity gating
- Reversible 2×/3× space-to-depth animations using manifold.py
- Kernels: Mandelbrot, Julia (48-phase parameter sweep), Newton fractal for z^3−1
- Smooth, parity-aware coloring with export capabilities and provenance JSON
"""

from .config import FractalConfig
from .render import render_frame, render_loop

__version__ = "0.1.0"
__all__ = ["FractalConfig", "render_frame", "render_loop"]