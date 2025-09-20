"""
Configuration and validation for fractal rendering.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
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
    
    # Performance cache (not user-configurable)
    _cache: Optional[Dict[str, Any]] = None
    
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
        
        # Initialize cache
        self._cache = None
    
    def get_cache(self) -> Dict[str, Any]:
        """Get or create performance cache for invariant maps."""
        if self._cache is None:
            self._cache = self._build_cache()
        return self._cache
    
    def _build_cache(self) -> Dict[str, Any]:
        """Build cached invariant maps for performance."""
        from .crt import phi_field, keven_kodd_mask, iters_for_phi
        
        H, W = self.height, self.width
        
        # Cache 1: phi_map ∈ [0..47] and parity_map ∈ {0,1}
        # Use vectorized approach as suggested in issue
        xv = np.arange(W, dtype=np.int32)[None, :]
        yv = np.arange(H, dtype=np.int32)[:, None]
        phi_map = ((xv % 16) * 3 + (yv % 3)).astype(np.uint8)
        parity_map = ((xv + yv) & 1).astype(np.uint8)
        
        # Cache 2: complex coordinates grid (np.complex64)
        # Build centered X,Y and apply rotation/center in array ops
        dx = (xv - W/2).astype(np.float32) / self.scale
        dy = (yv - H/2).astype(np.float32) / self.scale
        
        # Apply rotation if specified
        if self.rotation != 0.0:
            angle = np.radians(self.rotation)
            cos_theta = np.cos(angle).astype(np.float32)
            sin_theta = np.sin(angle).astype(np.float32)
            xr = cos_theta * dx - sin_theta * dy
            yr = sin_theta * dx + cos_theta * dy
        else:
            xr, yr = dx, dy
        
        # Translate to center and create complex64 grid
        complex_coords = (xr + 1j*yr + (self.center[0] + 1j*self.center[1])).astype(np.complex64)
        
        # Cache 3: iters_map from 48-phase scheduler
        iters_map = np.zeros((H, W), dtype=np.int32)
        for phi in range(48):
            mask = (phi_map == phi)
            iters_map[mask] = iters_for_phi(phi, self.max_iters)
        
        # Cache 4: hue_by_phi[48] table 
        hue_by_phi = np.array([(self.base_hue + 360.0 * phi / 48) % 360.0 
                               for phi in range(48)], dtype=np.float32)
        
        return {
            'phi_map': phi_map,
            'parity_map': parity_map,  
            'complex_coords': complex_coords,
            'iters_map': iters_map,
            'hue_by_phi': hue_by_phi,
            # Cache key for invalidation
            'cache_key': self._get_cache_key()
        }
    
    def _get_cache_key(self) -> str:
        """Generate cache key based on parameters that affect invariant maps."""
        import hashlib
        
        # Only include parameters that affect the cached maps
        key_params = (
            self.width, self.height, self.scale, self.rotation,
            self.center, self.max_iters, self.base_hue
        )
        key_str = str(key_params)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def invalidate_cache(self):
        """Invalidate performance cache (e.g., when config changes)."""
        self._cache = None
    
    def pixel_to_complex(self, x: int, y: int) -> complex:
        """Convert pixel coordinates to complex plane coordinates.
        
        Note: This method is kept for compatibility but is slow.
        Use get_cache()['complex_coords'] for vectorized operations.
        """
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