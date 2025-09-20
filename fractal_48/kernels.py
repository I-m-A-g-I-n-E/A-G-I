"""
Fractal kernel implementations for the 48-manifold system.
"""

import numpy as np
import math
from typing import Tuple, Union


def mandelbrot(c: complex, max_iters: int, bailout: float = 4.0) -> Tuple[int, complex]:
    """
    Mandelbrot set iteration: z ← z^2 + c
    
    Args:
        c: Complex constant (pixel-mapped)
        max_iters: Maximum iterations
        bailout: Escape threshold (typically 2-4)
        
    Returns:
        (n, z): Escape iteration and final z value
    """
    z = 0+0j
    for n in range(max_iters):
        if z.real*z.real + z.imag*z.imag > bailout*bailout:
            return n, z
        z = z*z + c
    return max_iters, z


def julia(z0: complex, c: complex, max_iters: int, bailout: float = 4.0) -> Tuple[int, complex]:
    """
    Julia set iteration: z ← z^2 + c
    
    Args:
        z0: Initial z value (pixel-mapped)
        c: Julia constant
        max_iters: Maximum iterations
        bailout: Escape threshold
        
    Returns:
        (n, z): Escape iteration and final z value
    """
    z = z0
    for n in range(max_iters):
        if z.real*z.real + z.imag*z.imag > bailout*bailout:
            return n, z
        z = z*z + c
    return max_iters, z


def newton3(z0: complex, max_iters: int, eps: float = 1e-6) -> Tuple[int, int]:
    """
    Newton fractal for z^3 - 1 = 0
    
    Iteration: z ← z - (z^3 - 1) / (3z^2)
    
    Args:
        z0: Initial z value
        max_iters: Maximum iterations
        eps: Convergence threshold
        
    Returns:
        (steps, basin): Number of steps and basin index (0, 1, 2)
    """
    # Three roots of z^3 - 1 = 0
    roots = [
        1.0 + 0.0j,  # 1
        -0.5 + 0.8660254037844387j,  # exp(2πi/3)
        -0.5 - 0.8660254037844387j,  # exp(4πi/3)
    ]
    
    z = z0
    for step in range(max_iters):
        # Newton iteration for f(z) = z^3 - 1
        z3 = z * z * z
        dz = z * z * 3  # f'(z) = 3z^2
        
        if abs(dz) < eps:  # Avoid division by zero
            break
            
        z_new = z - (z3 - 1) / dz
        
        # Check convergence to any root
        for i, root in enumerate(roots):
            if abs(z_new - root) < eps:
                return step, i
        
        z = z_new
    
    # Default to root 0 if no convergence
    return max_iters, 0


def smooth_escape_time(n: int, z: complex, bailout: float = 4.0) -> float:
    """
    Calculate smooth escape time for continuous coloring.
    
    Args:
        n: Raw iteration count
        z: Final z value
        bailout: Escape threshold
        
    Returns:
        Smooth escape time
    """
    if abs(z) <= bailout:
        return float(n)
    
    # Smooth escape time formula
    return n + 1 - math.log(math.log(abs(z), 2), 2)


def cphi(phi: int, r: float = 0.4, theta: float = 0.0) -> complex:
    """
    Generate Julia constant based on phi (48-phase parameter sweep).
    
    Args:
        phi: Phase index [0, 47]
        r: Radius for parameter sweep
        theta: Additional phase offset
        
    Returns:
        Complex Julia constant
    """
    angle = 2 * math.pi * phi / 48 + theta
    return r * complex(math.cos(angle), math.sin(angle))