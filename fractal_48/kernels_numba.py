"""
Numba JIT-compiled fractal kernels for high-performance rendering.

This module provides Numba-accelerated versions of the fractal algorithms
with parallel execution support for multi-core CPUs.
"""

try:
    import numba as nb
    import numpy as np
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    
    @nb.njit(parallel=True, fastmath=True)
    def mandelbrot_numba(c_real, c_imag, iters_map, bailout2):
        """
        Numba-accelerated Mandelbrot set computation with parallel execution.
        
        Args:
            c_real: Real part of complex coordinates (H, W)
            c_imag: Imaginary part of complex coordinates (H, W)
            iters_map: Maximum iterations per pixel (H, W)
            bailout2: Squared bailout threshold
            
        Returns:
            (iters, zr_final, zi_final): Arrays with escape iterations and final z values
        """
        H, W = c_real.shape
        iters = np.zeros((H, W), dtype=np.int32)
        zr_final = np.zeros((H, W), dtype=np.float32)
        zi_final = np.zeros((H, W), dtype=np.float32)
        
        for y in nb.prange(H):
            for x in range(W):
                cr, ci = c_real[y, x], c_imag[y, x]
                zr, zi = 0.0, 0.0
                max_it = iters_map[y, x]
                
                for n in range(max_it):
                    zr2, zi2 = zr*zr, zi*zi
                    if zr2 + zi2 > bailout2:
                        iters[y, x] = n
                        zr_final[y, x], zi_final[y, x] = zr, zi
                        break
                    zi = 2*zr*zi + ci
                    zr = zr2 - zi2 + cr
                else:
                    iters[y, x] = max_it
                    zr_final[y, x], zi_final[y, x] = zr, zi
                    
        return iters, zr_final, zi_final

    @nb.njit(parallel=True, fastmath=True)
    def julia_numba(z0_real, z0_imag, c_real, c_imag, iters_map, bailout2):
        """
        Numba-accelerated Julia set computation with parallel execution.
        
        Args:
            z0_real: Real part of initial z values (H, W)
            z0_imag: Imaginary part of initial z values (H, W)
            c_real: Real part of Julia constant (H, W) or scalar
            c_imag: Imaginary part of Julia constant (H, W) or scalar
            iters_map: Maximum iterations per pixel (H, W)
            bailout2: Squared bailout threshold
            
        Returns:
            (iters, zr_final, zi_final): Arrays with escape iterations and final z values
        """
        H, W = z0_real.shape
        iters = np.zeros((H, W), dtype=np.int32)
        zr_final = np.zeros((H, W), dtype=np.float32)
        zi_final = np.zeros((H, W), dtype=np.float32)
        
        for y in nb.prange(H):
            for x in range(W):
                zr, zi = z0_real[y, x], z0_imag[y, x]
                
                # Handle both scalar and array c values
                if c_real.ndim == 0:  # scalar
                    cr, ci = c_real, c_imag
                else:  # array
                    cr, ci = c_real[y, x], c_imag[y, x]
                
                max_it = iters_map[y, x]
                
                for n in range(max_it):
                    zr2, zi2 = zr*zr, zi*zi
                    if zr2 + zi2 > bailout2:
                        iters[y, x] = n
                        zr_final[y, x], zi_final[y, x] = zr, zi
                        break
                    zi = 2*zr*zi + ci
                    zr = zr2 - zi2 + cr
                else:
                    iters[y, x] = max_it
                    zr_final[y, x], zi_final[y, x] = zr, zi
                    
        return iters, zr_final, zi_final

    @nb.njit(parallel=True, fastmath=True)
    def newton3_numba(z0_real, z0_imag, iters_map, eps=1e-6):
        """
        Numba-accelerated Newton fractal for z^3 - 1 = 0 with parallel execution.
        
        Args:
            z0_real: Real part of initial z values (H, W)
            z0_imag: Imaginary part of initial z values (H, W)
            iters_map: Maximum iterations per pixel (H, W)
            eps: Convergence threshold
            
        Returns:
            (steps, basin): Arrays with convergence steps and basin index
        """
        H, W = z0_real.shape
        steps = np.zeros((H, W), dtype=np.int32)
        basin = np.zeros((H, W), dtype=np.int32)
        
        # Three roots of z^3 - 1 = 0
        roots_real = np.array([1.0, -0.5, -0.5], dtype=np.float32)
        roots_imag = np.array([0.0, 0.8660254037844387, -0.8660254037844387], dtype=np.float32)
        
        for y in nb.prange(H):
            for x in range(W):
                zr, zi = z0_real[y, x], z0_imag[y, x]
                max_it = iters_map[y, x]
                
                for step in range(max_it):
                    # Newton iteration for f(z) = z^3 - 1
                    # f(z) = z^3 - 1, f'(z) = 3z^2
                    z2r = zr*zr - zi*zi  # z^2 real part
                    z2i = 2*zr*zi        # z^2 imag part
                    z3r = z2r*zr - z2i*zi - 1.0  # z^3 - 1 real part
                    z3i = z2r*zi + z2i*zr        # z^3 - 1 imag part
                    
                    # 3z^2 for derivative
                    dz_r = 3*z2r
                    dz_i = 3*z2i
                    
                    # Check for division by zero
                    dz_mag2 = dz_r*dz_r + dz_i*dz_i
                    if dz_mag2 < eps*eps:
                        break
                    
                    # Newton update: z_new = z - (z^3 - 1) / (3z^2)
                    # Complex division: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
                    inv_dz_mag2 = 1.0 / dz_mag2
                    dr = (z3r*dz_r + z3i*dz_i) * inv_dz_mag2
                    di = (z3i*dz_r - z3r*dz_i) * inv_dz_mag2
                    
                    zr_new = zr - dr
                    zi_new = zi - di
                    
                    # Check convergence to any root
                    converged = False
                    for i in range(3):
                        dr = zr_new - roots_real[i]
                        di = zi_new - roots_imag[i]
                        if dr*dr + di*di < eps*eps:
                            steps[y, x] = step
                            basin[y, x] = i
                            converged = True
                            break
                    
                    if converged:
                        break
                    
                    zr, zi = zr_new, zi_new
                else:
                    # No convergence, default to root 0
                    steps[y, x] = max_it
                    basin[y, x] = 0
                    
        return steps, basin

else:
    # Fallback functions when Numba is not available
    def mandelbrot_numba(*args, **kwargs):
        raise ImportError("Numba is not available. Install numba>=0.56.0 or use numpy backend.")
    
    def julia_numba(*args, **kwargs):
        raise ImportError("Numba is not available. Install numba>=0.56.0 or use numpy backend.")
    
    def newton3_numba(*args, **kwargs):
        raise ImportError("Numba is not available. Install numba>=0.56.0 or use numpy backend.")